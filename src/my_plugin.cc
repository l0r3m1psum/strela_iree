#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

#include "StrelaDialect.h.inc"
#define GET_OP_CLASSES
#include "StrelaOps.h.inc"
#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
#include "StrelaDialect.cpp.inc"

namespace mlir::strela {
  void StrelaDialect::initialize() {
    addOperations<
#define GET_OP_LIST
#include "StrelaOps.cpp.inc"
    >();
  }
}

/*
 * +--+--+--+--+
 * | 0| 1| 2| 3|
 * +--+--+--+--+
 * | 4| 5| 6| 7|
 * +--+--+--+--+
 * | 8| 9|10|11|
 * +--+--+--+--+
 * |12|13|14|15|
 * +--+--+--+--+
 */
// Constants are -1=0xFFFFFFFF and delays are set to 0xDDEE
static const std::array<uint32_t, 5*4*4> centered_matmul_kernel {
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 12
  0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 8
  0x00000041, 0x02000000, 0x00000000, 0x00000000, 0x00000000, // 4
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 0

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 9
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 5
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 1

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 10
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 6
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 2

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 11
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 7
  0x00000201, 0x020C0300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 3
};

static std::mutex printMutex;

static void
print(mlir::ModuleOp moduleOp) {
  std::lock_guard<std::mutex> lock(printMutex);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  moduleOp.print(llvm::errs(), flags);
  llvm::errs() << '\n';
}

static void
print(mlir::func::FuncOp funcOp) {
  print(funcOp->getParentOfType<mlir::ModuleOp>());
}

static mlir::LogicalResult
allInputsInteger32(mlir::linalg::GenericOp genericOp) {
  for (mlir::Value input : genericOp.getDpsInputs()) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.getElementType().isInteger(32)) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

static mlir::LogicalResult
isStrelaLinalgAdd(mlir::linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return mlir::failure();
  }

  if (failed(allInputsInteger32(genericOp))) {
    return mlir::failure();
  }

  if (!genericOp.hasPureTensorSemantics()) {
    return mlir::failure();
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(), mlir::linalg::isParallelIterator)) {
    return mlir::failure();
  }

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return mlir::failure();
  }

  auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&body->front());
  if (!addOp) {
    return mlir::failure();
  }

  auto yieldOp = mlir::dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != addOp.getResult()) {
    return mlir::failure();
  }

  return mlir::success();
}

static mlir::LogicalResult
isStrelaLinalgRelu(mlir::linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1) {
    return mlir::failure();
  }

  if (failed(allInputsInteger32(genericOp))) {
    return mlir::failure();
  }

  if (!genericOp.hasPureTensorSemantics()) {
    return mlir::failure();
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(), mlir::linalg::isParallelIterator)) {
    return mlir::failure();
  }

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 3) {
    return mlir::failure();
  }

  auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(&body->front());
  if (!constOp) {
    return mlir::failure();
  } else {
    auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(constOp.getValue());
    if (!intAttr || intAttr.getInt() != 0) {
      return mlir::failure();
    }
  }

  auto maxOp = mlir::dyn_cast<mlir::arith::MaxSIOp>(std::next(body->begin()));
  if (!maxOp) {
    return mlir::failure();
  }

  {
    mlir::Value inputElem = body->getArgument(0);
    mlir::Value zeroVal = constOp.getResult();
    bool isReluOperands = (maxOp.getLhs() == inputElem && maxOp.getRhs() == zeroVal) ||
                          (maxOp.getLhs() == zeroVal && maxOp.getRhs() == inputElem);
    if (!isReluOperands) {
      return mlir::failure();
    }
  }

  auto yieldOp = mlir::dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != maxOp.getResult()) {
    return mlir::failure();
  }

  return mlir::success();
}

using namespace mlir;
using namespace mlir::iree_compiler;

// TODO: implement loop fission for this two patterns.

struct ConvertLinalgAddToStrela : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(
    linalg::GenericOp genericOp, PatternRewriter &rewriter
  ) const override {
    LogicalResult result = isStrelaLinalgAdd(genericOp);

    if (succeeded(result)) {
      rewriter.replaceOpWithNewOp<strela::AddOp>(
        genericOp,
        genericOp.getResultTypes(),
        genericOp.getDpsInputs()[0],
        genericOp.getDpsInputs()[1]
      );
    }

    return result;
  }
};

struct ConvertLinalgReluToStrela : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(
    linalg::GenericOp genericOp, PatternRewriter &rewriter
  ) const override {
    LogicalResult result = isStrelaLinalgRelu(genericOp);

    if (succeeded(result)) {
      rewriter.replaceOpWithNewOp<strela::ReluOp>(
        genericOp,
        genericOp.getResultTypes(),
        genericOp.getDpsInputs()[0]
      );
    }

    return result;
  }
};

struct LinalgToStrelaPass
    : public PassWrapper<LinalgToStrelaPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToStrelaPass)

  StringRef getArgument() const override { return "iree-strela-convert-linalg"; }
  StringRef getDescription() const override {
    return "Converts linalg ops to strela backend ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<strela::StrelaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ConvertLinalgAddToStrela, ConvertLinalgReluToStrela>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }

    print(funcOp);
  }
};

namespace mlir::iree_compiler {

struct StrelaTargetBackend : public IREE::HAL::TargetBackend {

  std::string getLegacyDefaultDeviceID() const override { return "strela"; }

  void buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetAttr exectutableTargetAttr,
    OpPassManager &passManager
  ) override {
    OpPassManager &modulePassManager = passManager.nest<ModuleOp>();

    modulePassManager.addNestedPass<func::FuncOp>(
      std::make_unique<LinalgToStrelaPass>()
    );
  }

  // TODO: how does this relate to StrelaTargetDevice::getDefaultDeviceTarget?
  // Can some code between the two be reused?
  void
  getDefaultExecutableTargets(
    MLIRContext *context,
    StringRef deviceID,
    DictionaryAttr deviceConfigAttr,
    SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs
  ) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto executableTargetAttr = b.getAttr<IREE::HAL::ExecutableTargetAttr>(
      b.getStringAttr("strela"), b.getStringAttr("custom"), b.getDictionaryAttr(configItems)
    );
    executableTargetAttrs.push_back(executableTargetAttr);
  }

  LogicalResult serializeExecutable(
    const SerializationOptions &options,
    IREE::HAL::ExecutableVariantOp variantOp,
    OpBuilder &executableBuilder
  ) override {
    uint32_t detected_opcode = 0;

    mlir::ModuleOp innerModule = variantOp.getInnerModule();
    if (innerModule) {
      innerModule.walk([&detected_opcode](Operation *op) {
        if (isa<strela::AddOp>(op)) {
          detected_opcode = 1; // e.g. 1 = ADD
        } else if (isa<strela::ReluOp>(op)) {
          detected_opcode = 2; // e.g. 2 = RELU
        }
      });
    }

    if (detected_opcode != 0) {
      const uint8_t* byte_ptr = reinterpret_cast<const uint8_t *>(centered_matmul_kernel.data());
      std::vector<uint8_t> binary_payload(byte_ptr, byte_ptr + sizeof centered_matmul_kernel);

      IREE::HAL::ExecutableBinaryOp::create(
        executableBuilder,
        variantOp.getLoc(),
        variantOp.getSymNameAttr(),         // Inherit the symbol name ("strela")
        variantOp.getTarget().getFormat(),  // Inherit the format ("custom")
        binary_payload
      );
      return success();
    } else {
      // FIXME: when this is reached compilation fails. Region computable by
      // STRELA should be created with stream.affinity
      return failure();
    }

  }
};

struct StrelaTargetDevice : public IREE::HAL::TargetDevice {

  IREE::HAL::DeviceTargetAttr
  getDefaultDeviceTarget(
    MLIRContext *context,
    const IREE::HAL::TargetRegistry &targetRegistry
  ) const override {
    mlir::Builder b(context);

    // With
    // iree-compile --iree-plugin=example2  --iree-hal-target-device=strela --compile-to=stream 3rdparty/iree/samples/models/simple_abs.mlir   -o simple_abs_hal.mlir
    // this #stream.resource_config appears in the MLIR source.
    auto resourceConfigAttr = b.getAttr<IREE::Stream::ResourceConfigAttr>(
      // TODO: put real numbers...
      /*max_allocation_size=*/ 1ull * 1024 * 1024 * 1024,
      /*min_buffer_offset_alignment=*/ 256,
      /*max_buffer_range=*/ 256,
      /*min_buffer_range_alignment=*/ 256,
      /*index_bits=*/ 0,
      /*alias_mutable_bindings=*/ false,
      /*memory_model=*/ IREE::Stream::MemoryModel::Unified
    );

    SmallVector<NamedAttribute> configItems;
    configItems.emplace_back(
      b.getStringAttr("stream.resource_config"), resourceConfigAttr
    );

    auto configAttr = b.getDictionaryAttr(configItems);
    auto deviceID = b.getStringAttr("strela");

    // With
    // iree-compile --iree-plugin=example2  --iree-hal-target-device=strela --compile-to=hal 3rdparty/iree/samples/models/simple_abs.mlir   -o simple_abs_hal.mlir
    // this #hal.executable.target appears in the MLIR source.
    auto executableTargetAttr = b.getAttr<IREE::HAL::ExecutableTargetAttr>(
      /*backend=*/ deviceID,
      /*format=*/ b.getStringAttr("custom"),
      /*configuration=*/ b.getDictionaryAttr({})
    );

    return IREE::HAL::DeviceTargetAttr::get(
      context,
      /*deviceID=*/ deviceID,
      /*configuration=*/ configAttr,
      /*executable_targets=*/ {executableTargetAttr}
    );
  }
};

} // namespace mlir::iree_compiler

namespace {

struct Conv2DMatmulAnalysis {
  int32_t z_x_val;
  int32_t z_w_val;
  uint16_t in_chan;
  SmallVector<Value> operands;
};

static LogicalResult
isMatmulEquivalent(
  linalg::Conv2DNhwcHwcfQOp convOp, Conv2DMatmulAnalysis& analysis
) {
  Conv2DMatmulAnalysis localAnalysis{};

  Value x = convOp.getDpsInputOperand(0)->get();
  Value w = convOp.getDpsInputOperand(1)->get();
  Value z_x = convOp.getDpsInputOperand(2)->get();
  Value z_w = convOp.getDpsInputOperand(3)->get();
  Value y = convOp.getDpsInits()[0];

  std::array<Value, 2> zero_points {z_x, z_w};
  std::array<int32_t, 2> zero_point_vals {0, 0};
  for (size_t i = 0; i < zero_points.size(); ++i) {
    Value zp = zero_points[i];
    if (!zp.getType().isInteger(32)) return failure();
    auto constantOp = zp.getDefiningOp<arith::ConstantOp>();
    if (!constantOp) return failure();
    auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
    if (!intAttr) return failure();
    zero_point_vals[i] = static_cast<int32_t>(intAttr.getInt());
  }
  localAnalysis.z_x_val = zero_point_vals[0];
  localAnalysis.z_w_val = zero_point_vals[1];

  auto isAllOnes = [](DenseIntElementsAttr attr) {
    if (!attr) return false;
    return llvm::all_of(attr.getValues<int64_t>(), [](int64_t v) { return v == 1; });
  };
  if (!isAllOnes(convOp.getStrides()) || !isAllOnes(convOp.getDilations())) {
    return failure();
  }

  constexpr int wH = 0, wW = 1, wC = 2, wF = 3;

  auto wType = dyn_cast<RankedTensorType>(w.getType());
  if (!wType.getElementType().isInteger(8)) return failure();
  if (wType.getDimSize(wH) != 1 || wType.getDimSize(wW) != 1) {
    return failure();
  }

  int64_t wCDimSize = wType.getDimSize(wC);
  if (ShapedType::isDynamic(wCDimSize)
    || wCDimSize > std::numeric_limits<uint16_t>::max()) {
    return failure();
  }
  localAnalysis.in_chan = static_cast<uint16_t>(wCDimSize);

  constexpr int xH = 1, xW = 2;

  auto xType = dyn_cast<RankedTensorType>(x.getType());
  if (!xType.getElementType().isInteger(8)) return failure();
  if (xType.getDimSize(xH) != 1 || xType.getDimSize(xW) != 1) {
    return failure();
  }

  auto yType = dyn_cast<RankedTensorType>(y.getType());
  if (!yType.getElementType().isInteger(32)) return failure();

  auto transposeOp = w.getDefiningOp<linalg::TransposeOp>();
  if (!transposeOp) return failure();

  // Given that the weight layout is HWCF if the transposition has put the
  // output features F before the input channels C this means (since the
  // height H and width W are both 1 they do not matter for stride
  // calculations) that the operation is a matrix multiplication with
  // transposed RHS i.e. we can load from the RHS with unit stride.
  //
  // To detect this we just index in the permutation to detect where the F and
  // C dimensions were originally e.g. for the permutation [1,2,3,0] at index
  // 3=F there is 0 and this means that before applying F was at the first
  // dimension; for the permutation [0,1,2,3] at index 3=F there is 3 this
  // means that F has not been moved.
  ArrayRef<int64_t> permutation = transposeOp.getPermutation();
  assert(permutation.size() == 4);
  int64_t cBefore = permutation[wC];
  int64_t fBefore = permutation[wF];

  Value wToPass;
  if (fBefore < cBefore) {
    wToPass = transposeOp.getInput();
  } else {
    return failure();
  }

  localAnalysis.operands = {x, z_x, wToPass, z_w, y};

  analysis = localAnalysis;

  return success();
}

static Type
getDynamicTensorType(Type t) {
  if (auto rankedType = dyn_cast<RankedTensorType>(t)) {
    SmallVector<int64_t> dynShape(rankedType.getRank(), ShapedType::kDynamic);
    return RankedTensorType::get(dynShape, rankedType.getElementType());
  }
  assert(t.isIntOrIndexOrFloat() && "Expected a RankedTensorType or a scalar type");
  return t;
}

struct Conv2DOffload : public OpRewritePattern<linalg::Conv2DNhwcHwcfQOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    linalg::Conv2DNhwcHwcfQOp convOp,
    PatternRewriter& rewriter
  ) const override {

    Conv2DMatmulAnalysis analysis{};
    if (failed(isMatmulEquivalent(convOp, analysis))) {
      return failure();
    }

    convOp.emitRemark("Offloading Conv2D equivalent to Matmul");

    ModuleOp moduleOp = convOp->getParentOfType<ModuleOp>();
    Location loc = convOp.getLoc();
    Type i32 = rewriter.getI32Type();

    SymbolTable symbolTable(moduleOp);

    StringRef initFuncName = "custom.init_accelerator";
    auto initFuncDecl = moduleOp.lookupSymbol<func::FuncOp>(initFuncName);
    if (!initFuncDecl) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto unrankedTensorType = RankedTensorType::get({ShapedType::kDynamic}, i32);
      auto initFuncType = rewriter.getFunctionType({unrankedTensorType}, {i32});
      initFuncDecl = func::FuncOp::create(rewriter, loc, initFuncName, initFuncType);
      initFuncDecl.setPrivate();

      symbolTable.insert(initFuncDecl);
    }

    IREE::Util::GlobalOp globalDecl;
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      StringRef globalName = "kernel_handle";
      bool isMutable = true;
      globalDecl = IREE::Util::GlobalOp::create(
        rewriter, loc, globalName, isMutable, i32
      );
      globalDecl.setPrivate();

      symbolTable.insert(globalDecl);
    }

    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(moduleOp.getBody());

      auto initOp = IREE::Util::InitializerOp::create(rewriter, loc);
      rewriter.createBlock(&initOp.getBody());

      auto kernel = centered_matmul_kernel;
      kernel.at(5*(3+0) + 4) = analysis.z_x_val;
      kernel.at(5*(3+4) + 4) = kernel.at(5*(3+8) + 4) = kernel.at(5*(3+12) + 4) = analysis.z_w_val;
      kernel.at(5*(5+0) + 2) = kernel.at(5*(5+4) + 2) = kernel.at(5*(5+8)  + 2) = (analysis.in_chan << 16) | 0x0080;

      auto kernelTensorType = RankedTensorType::get(
        {static_cast<int64_t>(centered_matmul_kernel.size())}, i32
      );
      auto denseAttr = DenseElementsAttr::get(kernelTensorType, ArrayRef(centered_matmul_kernel));
      Value kernelBlob = arith::ConstantOp::create(rewriter, loc, kernelTensorType, denseAttr);

      auto unrankedTensorType = RankedTensorType::get({ShapedType::kDynamic}, i32);
      Value castedBlob = tensor::CastOp::create(rewriter, loc, unrankedTensorType, kernelBlob);
      auto callOp = func::CallOp::create(rewriter, loc, initFuncDecl, ValueRange{castedBlob});

      Value callResult = callOp.getResult(0);
      IREE::Util::GlobalStoreOp::create(rewriter, loc, callResult, globalDecl);

      IREE::Util::ReturnOp::create(rewriter, loc);
    }

    auto collapseTo2D = [&](Value val) -> Value {
      auto type = dyn_cast<RankedTensorType>(val.getType());
      ReassociationIndices merge_dims = {0, 1, 2}, keep_dims = {3};
      SmallVector<ReassociationIndices> reassoc = {merge_dims, keep_dims};

      int64_t dim0 = ShapedType::kDynamic;
      bool shape_is_comptime_know = !type.isDynamicDim(0)
        && !type.isDynamicDim(1) && !type.isDynamicDim(2);
      if (shape_is_comptime_know) {
        dim0 = type.getDimSize(0) * type.getDimSize(1) * type.getDimSize(2);
      }

      auto collapsedType = RankedTensorType::get({dim0, type.getDimSize(3)}, type.getElementType());
      return tensor::CollapseShapeOp::create(rewriter, loc, collapsedType, val, reassoc);
    };

    analysis.operands[0] = collapseTo2D(analysis.operands[0]); // x
    analysis.operands[2] = collapseTo2D(analysis.operands[2]); // wToPass
    analysis.operands[4] = collapseTo2D(analysis.operands[4]); // y

    StringRef gemmFuncName = "custom.my_centered_gemm";

    auto funcDecl = moduleOp.lookupSymbol<func::FuncOp>(gemmFuncName);
    SmallVector<Type> inputTypes = {i32};
    std::transform(
      analysis.operands.begin(), analysis.operands.end(),
      std::back_inserter(inputTypes), [](Value v){ return v.getType(); }
    );
    Type origResultType = convOp.getResultTypes()[0]; // The final 4D output type
    Type callResultType = analysis.operands[4].getType(); // The 2D collapsed type of 'y'

    SmallVector<Type> dynamicInputTypes;
    for (Type t : inputTypes) dynamicInputTypes.push_back(getDynamicTensorType(t));

    Type dynamicResultType = getDynamicTensorType(callResultType);

    if (!funcDecl) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = rewriter.getFunctionType(dynamicInputTypes, {dynamicResultType});
      funcDecl = func::FuncOp::create(rewriter, loc, gemmFuncName, funcType);
      funcDecl.setPrivate();
    }

    auto loadOp = IREE::Util::GlobalLoadOp::create(rewriter, loc, globalDecl);
    Value loadOpRes = loadOp.getResult();

    SmallVector<Value> originalOperands = {loadOpRes};
    std::copy(
      analysis.operands.begin(), analysis.operands.end(),
      std::back_inserter(originalOperands)
    );

    // We make tensor shape dynamic (i.e. remove shape information)
    SmallVector<Value> castedOperands;
    for (size_t i = 0; i < originalOperands.size(); ++i) {
      if (originalOperands[i].getType() != dynamicInputTypes[i]) {
        castedOperands.push_back(
          tensor::CastOp::create(
            rewriter, convOp.getLoc(), dynamicInputTypes[i], originalOperands[i]
          )
        );
      } else {
        castedOperands.push_back(originalOperands[i]);
      }
    }

    auto callOp = func::CallOp::create(rewriter, loc, funcDecl, castedOperands);

    Value callResult = callOp.getResult(0);

    // We add back the static shape information
    Value castedCallResult = callResult;
    if (callResult.getType() != callResultType) {
      castedCallResult = tensor::CastOp::create(
        rewriter, convOp.getLoc(), callResultType, callResult
      );
    }

    SmallVector<ReassociationIndices> reassoc = {{0, 1, 2}, {3}};
    Value finalResult = tensor::ExpandShapeOp::create(
      rewriter, loc, origResultType, castedCallResult, reassoc
    );

    rewriter.replaceOp(convOp, finalResult);

    return success();
  }
};

// NOTE: should OperationPass<T> be OperationPass<ModuleOp>?
struct MyFusionPass : public PassWrapper<MyFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyFusionPass)

  StringRef getArgument() const override { return "iree-example2-fusion-pass"; }
  StringRef getDescription() const override { return "Fuses centering subtractions into a GEMM ukernel"; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    llvm::errs() << "DEBUG: MyFusionPass is running on operation: "
               << funcOp.getName() << "\n";
    RewritePatternSet patterns(context);
    patterns.add<Conv2DOffload>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

struct DoubleRoundRewriter : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    tosa::RescaleOp rescaleOp,
    PatternRewriter& rewriter
  ) const override {

    if (rescaleOp.getRoundingMode() != mlir::tosa::RoundingMode::DOUBLE_ROUND) {
      return failure();
    }

    rescaleOp.emitWarning("Converting from DOUBLE_ROUND to SINGLE_ROUND");

    rewriter.modifyOpInPlace(rescaleOp, [&]() {
      rescaleOp->setAttr(
        "rounding_mode",
        mlir::tosa::RoundingModeAttr::get(
          rewriter.getContext(),
          mlir::tosa::RoundingMode::SINGLE_ROUND
        )
      );
    });

    return success();
  }
};

// TODO: does it make sense to support multiple dynamic leading dynamic
// dimensions (i.e. ?x?x1x1x128x)? The idea it would be to map each of them to a
// scf.for loop but I am not sure of the semantic of tosa.rescale in this case.
// Probably a better solution is to wrap this in a tensor.collapse_shape
// (flatten) and tensor.expand_shape (reshape) and then use the same loop as
// now.
struct DynamicBatchRewriter : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    tosa::RescaleOp rescaleOp,
    PatternRewriter& rewriter
  ) const override {

    auto inputType = dyn_cast<RankedTensorType>(rescaleOp.getInput().getType());
    if (!inputType || inputType.getRank() == 0 || !inputType.isDynamicDim(0)) {
      return failure();
    }

    rescaleOp.emitRemark("Lowering dynamic batch to an scf.for loop");

    Location loc = rescaleOp.getLoc();
    int64_t rank = inputType.getRank();
    auto outputType = cast<RankedTensorType>(rescaleOp.getResult().getType());

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value batchSize = tensor::DimOp::create(rewriter, loc, rescaleOp.getInput(), c0);

    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (outputType.isDynamicDim(i)) {
        Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value dynSize = tensor::DimOp::create(rewriter, loc, rescaleOp.getInput(), dimIdx);
        dynSizes.push_back(dynSize);
      }
    }
    Value initTensor = tensor::EmptyOp::create(
      rewriter, loc, outputType.getShape(), outputType.getElementType(), dynSizes
    );

    Value lowerBound = c0;
    Value upperBound = batchSize;
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto forOp = scf::ForOp::create(
      rewriter, loc, lowerBound, upperBound, step, ValueRange{initTensor},
      [&](OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs) {
        // currentIterArg holds the accumulated tensor for this iteration
        Value currentIterArg = iterArgs.front();

        SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes(rank, builder.getIndexAttr(1));
        SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

        offsets[0] = iv;
        sizes[0] = builder.getIndexAttr(1);

        for (int64_t i = 1; i < rank; ++i) {
          if (inputType.isDynamicDim(i)) {
             Value dimIdx = arith::ConstantIndexOp::create(builder, loc, i);
             sizes[i] = tensor::DimOp::create(builder, loc, rescaleOp.getInput(), dimIdx)->getResult(0);
          } else {
             sizes[i] = builder.getIndexAttr(inputType.getDimSize(i));
          }
        }

        SmallVector<int64_t> sliceInputShape(inputType.getShape());
        sliceInputShape[0] = 1;
        auto sliceInputType = RankedTensorType::get(sliceInputShape, inputType.getElementType());

        Value sliceInput = tensor::ExtractSliceOp::create(
          builder, loc, sliceInputType, rescaleOp.getInput(), offsets, sizes, strides
        );

        SmallVector<int64_t> sliceOutputShape(outputType.getShape());
        sliceOutputShape[0] = 1;
        auto sliceOutputType = RankedTensorType::get(sliceOutputShape, outputType.getElementType());

        SmallVector<Value> newOperands(rescaleOp->getOperands().begin(), rescaleOp->getOperands().end());
        newOperands[0] = sliceInput;

        OperationState state(loc, tosa::RescaleOp::getOperationName());
        state.addOperands(newOperands);
        state.addTypes(sliceOutputType);
        state.addAttributes(rescaleOp->getAttrs());

        Operation *newRescale = builder.create(state);
        Value rescaleResult = newRescale->getResult(0);

        Value insertedSlice = tensor::InsertSliceOp::create(
          builder, loc, rescaleResult, currentIterArg, offsets, sizes, strides
        );

        scf::YieldOp::create(builder, loc, insertedSlice);
      }
    );

    rewriter.replaceOp(rescaleOp, forOp.getResult(0));

    return success();
  }
};


struct MyRewritePass : public PassWrapper<MyRewritePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyRewritePass)

  StringRef getArgument() const override { return "iree-example2-rewrite-pass"; }
  StringRef getDescription() const override { return "Rewrites sruff"; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    llvm::errs() << "DEBUG: MyRewritePass is running on operation: "
               << funcOp.getName() << "\n";
    RewritePatternSet patterns(context);
    patterns.add<DoubleRoundRewriter, DynamicBatchRewriter>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

struct MyOptions {
  bool enable_fusion = false;
  void bindOptions(OptionsBinder& binder) {
    static llvm::cl::OptionCategory category("IREE Example 2 Plugin");
    binder.opt<bool>(
      "iree-example2-fusion",
      enable_fusion,
      llvm::cl::desc("Enable the custom centered-gemm microkernel fusion"),
      llvm::cl::cat(category)
    );
  }
};

struct MySession : public PluginSession<MySession, MyOptions> {

  void
  extendInputConversionPreprocessingPassPipeline(
    OpPassManager &passManager, InputDialectOptions::Type inputType
  ) override {
    passManager.addNestedPass<func::FuncOp>(std::make_unique<MyRewritePass>());
  }

  bool
  extendCustomInputConversionPassPipeline(
    OpPassManager& passManager, std::string_view typeMnemonic
  ) override {
      bool extensionsWereMade = false;
      if (options.enable_fusion) {
        passManager.addNestedPass<func::FuncOp>(std::make_unique<MyFusionPass>());
        extensionsWereMade = true;
      }
      return extensionsWereMade;
  }

  void
  onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<strela::StrelaDialect>();
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList& targets) override {
    targets.add(
      "strela",
      []() -> std::shared_ptr<IREE::HAL::TargetBackend> {
        return std::make_shared<StrelaTargetBackend>();
      }
    );
  }

  void
  populateHALTargetDevices(IREE::HAL::TargetDeviceList& targets) override {
    targets.add(
      "strela",
      []() -> std::shared_ptr<IREE::HAL::TargetDevice> {
        return std::make_shared<StrelaTargetDevice>();
      }
    );
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool
iree_register_compiler_plugin_example2(
  mlir::iree_compiler::PluginRegistrar *registrar
) {
  registrar->registerPlugin<MySession>("example2");
  return true;
}
