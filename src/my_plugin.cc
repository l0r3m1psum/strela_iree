#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

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
  0x00000201, 0x02040300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 0

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 13
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 9
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 5
  0x00000201, 0x02040300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 1

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 14
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 10
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 6
  0x00000201, 0x02040300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 2

  0x00000021, 0x00000000, 0x00000000, 0x00000000, 0x00000000, // 15
  0x00000201, 0xC0040400, 0xDDEE0080, 0x00000000, 0x00000000, // 11
  0x08800109, 0x003C0340, 0x00000082, 0x00000000, 0x00000000, // 7
  0x00000201, 0x02040300, 0x00000081, 0x00000000, 0xFFFFFFFF, // 3
};

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
    Value zp = zero_points[0];
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
      kernel[5*(3+0) + 4] = analysis.z_x_val;
      kernel[5*(3+4) + 4] = kernel[5*(3+8) + 4] = kernel[5*(3+12) + 4] = analysis.z_w_val;
      kernel[5*(5+0) + 2] = kernel[5*(5+4) + 2] = kernel[5*(5+8)  + 2] = (analysis.in_chan << 16) | 0x0080;

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

    StringRef gemmFuncName = "custom.my_centered_gemm";

    auto funcDecl = moduleOp.lookupSymbol<func::FuncOp>(gemmFuncName);
    SmallVector<Type> inputTypes = {i32};
    std::transform(
      analysis.operands.begin(), analysis.operands.end(),
      std::back_inserter(inputTypes), [](Value v){ return v.getType(); }
    );
    Type resultType = convOp.getResultTypes()[0];

    SmallVector<Type> dynamicInputTypes;
    for (Type t : inputTypes) dynamicInputTypes.push_back(getDynamicTensorType(t));

    Type dynamicResultType = getDynamicTensorType(resultType);

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
    Value finalResult = callResult;

    if (callResult.getType() != resultType) {
      finalResult = tensor::CastOp::create(
        rewriter, convOp.getLoc(), resultType, callResult
      );
    }

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

    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(16);
    funcOp->getParentOfType<ModuleOp>().print(llvm::errs(), flags);
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
    patterns.add<DoubleRoundRewriter>(context);
    patterns.add<DynamicBatchRewriter>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }

    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(16);
    funcOp->getParentOfType<ModuleOp>().print(llvm::errs(), flags);
  }
};

struct MyOptions {
  bool enable_fusion = false;
  void bindOptions(OptionsBinder& binder) {
    static llvm::cl::OptionCategory category("IREE Example 2 Plugin");
    binder.opt<bool>("iree-example2-fusion", enable_fusion,
                     llvm::cl::desc("Enable the custom centered-gemm microkernel fusion"),
                     llvm::cl::cat(category));
  }
};

struct MySession : public PluginSession<MySession, MyOptions> {

  void extendInputConversionPreprocessingPassPipeline(
    OpPassManager &passManager,
    InputDialectOptions::Type inputType
  ) override {
    passManager.addNestedPass<func::FuncOp>(std::make_unique<MyRewritePass>());
  }

  bool extendCustomInputConversionPassPipeline(
    OpPassManager& pm,
    std::string_view typeMnemonic
  ) override {
      llvm::errs()
        << "Custom input conversion pass pipeline type: "
        << typeMnemonic << "\n";
      bool extensionsWereMade = false;
      if (options.enable_fusion) {
        pm.addNestedPass<func::FuncOp>(std::make_unique<MyFusionPass>());
        extensionsWereMade = true;
      }
      return extensionsWereMade;
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool iree_register_compiler_plugin_example2(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<MySession>("example2");
  return true;
}
