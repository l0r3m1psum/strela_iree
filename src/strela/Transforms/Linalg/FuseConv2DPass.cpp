#include "strela/Transforms/Linalg/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "strela/Utils/Kernels.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using mlir::strela::centered_matmul_kernel;

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

} // namespace

namespace mlir::strela {

std::unique_ptr<Pass> createFuseConv2DPass() {
  return std::make_unique<MyFusionPass>();
}

} // namespace mlir::strela
