#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

static LogicalResult
isMatmulEquivalent(linalg::Conv2DNhwcHwcfQOp convOp) {
  Value x = convOp.getDpsInputOperand(0)->get();
  Value w = convOp.getDpsInputOperand(1)->get();
  Value z_x = convOp.getDpsInputOperand(2)->get();
  Value z_w = convOp.getDpsInputOperand(3)->get();
  Value y = convOp.getDpsInits()[0];

  if (!z_x.getType().isInteger(32) || !z_w.getType().isInteger(32)) {
    return failure();
  }

  auto isAllOnes = [](DenseIntElementsAttr attr) {
    if (!attr) return false;
    return llvm::all_of(attr.getValues<int64_t>(), [](int64_t v) { return v == 1; });
  };
  if (!isAllOnes(convOp.getStrides()) || !isAllOnes(convOp.getDilations())) {
    return failure();
  }

  // Check that the Filter (W) spatial dimensions are 1x1
  // Layout is HWCF: index 0 (H), index 1 (W)
  auto wType = dyn_cast<RankedTensorType>(w.getType());
  if (!wType.getElementType().isInteger(8)) return failure();
  if (wType.getDimSize(0) != 1 || wType.getDimSize(1) != 1) {
    return failure();
  }

  // Check that the Input (X) spatial dimensions are 1x1
  // Layout is NHWC: index 1 (H), index 2 (W)
  auto xType = dyn_cast<RankedTensorType>(x.getType());
  if (!xType.getElementType().isInteger(8)) return failure();
  if (xType.getDimSize(1) != 1 || xType.getDimSize(2) != 1) {
    return failure();
  }

  auto yType = dyn_cast<RankedTensorType>(y.getType());
  if (!yType.getElementType().isInteger(32)) return failure();

  return success();
}

struct Conv2DOffload : public OpRewritePattern<linalg::Conv2DNhwcHwcfQOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    linalg::Conv2DNhwcHwcfQOp convOp,
    PatternRewriter& rewriter
  ) const override {

    if (failed(isMatmulEquivalent(convOp))) {
      return failure();
    }

    convOp.emitRemark("Offloading Conv2D equivalent to Matmul");

    Value x = convOp.getDpsInputOperand(0)->get();
    Value w = convOp.getDpsInputOperand(1)->get();
    Value z_x = convOp.getDpsInputOperand(2)->get();
    Value z_w = convOp.getDpsInputOperand(3)->get();

    auto ukernel = IREE::Codegen::UKernelGenericOp::create(
        rewriter,
        convOp.getLoc(),
        convOp.getResultTypes(),
        rewriter.getStringAttr("my_centered_gemm"),
        ValueRange{x, z_x, w, z_w},
        convOp.getDpsInits(),
        ValueRange{},
        /*fn_def_attrs=*/nullptr,
        /*strided_outer_dims=*/nullptr
    );

    rewriter.replaceOp(convOp, ukernel.getResults());

    return success();
  }
};

struct SpyPattern : public RewritePattern {
  SpyPattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
    if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
      linalgOp->emitRemark("I am a Linalg Op: ") << linalgOp->getName();
    }

    return failure();
  }
};

struct MyFusionPass : public PassWrapper<MyFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyFusionPass)

  StringRef getArgument() const override { return "iree-example2-fusion-pass"; }
  StringRef getDescription() const override { return "Fuses centering subtractions into a GEMM ukernel"; }

  void runOnOperation() override {
    llvm::errs() << "DEBUG: MyFusionPass is running on operation: "
               << getOperation().getName() << "\n";
    mlir::MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<Conv2DOffload>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns), config))) {
      signalPassFailure();
    }
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
  bool extendCustomInputConversionPassPipeline(
    OpPassManager& pm,
    std::string_view typeMnemonic
  ) override {
      llvm::errs()
        << "Custom input conversion pass pipeline type: "
        << typeMnemonic << "\n";
      pm.addNestedPass<func::FuncOp>(std::make_unique<MyFusionPass>());
      bool extensionsWereMade = true;
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
