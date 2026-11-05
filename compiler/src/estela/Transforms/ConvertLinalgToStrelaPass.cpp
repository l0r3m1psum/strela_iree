#include "estela/Transforms/Passes.h"

#include <mutex>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"
#include "estela/Utils/KernelMatchers.h"

static std::mutex printMutex;

static void
print(mlir::ModuleOp moduleOp) {
  std::lock_guard<std::mutex> lock(printMutex);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  // Required, not cosmetic: without it AsmPrinter's findParent() walks past this
  // module all the way to the root and numbers SSA values over the *whole*
  // top-level module. Executable variants are translated in parallel, so that
  // reads IR another thread is busy rewriting. useLocalScope() stops the walk at
  // the first IsolatedFromAbove ancestor, i.e. this module.
  flags.useLocalScope();
  moduleOp.print(llvm::errs(), flags);
  llvm::errs() << '\n';
}

static void
print(mlir::func::FuncOp funcOp) {
  print(funcOp->getParentOfType<mlir::ModuleOp>());
}

namespace mlir::estela {
#define GEN_PASS_DEF_CONVERTLINALGTOSTRELAPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using mlir::estela::isStrelaLinalgAdd;
using mlir::estela::isStrelaLinalgRelu;
using mlir::estela::isStrelaLinalgMatvec;

namespace {

struct ConvertLinalgAddToStrela
  : public OpRewritePattern<linalg::GenericOp> {
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

struct ConvertLinalgReluToStrela
  : public OpRewritePattern<linalg::GenericOp> {
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

struct ConvertLinalgMatvecToStrela
  : public OpRewritePattern<linalg::BatchMatmulOp> {
  using OpRewritePattern<linalg::BatchMatmulOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(
    linalg::BatchMatmulOp matmulOp, PatternRewriter &rewriter
  ) const override {
    LogicalResult result = isStrelaLinalgMatvec(matmulOp);

    // The zero fill init is dropped: the kernel computes from zero, and the
    // fill is then dead and erased by the driver.
    if (succeeded(result)) {
      rewriter.replaceOpWithNewOp<strela::MatvecOp>(
        matmulOp,
        matmulOp.getResultTypes(),
        matmulOp.getDpsInputs()[0],
        matmulOp.getDpsInputs()[1]
      );
    }

    return result;
  }
};

struct ConvertLinalgToStrelaPass final
  : estela::impl::ConvertLinalgToStrelaPassBase<ConvertLinalgToStrelaPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<
      ConvertLinalgAddToStrela,
      ConvertLinalgReluToStrela,
      ConvertLinalgMatvecToStrela
    >(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace
