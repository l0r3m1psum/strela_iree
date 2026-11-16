#include "estela/Transforms/Passes.h"
#include "estela/Utils/Finders.h"
#include "estela/Utils/StrelaContraction.h"

#include <mutex>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"

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

// TODO: move this in utils...
static void
print(mlir::func::FuncOp funcOp) {
  print(funcOp->getParentOfType<mlir::ModuleOp>());
}

namespace mlir::estela {
#define GEN_PASS_DEF_CONVERTLINALGTOSTRELAPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;

namespace {

struct ConvertLinalgMatvecToStrela
  : public OpInterfaceRewritePattern<linalg::LinalgOp> {

  strela::TargetDescAttr desc;

  ConvertLinalgMatvecToStrela(MLIRContext *context, strela::TargetDescAttr desc)
      : OpInterfaceRewritePattern<linalg::LinalgOp>(context), desc(desc) {}

  LogicalResult
  matchAndRewrite(
    linalg::LinalgOp op, PatternRewriter &rewriter
  ) const override {
    if (!estela::isStrelaMatvec(op, desc)) {
      return rewriter.notifyMatchFailure(op, "not a STRELA matvec");
    }
    // The zero fill init is dropped: the kernel computes from zero, and the
    // fill is then dead and erased by the driver.
    SmallVector<Value> inputs = op.getDpsInputs();
    rewriter.replaceOpWithNewOp<strela::MatvecOp>(
      op, op->getResultTypes(), inputs[0], inputs[1]
    );
    return success();
  }
};

struct ConvertLinalgToStrelaPass final
  : estela::impl::ConvertLinalgToStrelaPassBase<ConvertLinalgToStrelaPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    strela::TargetDescAttr desc = estela::findStrelaTargetDesc(funcOp);

    RewritePatternSet patterns(context);
    patterns.add<ConvertLinalgMatvecToStrela>(context, desc);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace
