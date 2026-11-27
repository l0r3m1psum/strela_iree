#include "estela/Transforms/Passes.h"
#include "estela/Utils/Finders.h"
#include "estela/Utils/StrelaContraction.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"

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
    if (!op.hasPureBufferSemantics()) {
      return rewriter.notifyMatchFailure(op, "not bufferized");
    }
    FailureOr<estela::StrelaMatvecShape> shape =
      estela::matchStrelaContraction(op, desc);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "not a STRELA matvec");
    }
    // The lowering strategy made the contraction an N x C times C x F (or
    // F x C) matmul, so the tiles are already the kernel's operands.
    if (llvm::any_of(op->getOperands(), [](Value v) {
          return cast<MemRefType>(v.getType()).getRank() != 2;
        })) {
      return rewriter.notifyMatchFailure(op, "not the kernel's 2-D operands");
    }

    // The zero fill goes: the kernel computes from zero.
    rewriter.eraseOp(estela::findZeroFill(op));
    SmallVector<Value> inputs = op.getDpsInputs();
    strela::MatvecOp::create(
      rewriter, op.getLoc(), inputs[0], inputs[1], op.getDpsInits()[0],
      shape->weightsTransposed ? rewriter.getUnitAttr() : UnitAttr()
    );
    rewriter.eraseOp(op);
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
