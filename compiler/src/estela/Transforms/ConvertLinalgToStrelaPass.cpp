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
    FailureOr<estela::StrelaMatvecShape> shape =
      estela::matchStrelaContraction(op, desc);
    if (failed(shape)) {
      return rewriter.notifyMatchFailure(op, "not a STRELA matvec");
    }
    int64_t n = shape->n, c = shape->c, f = shape->f;

    // To N x C times C x F (or F x C) and back. matchStrelaContraction checked
    // that the operands differ from these shapes only by unit dims, so every
    // reshape exists. After the lowering strategy the tile already has the
    // kernel's shape and nothing is reshaped.
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    SmallVector<Value> inputs = op.getDpsInputs();
    bool transposed = shape->weightsTransposed;
    Value lhs = *estela::reshapeTo(
      rewriter, loc, inputs[0], RankedTensorType::get({n, c}, i32)
    );
    Value rhs = *estela::reshapeTo(
      rewriter, loc, inputs[1],
      RankedTensorType::get(transposed ? ArrayRef<int64_t>{f, c}
                                       : ArrayRef<int64_t>{c, f}, i32)
    );
    // The zero fill init is dropped: the kernel computes from zero.
    Value y = strela::MatvecOp::create(
      rewriter, loc, RankedTensorType::get({n, f}, i32), lhs, rhs,
      transposed ? rewriter.getUnitAttr() : UnitAttr()
    );
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    rewriter.replaceOp(op, *estela::reshapeTo(rewriter, loc, y, resultType));
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
