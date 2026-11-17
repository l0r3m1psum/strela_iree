#include "estela/Transforms/Passes.h"
#include "estela/Utils/Finders.h"
#include "estela/Utils/StrelaContraction.h"

#include <mutex>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
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

// collapse_shape when the target has lower rank, expand_shape when higher,
// nothing when equal.
static FailureOr<Value>
reshapeTo(OpBuilder &b, Location loc, Value v, RankedTensorType tgtType) {
  using Reassociation = std::optional<SmallVector<ReassociationIndices>>;

  FailureOr<Value> res = failure();
  auto srcType = cast<RankedTensorType>(v.getType());
  int64_t srcTypeRank = srcType.getRank();
  int64_t tgtTypeRank = tgtType.getRank();
  if (srcType == tgtType) {
    res = v;
  } else if (srcTypeRank == tgtTypeRank) { // e.g. [16, 1] becomes [1, 16]
    SmallVector<int64_t> squeezed;
    for (int64_t d : srcType.getShape()) {
      if (d != 1) squeezed.push_back(d);
    }
    assert(static_cast<int64_t>(squeezed.size()) < srcTypeRank);
    auto squeezedType = RankedTensorType::get(squeezed, srcType.getElementType());
    res = reshapeTo(b, loc, *reshapeTo(b, loc, v, squeezedType), tgtType);
  } else if (
    Reassociation reassoc = getReassociationIndicesForReshape(srcType, tgtType)
  ) {
    if (srcTypeRank > tgtTypeRank) {
      res = tensor::CollapseShapeOp::create(b, loc, tgtType, v, *reassoc)
        .getResult();
    } else {
      res = tensor::ExpandShapeOp::create(b, loc, tgtType, v, *reassoc)
        .getResult();
    }
  }

  return res;
}

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
    auto [n, c, f] = *shape;

    // To N x C times C x F and back. matchStrelaContraction checked that the
    // operands differ from these shapes only by unit dims, so every reshape
    // exists.
    Location loc = op.getLoc();
    Type i32 = rewriter.getI32Type();
    SmallVector<Value> inputs = op.getDpsInputs();
    Value lhs = *reshapeTo(rewriter, loc, inputs[0], RankedTensorType::get({n, c}, i32));
    Value rhs = *reshapeTo(rewriter, loc, inputs[1], RankedTensorType::get({c, f}, i32));
    // The zero fill init is dropped: the kernel computes from zero.
    Value y = strela::MatvecOp::create(
      rewriter, loc, RankedTensorType::get({n, f}, i32), lhs, rhs
    );
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    rewriter.replaceOp(op, *reshapeTo(rewriter, loc, y, resultType));
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
