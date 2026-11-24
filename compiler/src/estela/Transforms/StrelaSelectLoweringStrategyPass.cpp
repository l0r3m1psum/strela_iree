#include "estela/Transforms/Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"

#include "estela/Utils/Bitstreams.h"
#include "estela/Utils/Finders.h"
#include "estela/Utils/StrelaContraction.h"

namespace mlir::estela {
#define GEN_PASS_DEF_STRELASELECTLOWERINGSTRATEGYPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// The size of loop dim `dim` of `op`, from the init it writes: a constant when
// static, a tensor.dim otherwise.
static OpFoldResult
loopSize(OpBuilder &b, Location loc, linalg::LinalgOp op, unsigned dim) {
  int64_t size = op.getStaticLoopRanges()[dim];
  if (!ShapedType::isDynamic(size)) return b.getIndexAttr(size);
  OpOperand *init = op.getDpsInitOperand(0);
  AffineMap map = op.getMatchingIndexingMap(init);
  for (auto [i, expr] : llvm::enumerate(map.getResults())) {
    if (cast<AffineDimExpr>(expr).getPosition() == dim) {
      return tensor::getMixedSize(b, loc, init->get(), i);
    }
  }
  llvm_unreachable("a contraction's M dim indexes its init");
}

// lhs[N, C] times rhs[F, C] into init[N, F]: the loop dims are (n, f, c) in
// that order, as for linalg.matmul, so the same tile sizes apply.
static linalg::LinalgOp
createTransposedMatmul(RewriterBase &rewriter, Location loc, Value lhs,
                       Value rhs, Value init) {
  MLIRContext *context = rewriter.getContext();
  AffineExpr n, f, c;
  bindDims(context, n, f, c);
  SmallVector<AffineMap> maps = {
    AffineMap::get(3, 0, {n, c}, context),
    AffineMap::get(3, 0, {f, c}, context),
    AffineMap::get(3, 0, {n, f}, context),
  };
  SmallVector<utils::IteratorType> iterators = {
    utils::IteratorType::parallel, utils::IteratorType::parallel,
    utils::IteratorType::reduction,
  };
  return linalg::GenericOp::create(
    rewriter, loc, TypeRange{init.getType()}, ValueRange{lhs, rhs},
    ValueRange{init}, maps, iterators,
    [](OpBuilder &b, Location loc, ValueRange args) {
      Value product = arith::MulIOp::create(b, loc, args[0], args[1]);
      Value sum = arith::AddIOp::create(b, loc, args[2], product);
      linalg::YieldOp::create(b, loc, sum);
    }
  );
}

// Rewrites the contraction `op` into the N x C times C x F linalg.matmul the
// matvec kernel runs (times F x C weights when they are transposed, as a
// linalg.generic), between reshapes that only drop or add unit dims, so that
// the tile IREE cuts from it is already the kernel's shape. The reshapes fold
// into the bindings (FoldReshapeIntoInterfaceTensor).
static linalg::LinalgOp
normalizeToMatmul(RewriterBase &rewriter, linalg::LinalgOp op,
                  const estela::StrelaMatvecShape &shape) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  Type i32 = rewriter.getI32Type();
  SmallVector<Value> inputs = op.getDpsInputs();
  Value lhs = *estela::reshapeTo(
    rewriter, loc, inputs[0], RankedTensorType::get({shape.n, shape.c}, i32)
  );
  Value rhs = *estela::reshapeTo(
    rewriter, loc, inputs[1],
    RankedTensorType::get(shape.weightsTransposed
                            ? ArrayRef<int64_t>{shape.f, shape.c}
                            : ArrayRef<int64_t>{shape.c, shape.f}, i32)
  );
  OpFoldResult rows = shape.dims.m.empty()
    ? rewriter.getIndexAttr(1)
    : loopSize(rewriter, loc, op, shape.dims.m[0]);
  Value empty = tensor::EmptyOp::create(
    rewriter, loc, ArrayRef<OpFoldResult>{rows, rewriter.getIndexAttr(shape.f)},
    i32
  );
  Value zero = arith::ConstantIntOp::create(rewriter, loc, i32, 0);
  Value init = linalg::FillOp::create(rewriter, loc, zero, empty).getResult(0);
  linalg::LinalgOp matmulOp = shape.weightsTransposed
    ? createTransposedMatmul(rewriter, loc, lhs, rhs, init)
    : cast<linalg::LinalgOp>(linalg::MatmulOp::create(
        rewriter, loc, TypeRange{init.getType()}, ValueRange{lhs, rhs},
        ValueRange{init}
      ).getOperation());
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  rewriter.replaceOp(
    op, *estela::reshapeTo(rewriter, loc, matmulOp->getResult(0), resultType)
  );
  return matmulOp;
}

// Picks how a STRELA dispatch is cut into jobs: its contraction becomes an
// N x C times C x F linalg.matmul tiled by one row and three features, the
// shape of one matvec job; C is not tiled because STRELA does the reduction.
// IREE's TileAndDistributeToWorkgroupsUsingForallOp reads the tile sizes from
// the lowering_config, the grid it makes is (ceildiv(F, 3), N, 1).
struct StrelaSelectLoweringStrategyPass final
    : estela::impl::StrelaSelectLoweringStrategyPassBase<
        StrelaSelectLoweringStrategyPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    strela::TargetDescAttr desc = estela::findStrelaTargetDesc(funcOp);

    linalg::LinalgOp rootOp;
    FailureOr<estela::StrelaMatvecShape> shape = failure();
    WalkResult walk = funcOp.walk([&](linalg::LinalgOp op) {
      FailureOr<estela::StrelaMatvecShape> opShape =
        estela::matchStrelaContraction(op, desc);
      if (failed(opShape)) return WalkResult::advance();
      if (rootOp) {
        op.emitOpError("second STRELA contraction in one dispatch; the "
                       "backend runs a single kernel per entry point");
        return WalkResult::interrupt();
      }
      rootOp = op;
      shape = opShape;
      return WalkResult::advance();
    });
    if (walk.wasInterrupted()) return signalPassFailure();
    // Nothing STRELA can run: ConvertToStrelaABI reports it.
    if (!rootOp) return;

    IRRewriter rewriter(funcOp.getContext());
    linalg::LinalgOp matmulOp = normalizeToMatmul(rewriter, rootOp, *shape);

    MLIRContext *context = funcOp.getContext();
    auto config = IREE::Codegen::LoweringConfigAttr::get(
      context, TileSizesListType{{1, estela::strelaMatvecFeaturesPerJob, 0}}
    );
    auto translation = IREE::Codegen::TranslationInfoAttr::get(
      context, IREE::Codegen::NoPipelineAttr::get(context),
      /*codegenSpec=*/SymbolRefAttr(), /*workgroupSize=*/{},
      /*subgroupSize=*/0, /*configuration=*/DictionaryAttr()
    );
    if (failed(setOpConfigAndEntryPointFnTranslation(
      funcOp, matmulOp, config, translation
    ))) {
      return signalPassFailure();
    }
  }
};

} // namespace
