#include "estela/Utils/StrelaContraction.h"

#include "estela/Utils/Bitstreams.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Matchers.h"

namespace mlir::estela {

// inferContractionDims only looks at the indexing maps and iterator types;
// the body could be anything. This also checks it is a multiply-accumulate.
FailureOr<linalg::ContractionDimensions>
matchContraction(linalg::LinalgOp op) {
  if (!linalg::isaContractionOpInterface(op)) return failure();
  return linalg::inferContractionDims(op);
}

// Contraction naming. linalg::inferContractionDims sorts every loop dimension
// by which operands it appears in:
//
//   linalg    | appears in        | deep learning
//   ----------+-------------------+--------------------------
//   batch     | lhs, rhs, out     | B  (e.g. attention heads)
//   M         | lhs, out          | N  (rows: batch, tokens)
//   N         | rhs, out          | F  (output features)
//   K         | lhs, rhs          | C  (input channels)
//
//   out[b, M, N] = sum_K lhs[b, M, K] * rhs[b, K, N]
//     y[B, N, F] = sum_C   x[B, N, C]   * W[B, C, F]
// linalg::inferContractionDims is needed to classify things like all the
// indices in an einsum contraction (remember that it can always be implemented
// as a reshape -> matmul -> reshape)
bool
isStrelaMatvec(linalg::LinalgOp op, strela::TargetDescAttr desc) {
  if (!desc || desc.getRows() != strelaKernelRows
      || desc.getColumns() != strelaKernelColumns) {
    return false;
  }
  if (!op.hasPureTensorSemantics()) return false;

  FailureOr<linalg::ContractionDimensions> dims = matchContraction(op);
  if (failed(dims)) return false;

  for (Value value : op->getOperands()) {
    if (!getElementTypeOrSelf(value.getType()).isInteger(32)) return false;
  }

  // The shape strela.matvec takes today: one unit batch dim, one each of
  // M (N rows), N (F features) and K (C). Step 1c relaxes this.
  if (op.getNumLoops() != 4 || dims->batch.size() != 1 || dims->m.size() != 1
      || dims->n.size() != 1 || dims->k.size() != 1) {
    return false;
  }
  unsigned b = dims->batch[0], n = dims->m[0], f = dims->n[0], c = dims->k[0];

  // The layouts of linalg.batch_matmul: lhs (b, n, c), rhs (b, c, f) and
  // result (b, n, f). That is, the activations are C-contiguous, the weights
  // are C x F and the result is F-contiguous.
  MLIRContext *ctx = op.getContext();
  auto map = [&](unsigned d0, unsigned d1, unsigned d2) {
    return AffineMap::get(
      op.getNumLoops(), 0,
      {getAffineDimExpr(d0, ctx), getAffineDimExpr(d1, ctx), getAffineDimExpr(d2, ctx)},
      ctx
    );
  };
  SmallVector<AffineMap> expected{map(b, n, c), map(b, c, f), map(b, n, f)};
  if (op.getIndexingMapsArray() != expected) return false;

  // Static C and F, at least one full job of features, and a weight column
  // that fits the DMA word counter. N may be dynamic.
  SmallVector<int64_t> ranges = op.getStaticLoopRanges();
  int64_t channels = ranges[c], features = ranges[f];
  if (ranges[b] != 1 || ShapedType::isDynamic(channels)
      || ShapedType::isDynamic(features)) {
    return false;
  }
  if (features < strelaMatvecFeaturesPerJob
      || channels * features > static_cast<int64_t>(desc.getMaxColumnWords())) {
    return false;
  }

  // TODO: this is done in multiple places and should be put in a utility function.
  // The kernel has no accumulator input, so it can only compute from 0.
  auto fillOp = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  APInt init;
  return fillOp
      && matchPattern(fillOp.getInputs()[0], m_ConstantInt(&init))
      && init.isZero();
}

} // namespace mlir::estela
