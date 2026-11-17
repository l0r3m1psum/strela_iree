#include "estela/Utils/StrelaContraction.h"

#include "estela/Utils/Bitstreams.h"

#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/IR/Matchers.h"

namespace mlir::estela {

// `init + lhs * rhs` with two inputs, whatever the op is called: the loop dims
// that play batch, M, N (F) and K (C). inferContractionDims only looks at the
// indexing maps and iterator types; the body could be anything, so this also
// checks it is a multiply-accumulate.
static FailureOr<linalg::ContractionDimensions>
matchContraction(linalg::LinalgOp op) {
  if (!linalg::isaContractionOpInterface(op)) return failure();
  return linalg::inferContractionDims(op);
}

// The loop dims `map` indexes, in order, leaving out `batch`. Contraction maps
// are projected permutations, so every result is a plain dim.
static SmallVector<unsigned>
filterBatchDims(ArrayRef<AffineExpr> mapResults, ArrayRef<unsigned> batch) {
  SmallVector<unsigned> dims;
  for (AffineExpr expr : mapResults) {
    unsigned d = cast<AffineDimExpr>(expr).getPosition();
    if (!llvm::is_contained(batch, d)) dims.push_back(d);
  }
  return dims;
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
FailureOr<StrelaMatvecShape>
matchStrelaContraction(linalg::LinalgOp op, strela::TargetDescAttr desc) {
  if (!desc || desc.getRows() != strelaKernelRows
      || desc.getColumns() != strelaKernelColumns) {
    return failure();
  }
  if (!op.hasPureTensorSemantics()) return failure();

  FailureOr<linalg::ContractionDimensions> dims = matchContraction(op);
  if (failed(dims)) return failure();

  for (Value value : op->getOperands()) {
    if (!getElementTypeOrSelf(value.getType()).isInteger(32)) return failure();
  }

  SmallVector<int64_t> ranges = op.getStaticLoopRanges();

  for (unsigned d : dims->batch) {
    if (ranges[d] != 1) return failure();
  }
  // At most one M dim (M.size() == 0 implies x is a vector, N = 1), one N (F)
  // and one K (C).
  if (dims->m.size() > 1 || dims->n.size() != 1 || dims->k.size() != 1) {
    return failure();
  }
  unsigned f = dims->n[0], c = dims->k[0];

  // The layouts of linalg.batch_matmul once the unit batch dims are dropped:
  // lhs (n, c), rhs (c, f) and out (n, f), without n when M is empty. That
  // is, the activations (lhs) are C-contiguous, the weights (rhs) and the
  // result (out) are F-contiguous.
  SmallVector<unsigned> lhs, rhs{c, f}, out;
  if (!dims->m.empty()) {
    lhs.push_back(dims->m[0]);
    out.push_back(dims->m[0]);
  }
  lhs.push_back(c);
  out.push_back(f);

  SmallVector<AffineMap> maps = op.getIndexingMapsArray();
  if (   filterBatchDims(maps[0].getResults(), dims->batch) != lhs
      || filterBatchDims(maps[1].getResults(), dims->batch) != rhs
      || filterBatchDims(maps[2].getResults(), dims->batch) != out) {
    return failure();
  }

  // Static C and F, at least one full job of features, and a weight column
  // that fits the DMA word counter. N may be dynamic.
  int64_t channels = ranges[c], features = ranges[f];
  if (ShapedType::isDynamic(channels) || ShapedType::isDynamic(features)) {
    return failure();
  }
  if (features < strelaMatvecFeaturesPerJob
      || channels * features > static_cast<int64_t>(desc.getMaxColumnWords())) {
    return failure();
  }

  // TODO: Put in a utility function.
  // The kernel has no accumulator input, so it can only compute from 0.
  auto fillOp = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  APInt init;
  if (!fillOp || !matchPattern(fillOp.getInputs()[0], m_ConstantInt(&init))
      || !init.isZero()) {
    return failure();
  }

  // N is the M dim's size, dynamic or not, or 1 when M is empty.
  int64_t rows = dims->m.empty() ? 1 : ranges[dims->m[0]];
  return StrelaMatvecShape{rows, channels, features};
}

} // namespace mlir::estela
