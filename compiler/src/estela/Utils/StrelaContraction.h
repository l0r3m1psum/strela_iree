#ifndef ESTELA_UTILS_STRELACONTRACTION_H_
#define ESTELA_UTILS_STRELACONTRACTION_H_

#include "estela/StrelaIR/StrelaAttrs.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::estela {

// The sizes of strela.matvec, N x C times C x F. N may be dynamic, C and F are
// static.
struct StrelaMatvecShape {
  int64_t n, c, f;
  // The loop dims of the op that play batch, N (m), F (n) and C (k).
  linalg::ContractionDimensions dims;
  // The weights are F x C (each feature's C weights contiguous) rather than
  // C x F.
  bool weightsTransposed = false;
};

// The zero linalg.fill `op`'s init comes from, or null: on tensors the op that
// defines the init, on buffers the op right before, writing the same memref.
linalg::FillOp findZeroFill(linalg::LinalgOp op);

// The shape the matvec kernel runs `op` with on the hardware `desc` describes,
// or failure when it cannot run it. Failure for every op when `desc` is null.
// `op` may be on tensors (the frontend, the lowering strategy) or on buffers
// (ConvertLinalgToStrela, after bufferization).
// With `asIfTransposed` the size limit is checked as if the weights were
// already F x C, as IREE's TransposeMatmul makes them after
// StrelaWidenOperands. With `asIfWidened` the inputs may be integers narrower
// than i32, as if they were already extended. StrelaWidenOperands asks both.
FailureOr<StrelaMatvecShape>
matchStrelaContraction(linalg::LinalgOp op, strela::TargetDescAttr desc,
                       bool asIfTransposed = false, bool asIfWidened = false);

// `v` reshaped to `type`: a collapse_shape to a lower rank, an expand_shape to
// a higher one, nothing for the same type. Unit dims may move (16x1 becomes
// 1x16). Failure when no reassociation exists.
FailureOr<Value>
reshapeTo(OpBuilder &b, Location loc, Value v, RankedTensorType type);

} // namespace mlir::estela

#endif // ESTELA_UTILS_STRELACONTRACTION_H_
