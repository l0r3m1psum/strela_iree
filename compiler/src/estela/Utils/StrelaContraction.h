#ifndef ESTELA_UTILS_STRELACONTRACTION_H_
#define ESTELA_UTILS_STRELACONTRACTION_H_

#include "estela/StrelaIR/StrelaAttrs.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::estela {

// The sizes of strela.matvec, N x C times C x F. N may be dynamic, C and F are
// static.
struct StrelaMatvecShape {
  int64_t n, c, f;
};

// The shape the matvec kernel runs `op` with on the hardware `desc` describes,
// or failure when it cannot run it. Failure for every op when `desc` is null.
FailureOr<StrelaMatvecShape>
matchStrelaContraction(linalg::LinalgOp op, strela::TargetDescAttr desc);

} // namespace mlir::estela

#endif // ESTELA_UTILS_STRELACONTRACTION_H_
