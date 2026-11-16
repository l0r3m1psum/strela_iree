#ifndef ESTELA_UTILS_STRELACONTRACTION_H_
#define ESTELA_UTILS_STRELACONTRACTION_H_

#include "estela/StrelaIR/StrelaAttrs.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

namespace mlir::estela {

// `init + lhs * rhs` with two inputs, whatever the op is called: the loop dims
// that play batch, M, N (F) and K (C).
FailureOr<linalg::ContractionDimensions> matchContraction(linalg::LinalgOp op);

// Whether the matvec kernel runs `op` on the hardware `desc` describes.
// False for every op when `desc` is null.
bool isStrelaMatvec(linalg::LinalgOp op, strela::TargetDescAttr desc);

} // namespace mlir::estela

#endif // ESTELA_UTILS_STRELACONTRACTION_H_
