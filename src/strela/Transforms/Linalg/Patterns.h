// Matchers shared by the linalg passes: ConvertLinalgToStrela rewrites what
// they match, FormStrelaDispatches wraps it in a pinned dispatch region.

#ifndef STRELA_TRANSFORMS_LINALG_PATTERNS_H_
#define STRELA_TRANSFORMS_LINALG_PATTERNS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

namespace mlir::strela {

// A linalg.generic that is exactly an i32 elementwise add STRELA can run.
LogicalResult isStrelaLinalgAdd(linalg::GenericOp genericOp);

// A linalg.generic that is exactly an i32 relu. Matches both the two-operand
// form tosa.maximum lowers to and the one-operand form canonicalization leaves.
LogicalResult isStrelaLinalgRelu(linalg::GenericOp genericOp);

bool isSupportedByStrela(linalg::GenericOp genericOp);

} // namespace mlir::strela

#endif // STRELA_TRANSFORMS_LINALG_PATTERNS_H_
