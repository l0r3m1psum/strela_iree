// Matchers shared by the linalg passes: ConvertLinalgToStrela rewrites what
// they match, FormStrelaDispatches wraps it in a pinned dispatch region.

#ifndef ESTELA_UTILS_KERNELMATCHERS_H_
#define ESTELA_UTILS_KERNELMATCHERS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

namespace mlir::estela {

// A linalg.generic that is exactly an i32 elementwise add STRELA can run.
LogicalResult isStrelaLinalgAdd(linalg::GenericOp genericOp);

// A linalg.generic that is exactly an i32 relu. Matches both the two-operand
// form tosa.maximum lowers to and the one-operand form canonicalization leaves.
LogicalResult isStrelaLinalgRelu(linalg::GenericOp genericOp);

bool isSupportedByStrela(linalg::GenericOp genericOp);

} // namespace mlir::estela

#endif // ESTELA_UTILS_KERNELMATCHERS_H_
