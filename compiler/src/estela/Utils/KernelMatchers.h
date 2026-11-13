// Matchers shared by the linalg passes: ConvertLinalgToStrela rewrites what
// they match, FormStrelaDispatches wraps it in a pinned dispatch region.

#ifndef ESTELA_UTILS_KERNELMATCHERS_H_
#define ESTELA_UTILS_KERNELMATCHERS_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Support/LLVM.h"

namespace mlir::estela {

// A single-batch i32 linalg.batch_matmul with static C and F, accumulating into
// a zero fill.
LogicalResult isStrelaLinalgMatvec(linalg::BatchMatmulOp op);

bool isSupportedByStrela(linalg::LinalgOp linalgOp);

} // namespace mlir::estela

#endif // ESTELA_UTILS_KERNELMATCHERS_H_
