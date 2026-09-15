// Passes that run on linalg, before anything STRELA-specific exists in the IR.

#ifndef ESTELA_TRANSFORMS_LINALG_PASSES_H_
#define ESTELA_TRANSFORMS_LINALG_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::estela {

// Wraps each STRELA-supported op in a flow.dispatch.region pinned to the STRELA
// device, bracketed by transfers back to the host. Runs on a module.
std::unique_ptr<Pass> createFormStrelaDispatchesPass();

// Rewrites quantized conv2d into the centered-gemm form STRELA implements.
// Runs on a func.
std::unique_ptr<Pass> createFuseConv2DPass();

} // namespace mlir::estela

#endif // ESTELA_TRANSFORMS_LINALG_PASSES_H_
