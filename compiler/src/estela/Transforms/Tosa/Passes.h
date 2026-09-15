// Passes that run on TOSA, before input conversion turns it into linalg.

#ifndef ESTELA_TRANSFORMS_TOSA_PASSES_H_
#define ESTELA_TRANSFORMS_TOSA_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::estela {

// Normalizes tosa.rescale so the rest of the pipeline sees a shape it can
// handle: splits double-rounding rescales and rewrites the dynamic batch
// dimension. Runs on a func.
std::unique_ptr<Pass> createNormalizeRescalePass();

} // namespace mlir::estela

#endif // ESTELA_TRANSFORMS_TOSA_PASSES_H_
