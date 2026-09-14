// Passes that run on TOSA, before input conversion turns it into linalg.

#ifndef STRELA_TRANSFORMS_TOSA_PASSES_H_
#define STRELA_TRANSFORMS_TOSA_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::strela {

// Normalizes tosa.rescale so the rest of the pipeline sees a shape it can
// handle: splits double-rounding rescales and rewrites the dynamic batch
// dimension. Runs on a func.
std::unique_ptr<Pass> createNormalizeRescalePass();

} // namespace mlir::strela

#endif // STRELA_TRANSFORMS_TOSA_PASSES_H_
