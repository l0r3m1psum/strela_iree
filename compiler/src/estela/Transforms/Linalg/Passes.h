#ifndef ESTELA_TRANSFORMS_LINALG_PASSES_H_
#define ESTELA_TRANSFORMS_LINALG_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::estela {

std::unique_ptr<Pass> createFuseConv2DPass();

} // namespace mlir::estela

#endif // ESTELA_TRANSFORMS_LINALG_PASSES_H_
