#ifndef ESTELA_TRANSFORMS_LINALG_PASSES_H_
#define ESTELA_TRANSFORMS_LINALG_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::estela {

std::unique_ptr<Pass> createFuseConv2DPass();

std::unique_ptr<Pass> createTransferWrapPass();

std::unique_ptr<Pass> createAssignTopologyPass();

} // namespace mlir::estela

#endif // ESTELA_TRANSFORMS_LINALG_PASSES_H_
