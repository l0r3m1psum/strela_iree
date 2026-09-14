// Passes that run on a STRELA executable, i.e. after LinalgToStrela has already
// produced strela ops.

#ifndef STRELA_TRANSFORMS_STRELA_PASSES_H_
#define STRELA_TRANSFORMS_STRELA_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::strela {

// Rewrites the dispatch interface into strela.binding_load / binding_store,
// deleting the hal.interface.* ops and the arithmetic that reassembled the byte
// offsets from the push constants. Runs on a func.
std::unique_ptr<Pass> createMaterializeBindingsPass();

// Replaces the export workgroup count region with a constant (1, 1, 1): a
// STRELA kernel always runs as a single unit of work. Runs on a
// hal.executable.variant.
std::unique_ptr<Pass> createResolveWorkgroupCountPass();

} // namespace mlir::strela

#endif // STRELA_TRANSFORMS_STRELA_PASSES_H_
