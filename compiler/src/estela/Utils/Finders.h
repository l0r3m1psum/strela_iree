// Lookups of the STRELA device and its description in the IR, shared by the
// passes so that every one of them finds the device the same way.
//
// All of them return null when there is nothing to find: no STRELA device just
// means nothing is offloaded.

#ifndef ESTELA_UTILS_FINDERS_H_
#define ESTELA_UTILS_FINDERS_H_

#include "estela/StrelaIR/StrelaAttrs.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "llvm/ADT/StringRef.h"

namespace mlir::estela {

// The key of #strela.target_desc in the #hal.executable.target configuration.
inline constexpr llvm::StringLiteral strelaTargetDescKey = "strela.target_desc";

// The device target a !hal.device global is initialized with, or null.
//
// If you ever pass a device list (--iree-hal-target-device=x=[a,b]) the
// global's initial value is a #hal.device.select<[...]> and this returns null,
// so none of the finders below see that device.
iree_compiler::IREE::HAL::DeviceTargetAttr
findDeviceTarget(iree_compiler::IREE::Util::GlobalOp globalOp);

// Whether `targetAttr` is a STRELA device.
bool isStrelaDevice(iree_compiler::IREE::HAL::DeviceTargetAttr targetAttr);

// The util.global the HAL device assignment pipeline created for the STRELA
// device, i.e. --iree-hal-target-device=NAME=strela, so the passes do not
// depend on the name given on the command line.
//
//   util.global private @device_b = #hal.device.target<"strela", ...> : !hal.device
//
// `op` is the module or any op inside it.
iree_compiler::IREE::Util::GlobalOp findStrelaDeviceGlobal(Operation *op);

// The affinity that places work on every queue of the STRELA device.
iree_compiler::IREE::HAL::DeviceAffinityAttr findStrelaAffinity(Operation *op);

// The STRELA description in scope at `op`: from the enclosing
// hal.executable.variant in the backend, from the STRELA device global before
// executables exist.
strela::TargetDescAttr findStrelaTargetDesc(Operation *op);

} // namespace mlir::estela

#endif // ESTELA_UTILS_FINDERS_H_
