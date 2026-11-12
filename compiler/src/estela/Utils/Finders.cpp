#include "estela/Utils/Finders.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir::iree_compiler;

namespace mlir::estela {

IREE::HAL::DeviceTargetAttr
findDeviceTarget(IREE::Util::GlobalOp globalOp) {
  return dyn_cast_if_present<IREE::HAL::DeviceTargetAttr>(
    globalOp.getInitialValueAttr()
  );
}

bool
isStrelaDevice(IREE::HAL::DeviceTargetAttr targetAttr) {
  return targetAttr && targetAttr.getDeviceID() == "strela";
}

IREE::Util::GlobalOp
findStrelaDeviceGlobal(Operation *op) {
  auto moduleOp = dyn_cast<ModuleOp>(op);
  if (!moduleOp) moduleOp = op->getParentOfType<ModuleOp>();
  if (!moduleOp) return {};
  // Device globals are always at the top level of the module.
  for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
    if (isStrelaDevice(findDeviceTarget(globalOp))) return globalOp;
  }
  return {};
}

IREE::HAL::DeviceAffinityAttr
findStrelaAffinity(Operation *op) {
  IREE::Util::GlobalOp globalOp = findStrelaDeviceGlobal(op);
  if (!globalOp) return {};
  int64_t allQueueMask = -1;
  return IREE::HAL::DeviceAffinityAttr::get(
    op->getContext(), FlatSymbolRefAttr::get(globalOp.getSymNameAttr()),
    allQueueMask
  );
}

// The description in a STRELA executable target's configuration, or null.
static strela::TargetDescAttr
findTargetDesc(IREE::HAL::ExecutableTargetAttr targetAttr) {
  if (!targetAttr || targetAttr.getBackend() != "strela") return {};
  DictionaryAttr config = targetAttr.getConfiguration();
  if (!config) return {};
  return config.getAs<strela::TargetDescAttr>(strelaTargetDescKey);
}

strela::TargetDescAttr
findStrelaTargetDesc(Operation *op) {
  // Backend: inside a hal.executable.variant, which carries the target.
  if (auto desc = findTargetDesc(IREE::HAL::ExecutableTargetAttr::lookup(op))) {
    return desc;
  }
  // Frontend: no executables yet, read it from the device global.
  IREE::Util::GlobalOp globalOp = findStrelaDeviceGlobal(op);
  if (!globalOp) return {};
  for (auto targetAttr : findDeviceTarget(globalOp).getExecutableTargets()) {
    if (auto desc = findTargetDesc(targetAttr)) return desc;
  }
  return {};
}

} // namespace mlir::estela
