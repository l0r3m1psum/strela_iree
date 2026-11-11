// This file is willingly not wired in the build system for now

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

using namespace mlir::iree_compiler;

namespace {

// Finds the `util.global` the HAL device assignment pipeline created for the
// device whose deviceID is `strela`, i.e. --iree-hal-target-device=NAME=strela.
// If you ever pass a device list (--iree-hal-target-device=x=[a,b]) the
// global's initial value is a #hal.device.select<[...]>, the
// dyn_cast_if_present<DeviceTargetAttr> fails and the pass won't find it.
static mlir::StringAttr
findStrelaDeviceGlobal(mlir::ModuleOp moduleOp) {
  mlir::StringAttr found;
  moduleOp.walk([&](IREE::Util::GlobalOpInterface globalOp) {
    auto targetAttr = dyn_cast_if_present<IREE::HAL::DeviceTargetAttr>(
      globalOp.getGlobalInitialValue()
    );
    if (targetAttr && targetAttr.getDeviceID().getValue() == "strela") {
      found = globalOp.getGlobalName();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

} // namespace
