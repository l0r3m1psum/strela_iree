#include "estela/Transforms/Linalg/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/Support/DebugLog.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct AssignTopologyPass
  : public PassWrapper<AssignTopologyPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AssignTopologyPass)

  StringRef getArgument() const override { return "iree-strela-assign-topology"; }
  StringRef getDescription() const override { return "Assigns the topology of the STRELA device"; }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    if (moduleOp->hasAttr("stream.topology")){
      LDBG() << "A topology is already present we do not overwrite it.";
      return;
    }

    SmallVector<FlatSymbolRefAttr> strelaDevices, otherDevices;
    for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
      auto targetAttr = dyn_cast_if_present<IREE::HAL::DeviceTargetAttr>(
          globalOp.getInitialValueAttr());
      if (!targetAttr) continue;
      auto ref = FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
      (targetAttr.getDeviceID() == "strela" ? strelaDevices : otherDevices)
          .push_back(ref);
    }

    if (strelaDevices.empty()) {
      mlir::emitWarning(moduleOp.getLoc()) << "No STRELA device found. It "
        "should be set via the --iree-hal-target-device=strela0=strela option.";
    }

    if (otherDevices.empty()) {
      mlir::emitWarning(moduleOp.getLoc()) << "No other device found. At least "
        "another device (capable of running any operation) should be given via "
        "the option --iree-hal-target-device=cpu0=local[0] option.";
    }

    // Other devices can access STRELA memory but not the contrary.
    MLIRContext *context = &getContext();
    SmallVector<IREE::HAL::DeviceLinkAttr> links;
    for (auto strela : strelaDevices) {
      for (auto other : otherDevices) {
        links.push_back(IREE::HAL::DeviceLinkAttr::get(
            context, other, strela, /*unified_memory=*/false,
            /*transparent_access=*/false, /*extra_properties=*/{}));
        // The host reads STRELA memory in place: elide the copies out of it.
        links.push_back(IREE::HAL::DeviceLinkAttr::get(
            context, strela, other, /*unified_memory=*/false,
            /*transparent_access=*/true, /*extra_properties=*/{}));
      }
    }
    moduleOp->setAttr("stream.topology",
                      IREE::HAL::DeviceTopologyAttr::get(context, links));
  }
};

} // namespace

namespace mlir::estela {

std::unique_ptr<Pass> createAssignTopologyPass() {
  return std::make_unique<AssignTopologyPass>();
}

} // namespace mlir::estela
