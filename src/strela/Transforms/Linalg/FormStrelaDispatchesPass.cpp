#include "strela/Transforms/Linalg/Passes.h"

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"

#include "strela/Transforms/Linalg/Patterns.h"

using namespace mlir;
using namespace mlir::iree_compiler;
using mlir::strela::isSupportedByStrela;

namespace {

// Finds the `util.global` the HAL device assignment pipeline created for the
// device whose deviceID is `strela`, i.e. --iree-hal-target-device=NAME=strela.
// If you ever pass a device list (--iree-hal-target-device=x=[a,b]) the
// global's initial value is a #hal.device.select<[...]>, the
// dyn_cast_if_present<DeviceTargetAttr> fails and the pass won't find it.
static StringAttr
findStrelaDeviceGlobal(mlir::ModuleOp moduleOp) {
  StringAttr found;
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

// PassWrapper::getArgument() returns a string that can be used with --mlir-print-ir-after?

// Converts this:
//
// %cst = arith.constant dense<0> : tensor<1xi32>
// %2 = tensor.empty() : tensor<8xi32>
// %3 = linalg.generic ins(%0, %1) outs(%2) { ^bb0(...): arith.addi  ... }   // add
// %4 = linalg.generic ins(%3, %cst) outs(%2) { ^bb0(...): arith.maxsi ... }  // relu
//
// to this:
//
// %3 = flow.tensor.transfer %0 : tensor<8xi32> to #hal.device.affinity<@cpu>
// %4 = flow.tensor.transfer %1 : tensor<8xi32> to #hal.device.affinity<@cpu>
// %5 = flow.dispatch.region -> (tensor<8xi32>) attributes {stream.affinity = #hal.device.affinity<@strela>} {
//   %11 = tensor.empty() : tensor<8xi32>                        // cloned in
//   %12 = linalg.generic ins(%3, %4) outs(%11) { ... addi ... }
//   flow.return %12 : tensor<8xi32>
// }
// %6 = flow.tensor.transfer %5 : tensor<8xi32> to #hal.device.affinity<@cpu>
// %7 = flow.tensor.transfer %6 : tensor<8xi32> to #hal.device.affinity<@cpu>
// %8 = flow.dispatch.region -> (tensor<8xi32>) attributes {stream.affinity = #hal.device.affinity<@strela>} {
//   %cst_0 = arith.constant dense<0> : tensor<1xi32>            // cloned in
//   %11 = tensor.empty() : tensor<8xi32>                        // cloned in
//   %12 = linalg.generic ins(%7, %cst_0) outs(%11) { ... maxsi ... }
//   flow.return %12 : tensor<8xi32>
// }
// %9 = flow.tensor.transfer %8 : tensor<8xi32> to #hal.device.affinity<@cpu>
//
// ElideRedundantTransfer should remove the redundant transfers later...
struct FormStrelaDispatchesPass
    : public PassWrapper<FormStrelaDispatchesPass, OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(FormStrelaDispatchesPass)

  StringRef getArgument() const override { return "iree-form-strela-dispatches"; }
  StringRef getDescription() const override {
    return "Isolates STRELA supported operations into explicit flow dispatch regions";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
      arith::ArithDialect,
      tensor::TensorDialect,
      IREE::Flow::FlowDialect
    >();
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    MLIRContext *context = &getContext();

    StringAttr strelaName = findStrelaDeviceGlobal(moduleOp);
    if (!strelaName) {
      moduleOp.emitWarning("No STRELA device was requested; nothing to partition");
      return;
    }

    auto hostAffinity = moduleOp->getAttrOfType<IREE::Stream::AffinityAttr>(
      "stream.affinity.default"
    );
    if (!hostAffinity) {
      moduleOp.emitError() << "no stream.affinity.default on the module; pass "
                              "--iree-hal-default-device to name the host device";
      return signalPassFailure();
    }

    auto strelaAffinity = IREE::HAL::DeviceAffinityAttr::get(
      context, FlatSymbolRefAttr::get(strelaName), /*queue_mask=*/-1
    );

    // TODO: if the module already has a topology it needs to be enriched with
    // this information
    if (!moduleOp->hasAttr("stream.topology")) {
      auto hostDeviceAffinity = cast<IREE::HAL::DeviceAffinityAttr>(hostAffinity);
      SymbolRefAttr hostSym = hostDeviceAffinity.getDevice();
      SymbolRefAttr strelaSym = FlatSymbolRefAttr::get(strelaName);
      bool unified_memory = true, transparent_access = true;
      auto emptyProperties = DictionaryAttr::get(context, {});
      SmallVector<IREE::HAL::DeviceLinkAttr> links{
        IREE::HAL::DeviceLinkAttr::get(
          context, hostSym, strelaSym, unified_memory, transparent_access, emptyProperties
        ),
        IREE::HAL::DeviceLinkAttr::get(
          context, strelaSym, hostSym, unified_memory, transparent_access, emptyProperties
        ),
      };
      moduleOp->setAttr(
        "stream.topology", IREE::HAL::DeviceTopologyAttr::get(context, links)
      );
    }

    // Collect first: the walk rewrites the IR as it goes. We want to avoid any
    // possible issue with modifing the underlying data structure while walking
    // on it.
    SmallVector<linalg::GenericOp> candidates;
    moduleOp.walk([&candidates](linalg::GenericOp genericOp) {
      if (isSupportedByStrela(genericOp)) {
        candidates.push_back(genericOp);
      }
    });

    for (linalg::GenericOp genericOp : candidates) {
      OpBuilder builder(genericOp);
      Location loc = genericOp.getLoc();

      SmallVector<Value> resultDims;
      for (auto [index, result] : llvm::enumerate(genericOp.getResults())) {
        auto tensorType = dyn_cast<RankedTensorType>(result.getType());
        if (tensorType) {
          Value init = genericOp.getDpsInits()[index];
          for (int64_t dim = 0; dim < tensorType.getRank(); ++dim) {
            if (tensorType.isDynamicDim(dim)) {
              Value dimIdx = arith::ConstantIndexOp::create(builder, loc, dim);
              resultDims.push_back(
                tensor::DimOp::create(builder, loc, init, dimIdx)
              );
            }
          }
        }
      }

      IRMapping mapping;
      SmallVector<Operation *> opsToCloneIntoRegion;
      for (OpOperand &operand : genericOp->getOpOperands()) {
        Value value = operand.get();
        if (!isa<RankedTensorType>(value.getType())) continue;

        Operation *definingOp = value.getDefiningOp();
        DenseElementsAttr constantAttr;
        if (definingOp &&
            (isa<tensor::EmptyOp>(definingOp) ||
             matchPattern(value, m_Constant(&constantAttr)))) {
          opsToCloneIntoRegion.push_back(definingOp);
          continue;
        }

        mapping.map(value, IREE::Flow::TensorTransferOp::create(
          builder, loc, value, hostAffinity
        ));
      }

      auto dispatchRegion = IREE::Flow::DispatchRegionOp::create(
        builder,
        loc,
        genericOp.getResultTypes(),
        resultDims,
        /*workload=*/ValueRange{}
      );

      dispatchRegion->setAttr("stream.affinity", strelaAffinity);

      Block *body = builder.createBlock(&dispatchRegion.getBody());
      builder.setInsertionPointToStart(body);
      for (Operation *op : opsToCloneIntoRegion) {
        Operation *clonedConstant = builder.clone(*op, mapping);
        for (auto [oldResult, newResult] :
             llvm::zip_equal(op->getResults(), clonedConstant->getResults())) {
          mapping.map(oldResult, newResult);
        }
      }
      Operation *clonedOp = builder.clone(*genericOp, mapping);
      IREE::Flow::ReturnOp::create(builder, loc, clonedOp->getResults());

      builder.setInsertionPointAfter(dispatchRegion);
      SmallVector<Value> replacements;
      for (Value result : dispatchRegion.getResults()) {
        replacements.push_back(
          IREE::Flow::TensorTransferOp::create(builder, loc, result, hostAffinity)
        );
      }

      genericOp.replaceAllUsesWith(replacements);
      genericOp.erase();
    }
  }

};

} // namespace

namespace mlir::strela {

std::unique_ptr<Pass> createFormStrelaDispatchesPass() {
  return std::make_unique<FormStrelaDispatchesPass>();
}

} // namespace mlir::strela
