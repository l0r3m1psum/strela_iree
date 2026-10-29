#include "estela/Transforms/Linalg/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"

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

static bool
isStrelaAdd(linalg::GenericOp op) {
  if (!linalg::isaElemwiseSingleBinaryOpInterface(op)) return false;
  if (!llvm::all_of(op.getIndexingMapsArray(),
                    [](AffineMap map) { return map.isIdentity(); })) {
    return false;
  }

  auto addOp = dyn_cast<arith::AddIOp>(op.getBody()->front());
  if (!addOp) return false;
  auto args = op.getRegionInputArgs();
  bool addsInputs =
      (addOp.getLhs() == args[0] && addOp.getRhs() == args[1]) ||
      (addOp.getLhs() == args[1] && addOp.getRhs() == args[0]);
  if (!addsInputs) return false;

  auto type = dyn_cast<RankedTensorType>(op.getResultTypes()[0]);
  return type && type.getElementType().isInteger(32);
}

// Affinity of the !hal.device global whose target is STRELA, so the pass
// does not depend on what the device is called on the command line.
// util.global private @device_b = #hal.device.target<"strela", {...}, [#hal.executable.target<"strela", "strela_bitstream">]> : !hal.device
static IREE::HAL::DeviceAffinityAttr
findStrelaAffinity(Operation *op) {
  auto moduleOp = op->getParentOfType<ModuleOp>();
  for (auto globalOp : moduleOp.getOps<IREE::Util::GlobalOp>()) {
    auto targetAttr = dyn_cast_if_present<IREE::HAL::DeviceTargetAttr>(
        globalOp.getInitialValueAttr());
    if (targetAttr && targetAttr.getDeviceID() == "strela") {
      SymbolRefAttr device = FlatSymbolRefAttr::get(globalOp.getSymNameAttr());
      int64_t all_queue_mask = -1;
      return IREE::HAL::DeviceAffinityAttr::get(
        op->getContext(), device, all_queue_mask
      );
    }
  }
  return {};
}

static void
wrapInTransfers(
  RewriterBase &rewriter, linalg::GenericOp op, Attribute strela, Attribute host
) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  for (OpOperand *input : op.getDpsInputOperands()) {
    SmallVector<Value> dims = IREE::Util::buildDynamicDimsForValue(
      loc, op.getDpsInitOperand(0)->get(), rewriter
    );
    Value transfer = IREE::Flow::TensorTransferOp::create(
      rewriter, loc, input->get().getType(), input->get(), dims, strela
    );
    rewriter.modifyOpInPlace(op, [&] { input->set(transfer); });
  }

  rewriter.setInsertionPointAfter(op);
  Value result = op->getResult(0);
  auto back = IREE::Flow::TensorTransferOp::create(rewriter, loc, result, host);
  rewriter.replaceAllUsesExcept(result, back, back);
}

static void
pinAbiToDefaultDevice(IREE::Util::FuncOp funcOp) {
  funcOp.walk([](Operation *op) {
    if (!isa<IREE::HAL::TensorImportOp, IREE::HAL::TensorExportOp,
             IREE::HAL::TensorAliasOp>(op)) {
      return;
    }
    auto affinityOp = cast<IREE::Stream::AffinityOpInterface>(op);
    if (affinityOp.getAffinityAttr()) {
      LDBG() << "A affinity is already present we do not overwrite it.";
      return;
    }
    // lookupOrDefault walks up from the op to the module's
    // stream.affinity.default which should be the CPU one.
    // Without it, Stream's affinity analysis places these unpinned ops
    // (effectively expanding flow.tensor.transfer) next to their users, i.e. on
    // STRELA e.g broadcast operations can be moved inside the "STRELA region".
    IREE::Stream::AffinityAttr affAttr = IREE::Stream::AffinityAttr::lookupOrDefault(op);
    if (!affAttr) {
      LDBG() << "No default affinity in scope, it should be set via the command line.";
    }
    affinityOp.setAffinityAttr(affAttr);
  });
}

struct TransferWrapPass
  : public PassWrapper<TransferWrapPass, OperationPass<IREE::Util::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TransferWrapPass)

  StringRef getArgument() const override { return "iree-strela-transfer-wrap"; }
  StringRef getDescription() const override { return "Wraps the supported operations in flow.tensor.transfer operations"; }

  void runOnOperation() override {
    IREE::Util::FuncOp funcOp = getOperation();
    IREE::HAL::DeviceAffinityAttr strela = findStrelaAffinity(funcOp);
    if (!strela) {
      mlir::emitWarning(funcOp.getLoc()) << "No STRELA device found. It "
        "should be set via the --iree-hal-target-device=strela0=strela option.";
      return;
    }

    // I am sure this makes naive assumptions and in some case it can fail but
    // for not it is good enough.
    pinAbiToDefaultDevice(funcOp);

    SmallVector<linalg::GenericOp> candidates;
    funcOp.walk([&](linalg::GenericOp op) {
      if (isStrelaAdd(op)) candidates.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::GenericOp op : candidates) {
      auto host = IREE::Stream::AffinityAttr::lookupOrDefault(op);
      if (!host) continue;
      wrapInTransfers(rewriter, op, strela, host);
    }
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect>();
  }
};

} // namespace

namespace mlir::estela {

std::unique_ptr<Pass> createTransferWrapPass() {
  return std::make_unique<TransferWrapPass>();
}

} // namespace mlir::estela
