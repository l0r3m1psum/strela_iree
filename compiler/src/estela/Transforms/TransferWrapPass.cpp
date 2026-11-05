#include "estela/Transforms/Passes.h"
#include "estela/Utils/KernelMatchers.h"

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

namespace mlir::estela {
#define GEN_PASS_DEF_TRANSFERWRAPPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

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

static bool
isInt32Immediate(Value value) {
  DenseIntElementsAttr attr;
  return matchPattern(value, m_Constant(&attr)) && attr.isSplat() &&
         attr.getElementType().isInteger(32);
}

static void
wrapInTransfers(
  RewriterBase &rewriter, linalg::LinalgOp op, Attribute strela, Attribute host
) {
  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);

  // The transfers back need the result's dynamic dims. Building them from the
  // result itself may create problems.
  // A result has its tied init's shape, so take the dims from the init.
  SmallVector<SmallVector<Value>> resultDims;
  for (OpOperand &init : op.getDpsInitsMutable()) {
    // For STRELA it is almost never efficient to "accumulate" in the output
    // buffer like it is done in linalg.conv_2d. Hence if we use the init it
    // has to be when it is filled with a uniform value that dispatch creation
    // folds inside.
    assert(!op.payloadUsesValueFromOperand(&init) ||
         init.get().getDefiningOp<linalg::FillOp>());

    resultDims.push_back(
      IREE::Util::buildDynamicDimsForValue(loc, init.get(), rewriter)
    );
  }

  for (OpOperand *input : op.getDpsInputOperands()) {
    Value value = input->get();
    // A splat is folded into the op, during dispatch formation
    // (--compile-to=dispatch-creation), and becomes a STRELA immediate, which a
    // transfer would prevent. Other constants have to arrive through a binding.
    if (isInt32Immediate(value)) continue;

    SmallVector<Value> dims =
      IREE::Util::buildDynamicDimsForValue(loc, value, rewriter);
    Value transfer = IREE::Flow::TensorTransferOp::create(
      rewriter, loc, value.getType(), value, dims, strela
    );
    rewriter.modifyOpInPlace(op, [&] { input->set(transfer); });
  }

  rewriter.setInsertionPointAfter(op);
  for (OpResult result : op->getResults()) {
    auto back = IREE::Flow::TensorTransferOp::create(
      rewriter, loc, result, resultDims[result.getResultNumber()], host
    );
    rewriter.replaceAllUsesExcept(result, back, back);
  }
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

struct TransferWrapPass final
  : estela::impl::TransferWrapPassBase<TransferWrapPass> {
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

    SmallVector<linalg::LinalgOp> candidates;
    funcOp.walk([&](linalg::LinalgOp op) {
      if (estela::isSupportedByStrela(op)) candidates.push_back(op);
    });

    IRRewriter rewriter(&getContext());
    for (linalg::LinalgOp op : candidates) {
      auto host = IREE::Stream::AffinityAttr::lookupOrDefault(op);
      if (!host) continue;
      wrapInTransfers(rewriter, op, strela, host);
    }
  }
};

} // namespace
