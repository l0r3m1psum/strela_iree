#include "estela/Transforms/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "estela/Utils/ElementRange.h"
#include "estela/Utils/Finders.h"
#include "estela/Utils/StrelaContraction.h"

namespace mlir::estela {
#define GEN_PASS_DEF_STRELAWIDENOPERANDSPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// extsi(input) as i32, elementwise: the widening the host runs (or
// const-eval, for constant weights). The sizes of a dynamic input are taken
// from `shapeSource`, a tensor of the same shape: the init of the op producing
// `input`, so that the widening is its only user.
static linalg::GenericOp
widen(RewriterBase &rewriter, Location loc, Value input, Value shapeSource) {
  auto type = cast<RankedTensorType>(input.getType());
  Type i32 = rewriter.getI32Type();
  Value empty = tensor::EmptyOp::create(
    rewriter, loc, tensor::getMixedSizes(rewriter, loc, shapeSource), i32
  );
  AffineMap id = rewriter.getMultiDimIdentityMap(type.getRank());
  return linalg::GenericOp::create(
    rewriter, loc, TypeRange{empty.getType()}, ValueRange{input},
    ValueRange{empty}, ArrayRef<AffineMap>{id, id},
    SmallVector<utils::IteratorType>(type.getRank(), utils::IteratorType::parallel),
    [&](OpBuilder &b, Location loc, ValueRange args) {
      linalg::YieldOp::create(
        b, loc, ValueRange{arith::ExtSIOp::create(b, loc, i32, args[0])}
      );
    }
  );
}

struct StrelaWidenOperandsPass final
    : estela::impl::StrelaWidenOperandsPassBase<StrelaWidenOperandsPass> {
  void runOnOperation() override {
    IREE::Util::FuncOp funcOp = getOperation();
    strela::TargetDescAttr desc = estela::findStrelaTargetDesc(funcOp);
    if (!desc) {
      funcOp.emitError() << "no STRELA device: set one with "
                            "--iree-hal-target-device=strela0=strela";
      return signalPassFailure();
    }

    // Only what TransferWrap will pin once widened: a matmul left on the CPU
    // keeps its narrow operands. The multiplier check holds on the narrow
    // types already, and the extsi keeps it provable afterwards.
    SmallVector<linalg::MatmulOp> ops;
    funcOp.walk([&](linalg::MatmulOp op) {
      bool narrow = llvm::any_of(op.getDpsInputs(), [](Value v) {
        return !getElementTypeOrSelf(v.getType()).isInteger(32);
      });
      if (narrow && !op.hasUserDefinedMaps()
          && op.getCast() == linalg::TypeFn::cast_signed
          && succeeded(estela::matchStrelaContraction(
               op, desc, /*asIfTransposed=*/true, /*asIfWidened=*/true))
          && estela::operandsFitMultiplier(op, desc)) {
        ops.push_back(op);
      }
    });

    IRRewriter rewriter(&getContext());
    for (linalg::MatmulOp op : ops) {
      rewriter.setInsertionPoint(op);
      Location loc = op.getLoc();
      SmallVector<Value> inputs;
      SmallVector<linalg::GenericOp> widenOps;
      for (Value input : op.getDpsInputs()) {
        if (getElementTypeOrSelf(input.getType()).isInteger(32)) {
          inputs.push_back(input);
          continue;
        }
        auto producer = input.getDefiningOp<linalg::GenericOp>();
        Value shapeSource = producer && producer->getNumResults() == 1
                              ? producer.getDpsInits()[0] : input;
        linalg::GenericOp widenOp = widen(rewriter, loc, input, shapeSource);
        inputs.push_back(widenOp.getResult(0));
        widenOps.push_back(widenOp);
      }
      rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, op.getResultTypes(), inputs, op.getOutputs()
      );

      // Where the widening is now the only user of an elementwise op (the
      // rescale of the layer before), that op writes i32 itself: the extsi
      // goes into its body, right after the truncation, and no i8 tensor is
      // left between the two. (Dispatch formation never fuses a truncation
      // with an extension, so the widening would be a dispatch of its own.)
      for (linalg::GenericOp widenOp : widenOps) {
        OpOperand *widenInput = widenOp.getDpsInputOperand(0);
        auto producer = widenInput->get().getDefiningOp<linalg::GenericOp>();
        if (!producer || producer->getNumResults() != 1
            || !widenInput->get().hasOneUse()
            || !linalg::areElementwiseOpsFusable(widenInput)) {
          continue;
        }
        rewriter.setInsertionPoint(widenOp);
        FailureOr<linalg::ElementwiseOpFusionResult> fused =
          linalg::fuseElementwiseOps(rewriter, widenInput);
        if (failed(fused)) continue;
        rewriter.replaceOp(
          widenOp, fused->replacements.lookup(widenOp.getResult(0))
        );
        rewriter.eraseOp(producer);
      }
    }
  }
};

} // namespace
