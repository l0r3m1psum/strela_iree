#include "estela/Dialect/Strela/Transforms/Passes.h"

#include <optional>
#include <utility>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "estela/Dialect/Strela/IR/StrelaDialect.h"
#include "estela/Dialect/Strela/IR/StrelaOps.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

//===--------------------------------------------------------------------===//
// Executable interface extraction
//===--------------------------------------------------------------------===//
//
// The HAL hands a dispatch to the driver as (workgroup count, push constants,
// bindings). Nothing about which binding is which kernel argument survives into
// the binary unless we put it there, so these helpers read it back out of the
// executable source and StrelaMaterializeBindingsPass restates it as
// strela.binding_load / strela.binding_store.

// Where one STRELA kernel argument comes from, as far as the compiler is
// concerned. How this eventually reaches the driver is a serialization concern
// and deliberately not modelled here.
struct StrelaBinding {
  // Index into the dispatch binding list, i.e. the hal.pipeline.layout ordinal.
  uint32_t binding = 0;
  // Push constant ordinals holding the low and high halves of the subspan byte
  // offset, or empty when that offset is a compile-time zero. The two are not
  // necessarily adjacent: IREE dedupes the constant loads, so two bindings whose
  // offsets share a low word also share that ordinal.
  std::optional<std::pair<uint32_t, uint32_t>> offsets;
};

// `util.assume.int` forwards its operands unchanged; look through it so the
// patterns below see the real producers.
static Value
lookThroughAssume(Value value) {
  while (auto assumeOp = value.getDefiningOp<IREE::Util::AssumeIntOp>()) {
    value = assumeOp->getOperand(cast<OpResult>(value).getResultNumber());
  }
  return value;
}

// Recognizes the two forms a `hal.interface.binding.subspan` byte offset takes
// in an executable source and reports where the driver can find it:
//   * absent or a literal zero                       -> offset is zero
//   * index_castui(ori(extui(load LO),
//                      shli(extui(load HI), 32)))    -> ordinals LO and HI
// Anything else we cannot describe to the driver, so serialization fails rather
// than emitting a kernel that would read from the wrong address.
//
// LO and HI are recorded independently: IREE dedupes the constant loads, so in
// a dispatch where two offsets share a low word they also share that ordinal
// and the pair is not adjacent.
// Appended to every diagnostic that comes from failing to read a subspan byte
// offset. Those offsets are only dynamic because Stream's FuseDispatchBindings
// merges bindings and folds the per-dispatch-site offsets into push constants;
// with that pass off they are all literal zeros and the pattern below is never
// exercised. Naming the flag in the message keeps a network that hits an offset
// shape we do not recognize a one-flag fix rather than a compiler change.
static constexpr const char *strelaUnfusedBindingsHint =
  "; retry with --iree-scheduling-optimize-bindings=false, which stops "
  "FuseDispatchBindings from folding per-dispatch-site offsets into push "
  "constants and leaves every subspan offset a literal zero";

// On success `offsets` is left empty for a compile-time zero and otherwise holds
// the (low, high) push constant ordinals.
static LogicalResult
matchBindingOffset(Value byteOffset,
                   std::optional<std::pair<uint32_t, uint32_t>> &offsets) {
  offsets.reset();
  if (!byteOffset) return success();

  byteOffset = lookThroughAssume(byteOffset);

  llvm::APInt literal;
  if (matchPattern(byteOffset, m_ConstantInt(&literal))) {
    return success(literal.isZero());
  }

  auto castOp = byteOffset.getDefiningOp<arith::IndexCastUIOp>();
  if (!castOp) return failure();
  auto orOp = lookThroughAssume(castOp.getIn()).getDefiningOp<arith::OrIOp>();
  if (!orOp) return failure();

  auto lowExtOp = lookThroughAssume(orOp.getLhs()).getDefiningOp<arith::ExtUIOp>();
  auto shiftOp = lookThroughAssume(orOp.getRhs()).getDefiningOp<arith::ShLIOp>();
  if (!lowExtOp || !shiftOp) return failure();

  llvm::APInt shiftAmount;
  if (!matchPattern(shiftOp.getRhs(), m_ConstantInt(&shiftAmount)) ||
      shiftAmount != 32) {
    return failure();
  }

  auto highExtOp = lookThroughAssume(shiftOp.getLhs()).getDefiningOp<arith::ExtUIOp>();
  if (!highExtOp) return failure();

  auto lowLoadOp =
    lowExtOp.getIn().getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>();
  auto highLoadOp =
    highExtOp.getIn().getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>();
  if (!lowLoadOp || !highLoadOp) return failure();

  offsets = std::make_pair(
    static_cast<uint32_t>(lowLoadOp.getOrdinal().getZExtValue()),
    static_cast<uint32_t>(highLoadOp.getOrdinal().getZExtValue())
  );
  return success();
}

// True if a dispatch tensor load/store reads or writes the tensor densely from
// its origin, which is what lets the driver derive the element count from the
// binding length alone.
static bool
coversWholeTensor(ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> strides) {
  return llvm::all_of(offsets, [](OpFoldResult v) { return isConstantIntValue(v, 0); })
      && llvm::all_of(strides, [](OpFoldResult v) { return isConstantIntValue(v, 1); });
}

// Resolves a dispatch tensor to the binding it is a view of.
static std::optional<StrelaBinding>
matchBinding(Value dispatchTensor) {
  auto subspanOp =
    dispatchTensor.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp) return std::nullopt;

  StrelaBinding binding;
  binding.binding = static_cast<uint32_t>(subspanOp.getBinding().getZExtValue());
  if (failed(matchBindingOffset(subspanOp.getByteOffset(), binding.offsets))) {
    return std::nullopt;
  }
  return binding;
}

struct StrelaEntryPointInfo {
  // The strela.* op itself, and the load/store ops each argument came from.
  // StrelaMaterializeBindingsPass uses their locations so a binding that cannot
  // be resolved is reported against the op it came from.
  Operation *kernelOp = nullptr;
  SmallVector<Operation *> operandOps;
  Operation *resultOp = nullptr;
  SmallVector<StrelaBinding> operands;
  StrelaBinding result;
};

// Finds the single strela.* op of an entry point.
static FailureOr<Operation *>
findKernelOp(func::FuncOp funcOp) {
  Operation *kernelOp = nullptr;
  for (Operation &op : funcOp.getOps()) {
    if (isa<strela::AddOp, strela::ReluOp>(&op)) {
      if (kernelOp) {
        return op.emitOpError()
               << "second STRELA op in one dispatch; the backend can only "
                  "serialize a single kernel per entry point";
      }
      kernelOp = &op;
    }
  }
  if (!kernelOp) {
    return funcOp.emitOpError()
           << "no STRELA op in the dispatch body: LinalgToStrelaPass did not "
              "match anything, so there is no kernel to serialize";
  }
  return kernelOp;
}

// Reads the host <-> kernel interface of one entry point out of its executable
// source. Every failure is reported at the offending op: a STRELA dispatch that
// cannot be described exactly must not be silently handed a kernel.
static FailureOr<StrelaEntryPointInfo>
analyzeEntryPoint(func::FuncOp funcOp) {
  FailureOr<Operation *> maybeKernelOp = findKernelOp(funcOp);
  if (failed(maybeKernelOp)) return failure();
  Operation *kernelOp = *maybeKernelOp;

  StrelaEntryPointInfo info;
  info.kernelOp = kernelOp;

  if (!isa<RankedTensorType>(kernelOp->getResult(0).getType())) {
    return kernelOp->emitOpError() << "expected a ranked tensor result";
  }

  // Every operand must come straight from a full-extent load of a binding.
  for (Value operand : kernelOp->getOperands()) {
    auto loadOp = operand.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp) {
      return kernelOp->emitOpError()
             << "operand is not loaded directly from a dispatch binding; the "
                "kernel would need a computation STRELA cannot perform";
    }
    if (!coversWholeTensor(loadOp.getMixedOffsets(), loadOp.getMixedStrides())) {
      return loadOp.emitOpError()
             << "reads a strided or offset slice; STRELA kernels only take "
                "dense whole-tensor operands";
    }
    std::optional<StrelaBinding> binding =
      matchBinding(loadOp.getSource());
    if (!binding) {
      return loadOp.emitOpError()
             << "could not resolve the binding and byte offset this operand "
                "comes from" << strelaUnfusedBindingsHint;
    }
    info.operands.push_back(*binding);
    info.operandOps.push_back(loadOp);
  }

  // The result must go straight back out to a binding.
  if (!kernelOp->getResult(0).hasOneUse()) {
    return kernelOp->emitOpError()
           << "result must be stored to a binding exactly once";
  }
  auto storeOp = dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(
    *kernelOp->getResult(0).user_begin()
  );
  if (!storeOp) {
    return kernelOp->emitOpError()
           << "result is consumed by something other than a dispatch store";
  }
  if (!coversWholeTensor(storeOp.getMixedOffsets(), storeOp.getMixedStrides())) {
    return storeOp.emitOpError()
           << "writes a strided or offset slice; STRELA kernels only produce "
              "dense whole-tensor results";
  }
  std::optional<StrelaBinding> resultBinding =
    matchBinding(storeOp.getTarget());
  if (!resultBinding) {
    return storeOp.emitOpError()
           << "could not resolve the binding and byte offset the result goes to"
           << strelaUnfusedBindingsHint;
  }
  info.result = *resultBinding;
  info.resultOp = storeOp;

  return info;
}

// Rewrites the dispatch interface of a STRELA kernel into strela.binding_load /
// strela.binding_store, deleting the `hal.interface.*` ops, the push constant
// loads and the arithmetic that reassembled the byte offsets from them.
struct StrelaMaterializeBindingsPass
    : public PassWrapper<StrelaMaterializeBindingsPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrelaMaterializeBindingsPass)

  StringRef getArgument() const override {
    return "iree-strela-materialize-bindings";
  }
  StringRef getDescription() const override {
    return "Replaces the STRELA dispatch interface with strela.binding_load/store";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<strela::StrelaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    FailureOr<StrelaEntryPointInfo> info = analyzeEntryPoint(funcOp);
    if (failed(info)) return signalPassFailure();

    OpBuilder builder(info->kernelOp);
    auto offsetsAttrOf = [&](const StrelaBinding &binding) {
      if (!binding.offsets) return DenseI64ArrayAttr();
      return builder.getDenseI64ArrayAttr(
        {binding.offsets->first, binding.offsets->second}
      );
    };

    // Operands: the load takes over from subspan + dispatch.tensor.load, and
    // keeps that load's location so diagnostics still point at the original.
    for (auto [index, binding] : llvm::enumerate(info->operands)) {
      Operation *loadOp = info->operandOps[index];
      auto bindingLoadOp = strela::BindingLoadOp::create(
        builder, loadOp->getLoc(), loadOp->getResult(0).getType(),
        static_cast<int64_t>(binding.binding), offsetsAttrOf(binding)
      );
      info->kernelOp->setOperand(index, bindingLoadOp.getResult());
    }

    // Result: the store takes over from dispatch.tensor.store.
    builder.setInsertionPointAfter(info->kernelOp);
    strela::BindingStoreOp::create(
      builder, info->resultOp->getLoc(), info->kernelOp->getResult(0),
      static_cast<int64_t>(info->result.binding), offsetsAttrOf(info->result)
    );
    info->resultOp->erase();

    // Everything that fed the old interface is now unused. Erase in reverse so
    // consumers go before their producers.
    SmallVector<Operation *> ops;
    for (Operation &op : funcOp.getOps()) ops.push_back(&op);
    for (Operation *op : llvm::reverse(ops)) {
      if (isOpTriviallyDead(op)) op->erase();
    }
  }
};

} // namespace

namespace mlir::estela {

std::unique_ptr<Pass> createMaterializeBindingsPass() {
  return std::make_unique<StrelaMaterializeBindingsPass>();
}

} // namespace mlir::estela
