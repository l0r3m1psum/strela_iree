#include "estela/Transforms/Passes.h"

#include <variant>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"
#include "estela/Utils/Bitstreams.h"
#include "estela/Utils/Finders.h"

namespace mlir::estela {
#define GEN_PASS_DEF_MATERIALIZEBINDINGSPASS
#include "Passes.h.inc"
} // namespace mlir::estela

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
// executable source and MaterializeBindingsPass restates it as the columns of a
// strela.job.

// The push constant holding an offset that is only known at dispatch time.
struct PushConstant {
  uint32_t ordinal;
};

// Where one STRELA kernel argument comes from, as far as the compiler is
// concerned. How this eventually reaches the driver is a serialization concern
// and deliberately not modeled here.
struct StrelaBinding {
  // Index into the dispatch binding list, i.e. the hal.pipeline.layout ordinal.
  uint32_t binding = 0;
  // The subspan byte offset: known at compile time (zero for most bindings,
  // non-zero after FuseDispatchBindings packs several operands into one
  // buffer), or varying per dispatch site and passed in a push constant.
  std::variant<uint32_t, PushConstant> offset = 0u;
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

// Appended to every diagnostic that comes from failing to read a subspan byte
// offset. Offsets are non-zero at all only because Stream's FuseDispatchBindings
// merges bindings and either folds the per-dispatch-site offset into a constant
// or passes it in push constants; with that pass off they are all literal zeros.
static constexpr const char *strelaUnfusedBindingsHint =
  "; retry with --iree-scheduling-optimize-bindings=false, which stops "
  "FuseDispatchBindings from folding per-dispatch-site offsets into push "
  "constants and leaves every subspan offset a literal zero";

// Recognizes the forms of a `hal.interface.binding.subspan` byte offset the
// executable format can describe, a 32 bit immediate or push constant:
//   * absent or a literal zero  -> offset is zero
//   * any other literal         -> that constant offset
//   * index_castui(load ORD)    -> push constant ORD
// A 64 bit offset arrives as two push constants, (hi << 32) | lo, whose high
// half is only known at dispatch time. Stream's PackDispatchOperands splits it
// unless --iree-stream-resource-index-bits=32, so the diagnostic says so.
static LogicalResult
matchBindingOffset(Operation *op, Value byteOffset, StrelaBinding &binding) {
  binding.offset = 0u;
  if (!byteOffset) return success();

  byteOffset = lookThroughAssume(byteOffset);

  llvm::APInt literal;
  if (matchPattern(byteOffset, m_ConstantInt(&literal))) {
    if (literal.isNegative() || literal.getActiveBits() > 32) {
      return op->emitOpError("byte offset ") << literal.getSExtValue()
             << " does not fit in 32 bits";
    }
    binding.offset = static_cast<uint32_t>(literal.getZExtValue());
    return success();
  }

  if (auto castOp = byteOffset.getDefiningOp<arith::IndexCastUIOp>()) {
    Value in = lookThroughAssume(castOp.getIn());
    if (auto loadOp = in.getDefiningOp<IREE::HAL::InterfaceConstantLoadOp>()) {
      binding.offset =
        PushConstant{static_cast<uint32_t>(loadOp.getOrdinal().getZExtValue())};
      return success();
    }
    if (in.getDefiningOp<arith::OrIOp>()) {
      return op->emitOpError(
               "byte offset is a 64 bit value split over two push constants; "
               "STRELA takes 32 bit offsets, retry with "
               "--iree-stream-resource-index-bits=32"
             );
    }
  }
  return op->emitOpError(
           "could not resolve the byte offset of this binding"
         ) << strelaUnfusedBindingsHint;
}

// True if a dispatch tensor load/store reads or writes the tensor densely from
// its origin, which is what lets the driver derive the element count from the
// binding length alone.
static bool
coversWholeTensor(ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> strides) {
  return llvm::all_of(offsets, [](OpFoldResult v) { return isConstantIntValue(v, 0); })
      && llvm::all_of(strides, [](OpFoldResult v) { return isConstantIntValue(v, 1); });
}

// Resolves the dispatch tensor `op` loads or stores to the binding it is a
// view of, reporting failures at `op`.
static FailureOr<StrelaBinding>
matchBinding(Operation *op, Value dispatchTensor) {
  auto subspanOp =
    dispatchTensor.getDefiningOp<IREE::HAL::InterfaceBindingSubspanOp>();
  if (!subspanOp) {
    return op->emitOpError("does not access a hal.interface.binding.subspan");
  }

  StrelaBinding binding;
  binding.binding = static_cast<uint32_t>(subspanOp.getBinding().getZExtValue());
  if (failed(matchBindingOffset(op, subspanOp.getByteOffset(), binding))) {
    return failure();
  }
  return binding;
}

struct StrelaEntryPointInfo {
  // The strela.* op itself and the store of its result, which
  // MaterializeBindingsPass replaces with the job.
  Operation *kernelOp = nullptr;
  Operation *resultOp = nullptr;
  SmallVector<StrelaBinding> operands;
  StrelaBinding result;
};

// Finds the single strela.* op of an entry point.
static FailureOr<Operation *>
findKernelOp(func::FuncOp funcOp) {
  Operation *kernelOp = nullptr;
  for (Operation &op : funcOp.getOps()) {
    if (isa<strela::MatvecOp>(&op)) {
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
    FailureOr<StrelaBinding> binding = matchBinding(loadOp, loadOp.getSource());
    if (failed(binding)) return failure();
    info.operands.push_back(*binding);
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
  FailureOr<StrelaBinding> resultBinding = matchBinding(storeOp, storeOp.getTarget());
  if (failed(resultBinding)) return failure();
  info.result = *resultBinding;
  info.resultOp = storeOp;

  return info;
}

//===--------------------------------------------------------------------===//
// The matvec job
//===--------------------------------------------------------------------===//

using estela::matvecBitstream;
using estela::strelaKernelColumns;
using estela::strelaKernelRows;
using estela::strelaMatvecFeaturesPerJob;

// Builds the column attributes of one job.
class ColumnBuilder {
public:
  explicit ColumnBuilder(MLIRContext *context) : context(context) {}

  // min(wg_d * step, limit) bytes; this default moves nothing.
  strela::WgAxisParamsAttr
  axis(uint32_t step = 0, uint32_t limit = UINT32_MAX) const {
    return strela::WgAxisParamsAttr::get(context, step, limit);
  }

  // The address of the column `delta` bytes into the tensor of `binding`, for
  // workgroup (0, 0), plus the two axis terms.
  strela::ColAddrAttr
  address(
    const StrelaBinding &binding, uint32_t delta,
    strela::WgAxisParamsAttr x, strela::WgAxisParamsAttr y
  ) const {
    Attribute offset;
    if (auto *constant = std::get_if<PushConstant>(&binding.offset)) {
      offset = strela::PushConstantAttr::get(context, constant->ordinal);
    } else {
      offset = IntegerAttr::get(
        IntegerType::get(context, 32), std::get<uint32_t>(binding.offset)
      );
    }
    return strela::ColAddrAttr::get(
      context, binding.binding, offset, delta, x, y
    );
  }

  // A column that moves no words.
  strela::OutColAttr
  unusedOutput() const {
    auto offset = IntegerAttr::get(IntegerType::get(context, 32), 0);
    auto addr = strela::ColAddrAttr::get(context, 0, offset, 0, axis(), axis());
    return strela::OutColAttr::get(context, addr, 0);
  }

private:
  MLIRContext *context;
};

// result[n, f] = sum_c lhs[n, c] * rhs[c, f], one job per workgroup (x, y) of
// the (ceildiv(F, 3), N, 1) grid ResolveWorkgroupCountPass sets. The job
// computes the features f0 .. f0 + 2, f0 = min(3x, F - 3), of row n = y: the row
// enters input column 0, the rhs columns f0 .. f0 + 2 (stride F) enter input
// columns 1-3, and their dot products with the row leave by output columns
// 1-3. The last job overlaps the one before it when 3 does not divide F,
// rewriting the same values.
static FailureOr<strela::JobOp>
buildMatvecJob(
  OpBuilder &builder,
  const StrelaEntryPointInfo &info,
  strela::TargetDescAttr desc
) {
  auto matvecOp = cast<strela::MatvecOp>(info.kernelOp);
  if (!desc) {
    return matvecOp.emitOpError("has no STRELA target description in scope");
  }
  if (desc.getRows() != strelaKernelRows ||
      desc.getColumns() != strelaKernelColumns) {
    return matvecOp.emitOpError("the hand-written STRELA kernels need a ")
           << strelaKernelRows << "x" << strelaKernelColumns << " fabric, got "
           << desc.getRows() << "x" << desc.getColumns()
           << "; other fabrics need the mapper";
  }
  auto rhsType = cast<RankedTensorType>(matvecOp.getRhs().getType());
  if (!rhsType.hasStaticShape()) {
    return matvecOp.emitOpError("needs a static rhs, got ") << rhsType;
  }
  int64_t c = rhsType.getDimSize(0);
  int64_t f = rhsType.getDimSize(1);
  if (f < strelaMatvecFeaturesPerJob) {
    return matvecOp.emitOpError("needs F of at least ")
           << strelaMatvecFeaturesPerJob << ", got " << f;
  }

  if (c * f > desc.getMaxColumnWords()) {
    return matvecOp.emitOpError("needs C * F of at most ")
           << desc.getMaxColumnWords() << " words, the span a strided STRELA "
           << "column can cover, got " << c << " * " << f;
  }

  MLIRContext *context = builder.getContext();
  ColumnBuilder columns(context);
  constexpr int64_t word = sizeof(uint32_t);
  // Each x workgroup moves 3 features further, clamped so the last job ends at F.
  strela::WgAxisParamsAttr features = columns.axis(
    word * strelaMatvecFeaturesPerJob, word * (f - strelaMatvecFeaturesPerJob)
  );

  SmallVector<Attribute> inputs, outputs;
  strela::ColAddrAttr row = columns.address(
    info.operands[0], 0, columns.axis(), columns.axis(word * c)
  );
  inputs.push_back(strela::InpColAttr::get(context, row, c, 1));
  outputs.push_back(columns.unusedOutput());

  for (int64_t k = 0; k < strelaMatvecFeaturesPerJob; ++k) {
    strela::ColAddrAttr weights = columns.address(
      info.operands[1], word * k, features, columns.axis()
    );
    inputs.push_back(strela::InpColAttr::get(context, weights, c, f));

    strela::ColAddrAttr result = columns.address(
      info.result, word * k, features, columns.axis(word * f)
    );
    outputs.push_back(strela::OutColAttr::get(context, result, 1));
  }

  auto bitstream = matvecBitstream(static_cast<uint16_t>(c));
  SmallVector<int32_t> words(bitstream.begin(), bitstream.end());
  return strela::JobOp::create(
    builder, info.resultOp->getLoc(), builder.getDenseI32ArrayAttr(words),
    builder.getArrayAttr(inputs), builder.getArrayAttr(outputs)
  );
}

// Replaces the dispatch interface of a STRELA kernel and the kernel itself with
// one strela.job, deleting the `hal.interface.*` ops, the push constant loads
// and the arithmetic that reassembled the byte offsets from them.
struct MaterializeBindingsPass final
    : estela::impl::MaterializeBindingsPassBase<MaterializeBindingsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    FailureOr<StrelaEntryPointInfo> info = analyzeEntryPoint(funcOp);
    if (failed(info)) return signalPassFailure();

    OpBuilder builder(info->resultOp);
    strela::TargetDescAttr desc = estela::findStrelaTargetDesc(funcOp);
    if (failed(buildMatvecJob(builder, *info, desc))) {
      return signalPassFailure();
    }
    info->resultOp->erase();
    info->kernelOp->erase();

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
