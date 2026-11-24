#include "estela/Transforms/Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"
#include "estela/Utils/Bitstreams.h"
#include "estela/Utils/Finders.h"

namespace mlir::estela {
#define GEN_PASS_DEF_CONVERTTOSTRELAABIPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

using estela::KernelColumn;
using estela::matvecBitstream;
using estela::matvecColumns;
using estela::strelaKernelColumns;
using estela::strelaKernelRows;
using estela::strelaMatvecFeaturesPerJob;

namespace {

//===--------------------------------------------------------------------===//
// Phase 1: the HAL interface, one op at a time
//===--------------------------------------------------------------------===//
//
// The HAL hands a dispatch to the driver as (workgroup count, push constants,
// bindings). Each interface op becomes its STRELA counterpart with the same
// result type, so its users don't change; the HAL dialect is then illegal, and
// an interface op without a counterpart (hal.interface.workgroup.count, say) is
// reported where it is.

struct ConvertConstantLoad final
    : OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(
    IREE::HAL::InterfaceConstantLoadOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter
  ) const override {
    if (!op.getType().isInteger(32)) {
      return rewriter.notifyMatchFailure(op, "push constants are i32");
    }
    rewriter.replaceOpWithNewOp<strela::PushConstantOp>(
      op, op.getType(),
      rewriter.getI32IntegerAttr(op.getOrdinal().getZExtValue())
    );
    return success();
  }
};

struct ConvertBindingSubspan final
    : OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(
    IREE::HAL::InterfaceBindingSubspanOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOpWithNewOp<strela::BindingOp>(
      op, op.getType(),
      rewriter.getI32IntegerAttr(op.getBinding().getZExtValue()),
      adaptor.getByteOffset(), adaptor.getDynamicDims()
    );
    return success();
  }
};

struct ConvertWorkgroupId final
    : OpConversionPattern<IREE::HAL::InterfaceWorkgroupIDOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(
    IREE::HAL::InterfaceWorkgroupIDOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter
  ) const override {
    rewriter.replaceOpWithNewOp<strela::WorkgroupIdOp>(
      op, op.getType(),
      rewriter.getI32IntegerAttr(op.getDimension().getZExtValue())
    );
    return success();
  }
};

static LogicalResult
convertInterface(func::FuncOp funcOp) {
  MLIRContext *context = funcOp.getContext();
  ConversionTarget target(*context);
  target.addIllegalDialect<IREE::HAL::HALDialect>();
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  RewritePatternSet patterns(context);
  patterns.add<ConvertConstantLoad, ConvertBindingSubspan, ConvertWorkgroupId>(
    context
  );
  return applyPartialConversion(funcOp, target, std::move(patterns));
}

//===--------------------------------------------------------------------===//
// Phase 2: the kernel and its loads and store become a job
//===--------------------------------------------------------------------===//

// One operand or the result of the kernel: the binding its tensor is in, and
// the tile of it the load or store moves.
struct KernelTensor {
  Operation *op = nullptr; // the dispatch.tensor.load or store
  strela::BindingOp binding;
  // The whole tensor's shape, which gives the strides.
  SmallVector<int64_t> shape;
  // The tile: where it starts (arithmetic on the workgroup ids) and its static
  // sizes.
  SmallVector<OpFoldResult> offsets;
  SmallVector<int64_t> sizes;
};

static FailureOr<KernelTensor>
matchKernelTensor(
  Operation *op, Value dispatchTensor, ArrayRef<OpFoldResult> offsets,
  ArrayRef<OpFoldResult> sizes, ArrayRef<OpFoldResult> strides
) {
  if (!llvm::all_of(strides, [](OpFoldResult s) { return isConstantIntValue(s, 1); })) {
    return op->emitOpError("moves a strided tile; STRELA kernels take and "
                           "produce dense tiles");
  }
  auto bindingOp = dispatchTensor.getDefiningOp<strela::BindingOp>();
  if (!bindingOp) {
    return op->emitOpError("does not access a strela.binding");
  }
  auto type = cast<IREE::TensorExt::DispatchTensorType>(dispatchTensor.getType());
  if (type.getShape().size() != 2) {
    return op->emitOpError("expected a rank 2 binding, got ") << type;
  }
  SmallVector<int64_t> staticSizes;
  for (OpFoldResult size : sizes) {
    std::optional<int64_t> constant = getConstantIntValue(size);
    if (!constant) {
      return op->emitOpError("moves a tile of dynamic size; every STRELA job "
                             "runs a whole tile");
    }
    staticSizes.push_back(*constant);
  }
  return KernelTensor{
    op, bindingOp, llvm::to_vector(type.getShape()), llvm::to_vector(offsets),
    staticSizes
  };
}

struct StrelaEntryPoint {
  strela::MatvecOp matvecOp;
  IREE::TensorExt::DispatchTensorStoreOp storeOp;
  // Indexed by KernelColumn::Operand.
  std::array<KernelTensor, 3> tensors;
};

// Finds the single strela.matvec of an entry point and the loads and store
// around it. Every failure is reported at the offending op: a STRELA dispatch
// that cannot be described exactly must not be silently handed a kernel.
static FailureOr<StrelaEntryPoint>
analyzeEntryPoint(func::FuncOp funcOp) {
  StrelaEntryPoint entry;
  for (auto matvecOp : funcOp.getOps<strela::MatvecOp>()) {
    if (entry.matvecOp) {
      return matvecOp.emitOpError()
             << "second STRELA op in one dispatch; the backend can only "
                "serialize a single kernel per entry point";
    }
    entry.matvecOp = matvecOp;
  }
  if (!entry.matvecOp) {
    return funcOp.emitOpError()
           << "no STRELA op in the dispatch body: ConvertLinalgToStrela did "
              "not match anything, so there is no kernel to serialize";
  }

  for (auto [i, operand] : llvm::enumerate(entry.matvecOp->getOperands())) {
    auto loadOp = operand.getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (!loadOp) {
      return entry.matvecOp.emitOpError()
             << "operand is not loaded directly from a dispatch binding; the "
                "kernel would need a computation STRELA cannot perform";
    }
    FailureOr<KernelTensor> tensor = matchKernelTensor(
      loadOp, loadOp.getSource(), loadOp.getMixedOffsets(),
      loadOp.getMixedSizes(), loadOp.getMixedStrides()
    );
    if (failed(tensor)) return failure();
    entry.tensors[i] = *tensor;
  }

  Value result = entry.matvecOp.getResult();
  if (!result.hasOneUse()) {
    return entry.matvecOp.emitOpError()
           << "result must be stored to a binding exactly once";
  }
  entry.storeOp =
    dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(*result.user_begin());
  if (!entry.storeOp) {
    return entry.matvecOp.emitOpError()
           << "result is consumed by something other than a dispatch store";
  }
  FailureOr<KernelTensor> tensor = matchKernelTensor(
    entry.storeOp, entry.storeOp.getTarget(), entry.storeOp.getMixedOffsets(),
    entry.storeOp.getMixedSizes(), entry.storeOp.getMixedStrides()
  );
  if (failed(tensor)) return failure();
  entry.tensors[KernelColumn::Result] = *tensor;
  return entry;
}

constexpr int64_t word = sizeof(uint32_t);

// Builds the column ops of one job from the kernel's column table.
class ColumnBuilder {
public:
  ColumnBuilder(OpBuilder &builder, StrelaEntryPoint &entry)
    : builder(builder), entry(entry),
      columnType(strela::ColumnType::get(builder.getContext())) {}

  FailureOr<Value> input(const KernelColumn &column);
  FailureOr<Value> output(const KernelColumn &column);
  Value unusedOutput();

private:
  // Row-major element strides of a tensor whose last dimension is static.
  FailureOr<std::array<int64_t, 2>> strides(KernelTensor &tensor);
  Value address(KernelTensor &tensor, const KernelColumn &column,
                std::array<int64_t, 2> strides);

  OpBuilder &builder;
  StrelaEntryPoint &entry;
  Type columnType;
};

FailureOr<std::array<int64_t, 2>>
ColumnBuilder::strides(KernelTensor &tensor) {
  if (ShapedType::isDynamic(tensor.shape[1])) {
    return tensor.op->emitOpError("needs a static row length");
  }
  return std::array<int64_t, 2>{tensor.shape[1], 1};
}

// The byte offset of the binding plus
//   word * sum_i (tile offset_i + start_i) * stride_i
// for the first word of `column`, composed with whatever computes the binding
// offset and the tile offsets, which are arithmetic on the workgroup ids.
// foldColumnAddress takes it apart again: the push constant and the constants
// into `offset + delta`, the workgroup ids into the workgroup terms.
Value
ColumnBuilder::address(
  KernelTensor &tensor, const KernelColumn &column,
  std::array<int64_t, 2> strides
) {
  MLIRContext *context = builder.getContext();
  Location loc = tensor.op->getLoc();
  Value byteOffset = tensor.binding.getByteOffset();
  SmallVector<OpFoldResult> operands = {
    byteOffset ? OpFoldResult(byteOffset) : builder.getIndexAttr(0)
  };
  AffineExpr expr = getAffineSymbolExpr(0, context);
  for (int i = 0; i < 2; ++i) {
    expr = expr + (getAffineSymbolExpr(i + 1, context) + column.start[i]) *
                    (word * strides[i]);
    operands.push_back(tensor.offsets[i]);
  }
  AffineMap map = AffineMap::get(0, operands.size(), expr);
  OpFoldResult result =
    affine::makeComposedFoldedAffineApply(builder, loc, map, operands);
  return getValueOrCreateConstantIndexOp(builder, loc, result);
}

FailureOr<Value>
ColumnBuilder::input(const KernelColumn &column) {
  KernelTensor &tensor = entry.tensors[column.operand];
  FailureOr<std::array<int64_t, 2>> s = strides(tensor);
  if (failed(s)) return failure();
  unsigned walk = *column.walk;
  return strela::InpColumnOp::create(
           builder, tensor.op->getLoc(), columnType,
           builder.getI32IntegerAttr(tensor.binding.getOrdinal()),
           address(tensor, column, *s),
           builder.getI32IntegerAttr(tensor.sizes[walk]),
           builder.getI32IntegerAttr((*s)[walk])
  ).getResult();
}

FailureOr<Value>
ColumnBuilder::output(const KernelColumn &column) {
  KernelTensor &tensor = entry.tensors[column.operand];
  FailureOr<std::array<int64_t, 2>> s = strides(tensor);
  if (failed(s)) return failure();
  return strela::OutColumnOp::create(
           builder, tensor.op->getLoc(), columnType,
           builder.getI32IntegerAttr(tensor.binding.getOrdinal()),
           address(tensor, column, *s), builder.getI32IntegerAttr(1)
  ).getResult();
}

Value
ColumnBuilder::unusedOutput() {
  return strela::OutColumnOp::create(
           builder, entry.storeOp.getLoc(), columnType, IntegerAttr(), Value(),
           IntegerAttr()
  ).getResult();
}

// result[n, f] = sum_c lhs[n, c] * rhs[c, f] on one tile: row n and features
// f0 .. f0 + 2, f0 = min(3x, F - 3), the job of workgroup (x, y = n) of the
// (ceildiv(F, 3), N, 1) grid IREE's tiling made. The last job overlaps the one
// before it when 3 does not divide F, rewriting the same values.
static LogicalResult
buildMatvecJob(
  OpBuilder &builder, StrelaEntryPoint &entry, strela::TargetDescAttr desc
) {
  strela::MatvecOp matvecOp = entry.matvecOp;
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
  bool transposed = matvecOp.getTransposeRhs();
  auto lhsType = cast<RankedTensorType>(matvecOp.getLhs().getType());
  auto rhsType = cast<RankedTensorType>(matvecOp.getRhs().getType());
  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      lhsType.getDimSize(0) != 1 ||
      rhsType.getDimSize(transposed ? 0 : 1) != strelaMatvecFeaturesPerJob) {
    return matvecOp.emitOpError("is not one job's tile (one row by ")
           << strelaMatvecFeaturesPerJob << " features), got " << lhsType
           << " and " << rhsType;
  }
  int64_t c = rhsType.getDimSize(transposed ? 1 : 0);
  // A weight column is C words: contiguous for F x C weights, a row of the
  // whole C x F weights apart otherwise.
  int64_t rowLength = entry.tensors[KernelColumn::Rhs].shape[1];
  int64_t span = transposed ? c : c * rowLength;
  if (ShapedType::isDynamic(rowLength) || span > desc.getMaxColumnWords()) {
    return matvecOp.emitOpError("needs a weight column of at most ")
           << desc.getMaxColumnWords() << " words, the span a STRELA column "
           << "can cover, got " << span;
  }

  ColumnBuilder columns(builder, entry);
  estela::KernelColumns table = matvecColumns(transposed);
  SmallVector<Value> inputs, outputs;
  for (const std::optional<KernelColumn> &column : table.inputs) {
    FailureOr<Value> value = columns.input(*column);
    if (failed(value)) return failure();
    inputs.push_back(*value);
  }
  for (const std::optional<KernelColumn> &column : table.outputs) {
    if (!column) {
      outputs.push_back(columns.unusedOutput());
      continue;
    }
    FailureOr<Value> value = columns.output(*column);
    if (failed(value)) return failure();
    outputs.push_back(*value);
  }

  auto bitstream = matvecBitstream(static_cast<uint16_t>(c));
  SmallVector<int32_t> words(bitstream.begin(), bitstream.end());
  strela::JobOp::create(
    builder, entry.storeOp.getLoc(), builder.getDenseI32ArrayAttr(words),
    inputs, outputs
  );
  return success();
}

// Replaces the dispatch interface of a STRELA kernel with its strela.*
// counterparts, then the kernel, its loads and its store with one strela.job
// whose column addresses are SSA index arithmetic on the push constants.
struct ConvertToStrelaABIPass final
    : estela::impl::ConvertToStrelaABIPassBase<ConvertToStrelaABIPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (failed(convertInterface(funcOp))) return signalPassFailure();

    FailureOr<StrelaEntryPoint> entry = analyzeEntryPoint(funcOp);
    if (failed(entry)) return signalPassFailure();

    OpBuilder builder(entry->storeOp);
    strela::TargetDescAttr desc = estela::findStrelaTargetDesc(funcOp);
    if (failed(buildMatvecJob(builder, *entry, desc))) {
      return signalPassFailure();
    }
    entry->storeOp->erase();
    entry->matvecOp->erase();

    // The loads and the bindings are now unused, and so is whatever computed
    // only their sizes. Erase in reverse so consumers go before producers.
    SmallVector<Operation *> ops;
    for (Operation &op : funcOp.getOps()) ops.push_back(&op);
    for (Operation *op : llvm::reverse(ops)) {
      if (isOpTriviallyDead(op)) op->erase();
    }
  }
};

} // namespace
