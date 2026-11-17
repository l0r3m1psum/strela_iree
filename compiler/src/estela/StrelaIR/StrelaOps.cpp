#include "estela/StrelaIR/StrelaOps.h"

namespace mlir::strela {

// Both binding ops describe their byte offset the same way: at most one of the
// two optional attributes, with neither meaning a compile-time zero. Both fit
// the 32 bit ValueDef of the executable format.
static LogicalResult
verifyBindingOffsets(Operation *op, std::optional<uint64_t> offsetOrdinal,
                     std::optional<uint64_t> byteOffset) {
  if (offsetOrdinal && byteOffset) {
    return op->emitOpError()
           << "carries both a push constant offset ordinal and a static byte "
              "offset; the offset comes from one place or the other";
  }
  if (offsetOrdinal && *offsetOrdinal > UINT32_MAX) {
    return op->emitOpError() << "offset ordinal does not fit in 32 bits";
  }
  if (byteOffset && *byteOffset == 0) {
    return op->emitOpError()
           << "a zero byte offset is expressed by omitting the attribute";
  }
  if (byteOffset && *byteOffset > UINT32_MAX) {
    return op->emitOpError() << "byte offset does not fit in 32 bits";
  }
  return success();
}

// TODO: is the dynamic shape comparison still needed for all dimensions?
static bool
compatible(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

// NxC times CxF gives NxF; a dynamic size is checked against nothing.
LogicalResult MatvecOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto outType = cast<RankedTensorType>(getResult().getType());
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2) {
    return emitOpError("expected rank 2 operands and result");
  }

  if (!compatible(lhsType.getDimSize(1), rhsType.getDimSize(0))) {
    return emitOpError("lhs and rhs disagree on the reduction size");
  }
  if (!compatible(outType.getDimSize(0), lhsType.getDimSize(0)) ||
      !compatible(outType.getDimSize(1), rhsType.getDimSize(1))) {
    return emitOpError("result shape does not match lhs rows and rhs columns");
  }
  return success();
}

LogicalResult BindingLoadOp::verify() {
  return verifyBindingOffsets(*this, getOffsetOrdinal(), getByteOffset());
}

LogicalResult BindingStoreOp::verify() {
  return verifyBindingOffsets(*this, getOffsetOrdinal(), getByteOffset());
}

} // namespace mlir::strela

#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
