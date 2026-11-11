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

// 1xNxC times 1xCxF gives 1xNxF; a dynamic size is checked against nothing.
LogicalResult MatvecOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto resultType = cast<RankedTensorType>(getResult().getType());
  if (lhsType.getRank() != 3 || rhsType.getRank() != 3 || resultType.getRank() != 3) {
    return emitOpError("expected rank 3 operands and result");
  }

  auto compatible = [](int64_t a, int64_t b) {
    return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
  };
  if (lhsType.getDimSize(0) != 1 || !compatible(rhsType.getDimSize(0), 1) ||
      !compatible(resultType.getDimSize(0), 1)) {
    return emitOpError("expected a batch of one");
  }
  if (!compatible(lhsType.getDimSize(2), rhsType.getDimSize(1))) {
    return emitOpError("lhs and rhs disagree on the reduction size");
  }
  if (!compatible(resultType.getDimSize(1), lhsType.getDimSize(1)) ||
      !compatible(resultType.getDimSize(2), rhsType.getDimSize(2))) {
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
