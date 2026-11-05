#include "estela/StrelaIR/StrelaOps.h"

namespace mlir::strela {

// Both binding ops describe their byte offset the same way: at most one of the
// two optional attributes, with neither meaning a compile-time zero.
static LogicalResult
verifyBindingOffsets(Operation *op, std::optional<ArrayRef<int64_t>> offsets,
                     std::optional<uint64_t> byteOffset) {
  if (offsets && offsets->size() != 1 && offsets->size() != 2) {
    return op->emitOpError()
           << "expected one offset ordinal, or two for the low and high halves "
              "of the byte offset, but got " << offsets->size();
  }
  if (offsets && byteOffset) {
    return op->emitOpError()
           << "carries both push constant offset ordinals and a static byte "
              "offset; the offset comes from one place or the other";
  }
  if (byteOffset && *byteOffset == 0) {
    return op->emitOpError()
           << "a zero byte offset is expressed by omitting the attribute";
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
  return verifyBindingOffsets(*this, getOffsets(), getByteOffset());
}

LogicalResult BindingStoreOp::verify() {
  return verifyBindingOffsets(*this, getOffsets(), getByteOffset());
}

} // namespace mlir::strela

#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
