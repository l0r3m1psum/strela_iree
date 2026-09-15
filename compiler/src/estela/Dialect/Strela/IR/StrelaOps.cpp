#include "estela/Dialect/Strela/IR/StrelaOps.h"

namespace mlir::strela {

// Both binding ops describe their byte offset the same way: at most one of the
// two optional attributes, with neither meaning a compile-time zero.
static LogicalResult
verifyBindingOffsets(Operation *op, std::optional<ArrayRef<int64_t>> offsets,
                     std::optional<uint64_t> byteOffset) {
  if (offsets && offsets->size() != 2) {
    return op->emitOpError()
           << "expected exactly two offset ordinals, the low and high halves "
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

LogicalResult BindingLoadOp::verify() {
  return verifyBindingOffsets(*this, getOffsets(), getByteOffset());
}

LogicalResult BindingStoreOp::verify() {
  return verifyBindingOffsets(*this, getOffsets(), getByteOffset());
}

} // namespace mlir::strela

#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
