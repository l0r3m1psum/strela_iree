#include "estela/Dialect/Strela/IR/StrelaOps.h"

namespace mlir::strela {

// Both binding ops carry the same optional ordinal pair.
static LogicalResult
verifyBindingOffsets(Operation *op, std::optional<ArrayRef<int64_t>> offsets) {
  if (offsets && offsets->size() != 2) {
    return op->emitOpError()
           << "expected exactly two offset ordinals, the low and high halves "
              "of the byte offset, but got " << offsets->size();
  }
  return success();
}

LogicalResult BindingLoadOp::verify() {
  return verifyBindingOffsets(*this, getOffsets());
}

LogicalResult BindingStoreOp::verify() {
  return verifyBindingOffsets(*this, getOffsets());
}

} // namespace mlir::strela

#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
