#include "estela/Dialect/Strela/IR/StrelaDialect.h"

#include "estela/Dialect/Strela/IR/StrelaOps.h"

#include "StrelaDialect.cpp.inc"

namespace mlir::strela {

void StrelaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "StrelaOps.cpp.inc"
  >();
}

} // namespace mlir::strela
