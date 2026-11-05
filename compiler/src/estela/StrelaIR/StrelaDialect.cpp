#include "estela/StrelaIR/StrelaDialect.h"

#include "estela/StrelaIR/StrelaOps.h"

#include "StrelaDialect.cpp.inc"

namespace mlir::strela {

void StrelaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "StrelaOps.cpp.inc"
  >();
}

} // namespace mlir::strela
