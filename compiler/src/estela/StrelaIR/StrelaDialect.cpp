#include "estela/StrelaIR/StrelaDialect.h"

#include "estela/StrelaIR/StrelaOps.h"
#include "estela/StrelaIR/StrelaAttrs.h"

#include "StrelaDialect.cpp.inc"

namespace mlir::strela {

void StrelaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "StrelaOps.cpp.inc"
  >();
  registerAttributes();
}

} // namespace mlir::strela
