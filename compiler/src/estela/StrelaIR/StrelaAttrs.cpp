#include "estela/StrelaIR/StrelaAttrs.h"
#include "estela/StrelaIR/StrelaDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir::strela {

void
StrelaDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "StrelaAttrs.cpp.inc"
  >();
}

void
StrelaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "StrelaTypes.cpp.inc"
  >();
}

} // namespace mlir::strela

#define GET_ATTRDEF_CLASSES
#include "StrelaAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "StrelaTypes.cpp.inc"
