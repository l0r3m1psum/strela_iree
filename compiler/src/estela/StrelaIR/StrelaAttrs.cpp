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

LogicalResult
ColAddrAttr::verify(
  function_ref<InFlightDiagnostic()> emitError, uint32_t binding,
  Attribute offset, WgAxisParamsAttr wgXParams, WgAxisParamsAttr wgYParams
) {
  auto intAttr = dyn_cast<IntegerAttr>(offset);
  if (!isa<PushConstantAttr>(offset) &&
      !(intAttr && intAttr.getType().isSignlessInteger(32))) {
    return emitError() << "offset must be an i32 or a #strela.push_constant, "
                       << "got " << offset;
  }
  return success();
}

} // namespace mlir::strela

#define GET_ATTRDEF_CLASSES
#include "StrelaAttrs.cpp.inc"
