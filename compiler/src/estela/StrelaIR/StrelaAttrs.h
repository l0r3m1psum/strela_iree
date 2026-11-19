#ifndef ESTELA_STRELAIR_STRELAATTRS_H_
#define ESTELA_STRELAIR_STRELAATTRS_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "StrelaAttrs.h.inc"

namespace mlir::strela {

// The key of #strela.target_desc in the #hal.executable.target configuration.
inline constexpr llvm::StringLiteral targetDescKey = "strela.target_desc";

} // namespace mlir::strela

#endif // ESTELA_STRELAIR_STRELAATTRS_H_
