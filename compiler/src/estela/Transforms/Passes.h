// Every Estela pass. The declarations and the registration are generated from
// Passes.td.

#ifndef ESTELA_TRANSFORMS_PASSES_H_
#define ESTELA_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::estela {

#define GEN_PASS_DECL
#include "Passes.h.inc" // IWYU pragma: export

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc" // IWYU pragma: export

} // namespace mlir::estela

#endif // ESTELA_TRANSFORMS_PASSES_H_
