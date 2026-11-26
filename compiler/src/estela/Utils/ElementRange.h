// What the compiler can prove about the values a tensor holds: the STRELA
// multipliers use only the low bits of each i32 operand (principle 8), so a
// contraction is offloaded only if its operands are proven to fit.

#ifndef ESTELA_UTILS_ELEMENTRANGE_H_
#define ESTELA_UTILS_ELEMENTRANGE_H_

#include "estela/StrelaIR/StrelaAttrs.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir::estela {

// True if both inputs of the contraction `op` are proven to fit the
// multiplier width of `desc`. Only the frontend can prove it: in the backend
// every operand is loaded from a binding and its range is unknown, so the
// backend trusts what the frontend pinned to STRELA.
bool operandsFitMultiplier(linalg::LinalgOp op, strela::TargetDescAttr desc);

} // namespace mlir::estela

#endif // ESTELA_UTILS_ELEMENTRANGE_H_
