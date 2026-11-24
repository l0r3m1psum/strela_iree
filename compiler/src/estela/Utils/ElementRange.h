// What the compiler can prove about the values a tensor holds: the STRELA
// multipliers use only the low bits of each i32 operand (principle 8), so a
// contraction is offloaded only if its operands are proven to fit.

#ifndef ESTELA_UTILS_ELEMENTRANGE_H_
#define ESTELA_UTILS_ELEMENTRANGE_H_

#include "estela/StrelaIR/StrelaAttrs.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir::estela {

// The signed range of the elements of `value`, a tensor or a scalar of
// signless integers, at the bit width of its element type; nullopt for other
// element types. At worst the range of the element type:
//   - a constant: the range of its values;
//   - a narrower element type: its range, so i16 values are always in
//     [-2^15, 2^15);
//   - a linalg op (tosa.cast, the widening the frontend inserts): its body
//     evaluated with each op's InferIntRangeInterface, the block arguments
//     seeded from the ranges of the operands, recursively up to `depth`;
//   - a reshape or slice: the range of its source;
//   - a tensor.concat: the union of its inputs' ranges.
std::optional<ConstantIntRanges> elementRange(Value value, unsigned depth = 6);

// True if every value `range` allows fits in `width` signed bits.
bool fitsSigned(const ConstantIntRanges &range, unsigned width);

// True if both inputs of the contraction `op` are proven to fit the
// multiplier width of `desc`. Only the frontend can prove it: in the backend
// every operand is loaded from a binding and its range is unknown, so the
// backend trusts what the frontend pinned to STRELA.
bool operandsFitMultiplier(linalg::LinalgOp op, strela::TargetDescAttr desc);

} // namespace mlir::estela

#endif // ESTELA_UTILS_ELEMENTRANGE_H_
