// The byte address of a DMA column, folded from the index arithmetic that
// computes it in the IR into the form the runtime evaluates (ColumnAddressDef in
// strela_executable_def.fbs). The job verifier calls it to check every column,
// the serializer to write them.

#ifndef ESTELA_STRELAIR_COLUMNADDRESS_H_
#define ESTELA_STRELAIR_COLUMNADDRESS_H_

#include <array>
#include <cstdint>
#include <optional>

#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"

namespace mlir::strela {

// constant + [push constant] + min(wg_x * x.step, x.limit)
//                            + min(wg_y * y.step, y.limit)   bytes
struct ColumnAddress {
  // min(wg_d * step, limit); the default moves nothing.
  struct Term {
    uint32_t step = 0;
    uint32_t limit = UINT32_MAX;
  };

  // The ordinal of the push constant, if the address has one.
  std::optional<uint32_t> pushConstant;
  uint32_t constant = 0;
  std::array<Term, 2> wg;
};

// Folds `address`, an index computed by arith and affine ops from constants,
// strela.push_constant and strela.workgroup_id, into a ColumnAddress,
// reporting with `emitError` the part that doesn't fit.
//
// It looks through affine.apply, affine.min, arith.minsi/minui, arith.addi,
// arith.muli and arith.shli by a constant, arith.index_castui, arith.extui,
// util.assume.int, and arith.ori of operands with disjoint bits
// (lo + 2^32 * hi), so it reads the address before and after the affine ops
// are lowered. The result must be
//   c + [push constant] + sum over d in {x, y} of min(wg_d * step, limit)
// with c a constant in [0, 2^32): a positive multiple of a workgroup id is a
// term with no limit, a positive constant times min(k * wg_d, limit) the term
// min(constant * k * wg_d, constant * limit). A 64 bit offset, two push
// constants, is refused with a hint to keep offsets in 32 bits.
FailureOr<ColumnAddress>
foldColumnAddress(Value address, function_ref<InFlightDiagnostic()> emitError);

// The folded address of a used strela.inp_column or strela.out_column. Errors
// are reported at the column op.
FailureOr<ColumnAddress> getColumnAddress(Operation *columnOp);

} // namespace mlir::strela

#endif // ESTELA_STRELAIR_COLUMNADDRESS_H_
