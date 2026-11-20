#include "estela/StrelaIR/StrelaOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

namespace mlir::strela {

// TODO: is the dynamic shape comparison still needed for all dimensions?
static bool
compatible(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

// NxC times CxF gives NxF; a dynamic size is checked against nothing.
LogicalResult MatvecOp::verify() {
  auto lhsType = cast<RankedTensorType>(getLhs().getType());
  auto rhsType = cast<RankedTensorType>(getRhs().getType());
  auto outType = cast<RankedTensorType>(getResult().getType());
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2) {
    return emitOpError("expected rank 2 operands and result");
  }

  if (!compatible(lhsType.getDimSize(1), rhsType.getDimSize(0))) {
    return emitOpError("lhs and rhs disagree on the reduction size");
  }
  if (!compatible(outType.getDimSize(0), lhsType.getDimSize(0)) ||
      !compatible(outType.getDimSize(1), rhsType.getDimSize(1))) {
    return emitOpError("result shape does not match lhs rows and rhs columns");
  }
  return success();
}

// One column of each kind per fabric column. Checked only inside a STRELA
// executable variant, the only place the description can be found from here.
LogicalResult JobOp::verify() {
  auto targetAttr = iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(*this);
  DictionaryAttr config = targetAttr ? targetAttr.getConfiguration() : nullptr;
  auto desc = config ? config.getAs<TargetDescAttr>(targetDescKey) : nullptr;
  if (!desc) return success();
  if (getInputs().size() != desc.getColumns() ||
      getOutputs().size() != desc.getColumns()) {
    return emitOpError() << "has " << getInputs().size() << " input and "
                         << getOutputs().size() << " output columns, the "
                         << "fabric has " << desc.getColumns();
  }
  return success();
}

} // namespace mlir::strela

#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
