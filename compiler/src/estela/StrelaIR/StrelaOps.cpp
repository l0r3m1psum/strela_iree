#include "estela/StrelaIR/StrelaOps.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "estela/StrelaIR/ColumnAddress.h"

namespace mlir::strela {

// TODO: is the dynamic shape comparison still needed for all dimensions?
static bool
compatible(int64_t a, int64_t b) {
  return ShapedType::isDynamic(a) || ShapedType::isDynamic(b) || a == b;
}

// NxC times CxF gives NxF; a dynamic size is checked against nothing.
LogicalResult MatvecOp::verify() {
  auto lhsType = cast<MemRefType>(getLhs().getType());
  auto rhsType = cast<MemRefType>(getRhs().getType());
  auto outType = cast<MemRefType>(getOut().getType());
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2) {
    return emitOpError("expected rank 2 operands");
  }

  // C and F of the rhs: C x F, or F x C with transpose_rhs.
  int64_t rhsC = rhsType.getDimSize(getTransposeRhs() ? 1 : 0);
  int64_t rhsF = rhsType.getDimSize(getTransposeRhs() ? 0 : 1);
  if (!compatible(lhsType.getDimSize(1), rhsC)) {
    return emitOpError("lhs and rhs disagree on the reduction size");
  }
  if (!compatible(outType.getDimSize(0), lhsType.getDimSize(0)) ||
      !compatible(outType.getDimSize(1), rhsF)) {
    return emitOpError("result shape does not match lhs rows and rhs columns");
  }
  return success();
}

// An unused column has no binding, address or words at all; the custom syntax
// can't say otherwise, the generic one can.
LogicalResult OutColumnOp::verify() {
  bool used = getAddress() != nullptr;
  if (getBinding().has_value() != used || getWords().has_value() != used) {
    return emitOpError("needs a binding, an address and words, or none of "
                       "them for an unused column");
  }
  return success();
}

// The layout of the export whose function holds `op`, or null outside a HAL
// executable (dialect tests).
static iree_compiler::IREE::HAL::PipelineLayoutAttr
findLayout(Operation *op) {
  auto funcOp = op->getParentOfType<func::FuncOp>();
  if (!funcOp) return {};
  auto variantOp =
    funcOp->getParentOfType<iree_compiler::IREE::HAL::ExecutableVariantOp>();
  if (!variantOp) return {};
  for (auto exportOp : variantOp.getExportOps()) {
    if (exportOp.getSymName() == funcOp.getName()) return exportOp.getLayout();
  }
  return {};
}

// One column of each kind per fabric column, when the description is in scope
// (inside a STRELA executable variant). Every used column must fold into an
// address the schema can hold, with its binding and push constant in the
// layout of the export.
LogicalResult JobOp::verify() {
  auto targetAttr = iree_compiler::IREE::HAL::ExecutableTargetAttr::lookup(*this);
  DictionaryAttr config = targetAttr ? targetAttr.getConfiguration() : nullptr;
  auto desc = config ? config.getAs<TargetDescAttr>(targetDescKey) : nullptr;
  if (desc && (getInputs().size() != desc.getColumns() ||
               getOutputs().size() != desc.getColumns())) {
    return emitOpError() << "has " << getInputs().size() << " input and "
                         << getOutputs().size() << " output columns, the "
                         << "fabric has " << desc.getColumns();
  }

  iree_compiler::IREE::HAL::PipelineLayoutAttr layout = findLayout(*this);
  auto verifyColumn = [&](Value column, bool isInput,
                          unsigned index) -> LogicalResult {
    Operation *columnOp = column.getDefiningOp();
    if (isInput ? !isa_and_nonnull<InpColumnOp>(columnOp)
                : !isa_and_nonnull<OutColumnOp>(columnOp)) {
      return emitOpError() << (isInput ? "input " : "output ") << index
                           << " is not a "
                           << (isInput ? "strela.inp_column" : "strela.out_column");
    }
    uint32_t binding = 0;
    if (auto inpOp = dyn_cast<InpColumnOp>(columnOp)) {
      binding = inpOp.getBinding();
    } else {
      auto outOp = cast<OutColumnOp>(columnOp);
      if (outOp.isUnused()) return success();
      binding = *outOp.getBinding();
    }

    FailureOr<ColumnAddress> address = getColumnAddress(columnOp);
    if (failed(address)) return failure();
    if (!layout) return success();
    if (binding >= layout.getBindings().size()) {
      return columnOp->emitOpError()
             << "reads binding " << binding << ", the layout has "
             << layout.getBindings().size();
    }
    if (address->pushConstant && *address->pushConstant >= layout.getConstants()) {
      return columnOp->emitOpError()
             << "reads push constant " << *address->pushConstant
             << ", the layout has " << layout.getConstants();
    }
    return success();
  };

  for (auto [i, column] : llvm::enumerate(getInputs())) {
    if (failed(verifyColumn(column, /*isInput=*/true, i))) return failure();
  }
  for (auto [i, column] : llvm::enumerate(getOutputs())) {
    if (failed(verifyColumn(column, /*isInput=*/false, i))) return failure();
  }
  return success();
}

} // namespace mlir::strela

#define GET_OP_CLASSES
#include "StrelaOps.cpp.inc"
