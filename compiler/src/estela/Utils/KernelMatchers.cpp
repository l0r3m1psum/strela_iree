#include "estela/Utils/KernelMatchers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"

mlir::LogicalResult
mlir::estela::isStrelaLinalgMatvec(mlir::linalg::BatchMatmulOp op) {
  if (!op.hasPureTensorSemantics() || op.hasUserDefinedMaps()) {
    return mlir::failure();
  }

  auto lhsType = mlir::cast<mlir::RankedTensorType>(op.getDpsInputs()[0].getType());
  auto rhsType = mlir::cast<mlir::RankedTensorType>(op.getDpsInputs()[1].getType());
  auto outType = mlir::cast<mlir::RankedTensorType>(op.getDpsInits()[0].getType());
  if (!lhsType.getElementType().isInteger(32) ||
      !rhsType.getElementType().isInteger(32) ||
      !outType.getElementType().isInteger(32)) {
    return mlir::failure();
  }

  // The batch matrix multiplication BxNxC BxCxF. The linalg.batch_matmul
  // semantics force the two batches to have the same size but if one of the two
  // is dynamic it has to be checked at runtime.
  // C and F could be dynamic (the driver would patch the delay and derive the
  // column sizes, strides and grid from push constants), but for now they must
  // be static so every number in the export is a constant.
  if (lhsType.getDimSize(0) != 1 || !rhsType.hasStaticShape()) {
    return mlir::failure();
  }

  // The kernel has no accumulator input, so it can only compute A x B from 0.
  auto fillOp = op.getDpsInits()[0].getDefiningOp<mlir::linalg::FillOp>();
  llvm::APInt init;
  if (!fillOp ||
      !mlir::matchPattern(fillOp.getInputs()[0], mlir::m_ConstantInt(&init)) ||
      !init.isZero()) {
    return mlir::failure();
  }

  return mlir::success();
}

bool
mlir::estela::isSupportedByStrela(mlir::linalg::LinalgOp linalgOp) {
  if (auto batchedMatmulOp = dyn_cast<linalg::BatchMatmulOp>(linalgOp.getOperation())) {
    if (mlir::succeeded(isStrelaLinalgMatvec(batchedMatmulOp))) {
      return true;
    }
  }

  return false;
}
