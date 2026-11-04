#include "estela/Utils/KernelMatchers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"

static mlir::LogicalResult
allInputsInteger32(mlir::linalg::GenericOp genericOp) {
  for (mlir::Value input : genericOp.getDpsInputs()) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.getElementType().isInteger(32)) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult
mlir::estela::isStrelaLinalgAdd(mlir::linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return mlir::failure();
  }

  if (failed(allInputsInteger32(genericOp))) {
    return mlir::failure();
  }

  if (!genericOp.hasPureTensorSemantics()) {
    return mlir::failure();
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(), mlir::linalg::isParallelIterator)) {
    return mlir::failure();
  }

  // The add bitstream pairs the two input columns word by word, so a broadcast
  // operand would run out before the other one.
  if (!llvm::all_of(genericOp.getIndexingMapsArray(),
                    [](mlir::AffineMap map) { return map.isIdentity(); })) {
    return mlir::failure();
  }

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return mlir::failure();
  }

  auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&body->front());
  if (!addOp) {
    return mlir::failure();
  }

  auto args = genericOp.getRegionInputArgs();
  bool addsInputs =
      (addOp.getLhs() == args[0] && addOp.getRhs() == args[1]) ||
      (addOp.getLhs() == args[1] && addOp.getRhs() == args[0]);
  if (!addsInputs) {
    return mlir::failure();
  }

  auto yieldOp = mlir::dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != addOp.getResult()) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
mlir::estela::isStrelaLinalgRelu(mlir::linalg::GenericOp genericOp) {
  unsigned numInputs = genericOp.getNumDpsInputs();
  if ((numInputs != 1 && numInputs != 2) || genericOp.getNumDpsInits() != 1) {
    return mlir::failure();
  }

  if (failed(allInputsInteger32(genericOp))) {
    return mlir::failure();
  }

  if (!genericOp.hasPureTensorSemantics()) {
    return mlir::failure();
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(), mlir::linalg::isParallelIterator)) {
    return mlir::failure();
  }

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return mlir::failure();
  }

  auto maxOp = mlir::dyn_cast<mlir::arith::MaxSIOp>(&body->front());
  if (!maxOp) {
    return mlir::failure();
  }

  if (maxOp.getLhs() != body->getArgument(0)) {
    return mlir::failure();
  }

  if (numInputs == 2) {
    if (maxOp.getRhs() != body->getArgument(1)) {
      return mlir::failure();
    }
    mlir::DenseIntElementsAttr zeroAttr;
    if (
      !mlir::matchPattern(genericOp.getDpsInputs()[1], mlir::m_Constant(&zeroAttr))
      || !zeroAttr.isSplat()
      || !zeroAttr.getSplatValue<llvm::APInt>().isZero()
    ) {
      return mlir::failure();
    }
  } else {
    llvm::APInt zeroValue;
    if (!mlir::matchPattern(maxOp.getRhs(), mlir::m_ConstantInt(&zeroValue)) ||
        !zeroValue.isZero()) {
      return mlir::failure();
    }
  }

  auto yieldOp = mlir::dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != maxOp.getResult()) {
    return mlir::failure();
  }

  return mlir::success();
}

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
  if (auto genericOp = dyn_cast<linalg::GenericOp>(linalgOp.getOperation())) {
    if (mlir::succeeded(isStrelaLinalgAdd(genericOp))) {
      return true;
    }
    if (mlir::succeeded(isStrelaLinalgRelu(genericOp))) {
      return true;
    }
  }

  if (auto batchedMatmulOp = dyn_cast<linalg::BatchMatmulOp>(linalgOp.getOperation())) {
    if (mlir::succeeded(isStrelaLinalgMatvec(batchedMatmulOp))) {
      return true;
    }
  }

  return false;
}
