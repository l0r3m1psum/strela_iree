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

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return mlir::failure();
  }

  auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&body->front());
  if (!addOp) {
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

bool
mlir::estela::isSupportedByStrela(mlir::linalg::GenericOp genericOp) {
  if (mlir::succeeded(isStrelaLinalgAdd(genericOp))) {
    return true;
  }

  if (mlir::succeeded(isStrelaLinalgRelu(genericOp))) {
    return true;
  }

  return false;
}
