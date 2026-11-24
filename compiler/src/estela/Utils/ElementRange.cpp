#include "estela/Utils/ElementRange.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"

namespace mlir::estela {

static std::optional<ConstantIntRanges>
typeRange(Type type) {
  auto intType = dyn_cast<IntegerType>(getElementTypeOrSelf(type));
  if (!intType || !intType.isSignless()) return std::nullopt;
  return ConstantIntRanges::maxRange(intType.getWidth());
}

// The range of the constant `attr`, a dense tensor or an integer.
static std::optional<ConstantIntRanges>
constantRange(Attribute attr) {
  if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
    return ConstantIntRanges::constant(intAttr.getValue());
  }
  auto dense = dyn_cast<DenseIntElementsAttr>(attr);
  if (!dense || dense.empty()) return std::nullopt;
  APInt min = *dense.value_begin<APInt>(), max = min;
  for (const APInt &v : dense.getValues<APInt>()) {
    if (v.slt(min)) min = v;
    if (v.sgt(max)) max = v;
  }
  return ConstantIntRanges::fromSigned(min, max);
}

// The range of `value`, the result `resultIndex` of the linalg op `op`: its
// body evaluated op by op.
static std::optional<ConstantIntRanges>
linalgRange(linalg::LinalgOp op, unsigned resultIndex, unsigned depth) {
  if (!op.hasPureTensorSemantics() || op->getNumRegions() != 1) {
    return std::nullopt;
  }
  Block *body = op.getBlock();
  DenseMap<Value, ConstantIntRanges> ranges;
  for (OpOperand &operand : op->getOpOperands()) {
    BlockArgument arg = op.getMatchingBlockArgument(&operand);
    if (!arg || !op.payloadUsesValueFromOperand(&operand)) continue;
    std::optional<ConstantIntRanges> range =
      elementRange(operand.get(), depth - 1);
    if (range) ranges.try_emplace(arg, *range);
  }

  // A value the body uses: from the body, or from above (a captured scalar).
  auto lookup = [&](Value value) -> std::optional<ConstantIntRanges> {
    auto it = ranges.find(value);
    if (it != ranges.end()) return it->second;
    if (value.getParentBlock() != body) return elementRange(value, depth - 1);
    return typeRange(value.getType());
  };

  for (Operation &bodyOp : body->without_terminator()) {
    auto inferOp = dyn_cast<InferIntRangeInterface>(&bodyOp);
    SmallVector<ConstantIntRanges> argRanges;
    bool known = inferOp != nullptr;
    for (Value operand : bodyOp.getOperands()) {
      std::optional<ConstantIntRanges> range = lookup(operand);
      if (!range) {
        known = false;
        break;
      }
      argRanges.push_back(*range);
    }
    if (!known) continue; // its results stay unknown, the type's range
    inferOp.inferResultRanges(argRanges, [&](Value v, const ConstantIntRanges &r) {
      ranges.insert_or_assign(v, r);
    });
  }
  Value yielded = body->getTerminator()->getOperand(resultIndex);
  return lookup(yielded);
}

std::optional<ConstantIntRanges>
elementRange(Value value, unsigned depth) {
  std::optional<ConstantIntRanges> full = typeRange(value.getType());
  if (!full) return std::nullopt;

  Attribute constant;
  if (matchPattern(value, m_Constant(&constant))) {
    if (std::optional<ConstantIntRanges> range = constantRange(constant)) {
      return range;
    }
    return full;
  }
  Operation *op = value.getDefiningOp();
  if (!op || depth == 0) return full;

  std::optional<ConstantIntRanges> range;
  if (isa<tensor::CollapseShapeOp, tensor::ExpandShapeOp, tensor::CastOp,
          tensor::ExtractSliceOp, tensor::ReshapeOp>(op)) {
    range = elementRange(op->getOperand(0), depth - 1);
  } else if (auto concatOp = dyn_cast<tensor::ConcatOp>(op)) {
    // The bias rows StrelaPrepareWeights appends to the weights.
    for (Value input : concatOp.getInputs()) {
      std::optional<ConstantIntRanges> inputRange =
        elementRange(input, depth - 1);
      if (!inputRange) return full;
      range = range ? range->rangeUnion(*inputRange) : *inputRange;
    }
  } else if (auto linalgOp = dyn_cast<linalg::LinalgOp>(op)) {
    range = linalgRange(linalgOp, cast<OpResult>(value).getResultNumber(), depth);
  } else if (auto inferOp = dyn_cast<InferIntRangeInterface>(op);
             inferOp && !isa<ShapedType>(value.getType())) {
    // A scalar: what the op says about its operands' ranges.
    SmallVector<ConstantIntRanges> argRanges;
    for (Value operand : op->getOperands()) {
      std::optional<ConstantIntRanges> argRange = elementRange(operand, depth - 1);
      if (!argRange) return full;
      argRanges.push_back(*argRange);
    }
    inferOp.inferResultRanges(argRanges, [&](Value v, const ConstantIntRanges &r) {
      if (v == value) range = r;
    });
  }
  return range ? range : full;
}

bool
fitsSigned(const ConstantIntRanges &range, unsigned width) {
  unsigned bitWidth = range.smin().getBitWidth();
  if (width >= bitWidth) return true;
  APInt min = APInt::getSignedMinValue(width).sext(bitWidth);
  APInt max = APInt::getSignedMaxValue(width).sext(bitWidth);
  return range.smin().sge(min) && range.smax().sle(max);
}

bool
operandsFitMultiplier(linalg::LinalgOp op, strela::TargetDescAttr desc) {
  if (!desc) return false;
  return llvm::all_of(op.getDpsInputs(), [&](Value input) {
    std::optional<ConstantIntRanges> range = elementRange(input);
    return range && fitsSigned(*range, desc.getMultiplierWidth());
  });
}

} // namespace mlir::estela
