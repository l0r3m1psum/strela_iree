#include "StrelaModel.h"

#include "mlir/IR/Matchers.h"
#include "estela/Utils/Finders.h"

namespace mlir::estela {

static bool
isI32(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getElementType().isInteger(32);
}

static bool
hasZeroFillInit(linalg::LinalgOp op) {
  auto fillOp = op.getDpsInits()[0].getDefiningOp<linalg::FillOp>();
  return fillOp && matchPattern(fillOp.getInputs()[0], m_Zero());
}

// Contraction naming. linalg::inferContractionDims sorts every loop dimension
// by which operands it appears in:
//
//   linalg    | appears in        | deep learning
//   ----------+-------------------+--------------------------
//   batch     | lhs, rhs, out     | B  (e.g. attention heads)
//   M         | lhs, out          | N  (rows: batch, tokens)
//   N         | rhs, out          | F  (output features)
//   K         | lhs, rhs          | C  (input channels)
//
//   out[b, M, N] = sum_K lhs[b, M, K] * rhs[b, K, N]
//     y[B, N, F] = sum_C   x[B, N, C]   * W[B, C, F]
// linalg::inferContractionDims is needed to classify things like all the
// indices in an einsum contraction (remember that it can always be implemented
// as a reshape -> matmul -> reshape)
// We could allow multiple M dimension but I do not understand how...
static StrelaOpKind
classify(linalg::LinalgOp op) {
  StrelaOpKind res;
  if (op.getNumReductionLoops() == 0) {
    res = StrelaOpKind::Elementwise;
  } else if (!linalg::isaContractionOpInterface(op)) {
    res = StrelaOpKind::Unsupported;
  } else {
    FailureOr<linalg::ContractionDimensions> dims = linalg::inferContractionDims(op);
    if (failed(dims)) {
      res = StrelaOpKind::Unsupported;
    } else {
      SmallVector<int64_t> sizes = op.getStaticLoopRanges();
      bool unitBatch = true;
      for (unsigned d : dims->batch) unitBatch &= sizes[d] == 1;
      bool oneEach = dims->m.size() == 1 && dims->n.size() == 1 && dims->k.size() == 1;
      bool staticCF = oneEach
        && !ShapedType::isDynamic(sizes[dims->n[0]])
        && !ShapedType::isDynamic(sizes[dims->k[0]]);
      bool supported = unitBatch || staticCF || hasZeroFillInit(op);
      res = supported ? StrelaOpKind::Contraction : StrelaOpKind::Unsupported;
    }
  }

  return res;
}

// Computes the longest dependency chain in the op's body
// "each op's depth is one more than its deepest operand"
// ^bb0(%a: i32, %b: i32, %c: i32, %out: i32):
//   %0 = arith.addi %a, %b      // operands; %a (depth 0), b (depth 0) -> %0 (depth 1)
//   %1 = arith.muli %0, %c      // operands: %0 (depth 1), c (depth 0) -> %1 (depth 2)
//   %2 = arith.maxsi %1, %a     // operands: %1 (depth 2), a (depth 0) -> %2 (depth 3)
//   linalg.yield %2
static void
summarizeBody(linalg::LinalgOp op, StrelaOpSummary &summary) {
  DenseMap<Value, unsigned> depthOf;
  for (Operation &bodyOp : op.getBlock()->without_terminator()) {
    if (!isa<arith::ConstantOp>(bodyOp)) {
      unsigned depth = 0;
      for (Value operand : bodyOp.getOperands()) {
        depth = std::max(depth, depthOf.lookup(operand));
      }
      depthOf[bodyOp.getResult(0)] = depth + 1;
      summary.depth = std::max(summary.depth, depth + 1);
      summary.bodyOps.push_back(bodyOp.getName());
    }
  }
}

static void
collectInputs(linalg::LinalgOp op, StrelaOpSummary &summary) {
  llvm::SetVector<Value> inputs;
  for (OpOperand *operand : op.getDpsInputOperands()) {
    // It will be encoded as an immediate in the PE
    if (isInt32Immediate(operand->get())) continue;
    if (!op.payloadUsesValueFromOperand(operand)) continue;
    inputs.insert(operand->get());
  }
  summary.inputs = inputs.takeVector();
}

StrelaOpSummary
summarize(linalg::LinalgOp op) {
  StrelaOpSummary res;

  if (op.hasPureTensorSemantics() && llvm::all_of(op->getOperandTypes(), isI32)) {
    res.kind = classify(op);

    if (res.kind != StrelaOpKind::Unsupported) {
      summarizeBody(op, res);
      collectInputs(op, res);
      res.numResults = op->getNumResults();
    }

  }
  return res;
}

} // namespace mlir::estela
