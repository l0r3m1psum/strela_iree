#include "estela/Transforms/Passes.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "estela/Utils/ElementRange.h"
#include "estela/Utils/Finders.h"
#include "estela/Utils/StrelaContraction.h"

namespace mlir::estela {
#define GEN_PASS_DEF_STRELAPREPAREWEIGHTSPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

//===--------------------------------------------------------------------===//
// Zero points
//===--------------------------------------------------------------------===//

static SmallVector<utils::IteratorType>
parallelIterators(unsigned rank) {
  return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

// extsi(input) - zeroPoint, elementwise, as i32: the widening the host runs
// (or const-eval, for constant weights).
static Value
widen(RewriterBase &rewriter, Location loc, Value input, int64_t zeroPoint) {
  auto type = cast<RankedTensorType>(input.getType());
  Type i32 = rewriter.getI32Type();
  Value empty = tensor::EmptyOp::create(
    rewriter, loc, tensor::getMixedSizes(rewriter, loc, input), i32
  );
  AffineMap id = rewriter.getMultiDimIdentityMap(type.getRank());
  return linalg::GenericOp::create(
    rewriter, loc, TypeRange{empty.getType()}, ValueRange{input},
    ValueRange{empty}, ArrayRef<AffineMap>{id, id},
    parallelIterators(type.getRank()),
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value v = args[0];
      if (v.getType() != i32) v = arith::ExtSIOp::create(b, loc, i32, v);
      if (zeroPoint != 0) {
        v = arith::SubIOp::create(
          b, loc, v, arith::ConstantIntOp::create(b, loc, i32, zeroPoint)
        );
      }
      linalg::YieldOp::create(b, loc, v);
    }
  ).getResult(0);
}

// The integer constant `value` holds, if it is one.
static std::optional<int64_t>
constantInt(Value value) {
  APInt v;
  if (!matchPattern(value, m_ConstantInt(&v))) return std::nullopt;
  return v.getSExtValue();
}

// The values of `value`, a constant tensor of integers.
static std::optional<SmallVector<int64_t>>
constantValues(Value value) {
  Attribute attr;
  if (!matchPattern(value, m_Constant(&attr))) return std::nullopt;
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    return llvm::map_to_vector(dense.getValues<APInt>(), [](const APInt &v) {
      return v.getSExtValue();
    });
  }
  // What TOSA imports large constants as.
  if (auto resource = dyn_cast<DenseI32ResourceElementsAttr>(attr)) {
    if (std::optional<ArrayRef<int32_t>> values = resource.tryGetAsArrayRef()) {
      return llvm::map_to_vector(*values, [](int32_t v) { return int64_t(v); });
    }
  }
  return std::nullopt;
}

// The per-feature bias `init` broadcasts, if it is a constant: a linalg op
// copying a constant vector along the last dimension, as TOSA lowers the bias
// of a convolution to.
static std::optional<SmallVector<int64_t>>
constantBias(Value init, int64_t f) {
  auto op = init.getDefiningOp<linalg::LinalgOp>();
  if (!op || op.getNumDpsInputs() != 1 || op.getNumDpsInits() != 1 ||
      op.getNumReductionLoops() != 0) {
    return std::nullopt;
  }
  OpOperand *input = op.getDpsInputOperand(0);
  AffineMap inputMap = op.getMatchingIndexingMap(input);
  AffineMap initMap = op.getMatchingIndexingMap(op.getDpsInitOperand(0));
  auto yield = cast<linalg::YieldOp>(op.getBlock()->getTerminator());
  if (!initMap.isIdentity() || inputMap.getNumResults() != 1 ||
      inputMap.getResult(0) != getAffineDimExpr(initMap.getNumDims() - 1,
                                                op.getContext()) ||
      yield.getValues() != ValueRange{op.getMatchingBlockArgument(input)}) {
    return std::nullopt;
  }
  std::optional<SmallVector<int64_t>> bias = constantValues(input->get());
  if (!bias || int64_t(bias->size()) != f) return std::nullopt;
  return bias;
}

// True if extendWithOnes can extend the result `resultIndex` of `op`: an
// elementwise generic (projected permutation input maps, an identity result
// map, inits it doesn't read) with a static last dimension.
static bool
canExtendWithOnes(linalg::GenericOp op, unsigned resultIndex) {
  if (op.getNumLoops() != op.getNumParallelLoops() ||
      !op.hasPureTensorSemantics()) {
    return false;
  }
  OpOperand *init = op.getDpsInitOperand(resultIndex);
  auto type = cast<RankedTensorType>(init->get().getType());
  if (!op.getMatchingIndexingMap(init).isIdentity() ||
      ShapedType::isDynamic(type.getShape().back())) {
    return false;
  }
  for (OpOperand &operand : op.getDpsInitsMutable()) {
    if (op.payloadUsesValueFromOperand(&operand)) return false;
  }
  return llvm::all_of(op.getDpsInputOperands(), [&](OpOperand *input) {
    return !isa<RankedTensorType>(input->get().getType()) ||
           op.getMatchingIndexingMap(input).isProjectedPermutation();
  });
}

// The result `resultIndex` of the generic `op` (canExtendWithOnes), `ones`
// elements longer in its last dimension, which hold 1. One generic over the
// longer domain computes it: it reads op's inputs with tensor.extract, at the
// indices clamped to op's domain, runs op's body and yields 1 past the end.
// It has no inputs, so no tied operand either: the CPU writes it whole into a
// buffer of its own, and the dispatch stays on the CPU. (A tensor.pad would
// become a splat the dispatch writes into, which Stream then places on the
// device the padded tensor goes to, STRELA.)
static Value
extendWithOnes(RewriterBase &rewriter, linalg::GenericOp op,
               unsigned resultIndex, int64_t ones) {
  Location loc = op.getLoc();
  Value result = op.getResult(resultIndex);
  auto type = cast<RankedTensorType>(result.getType());
  unsigned rank = type.getRank();
  int64_t last = type.getShape().back();
  // From the init, not the result: op is erased afterwards.
  SmallVector<OpFoldResult> sizes = tensor::getMixedSizes(
    rewriter, loc, op.getDpsInitOperand(resultIndex)->get()
  );
  sizes.back() = rewriter.getIndexAttr(last + ones);
  Value empty =
    tensor::EmptyOp::create(rewriter, loc, sizes, type.getElementType());

  Block *body = op.getBlock();
  auto yield = cast<linalg::YieldOp>(body->getTerminator());
  return linalg::GenericOp::create(
    rewriter, loc, TypeRange{empty.getType()}, ValueRange{}, ValueRange{empty},
    ArrayRef<AffineMap>{rewriter.getMultiDimIdentityMap(rank)},
    parallelIterators(rank),
    [&](OpBuilder &b, Location loc, ValueRange) {
      SmallVector<Value> indices;
      for (unsigned d = 0; d < rank; ++d) {
        indices.push_back(linalg::IndexOp::create(b, loc, d));
      }
      Value inside = arith::CmpIOp::create(
        b, loc, arith::CmpIPredicate::ult, indices.back(),
        arith::ConstantIndexOp::create(b, loc, last)
      );
      // Clamped, so the extracts stay in bounds where the 1 is chosen.
      indices.back() = arith::MinUIOp::create(
        b, loc, indices.back(), arith::ConstantIndexOp::create(b, loc, last - 1)
      );

      IRMapping mapping;
      for (OpOperand *input : op.getDpsInputOperands()) {
        Value value = input->get();
        if (isa<RankedTensorType>(value.getType())) {
          SmallVector<Value> at;
          for (AffineExpr expr : op.getMatchingIndexingMap(input).getResults()) {
            at.push_back(indices[cast<AffineDimExpr>(expr).getPosition()]);
          }
          value = tensor::ExtractOp::create(b, loc, value, at);
        }
        mapping.map(op.getMatchingBlockArgument(input), value);
      }
      for (Operation &bodyOp : body->without_terminator()) {
        if (auto indexOp = dyn_cast<linalg::IndexOp>(bodyOp)) {
          mapping.map(indexOp.getResult(), indices[indexOp.getDim()]);
          continue;
        }
        b.clone(bodyOp, mapping);
      }
      Value value = mapping.lookup(yield.getOperand(resultIndex));
      Value one = arith::ConstantOp::create(
        b, loc, b.getIntegerAttr(type.getElementType(), 1)
      );
      linalg::YieldOp::create(
        b, loc, ValueRange{arith::SelectOp::create(b, loc, inside, value, one)}
      );
    }
  ).getResult(0);
}

// [x - zx, 1, ..., 1]: the activations `x` (N x H x W x C, i8) widened, minus
// their zero point, with `ones` trailing ones per pixel, as P x (C + ones) i32.
// It must not cost a dispatch, so it is computed where the host computes x:
// the widening is fused here into the elementwise op producing x (the previous
// layer's rescale and clamp), which IREE would not do itself (by element types
// the rescale is a truncation and the widening an extension, which it never
// fuses), then extended with the ones (extendWithOnes). For the model's input
// that is the widening alone, a dispatch of its own as it is anyway.
static Value
activationsWithOnes(RewriterBase &rewriter, Location loc, Value x,
                    int64_t zeroPoint, int64_t ones) {
  auto producer = x.getDefiningOp<linalg::GenericOp>();
  bool fuse = producer && producer->hasOneUse() &&
              producer.getNumResults() == 1 && linalg::isElementwise(producer) &&
              canExtendWithOnes(producer, 0);

  auto op = widen(rewriter, loc, x, zeroPoint).getDefiningOp<linalg::GenericOp>();
  unsigned resultIndex = 0;
  if (fuse && linalg::areElementwiseOpsFusable(&op->getOpOperand(0))) {
    FailureOr<linalg::ElementwiseOpFusionResult> fused =
      linalg::fuseElementwiseOps(rewriter, &op->getOpOperand(0));
    if (succeeded(fused)) {
      // The fused op also yields x, which the convolution being replaced still
      // reads from the producer: both are dead afterwards.
      Value widened = fused->replacements.lookup(op->getResult(0));
      rewriter.eraseOp(op);
      op = cast<linalg::GenericOp>(fused->fusedOp);
      resultIndex = cast<OpResult>(widened).getResultNumber();
    }
  }
  assert(canExtendWithOnes(op, resultIndex));
  Value extended = extendWithOnes(rewriter, op, resultIndex, ones);
  rewriter.eraseOp(op);
  SmallVector<ReassociationIndices> pixels = {{0, 1, 2}, {3}};
  return tensor::CollapseShapeOp::create(rewriter, loc, extended, pixels);
}

// A quantized 1x1 convolution (as TFLite fully connected layers become) is a
// quantized matmul over its N * H * W pixels:
//
//   y[p, f] = init[p, f] + sum_c (x[p, c] - zx) * (w[c, f] - zw)
//           = init[p, f] + sum_c x[p, c] * w'[c, f] - zx * sum_c w'[c, f]
//
// with w' = w - zw. It is rewritten into that last form: a plain i32 matmul
// of the widened activations and w' from a zero fill, which the matvec kernel
// runs, then an elementwise add of the init and of the per-feature constant
// -zx * sum_c w'[c, f], which stays on the host. The weights and zx must be
// constants: w', its column sums and their product with zx are computed from
// constants alone, so const-eval folds them, and the input-dependent term
// zw * sum_c x[p, c] is gone. TFLite weights are symmetric (zw = 0), so
// w' = w.
//
// With `maxBiasPart` set (fold-bias) and a constant bias, the bias goes into
// the matmul instead, as m extra reduction elements:
//
//   y[p, f] = sum_c (x[p, c] - zx) * w'[c, f] + sum_j 1 * b_j[f]
//
// The activations keep their zero point (x - zx fits 16 bits as x does), so
// the bias is TFLite's own, with no -zx * sum_c w'[c, f] in it. It is split
// into m parts that each fit the multipliers, b_j = floor((b + j) / m), which
// sum to b exactly; m is the fewest that fit, 1 if b already does. The
// activations get m trailing ones (a tensor.pad, which becomes a fill and an
// insert_slice that fuse into the dispatch producing x - zx), the weights m
// trailing rows (a tensor.concat of constants, which const-eval folds). No
// host add is left: the matmul result is y.
static LogicalResult
lowerQuantizedConv(RewriterBase &rewriter, linalg::Conv2DNhwcHwcfQOp convOp,
                   int64_t maxBiasPart) {
  Value x = convOp.getInputs()[0], w = convOp.getInputs()[1];
  Value init = convOp.getOutputs()[0];
  auto xType = cast<RankedTensorType>(x.getType());
  auto wType = cast<RankedTensorType>(w.getType());
  auto initType = cast<RankedTensorType>(init.getType());
  std::optional<int64_t> zx = constantInt(convOp.getInputs()[2]);
  std::optional<int64_t> zw = constantInt(convOp.getInputs()[3]);
  auto isOne = [](DenseIntElementsAttr attr) {
    return llvm::all_of(attr.getValues<APInt>(), [](APInt v) { return v == 1; });
  };
  // A 1x1 filter with unit strides and dilations, static C and F, narrow
  // operands, constant zero points and weights.
  if (!zx || !zw || !wType.hasStaticShape() || wType.getDimSize(0) != 1 ||
      wType.getDimSize(1) != 1 || !isOne(convOp.getStrides()) ||
      !isOne(convOp.getDilations()) ||
      xType.getElementTypeBitWidth() > 16 || wType.getElementTypeBitWidth() > 8 ||
      !initType.getElementType().isInteger(32) ||
      ShapedType::isDynamic(xType.getDimSize(3))) {
    return failure();
  }

  Location loc = convOp.getLoc();
  rewriter.setInsertionPoint(convOp);
  MLIRContext *context = rewriter.getContext();
  Type i32 = rewriter.getI32Type();
  int64_t f = wType.getDimSize(3);
  // (N, H, W, C) -> (N * H * W, C), the filter (1, 1, C, F) -> (C, F).
  SmallVector<ReassociationIndices> pixels = {{0, 1, 2}, {3}};

  std::optional<SmallVector<int64_t>> bias;
  if (maxBiasPart > 0) bias = constantBias(init, f);

  int64_t parts = 0;
  if (bias) {
    int64_t maxAbs = 0;
    for (int64_t b : *bias) maxAbs = std::max(maxAbs, std::abs(b));
    parts = std::max<int64_t>(1, llvm::divideCeil(maxAbs, maxBiasPart));
  }

  // The activations, [x - zx, 1, ..., 1] when folding the bias, and w'.
  Value x2 =
    bias ? Value() : tensor::CollapseShapeOp::create(rewriter, loc, x, pixels);
  Value w2 = tensor::CollapseShapeOp::create(rewriter, loc, w, pixels);
  Value x32 = bias ? activationsWithOnes(rewriter, loc, x, *zx, parts)
                   : widen(rewriter, loc, x2, 0);
  Value w32 = widen(rewriter, loc, w2, *zw);
  if (bias) {
    // [w'; b_0; ...; b_{m-1}].
    SmallVector<int32_t> rows;
    for (int64_t j = 0; j < parts; ++j) {
      for (int64_t b : *bias) {
        rows.push_back(int32_t(llvm::divideFloorSigned(b + j, parts)));
      }
    }
    auto rowsType = RankedTensorType::get({parts, f}, i32);
    Value biasRows = arith::ConstantOp::create(
      rewriter, loc, DenseIntElementsAttr::get(rowsType, ArrayRef(rows))
    );
    w32 = tensor::ConcatOp::create(rewriter, loc, 0, ValueRange{w32, biasRows});
  }

  // sum_c x[p, c] * w'[c, f], from zero.
  SmallVector<OpFoldResult> accSizes = {
    tensor::getMixedSize(rewriter, loc, x32, 0), rewriter.getIndexAttr(f)
  };
  Value zero = arith::ConstantIntOp::create(rewriter, loc, i32, 0);
  Value accEmpty = tensor::EmptyOp::create(rewriter, loc, accSizes, i32);
  Value accInit = linalg::FillOp::create(rewriter, loc, zero, accEmpty).getResult(0);
  Value acc = linalg::MatmulOp::create(
    rewriter, loc, TypeRange{accInit.getType()}, ValueRange{x32, w32},
    ValueRange{accInit}
  ).getResult(0);

  if (bias) {
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      convOp, initType, acc, pixels, tensor::getMixedSizes(rewriter, loc, init)
    );
    return success();
  }

  // -zx * sum_c w'[c, f]: constants only.
  AffineExpr c, fd;
  bindDims(context, c, fd);
  Value sumEmpty = tensor::EmptyOp::create(rewriter, loc, ArrayRef<int64_t>{f}, i32);
  Value sumInit = linalg::FillOp::create(rewriter, loc, zero, sumEmpty).getResult(0);
  Value correction = linalg::GenericOp::create(
    rewriter, loc, TypeRange{sumInit.getType()}, ValueRange{w32},
    ValueRange{sumInit},
    ArrayRef<AffineMap>{AffineMap::get(2, 0, {c, fd}, context),
                        AffineMap::get(2, 0, {fd}, context)},
    SmallVector<utils::IteratorType>{utils::IteratorType::reduction,
                                     utils::IteratorType::parallel},
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value scaled = arith::MulIOp::create(
        b, loc, args[0], arith::ConstantIntOp::create(b, loc, i32, -*zx)
      );
      Value sum = arith::AddIOp::create(b, loc, args[1], scaled);
      linalg::YieldOp::create(b, loc, sum);
    }
  ).getResult(0);

  // init + acc + correction[f], on the host.
  Value init2 = tensor::CollapseShapeOp::create(rewriter, loc, init, pixels);
  Value outEmpty = tensor::EmptyOp::create(rewriter, loc, accSizes, i32);
  AffineExpr p, ff;
  bindDims(context, p, ff);
  AffineMap id = AffineMap::get(2, 0, {p, ff}, context);
  Value sum = linalg::GenericOp::create(
    rewriter, loc, TypeRange{outEmpty.getType()},
    ValueRange{acc, init2, correction}, ValueRange{outEmpty},
    ArrayRef<AffineMap>{id, id, AffineMap::get(2, 0, {ff}, context), id},
    parallelIterators(2),
    [&](OpBuilder &b, Location loc, ValueRange args) {
      Value v = arith::AddIOp::create(b, loc, args[0], args[1]);
      Value w = arith::AddIOp::create(b, loc, v, args[2]);
      linalg::YieldOp::create(b, loc, w);
    }
  ).getResult(0);

  Value y = tensor::ExpandShapeOp::create(
    rewriter, loc, initType, sum, pixels,
    tensor::getMixedSizes(rewriter, loc, init)
  );
  rewriter.replaceOp(convOp, y);
  return success();
}

// True if `value` is computed from constants alone, through ops without side
// effects (a tosa.cast, a broadcast, a reshape): const-eval folds whatever is
// applied to it into a new constant.
static bool
isConstantTree(Value value, unsigned depth = 8) {
  if (matchPattern(value, m_Constant())) return true;
  Operation *op = value.getDefiningOp();
  if (!op || depth == 0) return false;
  if (isa<tensor::EmptyOp>(op)) return true;
  if (!isMemoryEffectFree(op)) return false;
  return llvm::all_of(op->getOperands(), [&](Value operand) {
    return isConstantTree(operand, depth - 1);
  });
}

// Rewrites `op`, a contraction with C x F weights, to read them as F x C: the
// weights go through a linalg.transpose (which const-eval folds) and the
// contraction becomes a linalg.generic whose rhs map swaps C and F.
static LogicalResult
transposeWeights(RewriterBase &rewriter, linalg::LinalgOp op,
                 const estela::StrelaMatvecShape &shape) {
  OpOperand *rhsOperand = op.getDpsInputOperand(1);
  AffineMap rhsMap = op.getMatchingIndexingMap(rhsOperand);
  unsigned c = shape.dims.k[0], f = shape.dims.n[0];
  std::optional<unsigned> posC, posF;
  for (auto [i, expr] : llvm::enumerate(rhsMap.getResults())) {
    unsigned d = cast<AffineDimExpr>(expr).getPosition();
    if (d == c) posC = i;
    if (d == f) posF = i;
  }
  if (!posC || !posF) return failure();

  SmallVector<int64_t> permutation =
    llvm::to_vector(llvm::seq<int64_t>(0, rhsMap.getNumResults()));
  std::swap(permutation[*posC], permutation[*posF]);

  Location loc = op.getLoc();
  rewriter.setInsertionPoint(op);
  Value rhs = rhsOperand->get();
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  SmallVector<int64_t> shapeT = applyPermutation(rhsType.getShape(), permutation);
  Value empty = tensor::EmptyOp::create(
    rewriter, loc, shapeT, rhsType.getElementType()
  );
  Value rhsT = linalg::TransposeOp::create(
    rewriter, loc, rhs, empty, permutation
  )->getResult(0);

  FailureOr<linalg::GenericOp> genericOp = linalg::generalizeNamedOp(rewriter, op);
  if (failed(genericOp)) return failure();
  SmallVector<AffineMap> maps = genericOp->getIndexingMapsArray();
  SmallVector<AffineExpr> results = llvm::to_vector(maps[1].getResults());
  std::swap(results[*posC], results[*posF]);
  maps[1] = AffineMap::get(maps[1].getNumDims(), 0, results, op.getContext());
  rewriter.modifyOpInPlace(*genericOp, [&] {
    genericOp->setIndexingMapsAttr(rewriter.getAffineMapArrayAttr(maps));
    genericOp->getDpsInputOperand(1)->set(rhsT);
  });
  return success();
}

// Constant weights the matvec kernel will read are transposed from C x F (TOSA)
// to F x C, so that each feature's C weights are contiguous: a weight column
// then spans C words instead of C * F, which the 640 x 128 layers of AD need
// (C * F = 81920 words, a column covers at most 16383). Const-eval folds the
// transpose: nothing is transposed at run time. Weights that are already
// F x C, or not constant, are left alone.
struct StrelaPrepareWeightsPass final
    : estela::impl::StrelaPrepareWeightsPassBase<StrelaPrepareWeightsPass> {
  using StrelaPrepareWeightsPassBase::StrelaPrepareWeightsPassBase;

  void runOnOperation() override {
    IREE::Util::FuncOp funcOp = getOperation();
    strela::TargetDescAttr desc = estela::findStrelaTargetDesc(funcOp);
    if (!desc) return;
    IRRewriter rewriter(funcOp.getContext());
    // The largest bias part the multipliers take: 2^(width - 1) - 1.
    int64_t maxBiasPart =
      foldBias ? (int64_t(1) << (desc.getMultiplierWidth() - 1)) - 1 : 0;

    // Zero points first: the matmuls they leave are what the transposition
    // below and TransferWrap look for.
    SmallVector<linalg::Conv2DNhwcHwcfQOp> convOps;
    funcOp.walk([&](linalg::Conv2DNhwcHwcfQOp op) {
      if (op.hasPureTensorSemantics() && isConstantTree(op.getInputs()[1])) {
        convOps.push_back(op);
      }
    });
    for (linalg::Conv2DNhwcHwcfQOp convOp : convOps) {
      // Else left to the CPU.
      (void)lowerQuantizedConv(rewriter, convOp, maxBiasPart);
    }

    SmallVector<std::pair<linalg::LinalgOp, estela::StrelaMatvecShape>> ops;
    funcOp.walk([&](linalg::LinalgOp op) {
      // Sized as if already transposed: C x F weights wider than a strided
      // column (C * F words) still fit once each feature's are contiguous.
      FailureOr<estela::StrelaMatvecShape> shape =
        estela::matchStrelaContraction(op, desc, /*asIfTransposed=*/true);
      // Only what TransferWrap will pin: a matmul left on the CPU keeps its
      // layout.
      if (failed(shape) || shape->weightsTransposed ||
          !estela::operandsFitMultiplier(op, desc) ||
          !isConstantTree(op.getDpsInputs()[1])) {
        return;
      }
      ops.push_back({op, *shape});
    });

    for (auto &[op, shape] : ops) {
      if (failed(transposeWeights(rewriter, op, shape))) {
        op.emitOpError("could not transpose its constant weights");
        return signalPassFailure();
      }
    }
  }
};

} // namespace
