#include "estela/Transforms/Passes.h"

#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::estela {
#define GEN_PASS_DEF_STRELAOVERLAPLASTTILEPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// `value` as a single-result affine map of its operands, composed through the
// affine.apply ops that compute it.
static affine::AffineValueMap
toValueMap(Value value) {
  AffineMap map = AffineMap::get(
    0, 1, getAffineSymbolExpr(0, value.getContext())
  );
  SmallVector<Value> operands = {value};
  affine::fullyComposeAffineMapAndOperands(&map, &operands);
  affine::canonicalizeMapAndOperands(&map, &operands);
  return affine::AffineValueMap(map, operands);
}

// True if a + b is the constant `sum` for every value of the operands.
static bool
sumsTo(affine::AffineValueMap a, affine::AffineValueMap b, int64_t sum) {
  // a - (-b): difference() is the only arithmetic AffineValueMap offers.
  AffineMap negated = AffineMap::get(
    b.getAffineMap().getNumDims(), b.getAffineMap().getNumSymbols(),
    -b.getResult(0)
  );
  affine::AffineValueMap negatedB(negated, b.getOperands());
  affine::AffineValueMap result;
  affine::AffineValueMap::difference(a, negatedB, &result);
  AffineExpr expr = simplifyAffineExpr(
    result.getResult(0), result.getNumDims(), result.getNumSymbols()
  );
  auto constant = dyn_cast<AffineConstantExpr>(expr);
  return constant && constant.getValue() == sum;
}

// The tile size `size` of a dimension of extent `extent`, if it is IREE's tail
// clamp min(extent - offset, tile) for the tile offset `offset`.
static std::optional<int64_t>
matchTailClamp(Value size, Value offset, int64_t extent) {
  auto minOp = size.getDefiningOp<affine::AffineMinOp>();
  if (!minOp || minOp.getAffineMap().getNumResults() != 2) return std::nullopt;
  affine::AffineValueMap clamp(minOp.getAffineMap(), minOp.getOperands());
  std::optional<int64_t> tile;
  std::optional<AffineExpr> remaining;
  for (AffineExpr expr : minOp.getAffineMap().getResults()) {
    if (auto constant = dyn_cast<AffineConstantExpr>(expr)) {
      tile = constant.getValue();
    } else {
      remaining = expr;
    }
  }
  if (!tile || !remaining) return std::nullopt;
  AffineMap remainingMap = AffineMap::get(
    minOp.getAffineMap().getNumDims(), minOp.getAffineMap().getNumSymbols(),
    *remaining
  );
  SmallVector<Value> operands(minOp.getOperands());
  affine::fullyComposeAffineMapAndOperands(&remainingMap, &operands);
  affine::canonicalizeMapAndOperands(&remainingMap, &operands);
  if (!sumsTo(affine::AffineValueMap(remainingMap, operands), toValueMap(offset),
              extent)) {
    return std::nullopt;
  }
  return tile;
}

// The tail clamps of one dispatch.tensor.load or store, rewritten in place:
// the offset becomes min(offset, extent - tile) and the size the tile.
template <typename OpTy>
static void
overlapLastTile(RewriterBase &rewriter, OpTy op, ArrayRef<int64_t> shape,
                SmallVectorImpl<Value> &clampedSizes,
                SmallVectorImpl<int64_t> &tiles) {
  SmallVector<OpFoldResult> offsets = op.getMixedOffsets();
  SmallVector<OpFoldResult> sizes = op.getMixedSizes();
  for (auto [d, extent] : llvm::enumerate(shape)) {
    auto size = dyn_cast<Value>(sizes[d]);
    auto offset = dyn_cast<Value>(offsets[d]);
    if (!size || !offset || ShapedType::isDynamic(extent)) continue;
    std::optional<int64_t> tile = matchTailClamp(size, offset, extent);
    if (!tile || *tile > extent) continue;

    rewriter.setInsertionPoint(op);
    MLIRContext *context = rewriter.getContext();
    AffineMap map = AffineMap::get(
      0, 1,
      {getAffineSymbolExpr(0, context),
       getAffineConstantExpr(extent - *tile, context)},
      context
    );
    Value clamped = affine::makeComposedAffineMin(
      rewriter, op.getLoc(), map, {OpFoldResult(offset)}
    );
    rewriter.modifyOpInPlace(op, [&] {
      for (OpOperand &operand : op.getOffsetsMutable()) {
        if (operand.get() == offset) operand.set(clamped);
      }
    });
    clampedSizes.push_back(size);
    tiles.push_back(*tile);
  }
}

// IREE tiles F by 3 with offset 3·id and size min(3, F - 3·id), and the
// bitstream can't run the smaller last tile. Rewrite every such tile to
// offset min(3·id, F - 3) and size 3: when 3 does not divide F the last job
// overlaps the one before it, rewriting the same values. The clamped offset is
// exactly the min(wg · step, limit) term of a column address.
//
// Runs after ReconcileTranslationInfo, so the overlapping writes are plain
// stores rather than conflicting parallel_insert_slices. Correct when jobs run
// concurrently on several CGRAs: overlapping jobs write the same values to the
// same whole words.
struct StrelaOverlapLastTilePass final
    : estela::impl::StrelaOverlapLastTilePassBase<StrelaOverlapLastTilePass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());
    SmallVector<Value> clampedSizes;
    SmallVector<int64_t> tiles;
    funcOp.walk([&](Operation *op) {
      if (auto loadOp = dyn_cast<IREE::TensorExt::DispatchTensorLoadOp>(op)) {
        overlapLastTile(rewriter, loadOp, loadOp.getSourceType().getShape(),
                        clampedSizes, tiles);
      } else if (auto storeOp =
                   dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(op)) {
        overlapLastTile(rewriter, storeOp, storeOp.getTargetType().getShape(),
                        clampedSizes, tiles);
      }
    });
    // Every use of a clamped size, the tile's tensor.empty included, now sees
    // the full tile.
    for (auto [size, tile] : llvm::zip_equal(clampedSizes, tiles)) {
      if (!size.getDefiningOp()) continue;
      rewriter.setInsertionPoint(size.getDefiningOp());
      Value constant =
        arith::ConstantIndexOp::create(rewriter, size.getLoc(), tile);
      rewriter.replaceAllUsesWith(size, constant);
    }
  }
};

} // namespace
