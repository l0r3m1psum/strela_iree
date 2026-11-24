#include "estela/Transforms/Passes.h"

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtDialect.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::estela {
#define GEN_PASS_DEF_STRELASTOREWORKGROUPTILESPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

static bool
allConstant(ArrayRef<OpFoldResult> values, int64_t constant) {
  return llvm::all_of(values, [&](OpFoldResult v) {
    return isConstantIntValue(v, constant);
  });
}

// True for a store or load of the whole tensor: from the origin, unit strides.
// The sizes are the tensor's, or the op would not verify.
template <typename OpTy>
static bool
isWholeTensor(OpTy op) {
  return allConstant(op.getMixedOffsets(), 0) && allConstant(op.getMixedStrides(), 1);
}

// The workgroup scf.forall IREE tiles a dispatch into returns the tiles it
// computed, and the dispatch stores them whole:
//
//   %r = scf.forall (%y, %x) ... shared_outs(%o = %init) {
//     ...
//     scf.forall.in_parallel {
//       tensor.parallel_insert_slice %tile into %o[%y, %x] [1, 3] [1, 1]
//     }
//   }
//   iree_tensor_ext.dispatch.tensor.store %r, %target, offsets = [0, 0], ...
//
// ReconcileTranslationInfo only resolves a forall without results, which the
// CPU pipelines reach by bufferizing. STRELA stays on tensors: each tile is
// stored where the parallel_insert_slice put it, and the forall returns
// nothing.
static LogicalResult
storeTilesInForall(RewriterBase &rewriter, scf::ForallOp forallOp) {
  SmallVector<IREE::TensorExt::DispatchTensorStoreOp> storeOps;
  for (OpResult result : forallOp->getResults()) {
    if (!result.hasOneUse()) {
      return forallOp.emitOpError("result ") << result.getResultNumber()
             << " must be stored to a binding exactly once";
    }
    auto storeOp =
      dyn_cast<IREE::TensorExt::DispatchTensorStoreOp>(*result.user_begin());
    if (!storeOp || !isWholeTensor(storeOp)) {
      return forallOp.emitOpError("result ") << result.getResultNumber()
             << " must be stored whole to a binding";
    }
    storeOps.push_back(storeOp);
  }

  rewriter.setInsertionPoint(forallOp);
  auto newForallOp = scf::ForallOp::create(
    rewriter, forallOp.getLoc(), forallOp.getMixedLowerBound(),
    forallOp.getMixedUpperBound(), forallOp.getMixedStep(), ValueRange{},
    forallOp.getMapping()
  );

  // The shared outputs are only written through parallel_insert_slice, which
  // become stores; any other use reads the initial value.
  SmallVector<Value> replacements(newForallOp.getInductionVars());
  llvm::append_range(replacements, forallOp.getOutputs());
  scf::InParallelOp oldTerminator = forallOp.getTerminator();
  SmallVector<std::pair<tensor::ParallelInsertSliceOp, unsigned>> inserts;
  for (Operation &op : oldTerminator.getYieldingOps()) {
    auto insertOp = cast<tensor::ParallelInsertSliceOp>(&op);
    auto dest = cast<BlockArgument>(insertOp.getDest());
    inserts.push_back({insertOp, dest.getArgNumber() - forallOp.getRank()});
  }
  rewriter.inlineBlockBefore(
    forallOp.getBody(), newForallOp.getBody()->getTerminator(), replacements
  );

  rewriter.setInsertionPoint(oldTerminator);
  for (auto [insertOp, index] : inserts) {
    IREE::TensorExt::DispatchTensorStoreOp storeOp = storeOps[index];
    IREE::TensorExt::DispatchTensorStoreOp::create(
      rewriter, insertOp.getLoc(), insertOp.getSource(), storeOp.getTarget(),
      storeOp.getTargetDims(), insertOp.getMixedOffsets(),
      insertOp.getMixedSizes(), insertOp.getMixedStrides()
    );
  }
  rewriter.eraseOp(oldTerminator);
  for (auto storeOp : storeOps) rewriter.eraseOp(storeOp);
  rewriter.eraseOp(forallOp);
  return success();
}

// A tile cut from a whole-tensor load becomes a load of just the tile, so each
// workgroup reads what its job reads.
static void
loadTiles(RewriterBase &rewriter, func::FuncOp funcOp) {
  SmallVector<tensor::ExtractSliceOp> sliceOps;
  funcOp.walk([&](tensor::ExtractSliceOp sliceOp) {
    auto loadOp =
      sliceOp.getSource().getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    if (loadOp && isWholeTensor(loadOp)) sliceOps.push_back(sliceOp);
  });
  for (tensor::ExtractSliceOp sliceOp : sliceOps) {
    auto loadOp =
      sliceOp.getSource().getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
    rewriter.setInsertionPoint(sliceOp);
    rewriter.replaceOpWithNewOp<IREE::TensorExt::DispatchTensorLoadOp>(
      sliceOp, sliceOp.getType(), loadOp.getSource(), loadOp.getSourceDims(),
      sliceOp.getMixedOffsets(), sliceOp.getMixedSizes(),
      sliceOp.getMixedStrides()
    );
  }
}

static bool
isWorkgroupForall(scf::ForallOp forallOp) {
  std::optional<ArrayAttr> mapping = forallOp.getMapping();
  return mapping && llvm::all_of(*mapping, [](Attribute attr) {
    return isa<IREE::Codegen::WorkgroupMappingAttr>(attr);
  });
}

struct StrelaStoreWorkgroupTilesPass final
    : estela::impl::StrelaStoreWorkgroupTilesPassBase<
        StrelaStoreWorkgroupTilesPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(funcOp.getContext());

    SmallVector<scf::ForallOp> forallOps;
    funcOp.walk([&](scf::ForallOp forallOp) {
      if (isWorkgroupForall(forallOp) && forallOp->getNumResults() != 0) {
        forallOps.push_back(forallOp);
      }
    });
    for (scf::ForallOp forallOp : forallOps) {
      if (failed(storeTilesInForall(rewriter, forallOp))) {
        return signalPassFailure();
      }
    }
    loadTiles(rewriter, funcOp);
  }
};

} // namespace
