#include "estela/Transforms/Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/TensorExt/IR/TensorExtOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

#include "estela/StrelaIR/StrelaOps.h"

namespace mlir::estela {
#define GEN_PASS_DEF_RESOLVEWORKGROUPCOUNTPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// Output features one matvec job computes: output columns 1-3.
constexpr int64_t strelaMatvecFeaturesPerJob = 3;

// One count dimension as the region computes it: a constant if the size is
// static, else the workload argument the dispatch body read that size from.
static FailureOr<Value>
resolveSize(
  OpBuilder &builder, Location loc, Region &countRegion, OpFoldResult size
) {
  if (std::optional<int64_t> c = getConstantIntValue(size)) {
    return arith::ConstantIndexOp::create(builder, loc, *c).getResult();
  }
  auto ordinalOp = cast<Value>(size)
    .getDefiningOp<IREE::TensorExt::DispatchWorkloadOrdinalOp>();
  if (!ordinalOp) return failure();
  // Argument 0 is the !hal.device, the workload follows in ordinal order.
  unsigned index = 1 + ordinalOp.getOrdinal().getZExtValue();
  if (index >= countRegion.getNumArguments()) return failure();
  return countRegion.getArgument(index);
}

// The grid of one export: (ceildiv(F, 3), N, 1) for a matvec, one job otherwise.
static FailureOr<SmallVector<Value, 3>>
buildCount(
  OpBuilder &builder, Location loc, Region &countRegion, func::FuncOp funcOp
) {
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);

  strela::MatvecOp matvecOp;
  funcOp.walk([&](strela::MatvecOp op) { matvecOp = op; });
  if (!matvecOp) return SmallVector<Value, 3>{one, one, one};

  auto rhsType = cast<RankedTensorType>(matvecOp.getRhs().getType());
  int64_t features = rhsType.getDimSize(2);
  if (ShapedType::isDynamic(features) || features < strelaMatvecFeaturesPerJob) {
    return matvecOp.emitOpError("needs a static F of at least ")
           << strelaMatvecFeaturesPerJob;
  }
  Value x = arith::ConstantIndexOp::create(
    builder, loc, llvm::divideCeil(features, strelaMatvecFeaturesPerJob)
  );

  auto lhsLoad = matvecOp.getLhs()
    .getDefiningOp<IREE::TensorExt::DispatchTensorLoadOp>();
  if (!lhsLoad) {
    return matvecOp.emitOpError("lhs is not a dispatch tensor load; the "
                                "workgroup count must run before "
                                "MaterializeBindings");
  }
  FailureOr<Value> y =
    resolveSize(builder, loc, countRegion, lhsLoad.getMixedSizes()[1]);
  if (failed(y)) {
    return lhsLoad.emitOpError("cannot tell which workload value is N");
  }
  return SmallVector<Value, 3>{x, *y, one};
}

struct StrelaResolveWorkgroupCountPass
    : estela::impl::ResolveWorkgroupCountPassBase<StrelaResolveWorkgroupCountPass> {

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    ModuleOp innerModule = variantOp.getInnerModule();
    OpBuilder builder(&getContext());

    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      Region &countRegion = exportOp.getWorkgroupCount();
      if (countRegion.empty()) continue;
      auto funcOp = innerModule.lookupSymbol<func::FuncOp>(exportOp.getSymName());
      if (!funcOp) {
        exportOp.emitOpError("has no function in the inner module");
        return signalPassFailure();
      }

      auto fromSliceOps = llvm::to_vector(
        countRegion.front()
          .getOps<IREE::TensorExt::DispatchWorkgroupCountFromSliceOp>()
      );
      if (fromSliceOps.size() != 1) {
        exportOp.emitOpError(
          "expected one iree_tensor_ext.dispatch.workgroup_count_from_slice "
          "placeholder in the workgroup count region"
        );
        return signalPassFailure();
      }
      auto fromSliceOp = fromSliceOps.front();

      // Built in place of the placeholder, so the workload arguments of the
      // region are still in scope.
      builder.setInsertionPoint(fromSliceOp);
      FailureOr<SmallVector<Value, 3>> count =
        buildCount(builder, exportOp.getLoc(), countRegion, funcOp);
      if (failed(count)) return signalPassFailure();

      fromSliceOp->replaceAllUsesWith(*count);
      fromSliceOp->erase();
    }
  }
};

} // namespace
