#include "estela/Dialect/Strela/Transforms/Passes.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// Dispatch creation leaves an
// `iree_tensor_ext.dispatch.workgroup_count_from_slice()` placeholder in every
// export's workgroup count region; each codegen backend is expected to resolve
// it (llvm-cpu/vmvx do it in ResolveWorkgroupCountHints). If it survives it is
// inlined into the host program and --compile-to=vm aborts inside
// iree-vm-conversion with "callback produced incorrect number of values".
// A STRELA kernel always runs as a single unit of work, so return (1, 1, 1).
struct StrelaResolveWorkgroupCountPass
    : public PassWrapper<StrelaResolveWorkgroupCountPass,
                         OperationPass<IREE::HAL::ExecutableVariantOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(StrelaResolveWorkgroupCountPass)

  StringRef getArgument() const override {
    return "iree-strela-resolve-workgroup-count";
  }
  StringRef getDescription() const override {
    return "Replaces STRELA export workgroup count "
      "(iree_tensor_ext.dispatch.workgroup_count_from_slice) regions with (1, 1, 1)";
  }

  void runOnOperation() override {
    IREE::HAL::ExecutableVariantOp variantOp = getOperation();
    OpBuilder builder(&getContext());

    for (auto exportOp : variantOp.getOps<IREE::HAL::ExecutableExportOp>()) {
      Region &countRegion = exportOp.getWorkgroupCount();
      if (!countRegion.empty()) {
        Location loc = exportOp.getLoc();
        SmallVector<Type> argTypes(countRegion.getArgumentTypes());
        SmallVector<Location> argLocs(argTypes.size(), loc);

        countRegion.getBlocks().clear();
        Block *block = builder.createBlock(
          &countRegion, countRegion.end(), argTypes, argLocs
        );
        builder.setInsertionPointToStart(block);

        Value one = arith::ConstantIndexOp::create(builder, loc, 1);
        IREE::HAL::ReturnOp::create(builder, loc, ValueRange{one, one, one});
      }
    }
  }
};

} // namespace

namespace mlir::estela {

std::unique_ptr<Pass> createResolveWorkgroupCountPass() {
  return std::make_unique<StrelaResolveWorkgroupCountPass>();
}

} // namespace mlir::estela
