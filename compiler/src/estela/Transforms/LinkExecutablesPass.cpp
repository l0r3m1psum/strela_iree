#include "estela/Transforms/Passes.h"

#include "iree/compiler/Codegen/Utils/LinkingUtils.h"
#include "iree/compiler/Utils/ModuleUtils.h"

#include "llvm/Support/FormatVariadic.h"

namespace mlir::estela {
#define GEN_PASS_DEF_LINKEXECUTABLESPASS
#include "Passes.h.inc"
} // namespace mlir::estela

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

// The inner modules hold one func.func per export and nothing else, so
// merging their symbols is all there is to do.
static LogicalResult
mergeModuleFn(
  ModuleOp sourceInnerModule,
  ModuleOp linkedInnerModule,
  DenseMap<StringRef, Operation *> &symbolMap
) {
  return mergeModuleInto(sourceInnerModule, linkedInnerModule, symbolMap);
}

struct LinkExecutablesPass final
  : estela::impl::LinkExecutablesPassBase<LinkExecutablesPass> {
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    SmallVector<IREE::HAL::ExecutableOp> sourceExecutableOps =
      gatherExecutablesForTarget(moduleOp, "strela");
    if (sourceExecutableOps.size() <= 1) return;

    OpBuilder moduleBuilder = OpBuilder::atBlockBegin(moduleOp.getBody());
    auto linkedExecutableOp = IREE::HAL::ExecutableOp::create(
      moduleBuilder, moduleOp.getLoc(),
      llvm::formatv("{}_linked_strela", guessModuleName(moduleOp, "module")).str()
    );
    linkedExecutableOp.setVisibility(sourceExecutableOps.front().getVisibility());
    OpBuilder executableBuilder =
      OpBuilder::atBlockBegin(&linkedExecutableOp.getBlock());

    auto targetAttrs = gatherExecutableTargets(sourceExecutableOps);
    for (auto [index, targetAttr] : llvm::enumerate(targetAttrs)) {
      // Executables can only share a variant if their targets are identical, so
      // there is one linked variant per distinct target e.g.
      // #hal.executable.target<"strela", "strela_bitstream">
      // #hal.executable.target<"strela", "strela_bitstream", {some_option = 1}>
      // are different targets.
      std::string variantName = llvm::formatv("{}_{}", targetAttr.getSymbolNameFragment(), index).str();
      auto linkedVariantOp = IREE::HAL::ExecutableVariantOp::create(
        executableBuilder, moduleOp.getLoc(), variantName, targetAttr
      );
      OpBuilder variantBuilder =
        OpBuilder::atBlockBegin(&linkedVariantOp.getBlock());
      ModuleOp::create(variantBuilder, moduleOp.getLoc());

      if (failed(linkExecutablesInto(moduleOp, sourceExecutableOps,
                                     linkedExecutableOp, linkedVariantOp,
                                     mergeModuleFn))) {
        return signalPassFailure();
      }
    }

    if (linkedExecutableOp.getBody().empty()) linkedExecutableOp.erase();
  }
};

} // namespace
