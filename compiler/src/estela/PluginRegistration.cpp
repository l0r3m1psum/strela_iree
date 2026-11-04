#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

#include "estela/Dialect/Strela/IR/StrelaDialect.h"
#include "estela/Codegen/Strela/StrelaTarget.h"
#include "estela/Transforms/Linalg/Passes.h"
#include "estela/Transforms/Tosa/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

struct EstelaOptions {
  bool enable_fusion = false;
  bool partition = false;
  void bindOptions(OptionsBinder& binder) {
    static llvm::cl::OptionCategory category("IREE STRELA Plugin");
    binder.opt<bool>(
      "iree-estela-fusion",
      enable_fusion,
      llvm::cl::desc("Enable the custom centered-gemm fusion"),
      llvm::cl::cat(category)
    );
  }
};

struct EstelaSession : public PluginSession<EstelaSession, EstelaOptions> {

  // Lets iree-opt run the passes on their own, e.g. in LIT tests.
  static void
  registerPasses() {
    mlir::registerPass([] { return estela::createTransferWrapPass(); });
  }

  void
  extendInputConversionPreprocessingPassPipeline(
    OpPassManager &passManager, InputDialectOptions::Type inputType
  ) override {
    passManager.addNestedPass<func::FuncOp>(estela::createNormalizeRescalePass());
  }

  // --iree-input-type=typeMnemonic
  bool
  extendCustomInputConversionPassPipeline(
    OpPassManager& passManager, std::string_view typeMnemonic
  ) override {
    // FIXME(Diego): this should not be here but in extendPreprocessingPassPipeline
    // This pass should be used just to convert from things like
    // typeMnemonic=="stablehlo"
    bool extensionsWereMade = false;
    if (options.enable_fusion) {
      passManager.addNestedPass<func::FuncOp>(estela::createFuseConv2DPass());
      extensionsWereMade = true;
    }
    return extensionsWereMade;
  }

  // Happens between --compile-to=abi and --compile-to=preprocessing
  // iree-compile --iree-plugin=estela --iree-hal-target-device=strela \
  // --compile-to=preprocessing --mlir-print-ir-before=<first-pass> \
  //  --mlir-disable-threading -mlir-print-ir-module-scope \
  // <file-name> -o /dev/null
  void
  extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    // TODO: create a pass that converts all integers smaller than int32 to int32
    passManager.addPass(estela::createAssignTopologyPass());
    // TODO(Diego): implement a pass that adds this information so that the
    // compiler prove that some dynamic sizes are the same and needs to push
    // less constants when launching kernels.
#if 0
static void
tieInputsToInits(RewriterBase &rewriter, linalg::GenericOp op) {
  rewriter.setInsertionPoint(op);
  for (OpOperand *input : op.getDpsInputOperands()) {
    Value value = input->get();
    if (cast<ShapedType>(value.getType()).hasStaticShape()) continue;
    for (OpOperand &init : op.getDpsInitsMutable()) {
      if (init.get().getType() != value.getType() ||
          op.getMatchingIndexingMap(&init) != op.getMatchingIndexingMap(input)) {
        continue;
      }
      SmallVector<Value> dims = IREE::Util::buildDynamicDimsForValue(
        op.getLoc(), init.get(), rewriter
      );
      Value tied = IREE::Flow::TensorTieShapeOp::create(
        rewriter, op.getLoc(), value.getType(), value, dims
      );
      rewriter.modifyOpInPlace(op, [&] { input->set(tied); });
      break;
    }
  }
}
#endif
    passManager.addNestedPass<IREE::Util::FuncOp>(estela::createTransferWrapPass());
    // TODO:
    // populateDecomposeLinalgOpsPattern -> explode each generic to one statement per op.
    // Greedy re-fuse with your ControlFusionFn budget -> groups the fabric can place.
    // util.optimization_barrier on each group's results -> DispatchCreation can't merge them back.
  }

  void
  onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<strela::StrelaDialect>();
  }

  LogicalResult
  onActivate() override {
    const llvm::DenseMap<llvm::StringRef, llvm::cl::Option *> &registered = llvm::cl::getRegisteredOptions();
    auto *limit = static_cast<llvm::cl::opt<int> *>(
      registered.lookup("iree-flow-inline-constants-max-byte-length")
    );
    if (
      limit
      && limit->getNumOccurrences() != 0
      && limit->getValue() > sizeof (int32_t)
    ) {
      mlir::emitWarning(mlir::UnknownLoc::get(context))
        << "STRELA supports inlining only scalar constants.";
    }
    return success();
  }

  void
  populateHALTargetBackends(IREE::HAL::TargetBackendList& targets) override {
    targets.add(
      "strela",
      []() -> std::shared_ptr<IREE::HAL::TargetBackend> {
        return std::make_shared<estela::StrelaTargetBackend>();
      }
    );
  }

  void
  populateHALTargetDevices(IREE::HAL::TargetDeviceList& targets) override {
    targets.add(
      "strela",
      []() -> std::shared_ptr<IREE::HAL::TargetDevice> {
        return std::make_shared<estela::StrelaTargetDevice>();
      }
    );
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(EstelaOptions);

extern "C" bool
iree_register_compiler_plugin_estela(
  mlir::iree_compiler::PluginRegistrar *registrar
) {
  registrar->registerPlugin<EstelaSession>("estela");
  return true;
}
