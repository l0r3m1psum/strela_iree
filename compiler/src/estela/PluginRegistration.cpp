#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"

#include "estela/StrelaOptions.h"
#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaTarget.h"
#include "estela/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

using mlir::estela::EstelaOptions;

void
EstelaOptions::bindOptions(OptionsBinder &binder) {
  static llvm::cl::OptionCategory category("STRELA target");
  binder.opt<unsigned>(
    "iree-strela-rows", rows, llvm::cl::cat(category),
    llvm::cl::desc("Number of PE rows in the STRELA fabric.")
  );
  binder.opt<unsigned>(
    "iree-strela-columns", columns, llvm::cl::cat(category),
    llvm::cl::desc("Number of PE columns (and DMA columns) in the STRELA fabric.")
  );
  binder.opt<unsigned>(
    "iree-strela-max-column-words", maxColumnWords, llvm::cl::cat(category),
    llvm::cl::desc("Most 32-bit words a DMA column can move in one job.")
  );
  binder.opt<unsigned>(
    "iree-strela-multiplier-width", multiplierWidth, llvm::cl::cat(category),
    llvm::cl::desc("Signed bits of each operand the STRELA multipliers use; a "
                   "contraction is offloaded only if its operands are proven "
                   "to fit.")
  );
  binder.opt<bool>(
    "iree-strela-fold-bias", foldBias, llvm::cl::cat(category),
    llvm::cl::desc("Fold the constant bias of each quantized fully connected "
                   "layer into its STRELA matmul as extra reduction elements, "
                   "instead of adding it on the host.")
  );
}

namespace {

struct EstelaSession : public PluginSession<EstelaSession, EstelaOptions> {

  static void
  registerPasses() {
    estela::registerEstelaPasses();
  }

  // --iree-input-type=typeMnemonic
  bool
  extendCustomInputConversionPassPipeline(
    OpPassManager& passManager, std::string_view typeMnemonic
  ) override {
    // This pass should be used just to convert from things like
    // typeMnemonic=="stablehlo"
    bool extensionsWereMade = false;
    return extensionsWereMade;
  }

  // Happens between --compile-to=abi and --compile-to=preprocessing
  // iree-compile --iree-plugin=estela --iree-hal-target-device=strela \
  // --compile-to=preprocessing --mlir-print-ir-before=<first-pass> \
  //  --mlir-disable-threading -mlir-print-ir-module-scope \
  // <file-name> -o /dev/null
  void
  extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    passManager.addPass(estela::createAssignTopologyPass());
    // Zero points folded and constant weights transposed (each quantized fully
    // connected layer becomes an i32 matmul from a zero fill), then the
    // matmuls STRELA can run, with operands proven to fit its multipliers,
    // pinned to it.
    passManager.addNestedPass<IREE::Util::FuncOp>(
      estela::createStrelaPrepareWeightsPass({options.foldBias})
    );
    passManager.addNestedPass<IREE::Util::FuncOp>(estela::createTransferWrapPass());
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
      [options=this->options]() -> std::shared_ptr<IREE::HAL::TargetBackend> {
        return std::make_shared<estela::StrelaTargetBackend>(options);
      }
    );
  }

  void
  populateHALTargetDevices(IREE::HAL::TargetDeviceList& targets) override {
    targets.add(
      "strela",
      [options=this->options]() -> std::shared_ptr<IREE::HAL::TargetDevice> {
        return std::make_shared<estela::StrelaTargetDevice>(options);
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
