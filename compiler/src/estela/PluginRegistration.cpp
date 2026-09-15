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
    binder.opt<bool>(
      "iree-estela-partition",
      partition,
      llvm::cl::desc("Enable the partition of supported operations by STRELA from the CPU ones"),
      llvm::cl::cat(category)
    );
  }
};

struct EstelaSession : public PluginSession<EstelaSession, EstelaOptions> {

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
      bool extensionsWereMade = false;
      if (options.enable_fusion) {
        passManager.addNestedPass<func::FuncOp>(estela::createFuseConv2DPass());
        extensionsWereMade = true;
      }
      return extensionsWereMade;
  }

  // This is anchored on builtin.module
  void
  extendPreprocessingPassPipeline(OpPassManager &passManager) override {
    if (options.partition) {
      passManager.addPass(estela::createFormStrelaDispatchesPass());
    }
  }

  void
  onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<strela::StrelaDialect>();
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
