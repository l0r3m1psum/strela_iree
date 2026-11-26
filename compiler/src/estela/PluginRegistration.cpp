#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/PluginAPI/Client.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

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
    OpPassManager &funcPassManager = passManager.nest<IREE::Util::FuncOp>();
    // A quantized fully connected layer (a 1x1 conv_2d_nhwc_hwcf_q) becomes a
    // linalg.matmul of its i8 operands from a zero fill, then the bias add and
    // the zero point correction -zx * sum_c w[c, f] on the host, whose sum of
    // constants const-eval folds. IREE runs the first two in
    // GlobalOptimization anyway; here they run before STRELA picks its
    // matmuls.
    funcPassManager.addPass(
      GlobalOptimization::createLinalgQuantizedConvToConvPass()
    );
    funcPassManager.addPass(
      GlobalOptimization::createDetachElementwiseFromNamedOpsPass()
    );
    funcPassManager.addPass(
      Preprocessing::createConvert1X1FilterConv2DToMatmulPass()
    );
    funcPassManager.addPass(createLinalgFoldUnitExtentDimsPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createLinalgSpecializeGenericOpsPass());
    // The operands of the matmuls STRELA will run widened to i32, the weights
    // of every matmul transposed to F x C, then the matmuls STRELA can run
    // pinned to it. With F x C weights each feature's C weights are
    // contiguous, so a weight column spans C words instead of C * F. Constant
    // weights are transposed by const-eval; weights computed at run time would
    // get a transpose at run time too.
    funcPassManager.addPass(estela::createStrelaWidenOperandsPass());
    funcPassManager.addPass(Preprocessing::createTransposeMatmulPass(
      {.input = Preprocessing::TransposeMatmulInput::Rhs}
    ));
    funcPassManager.addPass(estela::createTransferWrapPass());
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
