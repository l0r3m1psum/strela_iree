// The STRELA HAL target: what the compiler needs to know to produce an
// executable for the accelerator, and what it advertises about the device.

#ifndef ESTELA_CODEGEN_STRELA_STRELATARGET_H_
#define ESTELA_CODEGEN_STRELA_STRELATARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

namespace mlir::estela {

// Drives executable translation and serialization for the "strela"/"strela_bitstream"
// executable target.
struct StrelaTargetBackend : public iree_compiler::IREE::HAL::TargetBackend {
  std::string getLegacyDefaultDeviceID() const override;

  void getDependentDialects(DialectRegistry &registry) const override;

  void buildTranslationPassPipeline(
    iree_compiler::IREE::HAL::ExecutableTargetAttr executableTargetAttr,
    OpPassManager &passManager
  ) override;

  void getDefaultExecutableTargets(
    MLIRContext *context,
    StringRef deviceID,
    DictionaryAttr deviceConfigAttr,
    SmallVectorImpl<iree_compiler::IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs
  ) const override;

  LogicalResult serializeExecutable(
    const SerializationOptions &options,
    iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
    OpBuilder &executableBuilder
  ) override;
};

// Describes the STRELA device itself: its resource limits and the executable
// formats it accepts.
struct StrelaTargetDevice : public iree_compiler::IREE::HAL::TargetDevice {
  iree_compiler::IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
    MLIRContext *context,
    const iree_compiler::IREE::HAL::TargetRegistry &targetRegistry
  ) const override;
};

} // namespace mlir::estela

#endif // ESTELA_CODEGEN_STRELA_STRELATARGET_H_
