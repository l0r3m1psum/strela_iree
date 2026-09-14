// The STRELA HAL target: what the compiler needs to know to produce an
// executable for the accelerator, and what it advertises about the device.

#ifndef STRELA_TARGET_STRELATARGET_H_
#define STRELA_TARGET_STRELATARGET_H_

#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "iree/compiler/Dialect/HAL/Target/TargetDevice.h"

namespace mlir::iree_compiler {

// Drives executable translation and serialization for the "strela"/"custom"
// executable target.
struct StrelaTargetBackend : public IREE::HAL::TargetBackend {
  std::string getLegacyDefaultDeviceID() const override;

  void getDependentDialects(DialectRegistry &registry) const override;

  void buildTranslationPassPipeline(
    IREE::HAL::ExecutableTargetAttr executableTargetAttr,
    OpPassManager &passManager
  ) override;

  void getDefaultExecutableTargets(
    MLIRContext *context,
    StringRef deviceID,
    DictionaryAttr deviceConfigAttr,
    SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs
  ) const override;

  LogicalResult serializeExecutable(
    const SerializationOptions &options,
    IREE::HAL::ExecutableVariantOp variantOp,
    OpBuilder &executableBuilder
  ) override;
};

// Describes the STRELA device itself: its resource limits and the executable
// formats it accepts.
struct StrelaTargetDevice : public IREE::HAL::TargetDevice {
  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
    MLIRContext *context,
    const IREE::HAL::TargetRegistry &targetRegistry
  ) const override;
};

} // namespace mlir::iree_compiler

#endif // STRELA_TARGET_STRELATARGET_H_
