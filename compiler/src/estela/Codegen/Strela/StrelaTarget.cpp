#include "estela/Codegen/Strela/StrelaTarget.h"

#include <vector>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "estela/Dialect/Strela/IR/StrelaDialect.h"
#include "estela/Dialect/Strela/IR/StrelaOps.h"
#include "estela/Dialect/Strela/Conversion/Passes.h"
#include "estela/Transforms/Linalg/Passes.h"
#include "estela/Dialect/Strela/Transforms/Passes.h"
#include "estela/Utils/Bitstreams.h"

using namespace mlir::iree_compiler;

namespace mlir::estela {

//===--------------------------------------------------------------------===//
// StrelaTargetBackend
//===--------------------------------------------------------------------===//

std::string StrelaTargetBackend::getLegacyDefaultDeviceID() const {
  return "strela";
}

void StrelaTargetBackend::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<strela::StrelaDialect>();
}

void StrelaTargetBackend::buildTranslationPassPipeline(
  iree_compiler::IREE::HAL::ExecutableTargetAttr executableTargetAttr,
  OpPassManager &passManager
) {
  OpPassManager &modulePassManager = passManager.nest<ModuleOp>();

  modulePassManager.addNestedPass<func::FuncOp>(
    estela::createConvertLinalgToStrelaPass()
  );

  // Resolve the dispatch interface while the full executable source is still
  // around, and leave the answer in the IR as strela.binding_load/store.
  modulePassManager.addNestedPass<func::FuncOp>(
    estela::createMaterializeBindingsPass()
  );

  passManager.addPass(estela::createResolveWorkgroupCountPass());
}

// TODO: how does this relate to StrelaTargetDevice::getDefaultDeviceTarget?
// Can some code between the two be reused?
void StrelaTargetBackend::getDefaultExecutableTargets(
  MLIRContext *context,
  StringRef deviceID,
  DictionaryAttr deviceConfigAttr,
  SmallVectorImpl<iree_compiler::IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs
) const {
  Builder b(context);
  SmallVector<NamedAttribute> configItems;

  auto executableTargetAttr = b.getAttr<iree_compiler::IREE::HAL::ExecutableTargetAttr>(
    b.getStringAttr("strela"), b.getStringAttr("strela_bitstream"),
    b.getDictionaryAttr(configItems)
  );
  executableTargetAttrs.push_back(executableTargetAttr);
}

LogicalResult StrelaTargetBackend::serializeExecutable(
  const SerializationOptions &options,
  IREE::HAL::ExecutableVariantOp variantOp,
  OpBuilder &executableBuilder
) {
  uint32_t detected_opcode = 0;

  mlir::ModuleOp innerModule = variantOp.getInnerModule();
  if (innerModule) {
    innerModule.walk([&detected_opcode](Operation *op) {
      if (isa<strela::AddOp>(op)) {
        detected_opcode = 1; // e.g. 1 = ADD
      } else if (isa<strela::ReluOp>(op)) {
        detected_opcode = 2; // e.g. 2 = RELU
      }
    });
  }

  if (detected_opcode != 0) {
    const uint8_t *byte_ptr =
      reinterpret_cast<const uint8_t *>(estela::centered_matmul_bitstream.data());
    std::vector<uint8_t> binary_payload(
      byte_ptr, byte_ptr + sizeof estela::centered_matmul_bitstream
    );

    IREE::HAL::ExecutableBinaryOp::create(
      executableBuilder,
      variantOp.getLoc(),
      variantOp.getSymNameAttr(),         // Inherit the symbol name ("strela")
      variantOp.getTarget().getFormat(),  // Inherit the format ("strela_bitstream")
      binary_payload
    );
    return success();
  } else {
    return failure();
  }
}

//===--------------------------------------------------------------------===//
// StrelaTargetDevice
//===--------------------------------------------------------------------===//

iree_compiler::IREE::HAL::DeviceTargetAttr StrelaTargetDevice::getDefaultDeviceTarget(
  MLIRContext *context,
  const IREE::HAL::TargetRegistry &targetRegistry
) const {
  mlir::Builder b(context);

  // With
  // iree-compile --iree-plugin=estela  --iree-hal-target-device=strela --compile-to=stream 3rdparty/iree/samples/models/simple_abs.mlir   -o simple_abs_hal.mlir
  // this #stream.resource_config appears in the MLIR source.
  auto resourceConfigAttr = b.getAttr<IREE::Stream::ResourceConfigAttr>(
    // TODO: put real numbers...
    /*max_allocation_size=*/ 1ull * 1024 * 1024 * 1024,
    /*min_buffer_offset_alignment=*/ 64,
    /*max_buffer_range=*/ 1ull * 1024 * 1024 * 1024, // the largest span a single binding may cover
    /*min_buffer_range_alignment=*/ 64,
    /*index_bits=*/ 64,
    /*alias_mutable_bindings=*/ false,
    /*memory_model=*/ IREE::Stream::MemoryModel::Unified
  );

  SmallVector<NamedAttribute> configItems;
  configItems.emplace_back(
    b.getStringAttr("stream.resource_config"), resourceConfigAttr
  );

  auto configAttr = b.getDictionaryAttr(configItems);
  auto deviceID = b.getStringAttr("strela");

  // With
  // iree-compile --iree-plugin=estela  --iree-hal-target-device=strela --compile-to=hal 3rdparty/iree/samples/models/simple_abs.mlir   -o simple_abs_hal.mlir
  // this #hal.executable.target appears in the MLIR source.
  auto executableTargetAttr = b.getAttr<iree_compiler::IREE::HAL::ExecutableTargetAttr>(
    /*backend=*/ deviceID,
    /*format=*/ b.getStringAttr("strela_bitstream"),
    /*configuration=*/ b.getDictionaryAttr({})
  );

  return IREE::HAL::DeviceTargetAttr::get(
    context,
    /*deviceID=*/ deviceID,
    /*configuration=*/ configAttr,
    /*executable_targets=*/ {executableTargetAttr}
  );
}

} // namespace mlir::estela
