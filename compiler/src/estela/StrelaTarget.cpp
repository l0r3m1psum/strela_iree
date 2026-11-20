#include "estela/StrelaTarget.h"

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/compiler/Codegen/Common/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"
#include "estela/StrelaIR/StrelaAttrs.h"
#include "estela/Transforms/Passes.h"
#include "estela/Utils/Finders.h"

#include "strela_executable_def_builder.h"

using namespace mlir::iree_compiler;

namespace mlir::estela {

// The single strela.job of an export. Everything the flatbuffer needs is in it.
static FailureOr<strela::JobOp>
getJob(func::FuncOp funcOp) {
  SmallVector<strela::JobOp> jobOps(funcOp.getOps<strela::JobOp>());
  if (jobOps.size() != 1) {
    return funcOp.emitOpError("expected exactly one strela.job, found ")
           << jobOps.size();
  }
  return jobOps.front();
}

// An immediate, or the ordinal of the push constant holding the value.
static iree_hal_strela_ValueDef_t
toValueDef(Attribute attr) {
  if (auto pc = dyn_cast<strela::PushConstantAttr>(attr)) {
    return {pc.getOrdinal(), true};
  }
  return {static_cast<uint32_t>(cast<IntegerAttr>(attr).getInt()), false};
}

static iree_hal_strela_ValueDef_t
immediate(uint32_t value) {
  return {value, false};
}

static iree_hal_strela_WorkgroupTermDef_t
toTermDef(strela::WgAxisParamsAttr params) {
  return {immediate(params.getStep()), immediate(params.getLimit())};
}

static iree_hal_strela_ColumnAddressDef_ref_t
createAddress(FlatbufferBuilder &builder, strela::ColAddrAttr address) {
  iree_hal_strela_ValueDef_t offset = toValueDef(address.getOffset());
  iree_hal_strela_WorkgroupTermDef_t x = toTermDef(address.getWgXParams());
  iree_hal_strela_WorkgroupTermDef_t y = toTermDef(address.getWgYParams());
  iree_hal_strela_ColumnAddressDef_start(builder);
  iree_hal_strela_ColumnAddressDef_binding_add(builder, address.getBinding());
  iree_hal_strela_ColumnAddressDef_offset_add(builder, &offset);
  iree_hal_strela_ColumnAddressDef_delta_add(builder, address.getDelta());
  iree_hal_strela_ColumnAddressDef_wg_x_params_add(builder, &x);
  iree_hal_strela_ColumnAddressDef_wg_y_params_add(builder, &y);
  return iree_hal_strela_ColumnAddressDef_end(builder);
}

static flatbuffers_vec_ref_t
createInputs(FlatbufferBuilder &builder, ArrayAttr columns) {
  SmallVector<iree_hal_strela_InputColumnDef_ref_t> refs;
  for (auto column : columns.getAsRange<strela::InpColAttr>()) {
    auto addressRef = createAddress(builder, column.getAddress());
    iree_hal_strela_ValueDef_t wordCount = immediate(column.getWords());
    iree_hal_strela_ValueDef_t wordStride = immediate(column.getStride());
    iree_hal_strela_InputColumnDef_start(builder);
    iree_hal_strela_InputColumnDef_address_add(builder, addressRef);
    iree_hal_strela_InputColumnDef_word_count_add(builder, &wordCount);
    iree_hal_strela_InputColumnDef_word_stride_add(builder, &wordStride);
    refs.push_back(iree_hal_strela_InputColumnDef_end(builder));
  }
  return builder.createOffsetVecDestructive(refs);
}

static flatbuffers_vec_ref_t
createOutputs(FlatbufferBuilder &builder, ArrayAttr columns) {
  SmallVector<iree_hal_strela_OutputColumnDef_ref_t> refs;
  for (auto column : columns.getAsRange<strela::OutColAttr>()) {
    auto addressRef = createAddress(builder, column.getAddress());
    iree_hal_strela_ValueDef_t wordCount = immediate(column.getWords());
    iree_hal_strela_OutputColumnDef_start(builder);
    iree_hal_strela_OutputColumnDef_address_add(builder, addressRef);
    iree_hal_strela_OutputColumnDef_word_count_add(builder, &wordCount);
    refs.push_back(iree_hal_strela_OutputColumnDef_end(builder));
  }
  return builder.createOffsetVecDestructive(refs);
}

static IREE::HAL::ExecutableTargetAttr
getStrelaExecutableTarget(MLIRContext *context, const EstelaOptions &options) {
  Builder b(context);
  auto desc = strela::TargetDescAttr::get(
    context, options.rows, options.columns, options.maxColumnWords
  );
  NamedAttribute config(b.getStringAttr(strela::targetDescKey), desc);
  return b.getAttr<IREE::HAL::ExecutableTargetAttr>(
    b.getStringAttr("strela"), b.getStringAttr("strela_bitstream"),
    b.getDictionaryAttr(config)
  );
}

//===--------------------------------------------------------------------===//
// StrelaTargetBackend
//===--------------------------------------------------------------------===//

StrelaTargetBackend::StrelaTargetBackend(const EstelaOptions &options)
  : options(options) {}

std::string
StrelaTargetBackend::getLegacyDefaultDeviceID() const {
  return "strela";
}

void
StrelaTargetBackend::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<strela::StrelaDialect>();
}

void
StrelaTargetBackend::buildTranslationPassPipeline(
  IREE::HAL::ExecutableTargetAttr executableTargetAttr,
  OpPassManager &passManager
) {
  passManager.nest<ModuleOp>().addNestedPass<func::FuncOp>(
    estela::createConvertLinalgToStrelaPass()
  );

  // Simplifies away {collapse,expand}_shape introduced by
  // createConvertLinalgToStrelaPass.
  // NOTE: maybe it is better to do it in the preprocessing and
  // global-optimization takes care of it?
  passManager.addPass(createFoldReshapeIntoInterfaceTensorPass());

  // Needs the workload ordinals MaterializeBindings erases. A second nest,
  // because adding to the first module pipeline would run before this.
  passManager.addPass(estela::createResolveWorkgroupCountPass());

  // Resolve the dispatch interface while the full executable source is still
  // around, and leave the answer in the IR as a strela.job.
  passManager.nest<ModuleOp>().addNestedPass<func::FuncOp>(
    estela::createMaterializeBindingsPass()
  );
}

void
StrelaTargetBackend::buildLinkingPassPipeline(OpPassManager &passManager) {
  passManager.addPass(estela::createLinkExecutablesPass());
}

void
StrelaTargetBackend::getDefaultExecutableTargets(
  MLIRContext *context,
  StringRef deviceID,
  DictionaryAttr deviceConfigAttr,
  SmallVectorImpl<IREE::HAL::ExecutableTargetAttr> &executableTargetAttrs
) const {
  executableTargetAttrs.push_back(getStrelaExecutableTarget(context, options));
}

LogicalResult
StrelaTargetBackend::serializeExecutable(
  const SerializationOptions &options,
  IREE::HAL::ExecutableVariantOp variantOp,
  OpBuilder &executableBuilder
) {
  mlir::ModuleOp innerModule = variantOp.getInnerModule();
  if (!innerModule) {
    return variantOp.emitOpError("has no inner module to serialize");
  }

  FlatbufferBuilder builder;
  iree_hal_strela_ExecutableDef_start_as_root(builder);

  SmallVector<IREE::HAL::ExecutableExportOp> exportOps =
      llvm::to_vector(variantOp.getExportOps());
  SmallVector<iree_hal_strela_ExportDef_ref_t> exportRefs(exportOps.size(), 0);
  for (auto exportOp : exportOps) {
    auto ordinalAttr = exportOp.getOrdinalAttr();
    if (!ordinalAttr) {
      return exportOp.emitOpError("is missing its ordinal");
    }
    auto funcOp = innerModule.lookupSymbol<func::FuncOp>(exportOp.getSymName());
    if (!funcOp) {
      return exportOp.emitOpError("has no function in the inner module");
    }
    FailureOr<strela::JobOp> jobOp = getJob(funcOp);
    if (failed(jobOp)) return failure();

    auto layoutAttr = exportOp.getLayoutAttr();
    auto nameRef = builder.createString(exportOp.getSymName());
    SmallVector<uint32_t> bitstream(jobOp->getBitstream());
    auto bitstreamRef =
        flatbuffers_uint32_vec_create(builder, bitstream.data(), bitstream.size());
    flatbuffers_vec_ref_t inputsRef = createInputs(builder, jobOp->getInputs());
    flatbuffers_vec_ref_t outputsRef = createOutputs(builder, jobOp->getOutputs());

    iree_hal_strela_ExportDef_start(builder);
    iree_hal_strela_ExportDef_name_add(builder, nameRef);
    iree_hal_strela_ExportDef_bitstream_add(builder, bitstreamRef);
    iree_hal_strela_ExportDef_inputs_add(builder, inputsRef);
    iree_hal_strela_ExportDef_outputs_add(builder, outputsRef);
    iree_hal_strela_ExportDef_constant_count_add(
      builder, static_cast<uint32_t>(layoutAttr.getConstants())
    );
    iree_hal_strela_ExportDef_binding_count_add(
      builder, static_cast<uint32_t>(layoutAttr.getBindings().size())
    );
    exportRefs[ordinalAttr.getInt()] = iree_hal_strela_ExportDef_end(builder);
  }
  auto exportsRef = builder.createOffsetVecDestructive(exportRefs);

  iree_hal_strela_ExecutableDef_exports_add(builder, exportsRef);
  iree_hal_strela_ExecutableDef_end_as_root(builder);

  DenseIntElementsAttr bufferAttr = builder.getHeaderPrefixedBufferAttr(
    executableBuilder.getContext(),
    /*magic=*/iree_hal_strela_ExecutableDef_file_identifier,
    /*version=*/0
  );

  // --iree-hal-dump-executable-binaries-to: exactly the bytes the runtime
  // loads, IREE header included.
  if (!options.dumpBinariesPath.empty()) {
    IREE::HAL::dumpDataToPath<char>(
      options.dumpBinariesPath, options.dumpBaseName, variantOp.getName(),
      ".str1", bufferAttr.getRawData()
    );
  }

  auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
    executableBuilder,
    variantOp.getLoc(),
    variantOp.getSymNameAttr(),
    variantOp.getTarget().getFormat(),
    bufferAttr
  );
  binaryOp.setMimeTypeAttr(
    executableBuilder.getStringAttr("application/x-flatbuffers")
  );
  return success();
}

//===--------------------------------------------------------------------===//
// StrelaTargetDevice
//===--------------------------------------------------------------------===//

// TODO: move the two comments with the commands to an appropriate test.

StrelaTargetDevice::StrelaTargetDevice(const EstelaOptions &options)
  : options(options) {}

IREE::HAL::DeviceTargetAttr
StrelaTargetDevice::getDefaultDeviceTarget(
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
  auto executableTargetAttr = getStrelaExecutableTarget(context, options);

  return IREE::HAL::DeviceTargetAttr::get(
    context,
    /*deviceID=*/ deviceID,
    /*configuration=*/ configAttr,
    /*executable_targets=*/ {executableTargetAttr}
  );
}

} // namespace mlir::estela
