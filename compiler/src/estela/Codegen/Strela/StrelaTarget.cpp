#include "estela/Codegen/Strela/StrelaTarget.h"

#include <array>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "estela/Dialect/Strela/IR/StrelaDialect.h"
#include "estela/Dialect/Strela/IR/StrelaOps.h"
#include "estela/Dialect/Strela/Conversion/Passes.h"
#include "estela/Transforms/Linalg/Passes.h"
#include "estela/Dialect/Strela/Transforms/Passes.h"
#include "estela/Utils/Bitstreams.h"

#include "strela_executable_def_builder.h"

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

// The DMA programme of one CGRA column, before it is written out.
struct StrelaColumn {
  uint32_t binding = 0;
  uint64_t byteOffset = 0;
  SmallVector<uint32_t, 2> offsetConstants;
  uint32_t wordCount = 0;
  uint32_t stride = 0;
};

// The size register leaves 16 bits for the byte count (DESIGN.md H8), and the
// driver rejects anything larger.
static constexpr int64_t strelaMaxColumnWords = 0xFFFF / sizeof(uint32_t);

template <typename BindingOp>
static FailureOr<StrelaColumn> getColumn(BindingOp op, TensorType type,
                                         uint32_t stride) {
  if (!type.hasStaticShape()) {
    return op.emitOpError("STRELA needs a static shape, got ") << type;
  }
  int64_t words = type.getNumElements();
  if (words > strelaMaxColumnWords) {
    return op.emitOpError("moves ") << words << " words, more than the "
           << strelaMaxColumnWords << " a STRELA column can in one job";
  }

  StrelaColumn column;
  column.binding = static_cast<uint32_t>(op.getBinding());
  column.byteOffset = op.getByteOffset().value_or(0);
  if (std::optional<ArrayRef<int64_t>> offsets = op.getOffsets()) {
    for (int64_t ordinal : *offsets) {
      column.offsetConstants.push_back(static_cast<uint32_t>(ordinal));
    }
  }
  column.wordCount = static_cast<uint32_t>(words);
  column.stride = stride;
  return column;
}

// The single STRELA operation of an export and the columns it uses.
struct StrelaExport {
  const StrelaKernel *kernel = nullptr;
  std::array<StrelaColumn, 4> inputs;
  std::array<StrelaColumn, 4> outputs;
};

static FailureOr<StrelaExport> getExport(func::FuncOp funcOp) {
  SmallVector<Operation *> kernelOps;
  funcOp.walk([&](Operation *op) {
    if (isa<strela::AddOp, strela::ReluOp>(op)) kernelOps.push_back(op);
  });
  if (kernelOps.size() != 1) {
    return funcOp.emitOpError("expected exactly one STRELA operation, found ")
           << kernelOps.size();
  }
  Operation *kernelOp = kernelOps.front();

  StrelaExport result;
  result.kernel = isa<strela::AddOp>(kernelOp) ? &add_kernel : &relu_kernel;
  if (kernelOp->getNumOperands() != result.kernel->inputCount) {
    return kernelOp->emitOpError("operand count does not match its kernel");
  }

  for (auto [i, operand] : llvm::enumerate(kernelOp->getOperands())) {
    auto loadOp = operand.getDefiningOp<strela::BindingLoadOp>();
    if (!loadOp) {
      return kernelOp->emitOpError("operand ") << i
             << " is not read straight from a binding";
    }
    FailureOr<StrelaColumn> column =
        getColumn(loadOp, cast<TensorType>(loadOp.getType()), /*stride=*/1);
    if (failed(column)) return failure();
    result.inputs[result.kernel->inputColumns[i]] = *column;
  }

  Value value = kernelOp->getResult(0);
  if (!value.hasOneUse()) {
    return kernelOp->emitOpError("result must be stored to exactly one binding");
  }
  auto storeOp = dyn_cast<strela::BindingStoreOp>(*value.getUsers().begin());
  if (!storeOp) {
    return kernelOp->emitOpError("result is not stored straight to a binding");
  }
  FailureOr<StrelaColumn> column =
      getColumn(storeOp, cast<TensorType>(value.getType()), /*stride=*/0);
  if (failed(column)) return failure();
  result.outputs[result.kernel->outputColumn] = *column;

  return result;
}

static iree_hal_strela_ColumnDef_vec_ref_t
createColumns(FlatbufferBuilder &builder, ArrayRef<StrelaColumn> columns) {
  SmallVector<iree_hal_strela_ColumnDef_ref_t> refs;
  for (const StrelaColumn &column : columns) {
    auto constantsRef = flatbuffers_uint32_vec_create(
        builder, column.offsetConstants.data(), column.offsetConstants.size());
    iree_hal_strela_ColumnDef_start(builder);
    iree_hal_strela_ColumnDef_binding_add(builder, column.binding);
    iree_hal_strela_ColumnDef_byte_offset_add(builder, column.byteOffset);
    iree_hal_strela_ColumnDef_offset_constants_add(builder, constantsRef);
    iree_hal_strela_ColumnDef_word_count_add(builder, column.wordCount);
    iree_hal_strela_ColumnDef_stride_add(builder, column.stride);
    refs.push_back(iree_hal_strela_ColumnDef_end(builder));
  }
  return builder.createOffsetVecDestructive(refs);
}

LogicalResult StrelaTargetBackend::serializeExecutable(
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
    FailureOr<StrelaExport> strelaExport = getExport(funcOp);
    if (failed(strelaExport)) return failure();

    auto layoutAttr = exportOp.getLayoutAttr();
    auto nameRef = builder.createString(exportOp.getSymName());
    const auto &bitstream = *strelaExport->kernel->bitstream;
    auto bitstreamRef =
        flatbuffers_uint32_vec_create(builder, bitstream.data(), bitstream.size());
    auto inputsRef = createColumns(builder, strelaExport->inputs);
    auto outputsRef = createColumns(builder, strelaExport->outputs);

    iree_hal_strela_ExportDef_start(builder);
    iree_hal_strela_ExportDef_name_add(builder, nameRef);
    iree_hal_strela_ExportDef_bitstream_add(builder, bitstreamRef);
    iree_hal_strela_ExportDef_inputs_add(builder, inputsRef);
    iree_hal_strela_ExportDef_outputs_add(builder, outputsRef);
    iree_hal_strela_ExportDef_constant_count_add(
        builder, static_cast<uint32_t>(layoutAttr.getConstants()));
    iree_hal_strela_ExportDef_binding_count_add(
        builder, static_cast<uint32_t>(layoutAttr.getBindings().size()));
    exportRefs[ordinalAttr.getInt()] = iree_hal_strela_ExportDef_end(builder);
  }
  auto exportsRef = builder.createOffsetVecDestructive(exportRefs);

  iree_hal_strela_ExecutableDef_exports_add(builder, exportsRef);
  iree_hal_strela_ExecutableDef_end_as_root(builder);

  auto binaryOp = IREE::HAL::ExecutableBinaryOp::create(
    executableBuilder,
    variantOp.getLoc(),
    variantOp.getSymNameAttr(),
    variantOp.getTarget().getFormat(),
    builder.getHeaderPrefixedBufferAttr(
      executableBuilder.getContext(),
      /*magic=*/iree_hal_strela_ExecutableDef_file_identifier,
      /*version=*/0
    )
  );
  binaryOp.setMimeTypeAttr(
    executableBuilder.getStringAttr("application/x-flatbuffers")
  );
  return success();
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
