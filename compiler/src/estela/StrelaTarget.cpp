#include "estela/StrelaTarget.h"

#include <array>

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"

#include "estela/StrelaIR/StrelaDialect.h"
#include "estela/StrelaIR/StrelaOps.h"
#include "estela/StrelaIR/StrelaAttrs.h"
#include "estela/Transforms/Passes.h"
#include "estela/Utils/Bitstreams.h"
#include "estela/Utils/Finders.h"

#include "strela_executable_def_builder.h"

using namespace mlir::iree_compiler;

namespace mlir::estela {

// A ValueDef before it is written out: an immediate, or the ordinal of the push
// constant holding the value.
struct StrelaValue {
  uint32_t value = 0;
  bool isConstant = false;
};

static StrelaValue
immediate(int64_t value) {
  return {static_cast<uint32_t>(value), false};
}

// min(workgroup_id[d] * step, limit) bytes; the default adds nothing.
struct StrelaTerm {
  StrelaValue step;
  StrelaValue limit{UINT32_MAX, false};
};

// The DMA programme of one CGRA column, before it is written out. A word count
// of 0 leaves the column unused; outputs have no word stride.
struct StrelaColumn {
  uint32_t binding = 0;
  StrelaValue offset;
  std::array<StrelaTerm, 3> workgroup;
  StrelaValue wordCount;
  StrelaValue wordStride{1, false};
};

// Where a binding op reads or writes, for workgroup (0, 0, 0).
template <typename BindingOp> static StrelaColumn
getColumn(BindingOp op) {
  StrelaColumn column;
  column.binding = static_cast<uint32_t>(op.getBinding());
  if (std::optional<uint64_t> ordinal = op.getOffsetOrdinal()) {
    column.offset = {static_cast<uint32_t>(*ordinal), true};
  } else {
    column.offset = immediate(op.getByteOffset().value_or(0));
  }
  return column;
}

// `column` moved `bytes` further into its binding, which only an offset known
// at compile time can be.
static FailureOr<StrelaColumn>
offsetColumn(Operation *op, StrelaColumn column, int64_t bytes) {
  if (bytes == 0) return column;
  if (column.offset.isConstant) {
    return op->emitOpError(
      "needs a byte offset known at compile time; retry with "
      "--iree-scheduling-optimize-bindings=false"
    );
  }
  int64_t offset = column.offset.value + bytes;
  if (offset > UINT32_MAX) {
    return op->emitOpError("byte offset does not fit in 32 bits");
  }
  column.offset = immediate(offset);
  return column;
}

// The single STRELA operation of an export and the columns it uses.
struct StrelaExport {
  explicit StrelaExport(unsigned words, unsigned columns)
    : bitstream(words), inputs(columns), outputs(columns) {}
  SmallVector<uint32_t> bitstream;
  SmallVector<StrelaColumn> inputs;
  SmallVector<StrelaColumn> outputs;
};

// result[0, n, f] = sum_c lhs[0, n, c] * rhs[0, c, f], one job per workgroup
// (x, y) of the (ceildiv(F, 3), N, 1) grid. The job computes the features
// f0 .. f0 + 2, f0 = min(3x, F - 3), of row n = y: the row enters column 0, the
// rhs columns f0 .. f0 + 2 (stride F) enter columns 1-3, and their dot products
// with the row leave by output columns 1-3. The last job overlaps the one
// before it when 3 does not divide F, rewriting the same values.
static FailureOr<StrelaExport>
getMatvecExport(
  strela::MatvecOp matvecOp,
  strela::BindingLoadOp lhsOp,
  strela::BindingLoadOp rhsOp,
  strela::BindingStoreOp resultOp,
  strela::TargetDescAttr targetDesc
) {
  auto rhsType = cast<RankedTensorType>(matvecOp.getRhs().getType());
  if (!rhsType.hasStaticShape()) {
    return matvecOp.emitOpError("needs a static rhs, got ") << rhsType;
  }
  int64_t c = rhsType.getDimSize(1);
  int64_t f = rhsType.getDimSize(2);
  if (f < strelaMatvecFeaturesPerJob) {
    return matvecOp.emitOpError("needs F of at least ")
           << strelaMatvecFeaturesPerJob << ", got " << f;
  }
  // A weight column reads C words F apart, and the input size register holds
  // that whole span, C * F words, as a 16 bit byte count (DESIGN.md H8).
  if (c * f > targetDesc.getMaxColumnWords()) {
    return matvecOp.emitOpError("needs C * F of at most ")
           << targetDesc.getMaxColumnWords() << " words, the span a strided STRELA "
           << "column can cover, got " << c << " * " << f;
  }

  constexpr int64_t word = sizeof(uint32_t);
  StrelaTerm featureTerm{
    immediate(word * strelaMatvecFeaturesPerJob),
    immediate(word * (f - strelaMatvecFeaturesPerJob)),
  };

  StrelaExport result(strelaBitstreamWords, strelaKernelColumns);
  auto bitstream = matvecBitstream(static_cast<uint16_t>(c));
  result.bitstream.assign(bitstream.begin(), bitstream.end());

  StrelaColumn row = getColumn(lhsOp);
  row.wordCount = immediate(c);
  row.workgroup[1].step = immediate(word * c);
  result.inputs[0] = row;

  for (int64_t k = 0; k < strelaMatvecFeaturesPerJob; ++k) {
    FailureOr<StrelaColumn> weights = offsetColumn(rhsOp, getColumn(rhsOp), word * k);
    if (failed(weights)) return failure();
    weights->wordCount = immediate(c);
    weights->wordStride = immediate(f);
    weights->workgroup[0] = featureTerm;
    result.inputs[1 + k] = *weights;

    FailureOr<StrelaColumn> feature =
      offsetColumn(resultOp, getColumn(resultOp), word * k);
    if (failed(feature)) return failure();
    feature->wordCount = immediate(1);
    feature->workgroup[0] = featureTerm;
    feature->workgroup[1].step = immediate(word * f);
    result.outputs[1 + k] = *feature;
  }
  return result;
}

static FailureOr<StrelaExport>
getExport(func::FuncOp funcOp, strela::TargetDescAttr targetDesc) {
  if (targetDesc.getRows() != strelaKernelRows
      || targetDesc.getColumns() != strelaKernelColumns) {
    return funcOp.emitOpError("the hand-written STRELA kernels need a ")
      << strelaKernelRows << "x" << strelaKernelColumns
      << " fabric, got " << targetDesc.getRows() << "x" << targetDesc.getColumns()
      << "; other fabrics need the mapper";
  }
  SmallVector<Operation *> kernelOps;
  funcOp.walk([&kernelOps](Operation *op) {
    if (isa<strela::MatvecOp>(op)) {
      kernelOps.push_back(op);
    }
  });
  if (kernelOps.size() != 1) {
    return funcOp.emitOpError("expected exactly one STRELA operation, found ")
           << kernelOps.size();
  }
  Operation *kernelOp = kernelOps.front();

  SmallVector<strela::BindingLoadOp> loadOps;
  for (auto [i, operand] : llvm::enumerate(kernelOp->getOperands())) {
    auto loadOp = operand.getDefiningOp<strela::BindingLoadOp>();
    if (!loadOp) {
      return kernelOp->emitOpError("operand ") << i
             << " is not read straight from a binding";
    }
    loadOps.push_back(loadOp);
  }
  Value value = kernelOp->getResult(0);
  if (!value.hasOneUse()) {
    return kernelOp->emitOpError("result must be stored to exactly one binding");
  }
  auto storeOp = dyn_cast<strela::BindingStoreOp>(*value.getUsers().begin());
  if (!storeOp) {
    return kernelOp->emitOpError("result is not stored straight to a binding");
  }

  auto matvecOp = cast<strela::MatvecOp>(kernelOp);
  return getMatvecExport(matvecOp, loadOps[0], loadOps[1], storeOp, targetDesc);
}

static iree_hal_strela_ValueDef_t
toValueDef(StrelaValue value) {
  iree_hal_strela_ValueDef_t def{};
  def.value = value.value;
  def.is_constant = value.isConstant;
  return def;
}

static iree_hal_strela_AddressDef_ref_t
createAddress(FlatbufferBuilder &builder, const StrelaColumn &column) {
  iree_hal_strela_ValueDef_t offset = toValueDef(column.offset);
  std::array<iree_hal_strela_WorkgroupTermDef_t, 3> terms{};
  for (size_t d = 0; d < terms.size(); ++d) {
    terms[d].step = toValueDef(column.workgroup[d].step);
    terms[d].limit = toValueDef(column.workgroup[d].limit);
  }
  iree_hal_strela_AddressDef_start(builder);
  iree_hal_strela_AddressDef_binding_add(builder, column.binding);
  iree_hal_strela_AddressDef_offset_add(builder, &offset);
  iree_hal_strela_AddressDef_workgroup_x_add(builder, &terms[0]);
  iree_hal_strela_AddressDef_workgroup_y_add(builder, &terms[1]);
  iree_hal_strela_AddressDef_workgroup_z_add(builder, &terms[2]);
  return iree_hal_strela_AddressDef_end(builder);
}

static flatbuffers_vec_ref_t
createInputs(FlatbufferBuilder &builder, ArrayRef<StrelaColumn> columns) {
  SmallVector<iree_hal_strela_InputColumnDef_ref_t> refs;
  for (const StrelaColumn &column : columns) {
    auto addressRef = createAddress(builder, column);
    iree_hal_strela_ValueDef_t wordCount = toValueDef(column.wordCount);
    iree_hal_strela_ValueDef_t wordStride = toValueDef(column.wordStride);
    iree_hal_strela_InputColumnDef_start(builder);
    iree_hal_strela_InputColumnDef_address_add(builder, addressRef);
    iree_hal_strela_InputColumnDef_word_count_add(builder, &wordCount);
    iree_hal_strela_InputColumnDef_word_stride_add(builder, &wordStride);
    refs.push_back(iree_hal_strela_InputColumnDef_end(builder));
  }
  return builder.createOffsetVecDestructive(refs);
}

static flatbuffers_vec_ref_t
createOutputs(FlatbufferBuilder &builder, ArrayRef<StrelaColumn> columns) {
  SmallVector<iree_hal_strela_OutputColumnDef_ref_t> refs;
  for (const StrelaColumn &column : columns) {
    auto addressRef = createAddress(builder, column);
    iree_hal_strela_ValueDef_t wordCount = toValueDef(column.wordCount);
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
  NamedAttribute config(b.getStringAttr(strelaTargetDescKey), desc);
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

  // Needs the workload ordinals MaterializeBindings erases. A second nest,
  // because adding to the first module pipeline would run before this.
  passManager.addPass(estela::createResolveWorkgroupCountPass());

  // Resolve the dispatch interface while the full executable source is still
  // around, and leave the answer in the IR as strela.binding_load/store.
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

  strela::TargetDescAttr targetDesc = findStrelaTargetDesc(variantOp);
  if (!targetDesc) {
    return variantOp.emitOpError("has no target description");
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
    FailureOr<StrelaExport> strelaExport = getExport(funcOp, targetDesc);
    if (failed(strelaExport)) return failure();

    auto layoutAttr = exportOp.getLayoutAttr();
    auto nameRef = builder.createString(exportOp.getSymName());
    const auto &bitstream = strelaExport->bitstream;
    auto bitstreamRef =
        flatbuffers_uint32_vec_create(builder, bitstream.data(), bitstream.size());
    flatbuffers_vec_ref_t inputsRef = createInputs(builder, strelaExport->inputs);
    flatbuffers_vec_ref_t outputsRef = createOutputs(builder, strelaExport->outputs);

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
