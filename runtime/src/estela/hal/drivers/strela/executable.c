#include "iree/base/internal/flatcc/parsing.h"
#include "iree/hal/utils/executable_header.h"
#include "strela_executable_def_reader.h"
#include "strela_executable_def_verifier.h"

enum {
  // Columns per direction, as the schema lays them out.
  IREE_HAL_STRELA_COLUMN_COUNT = 4,
};

typedef struct {
  iree_hal_strela_ExportDef_table_t def;
  // The bitstream, loaded into the device's kernel memory.
  strela_kernel kernel;
} iree_hal_strela_export_t;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  strela_dev *dev;

  // A copy of the flatbuffer: the caller's data need not outlive creation, and
  // every def above points into it.
  uint8_t *flatbuffer_data;

  iree_host_size_t export_count;
  iree_hal_strela_export_t exports[];
} iree_hal_strela_executable_t;

static const iree_hal_executable_vtable_t iree_hal_strela_executable_vtable;

static iree_hal_strela_executable_t *
iree_hal_strela_executable_cast(iree_hal_executable_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_executable_vtable);
  return (iree_hal_strela_executable_t *)base_value;
}

static iree_status_t
iree_hal_strela_column_verify(
  iree_hal_strela_ColumnDef_table_t column,
  uint32_t binding_count,
  uint32_t constant_count
) {
  if (iree_hal_strela_ColumnDef_word_count_get(column) == 0) {
    return iree_ok_status();
  }
  if (iree_hal_strela_ColumnDef_binding_get(column) >= binding_count) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "column binding out of range");
  }
  if (iree_hal_strela_ColumnDef_byte_offset_get(column) % sizeof(strela_word) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "column offset is not word aligned");
  }
  flatbuffers_uint32_vec_t constants = iree_hal_strela_ColumnDef_offset_constants_get(column);
  size_t constants_len = flatbuffers_uint32_vec_len(constants);
  if (constants_len != 0 && constants_len != 2) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "a runtime offset takes two constants");
  }
  for (size_t i = 0; i < constants_len; ++i) {
    if (flatbuffers_uint32_vec_at(constants, i) >= constant_count) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "offset constant out of range");
    }
  }
  return iree_ok_status();
}

// Checks what the flatcc verifier cannot: that every export can be run as it
// stands, so dispatch need not check again.
static iree_status_t
iree_hal_strela_executable_flatbuffer_verify(iree_const_byte_span_t flatbuffer_data) {
  int verify_ret = iree_hal_strela_ExecutableDef_verify_as_root(
    flatbuffer_data.data, flatbuffer_data.data_length
  );
  if (verify_ret != flatcc_verify_ok) {
    return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT, "flatbuffer verification failed: %s",
      flatcc_verify_error_string(verify_ret)
    );
  }

  iree_hal_strela_ExecutableDef_table_t executable_def =
    iree_hal_strela_ExecutableDef_as_root(flatbuffer_data.data);
  iree_hal_strela_ExportDef_vec_t exports =
    iree_hal_strela_ExecutableDef_exports_get(executable_def);
  for (size_t i = 0; i < iree_hal_strela_ExportDef_vec_len(exports); ++i) {
    iree_hal_strela_ExportDef_table_t export_def = iree_hal_strela_ExportDef_vec_at(exports, i);
    if (!export_def || !iree_hal_strela_ExportDef_name_is_present(export_def)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "export %zu has no name", i);
    }
    if (flatbuffers_uint32_vec_len(iree_hal_strela_ExportDef_bitstream_get(export_def))
        != STRELA_KERNEL_SIZE) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "export %zu: bitstream is not %d words",
        i, STRELA_KERNEL_SIZE
      );
    }
    iree_hal_strela_ColumnDef_vec_t inputs = iree_hal_strela_ExportDef_inputs_get(export_def);
    iree_hal_strela_ColumnDef_vec_t outputs = iree_hal_strela_ExportDef_outputs_get(export_def);
    if (iree_hal_strela_ColumnDef_vec_len(inputs) != IREE_HAL_STRELA_COLUMN_COUNT
        || iree_hal_strela_ColumnDef_vec_len(outputs) != IREE_HAL_STRELA_COLUMN_COUNT) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "export %zu: expected %d input and output columns",
        i, IREE_HAL_STRELA_COLUMN_COUNT
      );
    }
    uint32_t binding_count = iree_hal_strela_ExportDef_binding_count_get(export_def);
    uint32_t constant_count = iree_hal_strela_ExportDef_constant_count_get(export_def);
    for (size_t c = 0; c < IREE_HAL_STRELA_COLUMN_COUNT; ++c) {
      IREE_RETURN_IF_ERROR(iree_hal_strela_column_verify(
        iree_hal_strela_ColumnDef_vec_at(inputs, c), binding_count, constant_count
      ));
      IREE_RETURN_IF_ERROR(iree_hal_strela_column_verify(
        iree_hal_strela_ColumnDef_vec_at(outputs, c), binding_count, constant_count
      ));
    }
  }
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_executable_infer_format(
  iree_const_byte_span_t executable_data,
  iree_host_size_t executable_format_capacity,
  char *executable_format,
  iree_host_size_t *out_inferred_size
) {
  const bool unsafe_infer_size = executable_data.data_length == 0;
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  IREE_RETURN_IF_ERROR(iree_hal_read_executable_flatbuffer_header(
    executable_data, unsafe_infer_size,
    iree_hal_strela_ExecutableDef_file_identifier, &flatbuffer_data
  ));
  IREE_RETURN_IF_ERROR(iree_hal_strela_executable_flatbuffer_verify(flatbuffer_data));

  iree_string_view_t format = IREE_SV("strela_bitstream");
  if (format.size >= executable_format_capacity) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "executable format buffer too small");
  }
  memcpy(executable_format, format.data, format.size + /*NUL*/ 1);
  *out_inferred_size = sizeof(iree_flatbuffer_file_header_t) + flatbuffer_data.data_length;
  return iree_ok_status();
}

static void
iree_hal_strela_executable_destroy(iree_hal_executable_t *base_executable);

iree_status_t iree_hal_strela_executable_create(
  strela_dev *dev,
  const iree_hal_executable_params_t* executable_params,
  iree_allocator_t host_allocator,
  iree_hal_executable_t **out_executable
) {
  TRACE_FUNC;
  iree_status_t status = iree_ok_status();
  iree_hal_strela_executable_t *executable = NULL;
  iree_const_byte_span_t flatbuffer_data = iree_const_byte_span_empty();
  iree_hal_strela_ExportDef_vec_t exports = NULL;
  iree_host_size_t export_count = 0;
  iree_host_size_t flatbuffer_offset = 0;

  *out_executable = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_hal_read_executable_flatbuffer_header(
      executable_params->executable_data, /*unsafe_infer_size=*/false,
      iree_hal_strela_ExecutableDef_file_identifier, &flatbuffer_data
    );
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_strela_executable_flatbuffer_verify(flatbuffer_data);
  }

  if (iree_status_is_ok(status)) {
    exports = iree_hal_strela_ExecutableDef_exports_get(
      iree_hal_strela_ExecutableDef_as_root(flatbuffer_data.data)
    );
    export_count = iree_hal_strela_ExportDef_vec_len(exports);
    // The copy keeps the flatbuffer alignment: it holds 64 bit fields, and an
    // unaligned 64 bit load faults on ARMv7.
    flatbuffer_offset = iree_host_align(
      sizeof *executable + export_count * sizeof executable->exports[0], 16
    );
    status = iree_allocator_malloc(
      host_allocator, flatbuffer_offset + flatbuffer_data.data_length,
      (void **)&executable
    );
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
      &iree_hal_strela_executable_vtable, &executable->resource
    );
    executable->host_allocator = host_allocator;
    executable->dev = dev;
    executable->export_count = export_count;
    executable->flatbuffer_data = (uint8_t *)executable + flatbuffer_offset;
    memcpy(executable->flatbuffer_data, flatbuffer_data.data, flatbuffer_data.data_length);
    exports = iree_hal_strela_ExecutableDef_exports_get(
      iree_hal_strela_ExecutableDef_as_root(executable->flatbuffer_data)
    );
    for (iree_host_size_t i = 0; i < export_count; ++i) {
      memset(&executable->exports[i].kernel, 0, sizeof executable->exports[i].kernel);
      executable->exports[i].def = iree_hal_strela_ExportDef_vec_at(exports, i);
    }
  }

  for (iree_host_size_t i = 0; iree_status_is_ok(status) && i < export_count; ++i) {
    iree_hal_strela_export_t *export = &executable->exports[i];
    flatbuffers_uint32_vec_t bitstream = iree_hal_strela_ExportDef_bitstream_get(export->def);
    export->kernel = strela_kernel_alloc(dev);
    strela_kernel_set(dev, export->kernel, bitstream);
    status = iree_hal_strela_status_from_dev(dev, "loading a STRELA bitstream");
  }

  if (iree_status_is_ok(status)) {
    *out_executable = (iree_hal_executable_t *)executable;
  } else if (executable) {
    iree_hal_strela_executable_destroy((iree_hal_executable_t *)executable);
  }

  return status;
}

static void
iree_hal_strela_executable_destroy(iree_hal_executable_t *base_executable) {
  TRACE_FUNC;
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;

  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    if (executable->exports[i].kernel.valid) {
      strela_kernel_free(executable->dev, executable->exports[i].kernel);
    }
  }
  iree_status_ignore(iree_hal_strela_status_from_dev(executable->dev, "freeing a STRELA kernel"));

  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t
iree_hal_strela_executable_function_count(iree_hal_executable_t *base_executable) {
  TRACE_FUNC;
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);
  return executable->export_count;
}

static iree_status_t
iree_hal_strela_executable_function_info(
  iree_hal_executable_t *base_executable,
  iree_hal_executable_function_t function,
  iree_hal_executable_function_info_t *out_info
) {
  TRACE_FUNC;
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);
  if (function.value >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "function %" PRIu64 " out of range", function.value);
  }
  iree_hal_strela_ExportDef_table_t def = executable->exports[function.value].def;
  flatbuffers_string_t name = iree_hal_strela_ExportDef_name_get(def);

  memset(out_info, 0, sizeof *out_info);
  out_info->name = iree_make_string_view(name, flatbuffers_string_len(name));
  out_info->constant_count = (uint16_t)iree_hal_strela_ExportDef_constant_count_get(def);
  out_info->binding_count = (uint16_t)iree_hal_strela_ExportDef_binding_count_get(def);
  out_info->workgroup_size[0] = 1;
  out_info->workgroup_size[1] = 1;
  out_info->workgroup_size[2] = 1;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_executable_function_parameters(
  iree_hal_executable_t *base_executable,
  iree_hal_executable_function_t function,
  iree_host_size_t capacity,
  iree_hal_executable_function_parameter_t* out_parameters
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_executable_lookup_function_by_name(
  iree_hal_executable_t *base_executable,
  iree_string_view_t name,
  iree_hal_executable_function_t *out_function
) {
  TRACE_FUNC;
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);

  for (iree_host_size_t i = 0; i < executable->export_count; ++i) {
    flatbuffers_string_t export_name =
      iree_hal_strela_ExportDef_name_get(executable->exports[i].def);
    if (iree_string_view_equal(
          name, iree_make_string_view(export_name, flatbuffers_string_len(export_name)))) {
      out_function->value = i;
      return iree_ok_status();
    }
  }
  return iree_make_status(
    IREE_STATUS_NOT_FOUND, "no STRELA function named '%.*s'", (int)name.size, name.data
  );
}

static iree_status_t
iree_hal_strela_executable_lookup_global_by_name(
  iree_hal_executable_t *base_executable,
  iree_string_view_t name,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_buffer_t **out_buffer
) {
  TRACE_FUNC;
  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_executable_vtable_t
iree_hal_strela_executable_vtable = {
  .destroy                 = iree_hal_strela_executable_destroy,
  .function_count          = iree_hal_strela_executable_function_count,
  .function_info           = iree_hal_strela_executable_function_info,
  .function_parameters     = iree_hal_strela_executable_function_parameters,
  .lookup_function_by_name = iree_hal_strela_executable_lookup_function_by_name,
  .lookup_global_by_name   = iree_hal_strela_executable_lookup_global_by_name,
};
