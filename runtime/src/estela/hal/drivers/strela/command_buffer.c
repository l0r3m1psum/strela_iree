// The command buffer that runs commands as they are recorded. The device
// records into an iree_hal_deferred_command_buffer and replays it onto one of
// these in queue_execute, so everything here happens synchronously, in order,
// on the calling thread.
typedef struct {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  strela_dev *dev;
} iree_hal_strela_command_buffer_t;

static const iree_hal_command_buffer_vtable_t iree_hal_strela_command_buffer_vtable;

static iree_hal_strela_command_buffer_t *
iree_hal_strela_command_buffer_cast(iree_hal_command_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_command_buffer_vtable);
  return (iree_hal_strela_command_buffer_t *)base_value;
}

static iree_status_t
iree_hal_strela_command_buffer_create(
  strela_dev *dev,
  iree_hal_allocator_t* device_allocator,
  iree_hal_command_buffer_mode_t mode,
  iree_hal_command_category_t command_categories,
  iree_hal_queue_affinity_t queue_affinity,
  iree_host_size_t binding_capacity,
  iree_allocator_t host_allocator,
  iree_hal_command_buffer_t **out_command_buffer
) {
  TRACE_FUNC;
  iree_status_t status = iree_ok_status();
  iree_hal_strela_command_buffer_t *command_buffer = NULL;
  iree_hal_command_buffer_t *command_buffer_base = NULL;

  if (iree_status_is_ok(status)) {
    iree_host_size_t command_buffer_validation_state_size
      = iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
    status = iree_allocator_malloc(
      host_allocator,
      sizeof *command_buffer + command_buffer_validation_state_size,
      (void **)&command_buffer
    );
  }

  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
      device_allocator,
      mode,
      command_categories,
      queue_affinity,
      binding_capacity,
      (uint8_t *)command_buffer + sizeof *command_buffer,
      &iree_hal_strela_command_buffer_vtable,
      &command_buffer->base
    );
    command_buffer->host_allocator = host_allocator;
    command_buffer->dev = dev;
    command_buffer_base = &command_buffer->base;
  }

  *out_command_buffer = command_buffer_base;
  return status;
}

static void
iree_hal_strela_command_buffer_destroy(iree_hal_command_buffer_t *base_command_buffer) {
  TRACE_FUNC;
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;

  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t
iree_hal_strela_command_buffer_begin(iree_hal_command_buffer_t *base_command_buffer) {
  TRACE_FUNC;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_end(iree_hal_command_buffer_t *base_command_buffer) {
  TRACE_FUNC;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_begin_debug_group(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_string_view_t label,
  iree_hal_label_color_t label_color,
  const iree_hal_label_location_t *location
) {
  TRACE_FUNC;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_end_debug_group(iree_hal_command_buffer_t* base_command_buffer) {
  TRACE_FUNC;
  return iree_ok_status();
}

// Every command finishes before the next one starts.
static iree_status_t
iree_hal_strela_command_buffer_execution_barrier(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_execution_stage_t source_stage_mask,
  iree_hal_execution_stage_t target_stage_mask,
  iree_hal_execution_barrier_flags_t flags,
  iree_host_size_t memory_barrier_count,
  const iree_hal_memory_barrier_t *memory_barriers,
  iree_host_size_t buffer_barrier_count,
  const iree_hal_buffer_barrier_t *buffer_barriers
) {
  TRACE_FUNC;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_signal_event(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_event_t *event,
  iree_hal_execution_stage_t source_stage_mask
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_reset_event(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_event_t *event,
  iree_hal_execution_stage_t source_stage_mask
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_wait_events(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_host_size_t event_count,
  const iree_hal_event_t **events,
  iree_hal_execution_stage_t source_stage_mask,
  iree_hal_execution_stage_t target_stage_mask,
  iree_host_size_t memory_barrier_count,
  const iree_hal_memory_barrier_t *memory_barriers,
  iree_host_size_t buffer_barrier_count,
  const iree_hal_buffer_barrier_t *buffer_barriers
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

// A hint; nothing to do. I guess this is like posix_madvise(3).
static iree_status_t
iree_hal_strela_command_buffer_advise_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t buffer_ref,
  iree_hal_memory_advise_flags_t flags,
  uint64_t arg0,
  uint64_t arg1
) {
  TRACE_FUNC;
  return iree_ok_status();
}

// Transfers run on the CPU through mappings; the map helpers flush what they
// write, since STRELA memory is not coherent.
static iree_status_t
iree_hal_strela_command_buffer_fill_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t target_ref,
  const void *pattern,
  iree_host_size_t pattern_length,
  iree_hal_fill_flags_t flags
) {
  TRACE_FUNC;
  return iree_hal_buffer_map_fill(
    target_ref.buffer, target_ref.offset, target_ref.length,
    pattern, pattern_length
  );
}

static iree_status_t
iree_hal_strela_command_buffer_update_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  const void *source_buffer,
  iree_host_size_t source_offset,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_update_flags_t flags
) {
  TRACE_FUNC;
  return iree_hal_buffer_map_write(
    target_ref.buffer, target_ref.offset,
    (const uint8_t *)source_buffer + source_offset, target_ref.length
  );
}

static iree_status_t
iree_hal_strela_command_buffer_copy_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t source_ref,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_copy_flags_t flags
) {
  TRACE_FUNC;
  return iree_hal_buffer_map_copy(
    source_ref.buffer, source_ref.offset,
    target_ref.buffer, target_ref.offset, target_ref.length
  );
}

static iree_status_t
iree_hal_strela_command_buffer_collective(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_channel_t *channel,
  iree_hal_collective_op_t op,
  uint32_t param,
  iree_hal_buffer_ref_t send_ref,
  iree_hal_buffer_ref_t recv_ref,
  iree_device_size_t element_count
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

// Where one CGRA column reads or writes, resolved for a dispatch. Workgroup id
// (x, y, z) moves its first word sum(min(id[d] * step[d], limit[d])) bytes past
// `offset`.
typedef struct {
  // NULL for an unused column.
  iree_hal_strela_buffer_t *buffer;
  // Byte offset of workgroup (0, 0, 0)'s first word in the allocated buffer.
  iree_device_size_t offset;
  // Bytes from there to the last word any workgroup touches.
  iree_device_size_t length;
  uint64_t step[3];
  uint64_t limit[3];
  size_t word_count;
  size_t word_stride;
} iree_hal_strela_column_t;

static iree_status_t
iree_hal_strela_value_get(
  iree_hal_strela_ValueDef_struct_t def,
  iree_const_byte_span_t constants,
  uint32_t *out_value
) {
  uint32_t value = iree_hal_strela_ValueDef_value_get(def);
  if (iree_hal_strela_ValueDef_is_constant_get(def)) {
    if (value >= constants.data_length / sizeof(uint32_t)) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "push constant %u is missing", value);
    }
    memcpy(&value, constants.data + value * sizeof(uint32_t), sizeof value);
  }
  *out_value = value;
  return iree_ok_status();
}

// Bytes workgroup `id` moves the column's first word.
static uint64_t
iree_hal_strela_column_shift(const iree_hal_strela_column_t *column, const uint32_t id[3]) {
  uint64_t shift = 0;
  for (int d = 0; d < 3; ++d) {
    shift += iree_min((uint64_t)id[d] * column->step[d], (uint64_t)column->limit[d]);
  }
  return shift;
}

// `word_stride` is NULL for outputs. Every workgroup count must be at least 1.
static iree_status_t
iree_hal_strela_column_resolve(
  iree_hal_strela_AddressDef_table_t address,
  iree_hal_strela_ValueDef_struct_t word_count,
  iree_hal_strela_ValueDef_struct_t word_stride,
  const uint32_t workgroup_count[3],
  iree_const_byte_span_t constants,
  iree_hal_buffer_ref_list_t bindings,
  iree_hal_strela_column_t *out_column
) {
  memset(out_column, 0, sizeof *out_column);
  uint32_t value = 0;
  IREE_RETURN_IF_ERROR(iree_hal_strela_value_get(word_count, constants, &value));
  out_column->word_count = value;
  value = 1;
  if (word_stride) {
    IREE_RETURN_IF_ERROR(iree_hal_strela_value_get(word_stride, constants, &value));
  }
  out_column->word_stride = value;
  if (out_column->word_count == 0) {
    return iree_ok_status();
  }

  uint32_t binding = iree_hal_strela_AddressDef_binding_get(address);
  if (binding >= bindings.count || !bindings.values[binding].buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding %u is missing", binding);
  }
  iree_hal_buffer_ref_t ref = bindings.values[binding];

  uint32_t byte_offset = 0;
  IREE_RETURN_IF_ERROR(iree_hal_strela_value_get(
    iree_hal_strela_AddressDef_offset_get(address), constants, &byte_offset
  ));
  iree_hal_strela_WorkgroupTermDef_struct_t terms[3] = {
    iree_hal_strela_AddressDef_workgroup_x_get(address),
    iree_hal_strela_AddressDef_workgroup_y_get(address),
    iree_hal_strela_AddressDef_workgroup_z_get(address),
  };
  uint32_t last_id[3];
  for (int d = 0; d < 3; ++d) {
    IREE_RETURN_IF_ERROR(iree_hal_strela_value_get(
      iree_hal_strela_WorkgroupTermDef_step_get(terms[d]), constants, &value
    ));
    out_column->step[d] = value;
    IREE_RETURN_IF_ERROR(iree_hal_strela_value_get(
      iree_hal_strela_WorkgroupTermDef_limit_get(terms[d]), constants, &value
    ));
    out_column->limit[d] = value;
    last_id[d] = workgroup_count[d] - 1;
  }

  // Every term grows with the id, so the last workgroup reaches farthest. The
  // driver sizes a stride of 0 as a stride of 1 (DESIGN.md D8).
  uint64_t stride = out_column->word_stride ? out_column->word_stride : 1;
  uint64_t span_words = (uint64_t)(out_column->word_count - 1) * stride + 1;
  out_column->length =
    iree_hal_strela_column_shift(out_column, last_id) + span_words * sizeof(strela_word);

  iree_device_size_t length = ref.length == IREE_HAL_WHOLE_BUFFER
    ? iree_hal_buffer_byte_length(ref.buffer) - ref.offset : ref.length;
  if (byte_offset + out_column->length > length) {
    return iree_make_status(
      IREE_STATUS_OUT_OF_RANGE, "binding %u: column runs past the bound range", binding
    );
  }

  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_resolve(
    ref.buffer, ref.offset + byte_offset, &out_column->offset
  );
  if (!buffer) {
    return iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "binding %u is not STRELA memory; the CGRA can only reach buffers from "
      "the STRELA allocator (see --device_lead_allocator and the device topology)",
      binding
    );
  }
  out_column->buffer = buffer;
  return iree_ok_status();
}

// What libstrela takes for workgroup `id`: words from the base of STRELA memory.
static iree_status_t
iree_hal_strela_column_place(
  const iree_hal_strela_column_t *column,
  const uint32_t id[3],
  size_t *out_words_from_base
) {
  *out_words_from_base = 0;
  if (!column->buffer) return iree_ok_status();
  uint64_t offset = column->offset + iree_hal_strela_column_shift(column, id);
  if (offset % sizeof(strela_word) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "column is not word aligned");
  }
  *out_words_from_base = column->buffer->s_buf.offset_words_from_base + offset / sizeof(strela_word);
  return iree_ok_status();
}

// Runs one job to completion.
static iree_status_t
iree_hal_strela_job_run(strela_dev *dev, strela_kernel kernel, strela_conf *conf) {
  strela_job job = strela_job_config(dev, kernel, conf);
  strela_job_submit(dev, job);
  // The driver's timeout bounds this: a job that hangs is cancelled.
  strela_job_wait(dev, job, -1);
  iree_status_t status = iree_hal_strela_status_from_dev(dev, "running a STRELA job");

  if (iree_status_is_ok(status)) {
    strela_job_state state = strela_job_poll(dev, job);
    status = iree_hal_strela_status_from_dev(dev, "polling a STRELA job");
    if (iree_status_is_ok(status) && state != STRELA_JOB_OK) {
      status = iree_make_status(
        IREE_STATUS_ABORTED,
        "STRELA job %s (the driver cancels jobs that do not finish in time)",
        state == STRELA_JOB_PENDING ? "did not finish" : "failed"
      );
    }
  }

  if (job.valid) {
    strela_job_free(dev, job);
    status = iree_status_join(status, iree_hal_strela_status_from_dev(dev, "freeing a STRELA job"));
  }
  return status;
}

static iree_status_t
iree_hal_strela_job_run_workgroup(
  strela_dev *dev,
  strela_kernel kernel,
  const iree_hal_strela_column_t inputs[IREE_HAL_STRELA_COLUMN_COUNT],
  const iree_hal_strela_column_t outputs[IREE_HAL_STRELA_COLUMN_COUNT],
  const uint32_t id[3]
) {
  size_t in[IREE_HAL_STRELA_COLUMN_COUNT], out[IREE_HAL_STRELA_COLUMN_COUNT];
  for (size_t c = 0; c < IREE_HAL_STRELA_COLUMN_COUNT; ++c) {
    IREE_RETURN_IF_ERROR(iree_hal_strela_column_place(&inputs[c], id, &in[c]));
    IREE_RETURN_IF_ERROR(iree_hal_strela_column_place(&outputs[c], id, &out[c]));
  }
  strela_conf conf = {
    .inp0_offset = in[0], .inp0_count = inputs[0].word_count, .inp0_stride = inputs[0].word_stride,
    .inp1_offset = in[1], .inp1_count = inputs[1].word_count, .inp1_stride = inputs[1].word_stride,
    .inp2_offset = in[2], .inp2_count = inputs[2].word_count, .inp2_stride = inputs[2].word_stride,
    .inp3_offset = in[3], .inp3_count = inputs[3].word_count, .inp3_stride = inputs[3].word_stride,
    .out0_offset = out[0], .out0_count = outputs[0].word_count,
    .out1_offset = out[1], .out1_count = outputs[1].word_count,
    .out2_offset = out[2], .out2_count = outputs[2].word_count,
    .out3_offset = out[3], .out3_count = outputs[3].word_count,
  };
  return iree_hal_strela_job_run(dev, kernel, &conf);
}

// One STRELA job per workgroup, one after the other.
static iree_status_t
iree_hal_strela_command_buffer_dispatch(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_executable_t *base_executable,
  iree_hal_executable_function_t executable_function,
  const iree_hal_dispatch_config_t dispatch_config,
  iree_const_byte_span_t constants,
  iree_hal_buffer_ref_list_t bindings,
  iree_hal_dispatch_flags_t dispatch_flags
) {
  TRACE_FUNC;
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);
  strela_dev *dev = command_buffer->dev;
  iree_status_t status = iree_ok_status();
  iree_hal_strela_column_t inputs[IREE_HAL_STRELA_COLUMN_COUNT];
  iree_hal_strela_column_t outputs[IREE_HAL_STRELA_COLUMN_COUNT];

  if (iree_hal_dispatch_uses_indirect_parameters(dispatch_flags)
      || iree_hal_dispatch_uses_custom_arguments(dispatch_flags)) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED, "indirect or custom STRELA dispatch");
  }
  if (executable_function.value >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "no such STRELA function");
  }
  const uint32_t *count = dispatch_config.workgroup_count;
  if (count[0] == 0 || count[1] == 0 || count[2] == 0) {
    return iree_ok_status();
  }
  const iree_hal_strela_export_t *export = &executable->exports[executable_function.value];
  iree_hal_strela_InputColumnDef_vec_t input_defs = iree_hal_strela_ExportDef_inputs_get(export->def);
  iree_hal_strela_OutputColumnDef_vec_t output_defs = iree_hal_strela_ExportDef_outputs_get(export->def);

  for (size_t c = 0; iree_status_is_ok(status) && c < IREE_HAL_STRELA_COLUMN_COUNT; ++c) {
    iree_hal_strela_InputColumnDef_table_t input = iree_hal_strela_InputColumnDef_vec_at(input_defs, c);
    iree_hal_strela_OutputColumnDef_table_t output = iree_hal_strela_OutputColumnDef_vec_at(output_defs, c);
    status = iree_hal_strela_column_resolve(
      iree_hal_strela_InputColumnDef_address_get(input),
      iree_hal_strela_InputColumnDef_word_count_get(input),
      iree_hal_strela_InputColumnDef_word_stride_get(input),
      count, constants, bindings, &inputs[c]
    );
    if (iree_status_is_ok(status)) {
      status = iree_hal_strela_column_resolve(
        iree_hal_strela_OutputColumnDef_address_get(output),
        iree_hal_strela_OutputColumnDef_word_count_get(output),
        NULL, count, constants, bindings, &outputs[c]
      );
    }
  }

  // The device must see what the CPU wrote, and nothing the CPU left dirty in
  // an output may be written back over what the device puts there later.
  for (size_t c = 0; iree_status_is_ok(status) && c < IREE_HAL_STRELA_COLUMN_COUNT; ++c) {
    if (inputs[c].buffer) {
      status = iree_hal_strela_buffer_flush_range(
        &inputs[c].buffer->base, inputs[c].offset, inputs[c].length
      );
    }
    if (iree_status_is_ok(status) && outputs[c].buffer) {
      status = iree_hal_strela_buffer_flush_range(
        &outputs[c].buffer->base, outputs[c].offset, outputs[c].length
      );
    }
  }

  uint32_t id[3];
  for (id[2] = 0; iree_status_is_ok(status) && id[2] < count[2]; ++id[2]) {
    for (id[1] = 0; iree_status_is_ok(status) && id[1] < count[1]; ++id[1]) {
      for (id[0] = 0; iree_status_is_ok(status) && id[0] < count[0]; ++id[0]) {
        status = iree_hal_strela_job_run_workgroup(dev, export->kernel, inputs, outputs, id);
      }
    }
  }

  // The CPU must see what the device wrote.
  for (size_t c = 0; iree_status_is_ok(status) && c < IREE_HAL_STRELA_COLUMN_COUNT; ++c) {
    if (outputs[c].buffer) {
      status = iree_hal_strela_buffer_invalidate_range(
        &outputs[c].buffer->base, outputs[c].offset, outputs[c].length
      );
    }
  }

  return status;
}

// Explicit cache maintenance (iree-org/iree#24945). Dispatch already does what
// STRELA needs, so only STRELA memory has anything to do here.
static iree_status_t
iree_hal_strela_command_buffer_flush_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t target_ref
) {
  TRACE_FUNC;
  iree_device_size_t offset = 0;
  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_resolve(target_ref.buffer, target_ref.offset, &offset);
  if (!buffer) return iree_ok_status();
  return iree_hal_strela_buffer_flush_range(&buffer->base, offset, target_ref.length);
}

static iree_status_t
iree_hal_strela_command_buffer_invalidate_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t target_ref
) {
  TRACE_FUNC;
  iree_device_size_t offset = 0;
  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_resolve(target_ref.buffer, target_ref.offset, &offset);
  if (!buffer) return iree_ok_status();
  return iree_hal_strela_buffer_invalidate_range(&buffer->base, offset, target_ref.length);
}

static const iree_hal_command_buffer_vtable_t
iree_hal_strela_command_buffer_vtable = {
  .destroy           = iree_hal_strela_command_buffer_destroy,
  .begin             = iree_hal_strela_command_buffer_begin,
  .end               = iree_hal_strela_command_buffer_end,
  .begin_debug_group = iree_hal_strela_command_buffer_begin_debug_group,
  .end_debug_group   = iree_hal_strela_command_buffer_end_debug_group,
  .execution_barrier = iree_hal_strela_command_buffer_execution_barrier,
  .signal_event      = iree_hal_strela_command_buffer_signal_event,
  .reset_event       = iree_hal_strela_command_buffer_reset_event,
  .wait_events       = iree_hal_strela_command_buffer_wait_events,
  .advise_buffer     = iree_hal_strela_command_buffer_advise_buffer,
  .fill_buffer       = iree_hal_strela_command_buffer_fill_buffer,
  .update_buffer     = iree_hal_strela_command_buffer_update_buffer,
  .copy_buffer       = iree_hal_strela_command_buffer_copy_buffer,
  .flush_buffer      = iree_hal_strela_command_buffer_flush_buffer,
  .invalidate_buffer = iree_hal_strela_command_buffer_invalidate_buffer,
  .collective        = iree_hal_strela_command_buffer_collective,
  .dispatch          = iree_hal_strela_command_buffer_dispatch,
};
