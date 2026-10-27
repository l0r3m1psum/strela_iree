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

// Where one CGRA column reads or writes, resolved for a dispatch.
typedef struct {
  // NULL for an unused column.
  iree_hal_strela_buffer_t *buffer;
  // Byte offset of the first word in the allocated buffer.
  iree_device_size_t offset;
  // Bytes from the first to the last word the column touches.
  iree_device_size_t length;
  // What libstrela takes: words from the base of STRELA memory.
  size_t words_from_base;
  size_t word_count;
  size_t stride;
} iree_hal_strela_column_t;

static iree_status_t
iree_hal_strela_column_resolve(
  iree_hal_strela_ColumnDef_table_t def,
  bool is_input,
  iree_const_byte_span_t constants,
  iree_hal_buffer_ref_list_t bindings,
  iree_hal_strela_column_t *out_column
) {
  memset(out_column, 0, sizeof *out_column);
  out_column->word_count = iree_hal_strela_ColumnDef_word_count_get(def);
  out_column->stride = is_input ? iree_hal_strela_ColumnDef_stride_get(def) : 0;
  if (out_column->word_count == 0) {
    return iree_ok_status();
  }

  uint32_t binding = iree_hal_strela_ColumnDef_binding_get(def);
  if (binding >= bindings.count || !bindings.values[binding].buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding %u is missing", binding);
  }
  iree_hal_buffer_ref_t ref = bindings.values[binding];

  // Compile-time part, then the [low, high] push constants of a runtime one.
  uint64_t byte_offset = iree_hal_strela_ColumnDef_byte_offset_get(def);
  flatbuffers_uint32_vec_t ordinals = iree_hal_strela_ColumnDef_offset_constants_get(def);
  if (flatbuffers_uint32_vec_len(ordinals) == 2) {
    const uint32_t *values = (const uint32_t *)constants.data;
    size_t lo = flatbuffers_uint32_vec_at(ordinals, 0);
    size_t hi = flatbuffers_uint32_vec_at(ordinals, 1);
    if ((hi + 1) * sizeof(uint32_t) > constants.data_length
        || (lo + 1) * sizeof(uint32_t) > constants.data_length) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "missing push constants");
    }
    byte_offset += (uint64_t)values[lo] | (uint64_t)values[hi] << 32;
  }

  // An input reads one word in every `stride`, the first one repeatedly for 0.
  size_t span_words = is_input && out_column->stride != 1
    ? (out_column->word_count - 1) * out_column->stride + 1
    : out_column->word_count;
  out_column->length = span_words * sizeof(strela_word);

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
  if (out_column->offset % sizeof(strela_word) != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "binding %u is not word aligned", binding);
  }
  out_column->buffer = buffer;
  out_column->words_from_base =
    buffer->s_buf.offset_words_from_base + out_column->offset / sizeof(strela_word);
  return iree_ok_status();
}

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

  if (executable_function.value >= executable->export_count) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE, "no such STRELA function");
  }
  const iree_hal_strela_export_t *export = &executable->exports[executable_function.value];
  iree_hal_strela_ColumnDef_vec_t input_defs = iree_hal_strela_ExportDef_inputs_get(export->def);
  iree_hal_strela_ColumnDef_vec_t output_defs = iree_hal_strela_ExportDef_outputs_get(export->def);

  for (size_t c = 0; iree_status_is_ok(status) && c < IREE_HAL_STRELA_COLUMN_COUNT; ++c) {
    status = iree_hal_strela_column_resolve(
      iree_hal_strela_ColumnDef_vec_at(input_defs, c), true, constants, bindings, &inputs[c]
    );
    if (iree_status_is_ok(status)) {
      status = iree_hal_strela_column_resolve(
        iree_hal_strela_ColumnDef_vec_at(output_defs, c), false, constants, bindings, &outputs[c]
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

  strela_job job;
  memset(&job, 0, sizeof job);
  if (iree_status_is_ok(status)) {
    strela_conf conf = {
      .inp0_offset = inputs[0].words_from_base, .inp0_count = inputs[0].word_count, .inp0_stride = inputs[0].stride,
      .inp1_offset = inputs[1].words_from_base, .inp1_count = inputs[1].word_count, .inp1_stride = inputs[1].stride,
      .inp2_offset = inputs[2].words_from_base, .inp2_count = inputs[2].word_count, .inp2_stride = inputs[2].stride,
      .inp3_offset = inputs[3].words_from_base, .inp3_count = inputs[3].word_count, .inp3_stride = inputs[3].stride,
      .out0_offset = outputs[0].words_from_base, .out0_count = outputs[0].word_count,
      .out1_offset = outputs[1].words_from_base, .out1_count = outputs[1].word_count,
      .out2_offset = outputs[2].words_from_base, .out2_count = outputs[2].word_count,
      .out3_offset = outputs[3].words_from_base, .out3_count = outputs[3].word_count,
    };
    job = strela_job_config(dev, export->kernel, &conf);
    strela_job_submit(dev, job);
    // The driver's timeout bounds this: a job that hangs is cancelled.
    strela_job_wait(dev, job, -1);
    status = iree_hal_strela_status_from_dev(dev, "running a STRELA job");
  }

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
