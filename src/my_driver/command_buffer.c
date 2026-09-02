typedef struct {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // Internal ring buffer or array to store recorded STRELA commands
} strela_command_buffer_t;

static iree_status_t
strela_command_buffer_dispatch(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_executable_t *executable,
  iree_hal_executable_function_t executable_function,
  const iree_hal_dispatch_config_t dispatch_config,
  iree_const_byte_span_t const_byte_span,
  iree_hal_buffer_ref_list_t buffer_ref_list,
  iree_hal_dispatch_flags_t dispatch_flags
) {

  [[maybe_unused]] strela_command_buffer_t *cmd = (strela_command_buffer_t*)base_command_buffer;

  // Initialize the configuration for the accelerator.
  [[maybe_unused]] strela_conf conf = {0};

  // Extract bound buffer offsets from the executable descriptor sets and
  // map them to inp0_offset, inp1_offset, out1_offset, etc.
  // conf.inp0_offset = ...

  // Push this `conf` struct into your command buffer's internal queue.
  // Do NOT execute the hardware here!
  // ...

  return iree_ok_status();
}

static void
strela_command_buffer_destroy(iree_hal_command_buffer_t *base_command_buffer) {
  [[maybe_unused]] strela_command_buffer_t *cmd = (strela_command_buffer_t *)base_command_buffer;

  // NOTE: Assuming you add `iree_allocator_t host_allocator` to strela_command_buffer_t
  // iree_allocator_free(cmd->host_allocator, cmd);
}

static iree_status_t
strela_command_buffer_begin(iree_hal_command_buffer_t *command_buffer) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_command_buffer_end(iree_hal_command_buffer_t *command_buffer) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_command_buffer_begin_debug_group(
  iree_hal_command_buffer_t *command_buffer,
  iree_string_view_t label,
  iree_hal_label_color_t label_color,
  const iree_hal_label_location_t *location
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_end_debug_group(iree_hal_command_buffer_t* command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_execution_barrier(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_execution_stage_t source_stage_mask,
  iree_hal_execution_stage_t target_stage_mask,
  iree_hal_execution_barrier_flags_t flags,
  iree_host_size_t memory_barrier_count,
  const iree_hal_memory_barrier_t *memory_barriers,
  iree_host_size_t buffer_barrier_count,
  const iree_hal_buffer_barrier_t *buffer_barriers
) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_command_buffer_signal_event(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_event_t *event,
  iree_hal_execution_stage_t source_stage_mask
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_reset_event(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_event_t *event,
  iree_hal_execution_stage_t source_stage_mask
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_wait_events(
  iree_hal_command_buffer_t *command_buffer,
  iree_host_size_t event_count,
  const iree_hal_event_t **events,
  iree_hal_execution_stage_t source_stage_mask,
  iree_hal_execution_stage_t target_stage_mask,
  iree_host_size_t memory_barrier_count,
  const iree_hal_memory_barrier_t *memory_barriers,
  iree_host_size_t buffer_barrier_count,
  const iree_hal_buffer_barrier_t *buffer_barriers
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_advise_buffer(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_ref_t buffer_ref,
  iree_hal_memory_advise_flags_t flags,
  uint64_t arg0,
  uint64_t arg1
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_fill_buffer(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_ref_t target_ref,
  const void *pattern,
  iree_host_size_t pattern_length,
  iree_hal_fill_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_update_buffer(
  iree_hal_command_buffer_t *command_buffer,
  const void *source_buffer,
  iree_host_size_t source_offset,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_update_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_command_buffer_copy_buffer(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_ref_t source_ref,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_copy_flags_t flags
) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_command_buffer_collective(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_channel_t *channel,
  iree_hal_collective_op_t op,
  uint32_t param,
  iree_hal_buffer_ref_t send_ref,
  iree_hal_buffer_ref_t recv_ref,
  iree_device_size_t element_count
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_command_buffer_vtable_t
strela_command_buffer_vtable = {
  .destroy           = strela_command_buffer_destroy,
  .begin             = strela_command_buffer_begin,
  .end               = strela_command_buffer_end,
  .begin_debug_group = strela_command_buffer_begin_debug_group,
  .end_debug_group   = strela_command_buffer_end_debug_group,
  .execution_barrier = strela_command_buffer_execution_barrier,
  .signal_event      = strela_command_buffer_signal_event,
  .reset_event       = strela_command_buffer_reset_event,
  .wait_events       = strela_command_buffer_wait_events,
  .advise_buffer     = strela_command_buffer_advise_buffer,
  .fill_buffer       = strela_command_buffer_fill_buffer,
  .update_buffer     = strela_command_buffer_update_buffer,
  .copy_buffer       = strela_command_buffer_copy_buffer,
  .collective        = strela_command_buffer_collective,
  .dispatch          = strela_command_buffer_dispatch,
};
