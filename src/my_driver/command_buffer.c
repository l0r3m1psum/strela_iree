typedef struct {
  iree_hal_command_buffer_t base;
  iree_allocator_t host_allocator;
  // Internal ring buffer or array to store recorded STRELA commands
} iree_hal_strela_command_buffer_t;

static const iree_hal_command_buffer_vtable_t iree_hal_strela_command_buffer_vtable;

static iree_hal_strela_command_buffer_t *
iree_hal_strela_command_buffer_cast(iree_hal_command_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_command_buffer_vtable);
  return (iree_hal_strela_command_buffer_t *)base_value;
}

static iree_status_t
iree_hal_strela_command_buffer_create(
  iree_hal_allocator_t* device_allocator,
  iree_hal_command_buffer_mode_t mode,
  iree_hal_command_category_t command_categories,
  iree_hal_queue_affinity_t queue_affinity,
  iree_host_size_t binding_capacity,
  iree_allocator_t host_allocator,
  iree_hal_command_buffer_t **out_command_buffer
) {
  iree_status_t status = iree_ok_status();
  iree_hal_strela_command_buffer_t *command_buffer = NULL;
  iree_hal_command_buffer_t *command_buffer_base = NULL;

  iree_host_size_t command_buffer_validation_state_size
    = iree_hal_command_buffer_validation_state_size(mode, binding_capacity);
  status = iree_allocator_malloc(
    host_allocator,
    sizeof *command_buffer + command_buffer_validation_state_size,
    (void **)&command_buffer
  );

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
    command_buffer_base = &command_buffer->base;
  }

  if (!iree_status_is_ok(status) && command_buffer_base) {
    iree_hal_command_buffer_release(command_buffer_base);
  }

  *out_command_buffer = command_buffer_base;
  return status;
}

static void
iree_hal_strela_command_buffer_destroy(iree_hal_command_buffer_t *base_command_buffer) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);
  iree_allocator_t host_allocator = command_buffer->host_allocator;

  iree_allocator_free(host_allocator, command_buffer);
}

static iree_status_t
iree_hal_strela_command_buffer_begin(iree_hal_command_buffer_t *base_command_buffer) {
  printf("%s\n", __func__);

  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_end(iree_hal_command_buffer_t *base_command_buffer) {
  printf("%s\n", __func__);

  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_begin_debug_group(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_string_view_t label,
  iree_hal_label_color_t label_color,
  const iree_hal_label_location_t *location
) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_end_debug_group(iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

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
  printf("%s\n", __func__);
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_command_buffer_signal_event(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_event_t *event,
  iree_hal_execution_stage_t source_stage_mask
) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_reset_event(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_event_t *event,
  iree_hal_execution_stage_t source_stage_mask
) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

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
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_advise_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t buffer_ref,
  iree_hal_memory_advise_flags_t flags,
  uint64_t arg0,
  uint64_t arg1
) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_fill_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t target_ref,
  const void *pattern,
  iree_host_size_t pattern_length,
  iree_hal_fill_flags_t flags
) {
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_update_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  const void *source_buffer,
  iree_host_size_t source_offset,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_update_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_copy_buffer(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_buffer_ref_t source_ref,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_copy_flags_t flags
) {
  printf("%s\n", __func__);
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_ok_status();
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
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_command_buffer_dispatch(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_executable_t *executable,
  iree_hal_executable_function_t executable_function,
  const iree_hal_dispatch_config_t dispatch_config,
  iree_const_byte_span_t const_byte_span,
  iree_hal_buffer_ref_list_t buffer_ref_list,
  iree_hal_dispatch_flags_t dispatch_flags
) {
  printf("%s\n", __func__);
  iree_hal_strela_command_buffer_t *command_buffer = iree_hal_strela_command_buffer_cast(base_command_buffer);

  (void)command_buffer;

  return iree_ok_status();
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
  .collective        = iree_hal_strela_command_buffer_collective,
  .dispatch          = iree_hal_strela_command_buffer_dispatch,
};
