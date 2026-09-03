typedef struct {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;
  iree_async_proactor_pool_t *proactor_pool;
  iree_async_proactor_t* proactor;
  iree_async_frontier_tracker_t *frontier_tracker;
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;
  iree_hal_channel_provider_t *channel_provider;
  iree_hal_device_topology_info_t topology_info;

  strela_dev *dev;
} iree_hal_strela_device_t;

static iree_status_t
iree_hal_strela_device_create_command_buffer(
  iree_hal_device_t *base_device,
  iree_hal_command_buffer_mode_t mode,
  iree_hal_command_category_t command_categories,
  iree_hal_queue_affinity_t queue_affinity,
  iree_host_size_t binding_capacity,
  iree_hal_command_buffer_t **out_command_buffer
) {
  printf("%s\n", __func__);

  IREE_TRACE_ZONE_BEGIN(z0);
  *out_command_buffer = NULL;
  iree_hal_strela_command_buffer_t* command_buffer = NULL;
  iree_hal_allocator_t *device_allocator = iree_hal_device_allocator(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_allocator_malloc(host_allocator,
                            sizeof(*command_buffer) +
                                iree_hal_command_buffer_validation_state_size(
                                    mode, binding_capacity),
                            (void**)&command_buffer));
  iree_hal_command_buffer_initialize(
      device_allocator, mode, command_categories, queue_affinity,
      binding_capacity, (uint8_t*)command_buffer + sizeof *command_buffer,
      &iree_hal_strela_command_buffer_vtable, &command_buffer->base);
  command_buffer->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_command_buffer = &command_buffer->base;
  } else {
    iree_hal_command_buffer_release(&command_buffer->base);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void
iree_hal_strela_device_destroy(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t *device = (iree_hal_strela_device_t *)base_device;
  iree_allocator_t host_allocator = device->host_allocator;

  // 1. Release the custom allocator (this decrements the refcount)
  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }

  // 2. Teardown the hardware handle
  if (device->dev) {
    // iree_hal_strela_dev_destroy(device->dev); // Replace with your actual hardware free function
  }

  // 3. Free the device memory
  iree_allocator_free(host_allocator, device);
}

static iree_status_t
iree_hal_strela_device_queue_execute(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_binding_table_t buffer_binding_table,
  iree_hal_execute_flags_t execute_flags
) {
  printf("%s\n", __func__);

  [[maybe_unused]] iree_hal_strela_device_t *device = (iree_hal_strela_device_t *)base_device;

  // 1. Wait on the provided wait_semaphore_list.
  // 2. Iterate over the recorded command_buffers.

  // For each recorded task, apply the configuration to the hardware.
  // iree_hal_strela_config(device->dev, kernel, &recorded_conf);

  // Trigger the accelerator execution.
  // iree_hal_strela_execute(device->dev);

  // Verify that the accelerator hasn't encountered a hardware fault.
  // if (!iree_hal_strela_dev_ok(device->dev)) {
  //     return iree_make_status(IREE_STATUS_INTERNAL, "STRELA execution failed");
  // }

  // 3. Signal the provided signal_semaphore_list to notify the VM of completion.

  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL));

  return iree_ok_status();
}

static iree_hal_allocator_t *
iree_hal_strela_device_allocator(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t* device = (iree_hal_strela_device_t*)base_device;
  return device->device_allocator;
}

static iree_string_view_t
iree_hal_strela_device_id(iree_hal_device_t *base_device) {
  return ((iree_hal_strela_device_t *)base_device)->identifier;
}

static iree_allocator_t
iree_hal_strela_device_host_allocator(iree_hal_device_t *base_device) {
  return ((iree_hal_strela_device_t *)base_device)->host_allocator;
}

static iree_status_t
iree_hal_strela_device_query_i64(
  iree_hal_device_t *base_device,
  iree_string_view_t category,
  iree_string_view_t key,
  int64_t *out_value
) {
  printf(
  "%s: category='%.*s', key='%.*s'\n", __func__,
   (int)category.size, category.data, (int)key.size, key.data
  );

  *out_value = 0;

  iree_string_view_t hal_executable_format = iree_string_view_literal("hal.executable.format");
  if (iree_string_view_equal(category, hal_executable_format)) {
    iree_string_view_t value = iree_string_view_literal("custom");
    if (iree_string_view_equal(key, value)) {
      *out_value = 1;
      return iree_ok_status();
    }
  }

  iree_string_view_t hal_device_id = iree_string_view_literal("hal.device.id");
  if (iree_string_view_equal(category, hal_device_id)) {
    iree_string_view_t value = iree_string_view_literal("strela");
    if (iree_string_view_equal(key, value)) {
      *out_value = 1;
      return iree_ok_status();
    }
  }

  return iree_make_status(
    IREE_STATUS_NOT_FOUND,
    "unknown device property '%.*s'.'%.*s'",
    (int)category.size, category.data, (int)key.size, key.data
  );
}

static iree_status_t
iree_hal_strela_device_create_semaphore(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  uint64_t initial_value,
  iree_hal_semaphore_flags_t flags,
  iree_hal_semaphore_t **out_semaphore
) {
  printf("%s\n", __func__);
  iree_hal_strela_device_t *device = (iree_hal_strela_device_t *) base_device;

  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  iree_hal_strela_semaphore_t *semaphore = NULL;
  iree_host_size_t frontier_offset = 0, total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
    z0,
    iree_async_semaphore_layout(sizeof *semaphore, 0, &frontier_offset, &total_size)
  );
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
    z0,
    iree_allocator_malloc(device->host_allocator, total_size, (void**)&semaphore)
  );
  iree_async_semaphore_initialize(
    (const iree_async_semaphore_vtable_t *) &iree_hal_strela_semaphore_vtable,
    device->proactor, initial_value, frontier_offset, 0, &semaphore->async
  );
  semaphore->host_allocator = device->host_allocator;

  iree_atomic_store(&semaphore->payload_value, initial_value, iree_memory_order_release);

  *out_semaphore = (iree_hal_semaphore_t *) &semaphore->async;
  IREE_TRACE_ZONE_END(z0);

  printf("semaphore->async = %p\n", &semaphore->async);
  printf("device->host_allocator.self = %p\n", device->host_allocator.self);

  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_device_create_executable_cache(
  iree_hal_device_t *base_device,
  iree_string_view_t identifier,
  iree_hal_executable_cache_t **out_executable_cache
) {
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_executable_cache = NULL;
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  iree_hal_null_executable_cache_t *executable_cache = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(*executable_cache),
                                (void**)&executable_cache));
  iree_hal_resource_initialize(&iree_hal_null_executable_cache_vtable,
                               &executable_cache->resource);
  executable_cache->host_allocator = host_allocator;

  iree_status_t status = iree_ok_status();

  if (iree_status_is_ok(status)) {
    *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  } else {
    iree_hal_executable_cache_release(
        (iree_hal_executable_cache_t*)executable_cache);
  }
  IREE_TRACE_ZONE_END(z0);

  return status;
}

static iree_status_t
iree_hal_strela_device_trim(iree_hal_device_t *device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_query_capabilities(
  iree_hal_device_t *device,
  iree_hal_device_capabilities_t *out_capabilities
) {
  memset(out_capabilities, 0, sizeof *out_capabilities);
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_device_refine_topology_edge(
  iree_hal_device_t *src_device,
  iree_hal_device_t *dst_device,
  iree_hal_topology_edge_t *edge
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_assign_topology_info(
  iree_hal_device_t *device,
  const iree_hal_device_topology_info_t *topology_info
) {
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_device_create_channel(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_channel_params_t params,
  iree_hal_channel_t **out_channel
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_create_event(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_event_flags_t flags,
  iree_hal_event_t **out_event
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_import_file(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_memory_access_t access,
  iree_io_file_handle_t *handle,
  iree_hal_external_file_flags_t flags,
  iree_hal_file_t **out_file
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_query_queue_pool_backend(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_queue_pool_backend_t *out_backend
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_alloca(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_pool_t *pool,
  iree_hal_buffer_params_t params,
  iree_device_size_t allocation_size,
  iree_hal_alloca_flags_t flags,
  iree_hal_buffer_t **IREE_RESTRICT out_buffer
) {
  return iree_hal_strela_allocator_allocate_buffer(
    ((iree_hal_strela_device_t *) device)->device_allocator,
    &params,
    allocation_size,
    out_buffer
  );
}

static iree_status_t
iree_hal_strela_device_queue_dealloca(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_buffer_t *buffer,
  iree_hal_dealloca_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_fill(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_buffer_t *target_buffer,
  iree_device_size_t target_offset,
  iree_device_size_t length,
  const void *pattern,
  iree_host_size_t pattern_length,
  iree_hal_fill_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_update(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  const void *source_buffer,
  iree_host_size_t source_offset,
  iree_hal_buffer_t *target_buffer,
  iree_device_size_t target_offset,
  iree_device_size_t length,
  iree_hal_update_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_copy(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_buffer_t *source_buffer,
  iree_device_size_t source_offset,
  iree_hal_buffer_t *target_buffer,
  iree_device_size_t target_offset,
  iree_device_size_t length,
  iree_hal_copy_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_read(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_file_t *source_file,
  uint64_t source_offset,
  iree_hal_buffer_t *target_buffer,
  iree_device_size_t target_offset,
  iree_device_size_t length,
  iree_hal_read_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_write(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_buffer_t *source_buffer,
  iree_device_size_t source_offset,
  iree_hal_file_t *target_file,
  uint64_t target_offset,
  iree_device_size_t length,
  iree_hal_write_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_host_call(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_host_call_t call,
  const uint64_t args[4],
  iree_hal_host_call_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_dispatch(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_executable_t *executable,
  iree_hal_executable_function_t function,
  const iree_hal_dispatch_config_t config,
  iree_const_byte_span_t constants,
  const iree_hal_buffer_ref_list_t bindings,
  iree_hal_dispatch_flags_t flags
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_flush(
  iree_hal_device_t *device,
  iree_hal_queue_affinity_t queue_affinity
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_begin(
  iree_hal_device_t *device,
  const iree_hal_device_profiling_options_t *options
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_flush(iree_hal_device_t *device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_end(iree_hal_device_t *device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_external_capture_begin(
  iree_hal_device_t *device,
  const iree_hal_device_external_capture_options_t *options
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_external_capture_end(iree_hal_device_t *device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static void
iree_hal_strela_device_replace_device_allocator(
  iree_hal_device_t *base_device,
  iree_hal_allocator_t *new_allocator
) {
  iree_hal_strela_device_t* device = (iree_hal_strela_device_t*)base_device;

  iree_hal_allocator_retain(new_allocator);

  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }

  device->device_allocator = new_allocator;
}

static void
iree_hal_strela_device_replace_channel_provider(
  iree_hal_device_t *base_device,
  iree_hal_channel_provider_t *new_provider
) {
  printf("%s\n", __func__);
  // If your device eventually supports MPI or async channels, you handle swapping here.
  // For now, it can be safely left as a no-op.
}

static const iree_hal_device_vtable_t
iree_hal_strela_device_vtable = {
  .destroy                  = iree_hal_strela_device_destroy,
  .id                       = iree_hal_strela_device_id,
  .host_allocator           = iree_hal_strela_device_host_allocator,
  .device_allocator         = iree_hal_strela_device_allocator,
  .replace_device_allocator = iree_hal_strela_device_replace_device_allocator,
  .replace_channel_provider = iree_hal_strela_device_replace_channel_provider,
  .trim                     = iree_hal_strela_device_trim,
  .query_i64                = iree_hal_strela_device_query_i64,
  .query_capabilities       = iree_hal_strela_device_query_capabilities,
  // const iree_hal_device_topology_info_t*(IREE_API_PTR* topology_info)(iree_hal_device_t* device);
  .refine_topology_edge     = iree_hal_strela_device_refine_topology_edge,
  .assign_topology_info     = iree_hal_strela_device_assign_topology_info,
  .create_channel           = iree_hal_strela_device_create_channel,
  .create_command_buffer    = iree_hal_strela_device_create_command_buffer,
  .create_event             = iree_hal_strela_device_create_event,
  .create_executable_cache  = iree_hal_strela_device_create_executable_cache,
  .import_file              = iree_hal_strela_device_import_file,
  .create_semaphore         = iree_hal_strela_device_create_semaphore,
  // iree_hal_semaphore_compatibility_t(IREE_API_PTR* query_semaphore_compatibility)(iree_hal_device_t* device, iree_hal_semaphore_t* semaphore);
  .query_queue_pool_backend = iree_hal_strela_device_query_queue_pool_backend,
  .queue_alloca             = iree_hal_strela_device_queue_alloca,
  .queue_dealloca           = iree_hal_strela_device_queue_dealloca,
  .queue_fill               = iree_hal_strela_device_queue_fill,
  .queue_update             = iree_hal_strela_device_queue_update,
  .queue_copy               = iree_hal_strela_device_queue_copy,
  .queue_read               = iree_hal_strela_device_queue_read,
  .queue_write              = iree_hal_strela_device_queue_write,
  .queue_host_call          = iree_hal_strela_device_queue_host_call,
  .queue_dispatch           = iree_hal_strela_device_queue_dispatch,
  .queue_execute            = iree_hal_strela_device_queue_execute,
  .queue_flush              = iree_hal_strela_device_queue_flush,
  .profiling_begin          = iree_hal_strela_device_profiling_begin,
  .profiling_flush          = iree_hal_strela_device_profiling_flush,
  .profiling_end            = iree_hal_strela_device_profiling_end,
  .external_capture_begin   = iree_hal_strela_device_external_capture_begin,
  .external_capture_end     = iree_hal_strela_device_external_capture_end,
};
