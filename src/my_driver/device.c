#include "iree/async/util/proactor_pool.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/async/frontier_tracker.h"

typedef struct iree_hal_strela_device_options_t {
  int reserved;
} iree_hal_strela_device_options_t;

static void
iree_hal_strela_device_options_initialize(
  iree_hal_strela_device_options_t *out_options
) {
  memset(out_options, 0, sizeof *out_options);
}

static iree_status_t
iree_hal_strela_device_options_verify(
  const iree_hal_strela_device_options_t *options
) {
  iree_status_t status = iree_ok_status();

  if (!is_all_zero(options, sizeof *options)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  return status;
}

typedef struct {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t *device_allocator;
  iree_async_proactor_pool_t *proactor_pool;
  iree_async_proactor_t *proactor;
  iree_async_frontier_tracker_t *frontier_tracker;
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;
  iree_hal_channel_provider_t *channel_provider;
  iree_hal_device_topology_info_t topology_info;

  strela_dev *dev; // TODO: does this goes here?

  // + trailing identifier string storage
} iree_hal_strela_device_t;

static const iree_hal_device_vtable_t iree_hal_strela_device_vtable;

static iree_hal_strela_device_t *
iree_hal_strela_device_cast(iree_hal_device_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_device_vtable);
  return (iree_hal_strela_device_t *)base_value;
}

static iree_status_t
iree_hal_strela_device_create(
  iree_string_view_t identifier,
  const iree_hal_strela_device_options_t *options,
  const iree_hal_device_create_params_t *create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t **out_device
) {
  iree_status_t status = iree_ok_status();
  iree_hal_strela_device_t *device = NULL;
  iree_host_size_t total_size = sizeof *device + identifier.size;

  if (iree_status_is_ok(status)) {
    status = iree_hal_strela_device_options_verify(options);
  }

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void **)&device);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_strela_device_vtable, &device->resource);
    iree_string_view_append_to_buffer(
      identifier, &device->identifier,
      (char *)device + total_size - identifier.size
    );
    device->host_allocator = host_allocator;
    device->proactor_pool = create_params->proactor_pool;
    iree_async_proactor_pool_retain(device->proactor_pool);
    iree_atomic_store(&device->epoch, 0, iree_memory_order_relaxed);
  }

  if (iree_status_is_ok(status)) {
    status = iree_async_proactor_pool_get(
      device->proactor_pool, 0, &device->proactor
    );
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_strela_allocator_create(
      host_allocator, &device->device_allocator
    );
  }

  if (!iree_status_is_ok(status)) {
    if (device) {
      iree_hal_device_release((iree_hal_device_t *)device);
    }
  }

  *out_device = (iree_hal_device_t *)device;

  return status;
}

static void
iree_hal_strela_device_clear_topology_info(iree_hal_strela_device_t *device) {
  if (device->frontier_tracker) {
    iree_async_frontier_tracker_retire_axis(
      device->frontier_tracker, device->axis,
      iree_status_from_code(IREE_STATUS_CANCELLED)
    );
    iree_async_frontier_tracker_release(device->frontier_tracker);
    device->frontier_tracker = NULL;
    device->axis = 0;
  }
  memset(&device->topology_info, 0, sizeof device->topology_info);
}

static void
iree_hal_strela_device_destroy(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t *device = (iree_hal_strela_device_t *)base_device;
  iree_allocator_t host_allocator = device->host_allocator;

  iree_hal_strela_device_clear_topology_info(device);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_async_proactor_pool_release(device->proactor_pool);

  iree_allocator_free(host_allocator, device);
}

static iree_string_view_t
iree_hal_strela_device_id(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t
iree_hal_strela_device_host_allocator(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t *
iree_hal_strela_device_allocator(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  return device->device_allocator;
}

static void
iree_hal_strela_device_replace_device_allocator(
  iree_hal_device_t *base_device,
  iree_hal_allocator_t *new_allocator
) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_allocator_retain(new_allocator);
  iree_hal_allocator_release(device->device_allocator);
  device->device_allocator = new_allocator;
}

static void
iree_hal_strela_device_replace_channel_provider(
  iree_hal_device_t *base_device,
  iree_hal_channel_provider_t *new_provider
) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t
iree_hal_strela_device_trim(iree_hal_device_t *base_device) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t
iree_hal_strela_device_query_i64(
  iree_hal_device_t *base_device,
  iree_string_view_t category,
  iree_string_view_t key,
  int64_t *out_value
) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  iree_status_t status = iree_make_status(
    IREE_STATUS_NOT_FOUND,
    "unknown device configuration key value '%.*s :: %.*s'",
    (int)category.size, category.data, (int)key.size, key.data
  );
  int64_t value = 0;

  if (iree_string_view_equal(category, IREE_SV("hal.device.id"))) {
    value = iree_string_view_match_pattern(device->identifier, key) ? 1 : 0;
    status = iree_ok_status();
  } else if (iree_string_view_equal(category, IREE_SV("hal.executable.format"))) {
    value = iree_string_view_equal(key, IREE_SV("custom"));
    status = iree_ok_status();
  } else if (iree_string_view_equal(category, IREE_SV("hal.device"))) {
    // TODO: verify if this is true or not for STRELA
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      value = 1;
      status = iree_ok_status();
    }
  } else if (iree_string_view_equal(category, IREE_SV("hal.dispatch"))) {
    // TODO: verify if this is true or not for STRELA
    if (iree_string_view_equal(key, IREE_SV("concurrency"))) {
      value = 1;
      status = iree_ok_status();
    }
  }

  *out_value = value;
  return status;
}

static iree_status_t
iree_hal_strela_device_query_capabilities(
  iree_hal_device_t *device,
  iree_hal_device_capabilities_t *out_capabilities
) {
  memset(out_capabilities, 0, sizeof *out_capabilities);
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
iree_hal_strela_device_topology_info(iree_hal_device_t* base_device) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t
iree_hal_strela_device_refine_topology_edge(
  iree_hal_device_t *src_device,
  iree_hal_device_t *dst_device,
  iree_hal_topology_edge_t *edge
) {
  (void)src_device;
  (void)dst_device;
  (void)edge;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_assign_topology_info(
  iree_hal_device_t *base_device,
  const iree_hal_device_topology_info_t *topology_info
) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  iree_status_t status = iree_ok_status();

  if (!topology_info) {
    iree_hal_strela_device_clear_topology_info(device);
  } else {
    iree_async_frontier_tracker_t *frontier_tracker =
        topology_info->frontier.tracker;
    iree_async_axis_t axis = topology_info->frontier.base_axis;

    status = iree_async_frontier_tracker_register_axis(
      frontier_tracker, axis, /*semaphore=*/NULL
    );

    if (iree_status_is_ok(status)) {
      device->topology_info = *topology_info;
      device->frontier_tracker = frontier_tracker;
      device->axis = axis;
      iree_async_frontier_tracker_retain(device->frontier_tracker);
    }
  }

  return status;
}

static iree_status_t
iree_hal_strela_device_create_channel(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_channel_params_t params,
  iree_hal_channel_t **out_channel
) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);

  (void)device;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

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
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_allocator_t *device_allocator = iree_hal_device_allocator(base_device);

  return iree_hal_strela_command_buffer_create(
    device_allocator,
    mode,
    command_categories,
    queue_affinity,
    binding_capacity,
    device->host_allocator,
    out_command_buffer
  );
}

static iree_status_t
iree_hal_strela_device_create_event(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_event_flags_t flags,
  iree_hal_event_t **out_event
) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);

  (void)device;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_create_executable_cache(
  iree_hal_device_t *base_device,
  iree_string_view_t identifier,
  iree_hal_executable_cache_t **out_executable_cache
) {
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  iree_allocator_t device_host_allocator = iree_hal_device_host_allocator(base_device);

  (void)device;

  return iree_hal_strela_executable_cache_create(
    identifier, device_host_allocator, out_executable_cache
  );
}

static iree_status_t
iree_hal_strela_device_import_file(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_memory_access_t access,
  iree_io_file_handle_t *handle,
  iree_hal_external_file_flags_t flags,
  iree_hal_file_t **out_file
) {
  iree_allocator_t device_host_allocator = iree_hal_device_host_allocator(base_device);
  return iree_hal_file_from_handle(
    /*device_allocator=*/NULL,
    queue_affinity,
    access,
    handle,
    /*proactor=*/NULL,
    device_host_allocator,
    out_file
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
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);

  return iree_hal_strela_semaphore_create(
    device->proactor,
    queue_affinity,
    initial_value,
    flags,
    device->host_allocator,
    out_semaphore
  );
}

static iree_hal_semaphore_compatibility_t
iree_hal_strela_device_query_semaphore_compatibility(
  iree_hal_device_t *base_device,
  iree_hal_semaphore_t *semaphore
) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);

  (void)device;

  iree_hal_semaphore_compatibility_t compatibility =
    IREE_HAL_SEMAPHORE_COMPATIBILITY_NONE;

  return compatibility;
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
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_pool_t *pool,
  iree_hal_buffer_params_t params,
  iree_device_size_t allocation_size,
  iree_hal_alloca_flags_t flags,
  iree_hal_buffer_t **out_buffer
) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);

  (void)device;

  // TODO: this is necessary to make progress in executing simple_abs and it has to be implemented...

  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_device_queue_dealloca(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_buffer_t *buffer,
  iree_hal_dealloca_flags_t flags
) {
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);

  (void)device;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_queue_fill(
  iree_hal_device_t *base_device,
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
  return iree_hal_device_queue_emulated_fill(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    target_buffer,
    target_offset,
    length,
    pattern,
    pattern_length,
    flags
  );
}

static iree_status_t
iree_hal_strela_device_queue_update(
  iree_hal_device_t *base_device,
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
  return iree_hal_device_queue_emulated_update(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    source_buffer,
    source_offset,
    target_buffer,
    target_offset,
    length,
    flags
  );
}

static iree_status_t
iree_hal_strela_device_queue_copy(
  iree_hal_device_t *base_device,
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
  return iree_hal_device_queue_emulated_copy(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    source_buffer,
    source_offset,
    target_buffer,
    target_offset,
    length,
    flags
  );
}

static iree_status_t
iree_hal_strela_device_queue_read(
  iree_hal_device_t *base_device,
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
  iree_hal_file_transfer_options_t options = {
    .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
    .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_read_streaming(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    source_file,
    source_offset,
    target_buffer,
    target_offset,
    length,
    flags,
    options
  );
}

static iree_status_t
iree_hal_strela_device_queue_write(
  iree_hal_device_t *base_device,
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
  iree_hal_file_transfer_options_t options = {
    .chunk_count = IREE_HAL_FILE_TRANSFER_CHUNK_COUNT_DEFAULT,
    .chunk_size = IREE_HAL_FILE_TRANSFER_CHUNK_SIZE_DEFAULT,
  };
  return iree_hal_device_queue_write_streaming(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    source_buffer,
    source_offset,
    target_file,
    target_offset,
    length,
    flags,
    options
  );
}

static iree_status_t
iree_hal_strela_device_queue_host_call(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_host_call_t call,
  const uint64_t args[4],
  iree_hal_host_call_flags_t flags
) {
  return iree_hal_device_queue_emulated_host_call(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    call,
    args,
    flags
  );
}

static iree_status_t
iree_hal_strela_device_queue_dispatch(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_executable_t *executable,
  iree_hal_executable_function_t export_ordinal,
  const iree_hal_dispatch_config_t config,
  iree_const_byte_span_t constants,
  const iree_hal_buffer_ref_list_t bindings,
  iree_hal_dispatch_flags_t flags
) {
  return iree_hal_device_queue_emulated_dispatch(
    base_device,
    queue_affinity,
    wait_semaphore_list,
    signal_semaphore_list,
    executable,
    export_ordinal,
    config,
    constants,
    bindings,
    flags
  );
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

  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);

  (void)device;

  return iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL);
}

static iree_status_t
iree_hal_strela_device_queue_flush(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_begin(
  iree_hal_device_t *base_device,
  const iree_hal_device_profiling_options_t *options
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_flush(iree_hal_device_t *base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_end(iree_hal_device_t *base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_external_capture_begin(
  iree_hal_device_t *base_device,
  const iree_hal_device_external_capture_options_t *options
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_external_capture_end(iree_hal_device_t *base_device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_device_vtable_t
iree_hal_strela_device_vtable = {
  .destroy                       = iree_hal_strela_device_destroy,
  .id                            = iree_hal_strela_device_id,
  .host_allocator                = iree_hal_strela_device_host_allocator,
  .device_allocator              = iree_hal_strela_device_allocator,
  .replace_device_allocator      = iree_hal_strela_device_replace_device_allocator,
  .replace_channel_provider      = iree_hal_strela_device_replace_channel_provider,
  .trim                          = iree_hal_strela_device_trim,
  .query_i64                     = iree_hal_strela_device_query_i64,
  .query_capabilities            = iree_hal_strela_device_query_capabilities,
  .topology_info                 = iree_hal_strela_device_topology_info,
  .refine_topology_edge          = iree_hal_strela_device_refine_topology_edge,
  .assign_topology_info          = iree_hal_strela_device_assign_topology_info,
  .create_channel                = iree_hal_strela_device_create_channel,
  .create_command_buffer         = iree_hal_strela_device_create_command_buffer,
  .create_event                  = iree_hal_strela_device_create_event,
  .create_executable_cache       = iree_hal_strela_device_create_executable_cache,
  .import_file                   = iree_hal_strela_device_import_file,
  .create_semaphore              = iree_hal_strela_device_create_semaphore,
  .query_semaphore_compatibility = iree_hal_strela_device_query_semaphore_compatibility,
  .query_queue_pool_backend      = iree_hal_strela_device_query_queue_pool_backend,
  .queue_alloca                  = iree_hal_strela_device_queue_alloca,
  .queue_dealloca                = iree_hal_strela_device_queue_dealloca,
  .queue_fill                    = iree_hal_strela_device_queue_fill,
  .queue_update                  = iree_hal_strela_device_queue_update,
  .queue_copy                    = iree_hal_strela_device_queue_copy,
  .queue_read                    = iree_hal_strela_device_queue_read,
  .queue_write                   = iree_hal_strela_device_queue_write,
  .queue_host_call               = iree_hal_strela_device_queue_host_call,
  .queue_dispatch                = iree_hal_strela_device_queue_dispatch,
  .queue_execute                 = iree_hal_strela_device_queue_execute,
  .queue_flush                   = iree_hal_strela_device_queue_flush,
  .profiling_begin               = iree_hal_strela_device_profiling_begin,
  .profiling_flush               = iree_hal_strela_device_profiling_flush,
  .profiling_end                 = iree_hal_strela_device_profiling_end,
  .external_capture_begin        = iree_hal_strela_device_external_capture_begin,
  .external_capture_end          = iree_hal_strela_device_external_capture_end,
};
