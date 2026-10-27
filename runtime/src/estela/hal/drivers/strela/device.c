#include "iree/async/util/proactor_pool.h"
#include "iree/hal/utils/file_registry.h"
#include "iree/hal/utils/queue_emulation.h"
#include "iree/hal/utils/queue_host_call_emulation.h"
#include "iree/hal/utils/file_transfer.h"
#include "iree/async/frontier_tracker.h"
#include "iree/base/internal/arena.h"
#include "iree/hal/utils/deferred_command_buffer.h"

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

  strela_dev *dev;
  // Backs the deferred command buffers.
  iree_arena_block_pool_t block_pool;

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
  TRACE_FUNC;
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
    iree_arena_block_pool_initialize(32 * 1024, host_allocator, &device->block_pool);
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
    // TODO(Diego): in the future in identifier or options there will be a way
    // to select which strela device to init (I guess)
    device->dev = strela_dev_init(0);
    if (!strela_dev_ok(device->dev)) {
      status = iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "cannot open STRELA (libstrela error %d); is strela2.ko loaded?",
        strela_dev_get_err(device->dev).errnum
      );
    }
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_strela_allocator_create(
      device->dev, host_allocator, &device->device_allocator
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_device_t *device = (iree_hal_strela_device_t *)base_device;
  iree_allocator_t host_allocator = device->host_allocator;

  iree_hal_strela_device_clear_topology_info(device);
  iree_hal_allocator_release(device->device_allocator);
  iree_hal_channel_provider_release(device->channel_provider);
  iree_async_proactor_pool_release(device->proactor_pool);
  if (device->dev) {
    strela_dev_deinit(device->dev);
  }
  iree_arena_block_pool_deinitialize(&device->block_pool);

  iree_allocator_free(host_allocator, device);
}

static iree_string_view_t
iree_hal_strela_device_id(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t
iree_hal_strela_device_host_allocator(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  return device->host_allocator;
}

static iree_hal_allocator_t *
iree_hal_strela_device_allocator(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  return device->device_allocator;
}

static void
iree_hal_strela_device_replace_device_allocator(
  iree_hal_device_t *base_device,
  iree_hal_allocator_t *new_allocator
) {
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_channel_provider_retain(new_provider);
  iree_hal_channel_provider_release(device->channel_provider);
  device->channel_provider = new_provider;
}

static iree_status_t
iree_hal_strela_device_trim(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t
iree_hal_strela_device_query_i64(
  iree_hal_device_t *base_device,
  iree_string_view_t category,
  iree_string_view_t key,
  int64_t *out_value
) {
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
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
    value = iree_string_view_equal(key, IREE_SV("strela_bitstream"));
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
  TRACE_FUNC;
  memset(out_capabilities, 0, sizeof *out_capabilities);
  return iree_ok_status();
}

static const iree_hal_device_topology_info_t *
iree_hal_strela_device_topology_info(iree_hal_device_t* base_device) {
  TRACE_FUNC;
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  return &device->topology_info;
}

static iree_status_t
iree_hal_strela_device_refine_topology_edge(
  iree_hal_device_t *src_device,
  iree_hal_device_t *dst_device,
  iree_hal_topology_edge_t *edge
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_assign_topology_info(
  iree_hal_device_t *base_device,
  const iree_hal_device_topology_info_t *topology_info
) {
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_allocator_t *device_allocator = iree_hal_device_allocator(base_device);

  // Recorded now, run by queue_execute.
  return iree_hal_deferred_command_buffer_create(
    device_allocator,
    mode,
    command_categories,
    queue_affinity,
    binding_capacity,
    &device->block_pool,
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_device_t* device = iree_hal_strela_device_cast(base_device);
  iree_allocator_t device_host_allocator = iree_hal_device_host_allocator(base_device);

  (void)device;

  return iree_hal_strela_executable_cache_create(
    device->dev, identifier, device_host_allocator, out_executable_cache
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);

  // TODO: implement this...
  iree_hal_deferred_work_queue_t *work_queue = NULL;
  return iree_hal_strela_semaphore_create(
    device->proactor,
    queue_affinity,
    initial_value,
    flags,
    work_queue,
    device->host_allocator,
    out_semaphore
  );
}

static iree_hal_semaphore_compatibility_t
iree_hal_strela_device_query_semaphore_compatibility(
  iree_hal_device_t *base_device,
  iree_hal_semaphore_t *semaphore
) {
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_status_t status = iree_ok_status();
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_allocator_t *device_allocator = iree_hal_device_allocator(base_device);
  iree_hal_buffer_t *buffer = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_wait(
      wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE
    );
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_allocate_buffer(
      device_allocator, params, allocation_size, &buffer
    );
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(
      signal_semaphore_list, /*frontier=*/NULL
    );
  }

  if (iree_status_is_ok(status)) {
    (void)device;
    // TODO: does this apply to STRELA?
    // iree_hal_cuda_device_advance_frontier(device);
  } else {
    iree_hal_semaphore_list_fail(
      signal_semaphore_list, iree_status_clone(status)
    );
  }

  if (!iree_status_is_ok(status)) {
    if (buffer) {
      iree_hal_buffer_release(buffer);
    }
  }

  *out_buffer = buffer;

  return status;
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
  TRACE_FUNC;
  // The buffer goes back to libstrela when its last reference is released, so
  // only the ordering is left to honour.
  iree_status_t status = iree_hal_semaphore_list_wait(
    wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE
  );
  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list, iree_status_clone(status));
  }
  return status;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_device_t *device = iree_hal_strela_device_cast(base_device);
  iree_hal_command_buffer_t *direct = NULL;

  // Synchronous: wait, run every command in order on this thread, signal.
  iree_status_t status = iree_hal_semaphore_list_wait(
    wait_semaphore_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE
  );

  if (iree_status_is_ok(status) && command_buffer) {
    if (!iree_hal_deferred_command_buffer_isa(command_buffer)) {
      status = iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT, "command buffer is not from a STRELA device"
      );
    }
  }

  if (iree_status_is_ok(status) && command_buffer) {
    status = iree_hal_strela_command_buffer_create(
      device->dev,
      iree_hal_device_allocator(base_device),
      IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT | IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED,
      iree_hal_command_buffer_allowed_categories(command_buffer),
      queue_affinity,
      /*binding_capacity=*/0,
      device->host_allocator,
      &direct
    );
  }

  if (iree_status_is_ok(status) && command_buffer) {
    status = iree_hal_deferred_command_buffer_apply(
      command_buffer, direct, buffer_binding_table
    );
  }

  if (direct) {
    iree_hal_command_buffer_release(direct);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL);
  }
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_list_fail(signal_semaphore_list, iree_status_clone(status));
  }
  return status;
}

static iree_status_t
iree_hal_strela_device_queue_flush(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity
) {
  TRACE_FUNC;
  /* Our driver is synchronous hence every operation executes immediatelly.
   * There is no need to flush. */
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_device_profiling_begin(
  iree_hal_device_t *base_device,
  const iree_hal_device_profiling_options_t *options
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_flush(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_profiling_end(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_external_capture_begin(
  iree_hal_device_t *base_device,
  const iree_hal_device_external_capture_options_t *options
) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_device_external_capture_end(iree_hal_device_t *base_device) {
  TRACE_FUNC;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

/* The device API for the HAL driver is large but the important parts are:
 *   - create_{channel,command_buffer,event,executable_cache,semaphore}
 *   - queue_{alloca,dealloca,fill,update,copy,read,write,host_call,dispatch,execute,flush}
 * The create functions make objects needed by other parts of the device API, in
 * particular command buffers, executable caches and semaphores.
 * The queue functions are for interacting with "hardware queues", what CUDA
 * calls CUstream, Vulkan VkQueue, DirectX ID3D12CommandQueue and Metal
 * MTLCommandQueue. All the queue functions (except queue_flush) have their
 * function signature begin with the following arguments
 *   - iree_hal_device_t               *base_device
 *   - iree_hal_queue_affinity_t        queue_affinity
 *   - const iree_hal_semaphore_list_t  wait_semaphore_list
 *   - const iree_hal_semaphore_list_t  signal_semaphore_list
 * queue_affinity is a 64 bits mask that is used to indicate on which queue the
 * command (i.e. one of {alloca,dealloca,fill,update,copy,read,write,host_call,
 * dispatch,execute,flush}) may execute. Instead wait_semaphore_list and
 * signal_semaphore_list are two, as their name implies, lists of "timeline
 * semaphore" [0] and value pairs. The first list is a series of (semaphore,
 * value) that the command has to wait on before executing i.e. the timeline
 * semaphore needs to become >= value; the other is a list of (semaphore, value)
 * pairs that the command should notify when it has finished executing i.e.
 * setting semaphore := value.
 *
 * In the HAL IR semaphore lists are represented as fences
 * %fence_x = hal.fence.create device(%dev_b : !hal.device) flags("None") : !hal.fence
 * %fence_y = hal.fence.create device(%dev_b : !hal.device) flags("None") : !hal.fence
 * %both = hal.fence.join at([%fence_x, %fence_y]) flags("None") -> !hal.fence
 * In this way every fence represents a set of values (timepoints) to reach in
 * the timeline (for waiting) or to set (for signaling).
 *
 * Arguably the most important piece of the queue API is queue_execute. This
 * function takes as input
 *   - iree_hal_command_buffer_t       *command_buffer
 *   - iree_hal_buffer_binding_table_t  buffer_binding_table
 * the command buffer is called CUgraph in CUDA, VkCommandBuffer in Vulkan,
 * ID3D12GraphicsCommandList in DirectX and MTLCommandBuffer in Metal. The
 * commands to execute are recorded once and (potentially) execute multiple
 * times. It is important to note that the commands recorded in the command
 * buffers are different from the commands of the queue functions. A command
 * buffer is a recorded list of device commands, submitted to a queue as a
 * single unit, whose order is fixed only by the barriers between them. The
 * recorded commands can either refer to I/O buffers directly (explicitly naming
 * them) or via an index in the binding table. The binding tables is a list of
 * (buffer, offset, length) triples where the buffer slot is filled at execution
 * time to allow the reuse of the recorded commands with different buffers.
 *
 * Below there is a diagram showing how barriers (---) define the partial order
 * of the commands (*) in the buffer.
 *
 *   ---
 *  / | \
 * *  *  *
 *  \ | /
 *   ---
 *
 * Below there is an example of how a command buffer is constructed. In this
 * specific example the variables that match this regex "%c[0-9]+"" are all
 * integer constants. This makes it so that the command can reference slots in
 * the binding table instead of explicitly name buffer. Note how mode is "None"
 * and not "OneShot".
 *
 * %cmd = hal.command_buffer.create device(%device_b : !hal.device) mode("None") categories("Transfer|Dispatch") affinity(%c-1_i64) bindings(%c4) : !hal.command_buffer
 * hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer> source(%c0 : index)[%c0] target(%c3 : index)[%c0] length(%c64) flags("None")
 * hal.command_buffer.copy_buffer<%cmd : !hal.command_buffer> source(%c1 : index)[%c0] target(%c3 : index)[%c64] length(%c64) flags("None")
 * hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")
 * hal.command_buffer.dispatch<%cmd : !hal.command_buffer> target(%dev_b : !hal.executable)[%function_id] workgroups([%c1, %c1, %c1]) bindings([
 *   (%c3 : index)[%c0, %c128],
 *   (%c2 : index)[%c0, %c64]
 * ]) flags("None")
 * hal.command_buffer.execution_barrier<%cmd : !hal.command_buffer> source("Dispatch|Transfer|CommandRetire") target("CommandIssue|Dispatch|Transfer") flags("None")
 * hal.command_buffer.finalize<%cmd : !hal.command_buffer>
 *
 * To then execute this command buffer this operation is used.
 *
 * hal.device.queue.execute.indirect<%device_b : !hal.device> affinity(%c-1_i64) wait(%fence_5) signal(%fence_6) commands(%cmd) bindings([
 *   (%buffer : !hal.buffer)[%c0, %c64],
 *   (%buffer_2 : !hal.buffer)[%c0, %c64],
 *   (%transient_buffer : !hal.buffer)[%c0, %c64],
 *   (%transient_buffer_4 : !hal.buffer)[%c0, %c128]
 * ]) flags("None")
 *
 * To allow for this execution model the IR of the accelerator does not need to
 * know about binding tables at all: it only needs to be able to address its
 * I/O buffers by ordinal. Inside an executable every buffer is reached through
 * hal.interface.binding.subspan, which names a binding ordinal and a byte
 * offset, never a concrete buffer. Hence the same compiled executable works for
 * both direct and indirect command buffers.
 *
 * Concretely the target IR must provide a translation of the "interface"
 * subset of the HAL dialect. For SPIR-V (Vulkan) it is:
 *
 * hal.interface.binding.subspan -> spirv.GlobalVariable bind(n, m)
 * hal.interface.constant.load   -> spirv.GlobalVariable
 * hal.interface.workgroup.count -> spirv.GlobalVariable built_in("NumWorkgroups")
 * hal.interface.workgroup.id    -> spirv.GlobalVariable built_in("WorkgroupId")
 * hal.interface.workgroup.size  -> spirv.Constant
 *
 * [0]: https://www.khronos.org/blog/vulkan-timeline-semaphores
 */
static const iree_hal_device_vtable_t
iree_hal_strela_device_vtable = {
  .destroy                       = iree_hal_strela_device_destroy,

  .id /*_get*/                   = iree_hal_strela_device_id,
  .host_allocator /*_get*/       = iree_hal_strela_device_host_allocator,
  .device_allocator /*_get*/     = iree_hal_strela_device_allocator,

  .replace_device_allocator      = iree_hal_strela_device_replace_device_allocator,
  .replace_channel_provider      = iree_hal_strela_device_replace_channel_provider,

  .trim /*_allocator*/           = iree_hal_strela_device_trim,

  .topology_info                 = iree_hal_strela_device_topology_info,
  .refine_topology_edge          = iree_hal_strela_device_refine_topology_edge,
  .assign_topology_info          = iree_hal_strela_device_assign_topology_info,

  .create_channel                = iree_hal_strela_device_create_channel,
  .create_command_buffer         = iree_hal_strela_device_create_command_buffer,
  .create_event                  = iree_hal_strela_device_create_event,
  .create_executable_cache       = iree_hal_strela_device_create_executable_cache,
  .create_semaphore              = iree_hal_strela_device_create_semaphore,

  .import_file                   = iree_hal_strela_device_import_file,

  .query_i64                     = iree_hal_strela_device_query_i64,
  .query_capabilities            = iree_hal_strela_device_query_capabilities,
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
