#include "strela.h"
#include "iree/hal/api.h"

typedef struct {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* device_allocator;
  iree_async_proactor_pool_t* proactor_pool;
  iree_async_proactor_t* proactor;
  iree_async_frontier_tracker_t* frontier_tracker;
  iree_async_axis_t axis;
  iree_atomic_int64_t epoch;
  iree_hal_channel_provider_t* channel_provider;
  iree_hal_device_topology_info_t topology_info;

  strela_dev *dev;
} strela_device_t;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  strela_dev *dev;
} strela_allocator_t;

////////////////////////////////////////////////////////////////////////////////
typedef struct strela_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;

  strela_buffer s_buf;
  void* host_ptr;
} strela_buffer_t;

static void
strela_buffer_recycle(iree_hal_buffer_t* buffer) {
  printf("%s\n", __func__);
}

static void
strela_buffer_destroy(iree_hal_buffer_t* buffer) {
  printf("%s\n", __func__);
}

static iree_status_t
strela_buffer_map_range(
  iree_hal_buffer_t *base_buffer,
  iree_hal_mapping_mode_t mapping_mode,
  iree_hal_memory_access_t memory_access,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length,
  iree_hal_buffer_mapping_t *mapping
) {
  printf("%s\n", __func__);

  strela_buffer_t *buffer = (strela_buffer_t *)base_buffer;
  if (!buffer->host_ptr) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "STRELA buffer is not host-accessible");
  }
  uint8_t *data_ptr = (uint8_t *)buffer->host_ptr + local_byte_offset;
  mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);

  return iree_ok_status();
}

static iree_status_t
strela_buffer_unmap_range(iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length, iree_hal_buffer_mapping_t* mapping) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_buffer_invalidate_range(iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_buffer_flush_range(iree_hal_buffer_t* buffer, iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_buffer_vtable_t
strela_buffer_vtable = {
  .recycle          = strela_buffer_recycle,
  .destroy          = strela_buffer_destroy,
  .map_range        = strela_buffer_map_range,
  .unmap_range      = strela_buffer_unmap_range,
  .invalidate_range = strela_buffer_invalidate_range,
  .flush_range      = strela_buffer_flush_range,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_allocator_allocate_buffer(
  iree_hal_allocator_t *IREE_RESTRICT base_allocator,
  const iree_hal_buffer_params_t *IREE_RESTRICT  params,
  iree_device_size_t allocation_size,
  iree_hal_buffer_t **IREE_RESTRICT out_buffer
) {
  printf("%s\n", __func__);

  strela_allocator_t *allocator = (strela_allocator_t *)base_allocator;

  // NOTE: You should move `strela_dev_init(0)` to your driver/allocator initialization!
  // Calling it here means you initialize the hardware on every single buffer allocation.
  if (!allocator->dev) {
    allocator->dev = strela_dev_init(0);
  }

  // Allocate contiguous hardware memory using your existing STRELA API.
  strela_buffer s_buf = strela_buffer_alloc(allocator->dev, allocation_size);

  // Retrieve the mapped host-visible pointer.
  void *host_ptr = strela_buffer_to_ptr(allocator->dev, s_buf);

  // 1. Allocate memory for your custom wrapper struct
  strela_buffer_t *buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator->host_allocator, sizeof(*buffer), (void**)&buffer));

  // 2. Save your hardware state and allocator config
  buffer->host_allocator = allocator->host_allocator;
  buffer->s_buf = s_buf;
  buffer->host_ptr = host_ptr;
  buffer->release_callback = (iree_hal_buffer_release_callback_t){0};

  iree_hal_memory_type_t actual_type = params->type | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  iree_hal_memory_access_t actual_access = IREE_HAL_MEMORY_ACCESS_ALL; // <--- The Fix
  iree_hal_buffer_usage_t actual_usage = params->usage | IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING;

  // 3. Initialize the base IREE buffer tracking fields
  iree_hal_buffer_initialize(
    iree_hal_buffer_placement_undefined(),
    (iree_hal_buffer_t *) buffer,     // Pointer to your newly allocated struct
    allocation_size,                  // Total allocation size
    0,                                // Byte offset
    allocation_size,                  // Byte length
    actual_type,                     // Memory type (host-visible, device-local, etc.)
    actual_access,                   // Allowed access (read/write)
    actual_usage,                    // Allowed usage (transfer, dispatch)
    &strela_buffer_vtable,            // Your custom vtable
    &buffer->base                     // Output pointer to the base iree_hal_buffer_t
  );

  // 4. Pass the initialized buffer back to the VM
  *out_buffer = &buffer->base;

  return iree_ok_status();
}

static void
strela_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  [[maybe_unused]] strela_allocator_t *allocator = (strela_allocator_t*)base_allocator;

  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t
strela_allocator_host_allocator(
  const iree_hal_allocator_t *base_allocator
) {
  printf("%s base_allocator: 0x%p\n", __func__, base_allocator);
  return ((strela_allocator_t *)base_allocator)->host_allocator;
}

static iree_status_t strela_allocator_trim(
  iree_hal_allocator_t *base_allocator
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
  // return iree_ok_status(); // No-op
}

static void
strela_allocator_deallocate_buffer(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *base_buffer
) {
  // We will implement this later when you handle actual memory freeing
}

static void
strela_allocator_query_statistics(
  iree_hal_allocator_t *base_allocator,
  iree_hal_allocator_statistics_t *out_statistics
) {
  memset(out_statistics, 0, sizeof *out_statistics);
}

static iree_status_t
strela_allocator_query_memory_heaps(
  iree_hal_allocator_t *base_allocator,
  iree_host_size_t capacity,
  iree_hal_allocator_memory_heap_t *heaps,
  iree_host_size_t *out_count
) {
  *out_count = 1;
  if (capacity > 0 && heaps != NULL) {
    heaps[0].type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    heaps[0].allowed_usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    heaps[0].max_allocation_size = 1024 * 1024 * 256; // 256 MB max allocation size
    heaps[0].min_alignment = 64;
  }
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
  // return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
strela_allocator_query_buffer_compatibility(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_params_t *params,
  iree_device_size_t *allocation_size
) {

  params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params->access = IREE_HAL_MEMORY_ACCESS_ALL; // Grants READ and WRITE
  params->usage |= IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING;

  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE
    | IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER
    | IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH
  ;
}

static iree_status_t
strela_allocator_import_buffer(
  iree_hal_allocator_t* IREE_RESTRICT allocator,
  const iree_hal_buffer_params_t* IREE_RESTRICT params,
  iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
  iree_hal_buffer_release_callback_t release_callback,
  iree_hal_buffer_t** IREE_RESTRICT out_buffer
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_export_buffer(
  iree_hal_allocator_t* IREE_RESTRICT allocator,
  iree_hal_buffer_t* IREE_RESTRICT buffer,
  iree_hal_external_buffer_type_t requested_type,
  iree_hal_external_buffer_flags_t requested_flags,
  iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static bool
strela_allocator_supports_virtual_memory(
      iree_hal_allocator_t* IREE_RESTRICT allocator) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_query_granularity(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_params_t params,
      iree_device_size_t* IREE_RESTRICT out_minimum_page_size,
      iree_device_size_t* IREE_RESTRICT out_recommended_page_size) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_reserve(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
      iree_hal_buffer_t** IREE_RESTRICT out_virtual_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_release(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT virtual_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_physical_memory_allocate(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_params_t params, iree_device_size_t size,
      iree_allocator_t host_allocator,
      iree_hal_physical_memory_t** IREE_RESTRICT out_physical_memory) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_physical_memory_free(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_physical_memory_t* IREE_RESTRICT physical_memory) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_map(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
      iree_device_size_t virtual_offset,
      iree_hal_physical_memory_t* IREE_RESTRICT physical_memory,
      iree_device_size_t physical_offset, iree_device_size_t size) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_unmap(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
      iree_device_size_t virtual_offset, iree_device_size_t size) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_protect(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
      iree_device_size_t virtual_offset, iree_device_size_t size,
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_memory_protection_t protection) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_allocator_virtual_memory_advise(
      iree_hal_allocator_t* IREE_RESTRICT allocator,
      iree_hal_buffer_t* IREE_RESTRICT virtual_buffer,
      iree_device_size_t virtual_offset, iree_device_size_t size,
      iree_hal_queue_affinity_t queue_affinity,
      iree_hal_memory_advice_t advice) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_allocator_vtable_t
strela_allocator_vtable = {
  .destroy                    = strela_allocator_destroy,
  .host_allocator             = strela_allocator_host_allocator,
  .trim                       = strela_allocator_trim,
  .query_statistics           = strela_allocator_query_statistics,
  .query_memory_heaps         = strela_allocator_query_memory_heaps,
  .query_buffer_compatibility = strela_allocator_query_buffer_compatibility,
  .allocate_buffer            = strela_allocator_allocate_buffer,
  .deallocate_buffer          = strela_allocator_deallocate_buffer,
  .import_buffer              = strela_allocator_import_buffer,
  .export_buffer              = strela_allocator_export_buffer,

  .supports_virtual_memory          = strela_allocator_supports_virtual_memory,
  .virtual_memory_query_granularity = strela_allocator_virtual_memory_query_granularity,
  .virtual_memory_reserve           = strela_allocator_virtual_memory_reserve,
  .virtual_memory_release           = strela_allocator_virtual_memory_release,
  .physical_memory_allocate         = strela_allocator_physical_memory_allocate,
  .physical_memory_free             = strela_allocator_physical_memory_free,
  .virtual_memory_map               = strela_allocator_virtual_memory_map,
  .virtual_memory_unmap             = strela_allocator_virtual_memory_unmap,
  .virtual_memory_protect           = strela_allocator_virtual_memory_protect,
  .virtual_memory_advise            = strela_allocator_virtual_memory_advise,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
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
strela_device_begin(iree_hal_command_buffer_t *command_buffer) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_device_end(iree_hal_command_buffer_t *command_buffer) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_device_begin_debug_group(iree_hal_command_buffer_t* command_buffer, iree_string_view_t label, iree_hal_label_color_t label_color, const iree_hal_label_location_t* location) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_end_debug_group(iree_hal_command_buffer_t* command_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_execution_barrier(
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
strela_device_signal_event(iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event, iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_reset_event(iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event, iree_hal_execution_stage_t source_stage_mask) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_wait_events(iree_hal_command_buffer_t* command_buffer, iree_host_size_t event_count, const iree_hal_event_t** events, iree_hal_execution_stage_t source_stage_mask, iree_hal_execution_stage_t target_stage_mask, iree_host_size_t memory_barrier_count, const iree_hal_memory_barrier_t* memory_barriers, iree_host_size_t buffer_barrier_count, const iree_hal_buffer_barrier_t* buffer_barriers) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_advise_buffer(iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t buffer_ref, iree_hal_memory_advise_flags_t flags, uint64_t arg0, uint64_t arg1) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_fill_buffer(iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_ref_t target_ref, const void* pattern, iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_update_buffer(iree_hal_command_buffer_t* command_buffer, const void* source_buffer, iree_host_size_t source_offset, iree_hal_buffer_ref_t target_ref, iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_copy_buffer(
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_ref_t source_ref,
  iree_hal_buffer_ref_t target_ref,
  iree_hal_copy_flags_t flags
) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_device_collective(iree_hal_command_buffer_t* command_buffer, iree_hal_channel_t* channel, iree_hal_collective_op_t op, uint32_t param, iree_hal_buffer_ref_t send_ref, iree_hal_buffer_ref_t recv_ref, iree_device_size_t element_count) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_command_buffer_vtable_t
strela_command_buffer_vtable = {
  .destroy           = strela_command_buffer_destroy,
  .begin             = strela_device_begin,
  .end               = strela_device_end,
  .begin_debug_group = strela_device_begin_debug_group,
  .end_debug_group   = strela_device_end_debug_group,
  .execution_barrier = strela_device_execution_barrier,
  .signal_event      = strela_device_signal_event,
  .reset_event       = strela_device_reset_event,
  .wait_events       = strela_device_wait_events,
  .advise_buffer     = strela_device_advise_buffer,
  .fill_buffer       = strela_device_fill_buffer,
  .update_buffer     = strela_device_update_buffer,
  .copy_buffer       = strela_device_copy_buffer,
  .collective        = strela_device_collective,
  .dispatch          = strela_command_buffer_dispatch,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
typedef struct strela_semaphore_t {
  iree_async_semaphore_t async;
  iree_allocator_t host_allocator;
  iree_atomic_int64_t payload_value;
} strela_semaphore_t;

static iree_status_t
strela_semaphore_wait(iree_hal_semaphore_t* semaphore, uint64_t value, iree_timeout_t timeout, iree_async_wait_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_semaphore_import_timepoint(iree_hal_semaphore_t* semaphore, uint64_t value, iree_hal_queue_affinity_t queue_affinity, iree_hal_external_timepoint_t external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_semaphore_export_timepoint(iree_hal_semaphore_t* semaphore, uint64_t value, iree_hal_queue_affinity_t queue_affinity, iree_hal_external_timepoint_type_t requested_type, iree_hal_external_timepoint_flags_t requested_flags, iree_hal_external_timepoint_t* IREE_RESTRICT out_external_timepoint) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static void
strela_semaphore_destroy(iree_async_semaphore_t* semaphore) {
  printf("%s\n", __func__);
}

static uint64_t
strela_semaphore_query(iree_async_semaphore_t *base_semaphore) {
  strela_semaphore_t* semaphore = (strela_semaphore_t*)base_semaphore;
  return iree_atomic_load(&semaphore->payload_value, iree_memory_order_acquire);
}

static iree_status_t
strela_semaphore_signal(
  iree_async_semaphore_t *base_semaphore,
  uint64_t value,
  const iree_async_frontier_t* frontier
) {
  printf("%s: signaling to %llu\n", __func__, (unsigned long long)value);

  strela_semaphore_t* semaphore = (strela_semaphore_t*)base_semaphore;

  // 1. Advance the simulated host-side tracking value.
  iree_atomic_store(&semaphore->payload_value, value, iree_memory_order_release);

  // 2. Notify the hardware (if you had a real device-side timeline semaphore).
  // strela_hw_signal_semaphore(..., value);

  iree_async_semaphore_advance_timeline(base_semaphore, value, frontier);

  // 3. Wake up the async proactor.
  // Note: Depending on your exact IREE revision, the base `iree_async_semaphore_t`
  // might automatically resolve waiting nodes when this vtable hook returns `OK`,
  // or you might need to explicitly call a proactor wake-up function like:
  // iree_async_semaphore_advance(base_semaphore, value, frontier);

  return iree_ok_status();
}

static void
strela_semaphore_on_fail(iree_async_semaphore_t* semaphore, iree_status_code_t status_code) {
  printf("%s\n", __func__);
}

static const iree_hal_semaphore_vtable_t
strela_semaphore_vtable = {
  .async = {
    .destroy = strela_semaphore_destroy,
    .query = strela_semaphore_query,
    .signal = strela_semaphore_signal,
    .on_fail = strela_semaphore_on_fail,
  },
  .wait             = strela_semaphore_wait,
  .import_timepoint = strela_semaphore_import_timepoint,
  .export_timepoint = strela_semaphore_export_timepoint,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_device_create_command_buffer(
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
  strela_command_buffer_t* command_buffer = NULL;
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
      &strela_command_buffer_vtable, &command_buffer->base);
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
strela_device_destroy(iree_hal_device_t *base_device) {
  strela_device_t *device = (strela_device_t *)base_device;
  iree_allocator_t host_allocator = device->host_allocator;

  // 1. Release the custom allocator (this decrements the refcount)
  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }

  // 2. Teardown the hardware handle
  if (device->dev) {
    // strela_dev_destroy(device->dev); // Replace with your actual hardware free function
  }

  // 3. Free the device memory
  iree_allocator_free(host_allocator, device);
}

static iree_status_t
strela_device_queue_execute(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_binding_table_t buffer_binding_table,
  iree_hal_execute_flags_t execute_flags
) {
  printf("%s\n", __func__);

  [[maybe_unused]] strela_device_t *device = (strela_device_t *)base_device;

  // 1. Wait on the provided wait_semaphore_list.
  // 2. Iterate over the recorded command_buffers.

  // For each recorded task, apply the configuration to the hardware.
  // strela_config(device->dev, kernel, &recorded_conf);

  // Trigger the accelerator execution.
  // strela_execute(device->dev);

  // Verify that the accelerator hasn't encountered a hardware fault.
  // if (!strela_dev_ok(device->dev)) {
  //     return iree_make_status(IREE_STATUS_INTERNAL, "STRELA execution failed");
  // }

  // 3. Signal the provided signal_semaphore_list to notify the VM of completion.

  IREE_RETURN_IF_ERROR(iree_hal_semaphore_list_signal(signal_semaphore_list, /*frontier=*/NULL));

  return iree_ok_status();
}

static iree_hal_allocator_t *
strela_device_allocator(iree_hal_device_t *base_device) {
  strela_device_t* device = (strela_device_t*)base_device;
  return device->device_allocator;
}

static iree_string_view_t
strela_device_id(iree_hal_device_t *base_device) {
  return ((strela_device_t *)base_device)->identifier;
}

static iree_allocator_t
strela_device_host_allocator(iree_hal_device_t *base_device) {
  return ((strela_device_t *)base_device)->host_allocator;
}

static iree_status_t
strela_device_query_i64(
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
strela_device_create_semaphore(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  uint64_t initial_value,
  iree_hal_semaphore_flags_t flags,
  iree_hal_semaphore_t **out_semaphore
) {
  printf("%s\n", __func__);
  strela_device_t *device = (strela_device_t *) base_device;

  IREE_ASSERT_ARGUMENT(out_semaphore);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_semaphore = NULL;

  strela_semaphore_t *semaphore = NULL;
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
    (const iree_async_semaphore_vtable_t *) &strela_semaphore_vtable,
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

typedef struct iree_hal_null_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_null_executable_cache_t;

static void
strela_executable_cache_destroy(iree_hal_executable_cache_t* executable_cache) {
  ;
}

static iree_status_t
strela_executable_cache_infer_format(iree_hal_executable_cache_t* executable_cache, iree_hal_executable_caching_mode_t caching_mode, iree_const_byte_span_t executable_data, iree_host_size_t executable_format_capacity, char* executable_format, iree_host_size_t* out_inferred_size) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static bool
strela_executable_cache_can_prepare_format(iree_hal_executable_cache_t* executable_cache, iree_hal_executable_caching_mode_t caching_mode, iree_string_view_t executable_format) {
  printf("%s\n", __func__);
  iree_string_view_t custom = iree_string_view_literal("custom");
  return iree_string_view_equal(executable_format, custom);
}

// Define the underlying STRELA executable structure
typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // TODO: Add fields here later to store your parsed STRELA binaries or kernel parameters
} strela_executable_t;

static void strela_executable_destroy(iree_hal_executable_t* base_executable) {
  strela_executable_t* executable = (strela_executable_t*)base_executable;
  iree_allocator_free(executable->host_allocator, executable);
}

static iree_host_size_t strela_executable_function_count(
    iree_hal_executable_t* base_executable) {
  // Stub: Pretend we successfully loaded 1 function
  return 1;
}

static iree_status_t strela_executable_function_info(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function,
    iree_hal_executable_function_info_t* out_info) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t strela_executable_function_parameters(
    iree_hal_executable_t* base_executable,
    iree_hal_executable_function_t function, iree_host_size_t capacity,
    iree_hal_executable_function_parameter_t* out_parameters) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t strela_executable_lookup_function_by_name(
    iree_hal_executable_t *base_executable, iree_string_view_t name,
    iree_hal_executable_function_t *out_function) {
  printf("%s: looking up kernel '%.*s'\n", __func__, (int)name.size, name.data);

  out_function->value = 1;
  return iree_ok_status();
}

static iree_status_t strela_executable_lookup_global_by_name(
    iree_hal_executable_t* base_executable, iree_string_view_t name,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

// Map the stubs to the vtable
static const iree_hal_executable_vtable_t
strela_executable_vtable = {
  .destroy = strela_executable_destroy,
  .function_count = strela_executable_function_count,
  .function_info = strela_executable_function_info,
  .function_parameters = strela_executable_function_parameters,
  .lookup_function_by_name = strela_executable_lookup_function_by_name,
  .lookup_global_by_name = strela_executable_lookup_global_by_name,
};

static iree_status_t
strela_executable_cache_prepare_executable(
  iree_hal_executable_cache_t* executable_cache,
  const iree_hal_executable_params_t* executable_params,
  iree_hal_executable_t** out_executable
) {
  printf("%s\n", __func__);

  // If you don't have a custom cache struct with a host_allocator yet,
  // you can safely fall back to the system allocator for this stub.
  iree_allocator_t host_allocator = iree_allocator_system();

  strela_executable_t* executable = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *executable, (void**)&executable));

  // Initialize the base resource using your new vtable
  iree_hal_resource_initialize(&strela_executable_vtable, &executable->resource);
  executable->host_allocator = host_allocator;

  // Pass the allocated and initialized executable back to IREE
  *out_executable = (iree_hal_executable_t*)executable;

  return iree_ok_status();
}

static const iree_hal_executable_cache_vtable_t
iree_hal_null_executable_cache_vtable = {
  .destroy = strela_executable_cache_destroy,
  .infer_format = strela_executable_cache_infer_format,
  .can_prepare_format = strela_executable_cache_can_prepare_format,
  .prepare_executable = strela_executable_cache_prepare_executable,
};

static iree_status_t
strela_device_create_executable_cache(
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
strela_device_trim(iree_hal_device_t* device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_query_capabilities(
  iree_hal_device_t* device,
  iree_hal_device_capabilities_t* out_capabilities
) {
  memset(out_capabilities, 0, sizeof *out_capabilities);
  return iree_ok_status();
}

static iree_status_t
strela_device_refine_topology_edge(iree_hal_device_t* src_device, iree_hal_device_t* dst_device, iree_hal_topology_edge_t* edge) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_assign_topology_info(iree_hal_device_t* device, const iree_hal_device_topology_info_t* topology_info) {
  return iree_ok_status();
}

static iree_status_t
strela_device_create_channel(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, iree_hal_channel_params_t params, iree_hal_channel_t** out_channel) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_create_event(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, iree_hal_event_flags_t flags, iree_hal_event_t** out_event) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_import_file(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_access_t access, iree_io_file_handle_t* handle, iree_hal_external_file_flags_t flags, iree_hal_file_t** out_file) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_query_queue_pool_backend(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, iree_hal_queue_pool_backend_t* out_backend) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_alloca(
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
  return strela_allocator_allocate_buffer(
    ((strela_device_t *) device)->device_allocator,
    &params,
    allocation_size,
    out_buffer
  );
}

static iree_status_t
strela_device_queue_dealloca(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* buffer, iree_hal_dealloca_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_fill(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset, iree_device_size_t length, const void* pattern, iree_host_size_t pattern_length, iree_hal_fill_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_update(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, const void* source_buffer, iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset, iree_device_size_t length, iree_hal_update_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_copy(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset, iree_device_size_t length, iree_hal_copy_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_read(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_file_t* source_file, uint64_t source_offset, iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset, iree_device_size_t length, iree_hal_read_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_write(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset, iree_hal_file_t* target_file, uint64_t target_offset, iree_device_size_t length, iree_hal_write_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_host_call(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_host_call_t call, const uint64_t args[4], iree_hal_host_call_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_dispatch(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity, const iree_hal_semaphore_list_t wait_semaphore_list, const iree_hal_semaphore_list_t signal_semaphore_list, iree_hal_executable_t* executable, iree_hal_executable_function_t function, const iree_hal_dispatch_config_t config, iree_const_byte_span_t constants, const iree_hal_buffer_ref_list_t bindings, iree_hal_dispatch_flags_t flags) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_queue_flush(iree_hal_device_t* device, iree_hal_queue_affinity_t queue_affinity) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_profiling_begin(iree_hal_device_t* device, const iree_hal_device_profiling_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_profiling_flush(iree_hal_device_t* device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_profiling_end(iree_hal_device_t* device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_external_capture_begin(iree_hal_device_t* device, const iree_hal_device_external_capture_options_t* options) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_device_external_capture_end(iree_hal_device_t* device) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static void
strela_device_replace_device_allocator(
  iree_hal_device_t* base_device,
  iree_hal_allocator_t* new_allocator
) {
  strela_device_t* device = (strela_device_t*)base_device;

  iree_hal_allocator_retain(new_allocator);

  if (device->device_allocator) {
    iree_hal_allocator_release(device->device_allocator);
  }

  device->device_allocator = new_allocator;
}

static void
strela_device_replace_channel_provider(
  iree_hal_device_t* base_device,
  iree_hal_channel_provider_t* new_provider
) {
  printf("%s\n", __func__);
  // If your device eventually supports MPI or async channels, you handle swapping here.
  // For now, it can be safely left as a no-op.
}

static const iree_hal_device_vtable_t
strela_device_vtable = {
  .destroy                  = strela_device_destroy,
  .id                       = strela_device_id,
  .host_allocator           = strela_device_host_allocator,
  .device_allocator         = strela_device_allocator,
  .replace_device_allocator = strela_device_replace_device_allocator,
  .replace_channel_provider = strela_device_replace_channel_provider,
  .trim                     = strela_device_trim,
  .query_i64                = strela_device_query_i64,
  .query_capabilities       = strela_device_query_capabilities,
  // const iree_hal_device_topology_info_t*(IREE_API_PTR* topology_info)(iree_hal_device_t* device);
  .refine_topology_edge     = strela_device_refine_topology_edge,
  .assign_topology_info     = strela_device_assign_topology_info,
  .create_channel           = strela_device_create_channel,
  .create_command_buffer    = strela_device_create_command_buffer,
  .create_event             = strela_device_create_event,
  .create_executable_cache  = strela_device_create_executable_cache,
  .import_file              = strela_device_import_file,
  .create_semaphore         = strela_device_create_semaphore,
  // iree_hal_semaphore_compatibility_t(IREE_API_PTR* query_semaphore_compatibility)(iree_hal_device_t* device, iree_hal_semaphore_t* semaphore);
  .query_queue_pool_backend = strela_device_query_queue_pool_backend,
  .queue_alloca             = strela_device_queue_alloca,
  .queue_dealloca           = strela_device_queue_dealloca,
  .queue_fill               = strela_device_queue_fill,
  .queue_update             = strela_device_queue_update,
  .queue_copy               = strela_device_queue_copy,
  .queue_read               = strela_device_queue_read,
  .queue_write              = strela_device_queue_write,
  .queue_host_call          = strela_device_queue_host_call,
  .queue_dispatch           = strela_device_queue_dispatch,
  .queue_execute            = strela_device_queue_execute,
  .queue_flush              = strela_device_queue_flush,
  .profiling_begin          = strela_device_profiling_begin,
  .profiling_flush          = strela_device_profiling_flush,
  .profiling_end            = strela_device_profiling_end,
  .external_capture_begin   = strela_device_external_capture_begin,
  .external_capture_end     = strela_device_external_capture_end,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_allocator_create(
  iree_allocator_t host_allocator,
  strela_dev *dev,
  iree_hal_allocator_t** out_allocator
) {

  strela_allocator_t *allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *allocator, (void**)&allocator));
  memset(allocator, 0, sizeof *allocator);

  // Initialize the resource tracking and map the vtable
  iree_hal_resource_initialize(&strela_allocator_vtable, &allocator->resource);

  allocator->dev = dev;
  allocator->host_allocator = host_allocator;

  *out_allocator = (iree_hal_allocator_t*)allocator;
  return iree_ok_status();
}

typedef struct iree_hal_strela_device_options_t {
  int reserved;
} iree_hal_strela_device_options_t;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_strela_device_options_t options;

  // add stuff here
} strela_driver_t;

static iree_status_t
strela_driver_create_device_by_id(
  iree_hal_driver_t* base_driver,
  iree_hal_device_id_t device_id,
  iree_host_size_t param_count,
  const iree_string_pair_t* params,
  const iree_hal_device_create_params_t *device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t** out_device
) {
  printf("%s\n", __func__);

  [[maybe_unused]] strela_driver_t* driver = (strela_driver_t*)base_driver;

  // 1. Allocate your custom strela_device_t
  strela_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *device, (void **)&device));
  memset(device, 0, sizeof *device);

  // 2. Initialize the basics
  iree_hal_resource_initialize(&strela_device_vtable, &device->resource);
  iree_string_view_t identifier = iree_string_view_literal("strela-fpga-0");
  device->host_allocator = host_allocator;
  device->identifier = identifier;

  // 3. Initialize your custom allocator HERE. The device owns it.
  IREE_RETURN_IF_ERROR(strela_allocator_create(host_allocator, device->dev, &device->device_allocator));

  *out_device = (iree_hal_device_t*)device;
  return iree_ok_status();
}

static void
strela_driver_destroy(iree_hal_driver_t *base_driver) {
  strela_driver_t *driver = (strela_driver_t *)base_driver;
  iree_allocator_t host_allocator = driver->host_allocator;

  // Perform any additional driver teardown here if necessary

  iree_allocator_free(host_allocator, driver);
}

static iree_status_t
strela_driver_query_available_devices(
  iree_hal_driver_t *base_driver,
  iree_allocator_t host_allocator,
  iree_host_size_t *out_device_info_count,
  iree_hal_device_info_t **out_device_infos
) {
  printf("%s\n", __func__);

  // We only have 1 FPGA device to expose
  *out_device_info_count = 1;

  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *device_infos * 1, (void**)&device_infos));

  iree_string_view_t name = iree_string_view_literal("strela-fpga-0");
  // Populate the dummy device info
  device_infos[0].device_id = 0; // Just use ID 0 for the first device
  device_infos[0].name = name;

  *out_device_infos = device_infos;
  return iree_ok_status();
}

static iree_status_t
strela_driver_dump_device_info(
  iree_hal_driver_t *base_driver,
  iree_hal_device_id_t device_id,
  iree_string_builder_t* builder
) {
  printf("%s\n", __func__);

  return iree_string_builder_append_cstring(builder, "STRELA Custom FPGA Accelerator\n");
}

static iree_status_t
strela_driver_create_device_by_path(
  iree_hal_driver_t *base_driver,
  iree_string_view_t driver_name,
  iree_string_view_t device_path,
  iree_host_size_t param_count,
  const iree_string_pair_t *params,
  const iree_hal_device_create_params_t * device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t **out_device
) {
  printf(
    "%s: \"%.*s\" \"%.*s\"\n",
    __func__,
    (int)driver_name.size, driver_name.data,
    (int)device_path.size, device_path.data
  );
  // return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);

  // Fall back to creating by ID (ID 0)
  return strela_driver_create_device_by_id(
   base_driver, 0, param_count, params,
   NULL, host_allocator, out_device
  );
}

static const iree_hal_driver_vtable_t
strela_driver_vtable = {
  .destroy = strela_driver_destroy,
  .query_available_devices = strela_driver_query_available_devices,
  .dump_device_info = strela_driver_dump_device_info,
  .create_device_by_id = strela_driver_create_device_by_id,
  .create_device_by_path = strela_driver_create_device_by_path,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_driver_factory_enumerate(
  void* self,
  iree_host_size_t* out_driver_info_count,
  const iree_hal_driver_info_t** out_driver_infos
) {
  printf("%s\n", __func__);

  static const iree_hal_driver_info_t driver_info = {
    .driver_name = iree_string_view_literal("strela"),
    .full_name = iree_string_view_literal("STRELA FPGA SoC Accelerator"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = (iree_hal_driver_info_t*)&driver_info;
  return iree_ok_status();
}

static iree_status_t
strela_driver_factory_try_create(
  void* self,
  iree_string_view_t driver_name,
  iree_allocator_t host_allocator,
  iree_hal_driver_t** out_driver
) {
  printf("%s\n", __func__);

  iree_string_view_t strela_name = iree_string_view_literal("strela");
  if (!iree_string_view_equal(driver_name, strela_name)) {
    return iree_make_status(
      IREE_STATUS_UNAVAILABLE, "no driver '%.*s' found",
      (int)driver_name.size, driver_name.data
    );
  }

  strela_driver_t *driver = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *driver, (void**)&driver));
  memset(driver, 0, sizeof *driver);
  iree_hal_resource_initialize(&strela_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  driver->identifier = strela_name;

  *out_driver = (iree_hal_driver_t *)driver;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_my_driver_module_register(iree_hal_driver_registry_t *registry) {
  printf("%s\n", __func__);

  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = strela_driver_factory_enumerate,
      .try_create = strela_driver_factory_try_create,
  };

  return iree_hal_driver_registry_register_factory(registry, &factory);
}
////////////////////////////////////////////////////////////////////////////////
