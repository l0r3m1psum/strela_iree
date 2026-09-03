typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  strela_dev *dev; // TODO: does this goes here?
} iree_hal_strela_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_strela_allocator_vtable;

static iree_hal_strela_allocator_t *
iree_hal_strela_allocator_cast(iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_allocator_vtable);
  return (iree_hal_strela_allocator_t *)base_value;
}

static iree_status_t
iree_hal_strela_allocator_create(
  iree_allocator_t host_allocator,
  iree_hal_allocator_t **out_allocator
) {
  iree_status_t status = iree_ok_status();

  iree_hal_strela_allocator_t* allocator = NULL;

  status = iree_allocator_malloc(
    host_allocator, sizeof *allocator, (void **)&allocator
  );

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
      &iree_hal_strela_allocator_vtable, &allocator->resource
    );
    allocator->host_allocator = host_allocator;
  }

  if (!iree_status_is_ok(status) && allocator) {
    iree_hal_allocator_release((iree_hal_allocator_t *)allocator);
  }

  *out_allocator = (iree_hal_allocator_t *)allocator;

  return status;
}

static iree_status_t
iree_hal_strela_allocator_allocate_buffer(
  iree_hal_allocator_t *IREE_RESTRICT base_allocator,
  const iree_hal_buffer_params_t *IREE_RESTRICT  params,
  iree_device_size_t allocation_size,
  iree_hal_buffer_t **IREE_RESTRICT out_buffer
) {
  printf("%s\n", __func__);

  iree_hal_strela_allocator_t *allocator = (iree_hal_strela_allocator_t *)base_allocator;

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
  iree_hal_strela_buffer_t *buffer = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(allocator->host_allocator, sizeof(*buffer), (void**)&buffer));

  // 2. Save your hardware state and allocator config
  buffer->host_allocator = allocator->host_allocator;
  buffer->s_buf = s_buf;
  buffer->host_ptr = host_ptr;
  buffer->release_callback = (iree_hal_buffer_release_callback_t){0};

  iree_hal_memory_type_t actual_type = params->type | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  iree_hal_memory_access_t actual_access = IREE_HAL_MEMORY_ACCESS_ALL;
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
    &iree_hal_strela_buffer_vtable,            // Your custom vtable
    &buffer->base                     // Output pointer to the base iree_hal_buffer_t
  );

  // 4. Pass the initialized buffer back to the VM
  *out_buffer = &buffer->base;

  return iree_ok_status();
}

static void
iree_hal_strela_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  [[maybe_unused]] iree_hal_strela_allocator_t *allocator = (iree_hal_strela_allocator_t*)base_allocator;

  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t
iree_hal_strela_allocator_host_allocator(
  const iree_hal_allocator_t *base_allocator
) {
  printf("%s base_allocator: 0x%p\n", __func__, base_allocator);
  return ((iree_hal_strela_allocator_t *)base_allocator)->host_allocator;
}

static iree_status_t
iree_hal_strela_allocator_trim(
  iree_hal_allocator_t *base_allocator
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
  // return iree_ok_status(); // No-op
}

static void
iree_hal_strela_allocator_deallocate_buffer(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *base_buffer
) {
  // We will implement this later when you handle actual memory freeing
}

static void
iree_hal_strela_allocator_query_statistics(
  iree_hal_allocator_t *base_allocator,
  iree_hal_allocator_statistics_t *out_statistics
) {
  memset(out_statistics, 0, sizeof *out_statistics);
}

static iree_status_t
iree_hal_strela_allocator_query_memory_heaps(
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
iree_hal_strela_allocator_query_buffer_compatibility(
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
iree_hal_strela_allocator_import_buffer(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  const iree_hal_buffer_params_t *IREE_RESTRICT params,
  iree_hal_external_buffer_t *IREE_RESTRICT external_buffer,
  iree_hal_buffer_release_callback_t release_callback,
  iree_hal_buffer_t **IREE_RESTRICT out_buffer
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_export_buffer(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_t *IREE_RESTRICT buffer,
  iree_hal_external_buffer_type_t requested_type,
  iree_hal_external_buffer_flags_t requested_flags,
  iree_hal_external_buffer_t *IREE_RESTRICT out_external_buffer
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static bool
iree_hal_strela_allocator_supports_virtual_memory(
  iree_hal_allocator_t *IREE_RESTRICT allocator
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_query_granularity(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_params_t params,
  iree_device_size_t *IREE_RESTRICT out_minimum_page_size,
  iree_device_size_t *IREE_RESTRICT out_recommended_page_size
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_reserve(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
  iree_hal_buffer_t **IREE_RESTRICT out_virtual_buffer
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_release(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_t *IREE_RESTRICT virtual_buffer
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_physical_memory_allocate(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_params_t params, iree_device_size_t size,
  iree_allocator_t host_allocator,
  iree_hal_physical_memory_t **IREE_RESTRICT out_physical_memory
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_physical_memory_free(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_physical_memory_t *IREE_RESTRICT physical_memory
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_map(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_t *IREE_RESTRICT virtual_buffer,
  iree_device_size_t virtual_offset,
  iree_hal_physical_memory_t *IREE_RESTRICT physical_memory,
  iree_device_size_t physical_offset, iree_device_size_t size
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_unmap(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_t *IREE_RESTRICT virtual_buffer,
  iree_device_size_t virtual_offset, iree_device_size_t size
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_protect(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_t *IREE_RESTRICT virtual_buffer,
  iree_device_size_t virtual_offset, iree_device_size_t size,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_memory_protection_t protection
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_advise(
  iree_hal_allocator_t *IREE_RESTRICT allocator,
  iree_hal_buffer_t *IREE_RESTRICT virtual_buffer,
  iree_device_size_t virtual_offset, iree_device_size_t size,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_memory_advice_t advice
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_allocator_vtable_t
iree_hal_strela_allocator_vtable = {
  .destroy                    = iree_hal_strela_allocator_destroy,
  .host_allocator             = iree_hal_strela_allocator_host_allocator,
  .trim                       = iree_hal_strela_allocator_trim,
  .query_statistics           = iree_hal_strela_allocator_query_statistics,
  .query_memory_heaps         = iree_hal_strela_allocator_query_memory_heaps,
  .query_buffer_compatibility = iree_hal_strela_allocator_query_buffer_compatibility,
  .allocate_buffer            = iree_hal_strela_allocator_allocate_buffer,
  .deallocate_buffer          = iree_hal_strela_allocator_deallocate_buffer,
  .import_buffer              = iree_hal_strela_allocator_import_buffer,
  .export_buffer              = iree_hal_strela_allocator_export_buffer,

  .supports_virtual_memory          = iree_hal_strela_allocator_supports_virtual_memory,
  .virtual_memory_query_granularity = iree_hal_strela_allocator_virtual_memory_query_granularity,
  .virtual_memory_reserve           = iree_hal_strela_allocator_virtual_memory_reserve,
  .virtual_memory_release           = iree_hal_strela_allocator_virtual_memory_release,
  .physical_memory_allocate         = iree_hal_strela_allocator_physical_memory_allocate,
  .physical_memory_free             = iree_hal_strela_allocator_physical_memory_free,
  .virtual_memory_map               = iree_hal_strela_allocator_virtual_memory_map,
  .virtual_memory_unmap             = iree_hal_strela_allocator_virtual_memory_unmap,
  .virtual_memory_protect           = iree_hal_strela_allocator_virtual_memory_protect,
  .virtual_memory_advise            = iree_hal_strela_allocator_virtual_memory_advise,
};
