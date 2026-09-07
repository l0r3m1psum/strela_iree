typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)

  strela_dev *dev; // TODO: does this goes here?
} iree_hal_strela_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_strela_allocator_vtable;

static iree_hal_strela_allocator_t *
iree_hal_strela_allocator_cast(iree_hal_allocator_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_allocator_vtable);
  return (iree_hal_strela_allocator_t *)base_value;
}

static const iree_hal_strela_allocator_t *
iree_hal_strela_allocator_const_cast(const iree_hal_allocator_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_allocator_vtable);
  return (iree_hal_strela_allocator_t *)base_value;
}

static iree_status_t
iree_hal_strela_allocator_create(
  iree_allocator_t host_allocator,
  iree_hal_allocator_t **out_allocator
) {
  iree_status_t status = iree_ok_status();

    // NOTE: can I reach the device that created this allocator and take it from there?
  strela_dev *dev = strela_dev_init(0);
  if (!strela_dev_ok(dev)) {
    status = iree_make_status(IREE_STATUS_FAILED_PRECONDITION, "Unable to initialize STRELA");
  }

  iree_hal_strela_allocator_t *allocator = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
      host_allocator, sizeof *allocator, (void **)&allocator
    );
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
      &iree_hal_strela_allocator_vtable, &allocator->resource
    );
    allocator->host_allocator = host_allocator;
    allocator->dev = dev;
  }

  if (!iree_status_is_ok(status) && allocator) {
    strela_dev_deinit(allocator->dev);
    iree_hal_allocator_release((iree_hal_allocator_t *)allocator);
  }

  *out_allocator = (iree_hal_allocator_t *)allocator;

  return status;
}

static void
iree_hal_strela_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);
  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t
iree_hal_strela_allocator_host_allocator(
  const iree_hal_allocator_t *base_allocator
) {
  const iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_const_cast(base_allocator);
  return allocator->host_allocator;
}

static iree_status_t
iree_hal_strela_allocator_trim(iree_hal_allocator_t *base_allocator) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_ok_status();
}

static void
iree_hal_strela_allocator_query_statistics(
  iree_hal_allocator_t *base_allocator,
  iree_hal_allocator_statistics_t *out_statistics
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);
  memcpy(out_statistics, &allocator->statistics, sizeof *out_statistics);
}

static iree_status_t
iree_hal_strela_allocator_query_memory_heaps(
  iree_hal_allocator_t *base_allocator,
  iree_host_size_t capacity,
  iree_hal_allocator_memory_heap_t *heaps,
  iree_host_size_t *out_count
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_hal_buffer_compatibility_t
iree_hal_strela_allocator_query_buffer_compatibility(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_params_t *params,
  iree_device_size_t *allocation_size
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params->access = IREE_HAL_MEMORY_ACCESS_ALL;
  params->usage |= IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING;

  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE
    | IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER
    | IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH
  ;
}

static iree_status_t
iree_hal_strela_allocator_allocate_buffer(
  iree_hal_allocator_t *base_allocator,
  const iree_hal_buffer_params_t *params,
  iree_device_size_t allocation_size,
  iree_hal_buffer_t **out_buffer
) {
  printf("%s\n", __func__);
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);
  iree_status_t status = iree_ok_status();

  iree_hal_buffer_params_t compat_params = *params;
  iree_hal_buffer_compatibility_t compatibility =
    iree_hal_strela_allocator_query_buffer_compatibility(
      base_allocator, &compat_params, &allocation_size
  );
  if (!iree_all_bits_set(compatibility, IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE)) {
    status = iree_make_status(
      IREE_STATUS_INVALID_ARGUMENT,
      "allocator cannot allocate a buffer with the given parameters"
    );
  }

  strela_buffer s_buf = {0};
  void *host_ptr = NULL;
  if (iree_status_is_ok(status)) {
    s_buf = strela_buffer_alloc(allocator->dev, allocation_size);
    if (s_buf.valid) {
      host_ptr = strela_buffer_to_ptr(allocator->dev, s_buf);
    } else {
      status = iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION, "Unable to allocate STRELA buffer"
      );
    }
  }

  iree_hal_buffer_t *buffer = NULL;
  if (iree_status_is_ok(status)) {
    iree_hal_memory_type_t actual_type = params->type | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    iree_hal_memory_access_t actual_access = IREE_HAL_MEMORY_ACCESS_ALL;
    iree_hal_buffer_usage_t actual_usage = params->usage | IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING;

    status = iree_hal_strela_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      actual_type,
      actual_access,
      actual_usage,
      allocation_size,
      /*byte_offset=*/0,
      allocation_size,
      s_buf,
      host_ptr,
      (iree_hal_buffer_release_callback_t){
        iree_hal_strela_buffer_release_fn, allocator->dev
      },
      allocator->host_allocator,
      &buffer
    );
  }

  if (!iree_status_is_ok(status)) {
    if (s_buf.valid) {
      strela_buffer_free(allocator->dev, s_buf);
    }
    if (buffer) {
      iree_hal_buffer_release(buffer);
    }
  }

  *out_buffer = buffer;
  return status;
}

static void
iree_hal_strela_allocator_deallocate_buffer(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *base_buffer
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  iree_hal_buffer_destroy(base_buffer);
}

static iree_status_t
iree_hal_strela_allocator_import_buffer(
  iree_hal_allocator_t *base_allocator,
  const iree_hal_buffer_params_t *params,
  iree_hal_external_buffer_t *external_buffer,
  iree_hal_buffer_release_callback_t release_callback,
  iree_hal_buffer_t **out_buffer
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_export_buffer(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *buffer,
  iree_hal_external_buffer_type_t requested_type,
  iree_hal_external_buffer_flags_t requested_flags,
  iree_hal_external_buffer_t *out_external_buffer
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static bool
iree_hal_strela_allocator_supports_virtual_memory(
  iree_hal_allocator_t *base_allocator
) {
  printf("%s\n", __func__);
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return false;
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_query_granularity(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_params_t params,
  iree_device_size_t *IREE_RESTRICT out_minimum_page_size,
  iree_device_size_t *IREE_RESTRICT out_recommended_page_size
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  *out_minimum_page_size = 0;
  *out_recommended_page_size = 0;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_reserve(
  iree_hal_allocator_t *base_allocator,
  iree_hal_queue_affinity_t queue_affinity,
  iree_device_size_t size,
  iree_hal_buffer_t **out_virtual_buffer
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  *out_virtual_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_release(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *virtual_buffer
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_physical_memory_allocate(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_params_t params,
  iree_device_size_t size,
  iree_allocator_t host_allocator,
  iree_hal_physical_memory_t **out_physical_memory
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  *out_physical_memory = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_physical_memory_free(
  iree_hal_allocator_t *base_allocator,
  iree_hal_physical_memory_t *physical_memory
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_map(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *virtual_buffer,
  iree_device_size_t virtual_offset,
  iree_hal_physical_memory_t *physical_memory,
  iree_device_size_t physical_offset,
  iree_device_size_t size
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_unmap(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *virtual_buffer,
  iree_device_size_t virtual_offset,
  iree_device_size_t size
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_protect(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *virtual_buffer,
  iree_device_size_t virtual_offset,
  iree_device_size_t size,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_memory_protection_t protection
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_allocator_virtual_memory_advise(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *virtual_buffer,
  iree_device_size_t virtual_offset, iree_device_size_t size,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_memory_advice_t advice
) {
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

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
