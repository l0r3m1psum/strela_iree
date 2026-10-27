typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)

  strela_dev *dev; // owned by the device
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
  strela_dev *dev,
  iree_allocator_t host_allocator,
  iree_hal_allocator_t **out_allocator
) {
  TRACE_FUNC;
  iree_status_t status = iree_ok_status();
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

  if (!iree_status_is_ok(status)) {
    if (allocator) {
      iree_hal_allocator_release((iree_hal_allocator_t *)allocator);
    }
  }

  *out_allocator = (iree_hal_allocator_t *)allocator;

  return status;
}

static void
iree_hal_strela_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  TRACE_FUNC;
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);
  iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t
iree_hal_strela_allocator_host_allocator(
  const iree_hal_allocator_t *base_allocator
) {
  TRACE_FUNC;
  const iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_const_cast(base_allocator);
  return allocator->host_allocator;
}

static iree_status_t
iree_hal_strela_allocator_trim(iree_hal_allocator_t *base_allocator) {
  TRACE_FUNC;
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_ok_status();
}

static void
iree_hal_strela_allocator_query_statistics(
  iree_hal_allocator_t *base_allocator,
  iree_hal_allocator_statistics_t *out_statistics
) {
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  // HOST_COHERENT is honoured even though the CPU maps STRELA memory cached and
  // the CGRA does not snoop: the CGRA is the only other party, and every
  // dispatch flushes all of its columns before the job and invalidates its
  // outputs after, so the CPU never sees anything stale.
  params->type |= IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
  params->access = IREE_HAL_MEMORY_ACCESS_ALL;
  // libstrela maps all of STRELA memory once, so a persistent mapping costs
  // nothing.
  params->usage |= IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH
    | IREE_HAL_BUFFER_USAGE_MAPPING | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;

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
  TRACE_FUNC;
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);
  iree_status_t status = iree_ok_status();
  strela_buffer s_buf = {0};
  void *host_ptr = NULL;
  iree_hal_buffer_t *buffer = NULL;

  if (iree_status_is_ok(status)) {
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
  }

  if (iree_status_is_ok(status)) {
    // Whole cache lines, so a sync never reaches the allocator's bookkeeping
    // or a neighbour. libstrela counts in words.
    iree_device_size_t padded_size =
      iree_device_align(allocation_size, IREE_HAL_STRELA_CACHE_LINE);
    s_buf = strela_buffer_alloc(allocator->dev, padded_size / sizeof(strela_word));
    status = iree_hal_strela_status_from_dev(allocator->dev, "allocating a STRELA buffer");
  }

  if (iree_status_is_ok(status)) {
    host_ptr = strela_buffer_to_ptr(allocator->dev, s_buf);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_memory_type_t actual_type = params->type | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    iree_hal_memory_access_t actual_access = IREE_HAL_MEMORY_ACCESS_ALL;
    iree_hal_buffer_usage_t actual_usage = params->usage | IREE_HAL_BUFFER_USAGE_TRANSFER
      | IREE_HAL_BUFFER_USAGE_DISPATCH | IREE_HAL_BUFFER_USAGE_MAPPING
      | IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT;

    status = iree_hal_strela_buffer_wrap(
      iree_hal_buffer_placement_undefined(),
      actual_type,
      actual_access,
      actual_usage,
      allocation_size,
      /*byte_offset=*/0,
      allocation_size,
      allocator->dev,
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
  iree_hal_strela_allocator_t *allocator = iree_hal_strela_allocator_cast(base_allocator);

  (void)allocator;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static bool
iree_hal_strela_allocator_supports_virtual_memory(
  iree_hal_allocator_t *base_allocator
) {
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
  TRACE_FUNC;
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
