typedef struct iree_hal_strela_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;

  strela_buffer s_buf;
  void *host_ptr; // NOTE: I don't think that this is needed since STRELA does not have a separate virtual memory
} iree_hal_strela_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_strela_buffer_vtable;

static iree_hal_strela_buffer_t *
iree_hal_strela_buffer_cast(iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_buffer_vtable);
  return (iree_hal_strela_buffer_t *)base_value;
}

static const iree_hal_strela_buffer_t *
iree_hal_strela_buffer_const_cast(const iree_hal_buffer_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_buffer_vtable);
  return (const iree_hal_strela_buffer_t *)base_value;
}

static iree_status_t
iree_hal_strela_buffer_wrap(
  iree_hal_buffer_placement_t placement,
  iree_hal_memory_type_t memory_type,
  iree_hal_memory_access_t allowed_access,
  iree_hal_buffer_usage_t allowed_usage,
  iree_device_size_t allocation_size,
  iree_device_size_t byte_offset,
  iree_device_size_t byte_length,

  strela_buffer s_buf,
  void *host_ptr,

  iree_hal_buffer_release_callback_t release_callback,
  iree_allocator_t host_allocator,
  iree_hal_buffer_t **out_buffer
) {
  iree_status_t status = iree_ok_status();

  iree_hal_strela_buffer_t* buffer = NULL;
  iree_hal_buffer_t *buffer_base = NULL;
  status = iree_allocator_malloc(
    host_allocator, sizeof *buffer, (void **)&buffer
  );

  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(
      placement,
      &buffer->base,
      allocation_size,
      byte_offset,
      byte_length,
      memory_type,
      allowed_access,
      allowed_usage,
      &iree_hal_strela_buffer_vtable,
      &buffer->base
    );
    buffer->host_allocator = host_allocator;
    buffer->release_callback = release_callback;
    buffer->s_buf = s_buf;
    buffer->host_ptr = host_ptr;
    buffer_base = &buffer->base;
  }

  if (!iree_status_is_ok(status) && buffer_base) {
    iree_hal_buffer_release(buffer_base);
  }

  *out_buffer = buffer_base;
  return status;
}

static void
iree_hal_strela_buffer_destroy(iree_hal_buffer_t *base_buffer) {
  printf("%s\n", __func__);
  iree_hal_strela_buffer_t* buffer = iree_hal_strela_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = buffer->host_allocator;

  if (buffer->release_callback.fn) {
    buffer->release_callback.fn(buffer->release_callback.user_data,
                                base_buffer);
  }

  iree_allocator_free(host_allocator, buffer);
}

static iree_status_t
iree_hal_strela_buffer_map_range(
  iree_hal_buffer_t *base_buffer,
  iree_hal_mapping_mode_t mapping_mode,
  iree_hal_memory_access_t memory_access,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length,
  iree_hal_buffer_mapping_t *mapping
) {
  printf("%s\n", __func__);

  iree_status_t status = iree_ok_status();
  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_cast(base_buffer);

  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(base_buffer), IREE_HAL_MEMORY_TYPE_HOST_VISIBLE
    );
  }

  if (iree_status_is_ok(status)) {
    iree_hal_buffer_usage_t required_usage
      = mapping_mode == IREE_HAL_MAPPING_MODE_PERSISTENT
        ? IREE_HAL_BUFFER_USAGE_MAPPING_PERSISTENT
        : IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    status = iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(base_buffer), required_usage
    );
  }

  if (iree_status_is_ok(status)) {
    uint8_t *data_ptr = (uint8_t *)buffer->host_ptr + local_byte_offset;
    mapping->contents = iree_make_byte_span(data_ptr, local_byte_length);
  }

  return status;
}

static iree_status_t
iree_hal_strela_buffer_unmap_range(
  iree_hal_buffer_t *buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length,
  iree_hal_buffer_mapping_t *mapping
) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_buffer_invalidate_range(
  iree_hal_buffer_t *buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length
) {
  // TODO: here cache invalidation
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_buffer_flush_range(
  iree_hal_buffer_t *buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static const iree_hal_buffer_vtable_t
iree_hal_strela_buffer_vtable = {
  .recycle          = iree_hal_buffer_recycle,
  .destroy          = iree_hal_strela_buffer_destroy,
  .map_range        = iree_hal_strela_buffer_map_range,
  .unmap_range      = iree_hal_strela_buffer_unmap_range,
  .invalidate_range = iree_hal_strela_buffer_invalidate_range,
  .flush_range      = iree_hal_strela_buffer_flush_range,
};

static void
iree_hal_strela_buffer_release_fn(
  void *user_data, struct iree_hal_buffer_t *base_buffer
) {
  strela_dev *dev = user_data;
  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_cast(base_buffer);

  strela_buffer_free(dev, buffer->s_buf);
}
