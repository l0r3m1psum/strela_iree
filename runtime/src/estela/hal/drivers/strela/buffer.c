typedef struct iree_hal_strela_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;

  strela_dev *dev;
  strela_buffer s_buf;
  void *host_ptr; // NOTE: I don't think that this is needed since STRELA does not have a separate virtual memory

  iree_hal_buffer_release_callback_t release_callback;
} iree_hal_strela_buffer_t;

static const iree_hal_buffer_vtable_t iree_hal_strela_buffer_vtable;

// Whether |base_value| itself is a STRELA buffer. A subspan of one is not: ask
// iree_hal_buffer_allocated_buffer() first.
static bool
iree_hal_strela_buffer_isa(const iree_hal_buffer_t *base_value) {
  return iree_hal_resource_is(base_value, &iree_hal_strela_buffer_vtable);
}

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

// TODO: rename to iree_hal_strela_buffer_create
static iree_status_t
iree_hal_strela_buffer_wrap(
  iree_hal_buffer_placement_t placement,
  iree_hal_memory_type_t memory_type,
  iree_hal_memory_access_t allowed_access,
  iree_hal_buffer_usage_t allowed_usage,
  iree_device_size_t allocation_size,
  iree_device_size_t byte_offset,
  iree_device_size_t byte_length,

  strela_dev *dev,
  strela_buffer s_buf,
  void *host_ptr,

  iree_hal_buffer_release_callback_t release_callback,
  iree_allocator_t host_allocator,
  iree_hal_buffer_t **out_buffer
) {
  TRACE_FUNC;
  iree_status_t status = iree_ok_status();
  iree_hal_strela_buffer_t* buffer = NULL;
  iree_hal_buffer_t *buffer_base = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
      host_allocator, sizeof *buffer, (void **)&buffer
    );
  }

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
    buffer->dev = dev;
    buffer->s_buf = s_buf;
    buffer->host_ptr = host_ptr;
    buffer_base = &buffer->base;
  }

  if (!iree_status_is_ok(status)) {
    if (buffer_base) {
      iree_hal_buffer_release(buffer_base);
    }
  }

  *out_buffer = buffer_base;
  return status;
}

static void
iree_hal_strela_buffer_destroy(iree_hal_buffer_t *base_buffer) {
  TRACE_FUNC;
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
  TRACE_FUNC;

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
  TRACE_FUNC;
  return iree_ok_status();
}

// Given a range [byte_offset,byte_offset+byte_length) it rounds down the
// start and rounds up the end so that the range size is a multiple of the cache
// cache size and it is aligned to it.
//  <------[offset,offset+length)------->
// |-------------------------------------|-------------------------------------|
static void
iree_hal_strela_buffer_lines(
  iree_device_size_t byte_offset,
  iree_device_size_t byte_length,
  size_t *out_offset_words,
  size_t *out_count_words
) {
  iree_device_size_t start = byte_offset & ~(iree_device_size_t)(IREE_HAL_STRELA_CACHE_LINE - 1);
  iree_device_size_t end = iree_device_align(byte_offset + byte_length, IREE_HAL_STRELA_CACHE_LINE);
  *out_offset_words = start / sizeof(strela_word);
  *out_count_words = (end - start) / sizeof(strela_word);
}

// Makes what the device wrote visible to the CPU. The lines at the edges may
// also hold CPU writes to the same buffer outside the range, so they are
// cleaned first and nothing the CPU wrote is lost.
static iree_status_t
iree_hal_strela_buffer_invalidate_range(
  iree_hal_buffer_t *base_buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length
) {
  TRACE_FUNC;
  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_cast(base_buffer);
  size_t offset = 0, count = 0;

  iree_hal_strela_buffer_lines(local_byte_offset, local_byte_length, &offset, &count);
  strela_buffer_flush(buffer->dev, buffer->s_buf, offset, count);
  strela_buffer_inval(buffer->dev, buffer->s_buf, offset, count);
  return iree_hal_strela_status_from_dev(buffer->dev, "invalidating a STRELA buffer");
}

// Makes what the CPU wrote visible to the device.
static iree_status_t
iree_hal_strela_buffer_flush_range(
  iree_hal_buffer_t *base_buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length
) {
  TRACE_FUNC;
  iree_hal_strela_buffer_t *buffer = iree_hal_strela_buffer_cast(base_buffer);
  size_t offset = 0, count = 0;

  iree_hal_strela_buffer_lines(local_byte_offset, local_byte_length, &offset, &count);
  strela_buffer_flush(buffer->dev, buffer->s_buf, offset, count);
  return iree_hal_strela_status_from_dev(buffer->dev, "flushing a STRELA buffer");
}

// The STRELA buffer behind |base_buffer| and the word its byte |offset| falls
// on, or NULL if the memory is not STRELA's: the CGRA cannot reach anything
// else.
static iree_hal_strela_buffer_t *
iree_hal_strela_buffer_resolve(
  iree_hal_buffer_t *base_buffer,
  iree_device_size_t offset,
  iree_device_size_t *out_allocated_offset
) {
  iree_hal_buffer_t *allocated = iree_hal_buffer_allocated_buffer(base_buffer);
  if (!iree_hal_strela_buffer_isa(allocated)) {
    return NULL;
  }
  *out_allocated_offset = iree_hal_buffer_byte_offset(base_buffer) + offset;
  return iree_hal_strela_buffer_cast(allocated);
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
