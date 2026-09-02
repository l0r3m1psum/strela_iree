typedef struct strela_buffer_t {
  iree_hal_buffer_t base;
  iree_allocator_t host_allocator;
  iree_hal_buffer_release_callback_t release_callback;

  strela_buffer s_buf;
  void *host_ptr;
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
strela_buffer_unmap_range(
  iree_hal_buffer_t *buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length,
  iree_hal_buffer_mapping_t *mapping
) {
  printf("%s\n", __func__);
  return iree_ok_status();
}

static iree_status_t
strela_buffer_invalidate_range(
  iree_hal_buffer_t *buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_buffer_flush_range(
  iree_hal_buffer_t *buffer,
  iree_device_size_t local_byte_offset,
  iree_device_size_t local_byte_length
) {
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
