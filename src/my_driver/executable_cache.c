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
