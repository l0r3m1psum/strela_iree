typedef struct iree_hal_strela_executable_cache_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} iree_hal_strela_executable_cache_t;

static const iree_hal_executable_cache_vtable_t iree_hal_strela_executable_cache_vtable;

static iree_hal_strela_executable_cache_t *
iree_hal_strela_executable_cache_cast(iree_hal_executable_cache_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_executable_cache_vtable);
  return (iree_hal_strela_executable_cache_t *)base_value;
}

static iree_status_t
iree_hal_strela_executable_cache_create(
  iree_string_view_t identifier,
  iree_allocator_t host_allocator,
  iree_hal_executable_cache_t **out_executable_cache
) {
  iree_status_t status = iree_ok_status();
  iree_hal_strela_executable_cache_t* executable_cache = NULL;

  status = iree_allocator_malloc(
    host_allocator, sizeof *executable_cache, (void **)&executable_cache
  );

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
      &iree_hal_strela_executable_cache_vtable, &executable_cache->resource
    );
    executable_cache->host_allocator = host_allocator;
  }

  if (!iree_status_is_ok(status) && executable_cache) {
    iree_hal_executable_cache_release(
      (iree_hal_executable_cache_t *)executable_cache
    );
  }

  *out_executable_cache = (iree_hal_executable_cache_t*)executable_cache;
  return status;
}


static void
iree_hal_strela_executable_cache_destroy(iree_hal_executable_cache_t *base_executable_cache) {
  iree_hal_strela_executable_cache_t* executable_cache = iree_hal_strela_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->host_allocator;

  iree_allocator_free(host_allocator, executable_cache);
}

static iree_status_t
iree_hal_strela_executable_cache_infer_format(
  iree_hal_executable_cache_t *base_executable_cache,
  iree_hal_executable_caching_mode_t caching_mode,
  iree_const_byte_span_t executable_data,
  iree_host_size_t executable_format_capacity,
  char *executable_format,
  iree_host_size_t *out_inferred_size
) {
  iree_hal_strela_executable_cache_t* executable_cache = iree_hal_strela_executable_cache_cast(base_executable_cache);

  (void)executable_cache;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static bool
iree_hal_strela_executable_cache_can_prepare_format(
  iree_hal_executable_cache_t *base_executable_cache,
  iree_hal_executable_caching_mode_t caching_mode,
  iree_string_view_t executable_format
) {
  printf("%s\n", __func__);
  iree_hal_strela_executable_cache_t* executable_cache = iree_hal_strela_executable_cache_cast(base_executable_cache);

  (void)executable_cache;

  iree_string_view_t custom = iree_string_view_literal("custom");
  return iree_string_view_equal(executable_format, custom);
}

static iree_status_t
iree_hal_strela_executable_cache_prepare_executable(
  iree_hal_executable_cache_t *base_executable_cache,
  const iree_hal_executable_params_t *executable_params,
  iree_hal_executable_t **out_executable
) {
  printf("%s\n", __func__);
  iree_hal_strela_executable_cache_t* executable_cache = iree_hal_strela_executable_cache_cast(base_executable_cache);

  (void)executable_cache;

  return iree_hal_strela_executable_create(
    executable_params, executable_cache->host_allocator, out_executable
  );
}

static const iree_hal_executable_cache_vtable_t
iree_hal_strela_executable_cache_vtable = {
  .destroy = iree_hal_strela_executable_cache_destroy,
  .infer_format = iree_hal_strela_executable_cache_infer_format,
  .can_prepare_format = iree_hal_strela_executable_cache_can_prepare_format,
  .prepare_executable = iree_hal_strela_executable_cache_prepare_executable,
};
