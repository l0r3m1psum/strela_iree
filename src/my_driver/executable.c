typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // TODO: Add fields here later to store your parsed STRELA binaries or kernel parameters
} iree_hal_strela_executable_t;

static const iree_hal_executable_vtable_t iree_hal_strela_executable_vtable;

static iree_hal_strela_executable_t *
iree_hal_strela_executable_cast(iree_hal_executable_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_executable_vtable);
  return (iree_hal_strela_executable_t *)base_value;
}

iree_status_t iree_hal_strela_executable_create(
  const iree_hal_executable_params_t* executable_params,
  iree_allocator_t host_allocator,
  iree_hal_executable_t **out_executable
) {
  iree_status_t status = iree_ok_status();
  iree_hal_strela_executable_t *executable = NULL;

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
      host_allocator, sizeof *executable, (void **)&executable
    );
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(
      &iree_hal_strela_executable_vtable, &executable->resource
    );
    executable->host_allocator = host_allocator;
  }

  if (!iree_status_is_ok(status)) {
    if (executable) {
      iree_hal_executable_destroy((iree_hal_executable_t *)executable);
    }
  }

  *out_executable = (iree_hal_executable_t *)executable;
  return status;
}

static void
iree_hal_strela_executable_destroy(iree_hal_executable_t *base_executable) {
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->host_allocator;

  iree_allocator_free(host_allocator, executable);
}

static iree_host_size_t
iree_hal_strela_executable_function_count(iree_hal_executable_t *base_executable) {
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);

  (void)executable;

  return 0;
}

static iree_status_t
iree_hal_strela_executable_function_info(
  iree_hal_executable_t *base_executable,
  iree_hal_executable_function_t function,
  iree_hal_executable_function_info_t *out_info
) {
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);

  (void)executable;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_executable_function_parameters(
  iree_hal_executable_t *base_executable,
  iree_hal_executable_function_t function,
  iree_host_size_t capacity,
  iree_hal_executable_function_parameter_t* out_parameters
) {
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);

  (void)executable;

  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
iree_hal_strela_executable_lookup_function_by_name(
  iree_hal_executable_t *base_executable,
  iree_string_view_t name,
  iree_hal_executable_function_t *out_function
) {
  printf("%s: looking up kernel '%.*s'\n", __func__, (int)name.size, name.data);
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);

  (void)executable;

  out_function->value = 1;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_executable_lookup_global_by_name(
  iree_hal_executable_t *base_executable,
  iree_string_view_t name,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_buffer_t **out_buffer
) {
  iree_hal_strela_executable_t *executable = iree_hal_strela_executable_cast(base_executable);

  (void)executable;

  *out_buffer = NULL;
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

// Map the stubs to the vtable
static const iree_hal_executable_vtable_t
iree_hal_strela_executable_vtable = {
  .destroy                 = iree_hal_strela_executable_destroy,
  .function_count          = iree_hal_strela_executable_function_count,
  .function_info           = iree_hal_strela_executable_function_info,
  .function_parameters     = iree_hal_strela_executable_function_parameters,
  .lookup_function_by_name = iree_hal_strela_executable_lookup_function_by_name,
  .lookup_global_by_name   = iree_hal_strela_executable_lookup_global_by_name,
};
