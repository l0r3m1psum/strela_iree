typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // TODO: Add fields here later to store your parsed STRELA binaries or kernel parameters
} strela_executable_t;

static void
strela_executable_destroy(iree_hal_executable_t *base_executable) {
  strela_executable_t* executable = (strela_executable_t*)base_executable;
  iree_allocator_free(executable->host_allocator, executable);
}

static iree_host_size_t
strela_executable_function_count(
  iree_hal_executable_t *base_executable
) {
  // Stub: Pretend we successfully loaded 1 function
  return 1;
}

static iree_status_t
strela_executable_function_info(
  iree_hal_executable_t *base_executable,
  iree_hal_executable_function_t function,
  iree_hal_executable_function_info_t *out_info
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_executable_function_parameters(
  iree_hal_executable_t* base_executable,
  iree_hal_executable_function_t function, iree_host_size_t capacity,
  iree_hal_executable_function_parameter_t* out_parameters
) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);
}

static iree_status_t
strela_executable_lookup_function_by_name(
  iree_hal_executable_t *base_executable, iree_string_view_t name,
  iree_hal_executable_function_t *out_function
) {
  printf("%s: looking up kernel '%.*s'\n", __func__, (int)name.size, name.data);

  out_function->value = 1;
  return iree_ok_status();
}

static iree_status_t
strela_executable_lookup_global_by_name(
  iree_hal_executable_t *base_executable,
  iree_string_view_t name,
  iree_hal_queue_affinity_t queue_affinity,
  iree_hal_buffer_t **out_buffer
) {
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
