typedef struct iree_hal_strela_driver_options_t {
  int reserved;
} iree_hal_strela_driver_options_t;

static void
iree_hal_strela_driver_options_initialize(
  iree_hal_strela_driver_options_t *out_options
) {
  memset(out_options, 0, sizeof *out_options);
}

static iree_status_t
iree_hal_strela_driver_options_verify(
  const iree_hal_strela_driver_options_t *options
) {
  iree_status_t status = iree_ok_status();

  if (!is_all_zero(options, sizeof *options)) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
  }

  return status;
}

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_strela_driver_options_t options;

  // + trailing identifier string storage
} iree_hal_strela_driver_t;

static const iree_hal_driver_vtable_t iree_hal_strela_driver_vtable;

static iree_hal_strela_driver_t *
iree_hal_strela_driver_cast(iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_strela_driver_vtable);
  return (iree_hal_strela_driver_t *)base_value;
}

static iree_status_t
iree_hal_strela_driver_create(
  iree_string_view_t identifier,
  const iree_hal_strela_driver_options_t *options,
  iree_allocator_t host_allocator,
  iree_hal_driver_t **out_driver
) {
  iree_status_t status = iree_ok_status();

  status = iree_hal_strela_driver_options_verify(options);

  iree_hal_strela_driver_t *driver = NULL;
  iree_host_size_t total_size = sizeof *driver + identifier.size;

  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(host_allocator, total_size, (void **)&driver);
  }

  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_strela_driver_vtable, &driver->resource);
    driver->host_allocator = host_allocator;
    iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char *)driver + total_size - identifier.size
    );

    memcpy(&driver->options, options, sizeof *options);
  }

  if (!iree_status_is_ok(status) && driver) {
    iree_hal_driver_release((iree_hal_driver_t *)driver);
  }

  *out_driver = (iree_hal_driver_t *)driver;

  return status;
}

static void
iree_hal_strela_driver_destroy(iree_hal_driver_t *base_driver) {
  iree_hal_strela_driver_t *driver = iree_hal_strela_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;

  // NOTE: if the driver loaded any libraries they should be closed here.

  iree_allocator_free(host_allocator, driver);
}

static iree_status_t
iree_hal_strela_driver_query_available_devices(
  iree_hal_driver_t *base_driver,
  iree_allocator_t host_allocator,
  iree_host_size_t *out_device_info_count,
  iree_hal_device_info_t **out_device_infos
) {
  printf("%s\n", __func__);

  iree_status_t status = iree_ok_status();

  unsigned count = 0;
  if (strela_device_count(&count) == -1) {
    status = iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }

  if (iree_status_is_ok(status)) {
    // TODO: is count != 0 we should return all the available devices.
    static const iree_hal_device_info_t device_infos[1] = {
      {
        .device_id = 0,
        .name = iree_string_view_literal("default_strela"),
      },
    };
    status = iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(device_infos, sizeof device_infos),
      (void **)out_device_infos
    );
    *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  }

  return status;
}

static iree_status_t
iree_hal_strela_driver_dump_device_info(
  iree_hal_driver_t *base_driver,
  iree_hal_device_id_t device_id,
  iree_string_builder_t *builder
) {
  printf("%s\n", __func__);
  iree_hal_strela_driver_t *driver = iree_hal_strela_driver_cast(base_driver);

  (void)driver;

  return iree_string_builder_append_cstring(builder, "STRELA Custom FPGA Accelerator\n");
}

static iree_status_t
iree_hal_strela_driver_create_device_by_id(
  iree_hal_driver_t *base_driver,
  iree_hal_device_id_t device_id,
  iree_host_size_t param_count,
  const iree_string_pair_t *params,
  const iree_hal_device_create_params_t *device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t **out_device
) {
  printf("%s\n", __func__);
  iree_hal_strela_driver_t *driver = iree_hal_strela_driver_cast(base_driver);

  iree_hal_strela_device_options_t options;
  iree_hal_strela_device_options_initialize(&options);

  (void)driver;

  return iree_hal_strela_device_create(
    driver->identifier, &options, device_create_params, host_allocator, out_device
  );
}

static iree_status_t
iree_hal_strela_driver_create_device_by_path(
  iree_hal_driver_t *base_driver,
  iree_string_view_t driver_name,
  iree_string_view_t device_path,
  iree_host_size_t param_count,
  const iree_string_pair_t *params,
  const iree_hal_device_create_params_t *device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t **out_device
) {
  printf(
    "%s: \"%.*s\" \"%.*s\"\n",
    __func__,
    (int)driver_name.size, driver_name.data,
    (int)device_path.size, device_path.data
  );
  // return iree_make_status(IREE_STATUS_UNIMPLEMENTED, __func__);

  // Fall back to creating by ID (ID 0)
  return iree_hal_strela_driver_create_device_by_id(
    base_driver, 0, param_count, params,
    device_create_params, host_allocator, out_device
  );
}

static const iree_hal_driver_vtable_t
iree_hal_strela_driver_vtable = {
  .destroy = iree_hal_strela_driver_destroy,
  .query_available_devices = iree_hal_strela_driver_query_available_devices,
  .dump_device_info = iree_hal_strela_driver_dump_device_info,
  .create_device_by_id = iree_hal_strela_driver_create_device_by_id,
  .create_device_by_path = iree_hal_strela_driver_create_device_by_path,
};
