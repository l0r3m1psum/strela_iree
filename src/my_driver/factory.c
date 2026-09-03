static iree_status_t
iree_hal_strela_driver_factory_enumerate(
  void *self,
  iree_host_size_t *out_driver_info_count,
  const iree_hal_driver_info_t **out_driver_infos
) {
  printf("%s\n", __func__);

  static const iree_hal_driver_info_t driver_info = {
    .driver_name = IREE_SVL("strela"),
    .full_name = IREE_SVL("STRELA FPGA SoC Accelerator"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = (iree_hal_driver_info_t *)&driver_info;
  return iree_ok_status();
}

static iree_status_t
iree_hal_strela_driver_factory_try_create(
  void *self,
  iree_string_view_t driver_name,
  iree_allocator_t host_allocator,
  iree_hal_driver_t **out_driver
) {
  printf("%s\n", __func__);

  iree_string_view_t iree_hal_strela_name = IREE_SVL("strela");
  if (!iree_string_view_equal(driver_name, iree_hal_strela_name)) {
    return iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "no driver '%.*s' is provided by this factory",
      (int)driver_name.size, driver_name.data
    );
  }

  {
    iree_hal_strela_driver_t *driver = NULL;
    iree_host_size_t total_size = sizeof *driver + driver_name.size;
    IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void **)&driver)
    );
    iree_hal_resource_initialize(&iree_hal_strela_driver_vtable, &driver->resource);
    driver->host_allocator = host_allocator;
    iree_string_view_append_to_buffer(
      driver_name, &driver->identifier,
      (char *)driver + total_size - driver_name.size
    );

    *out_driver = (iree_hal_driver_t *)driver;
  }

  return iree_ok_status();
}

static const iree_hal_driver_factory_t
factory = {
  .self = NULL,
  .enumerate = iree_hal_strela_driver_factory_enumerate,
  .try_create = iree_hal_strela_driver_factory_try_create,
};
