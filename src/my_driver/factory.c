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
  iree_status_t status = iree_ok_status();

  if (!iree_string_view_equal(driver_name, IREE_SV("strela"))) {
    status = iree_make_status(
      IREE_STATUS_UNAVAILABLE,
      "no driver '%.*s' is provided by this factory",
      (int)driver_name.size, driver_name.data
    );
  }

  iree_hal_strela_driver_options_t options;
  iree_hal_strela_driver_options_initialize(&options);

  if (iree_status_is_ok(status)) {
    status = iree_hal_strela_driver_create(
      driver_name, &options, host_allocator, out_driver
    );
  }

  return status;
}

static const iree_hal_driver_factory_t
factory = {
  .self = NULL,
  .enumerate = iree_hal_strela_driver_factory_enumerate,
  .try_create = iree_hal_strela_driver_factory_try_create,
};
