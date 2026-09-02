typedef struct iree_hal_strela_device_options_t {
  int reserved;
} iree_hal_strela_device_options_t;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;

  iree_string_view_t identifier;
  iree_hal_strela_device_options_t options;

  // add stuff here
} strela_driver_t;

static iree_status_t
strela_driver_create_device_by_id(
  iree_hal_driver_t *base_driver,
  iree_hal_device_id_t device_id,
  iree_host_size_t param_count,
  const iree_string_pair_t *params,
  const iree_hal_device_create_params_t *device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t **out_device
) {
  printf("%s\n", __func__);

  [[maybe_unused]] strela_driver_t* driver = (strela_driver_t*)base_driver;

  // 1. Allocate your custom strela_device_t
  strela_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *device, (void **)&device));
  memset(device, 0, sizeof *device);

  // 2. Initialize the basics
  iree_hal_resource_initialize(&strela_device_vtable, &device->resource);
  iree_string_view_t identifier = iree_string_view_literal("strela-fpga-0");
  device->host_allocator = host_allocator;
  device->identifier = identifier;

  {
    strela_allocator_t *allocator = NULL;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *allocator, (void**)&allocator));
    memset(allocator, 0, sizeof *allocator);

    iree_hal_resource_initialize(&strela_allocator_vtable, &allocator->resource);

    allocator->dev = device->dev;
    allocator->host_allocator = host_allocator;

    device->device_allocator = (iree_hal_allocator_t *)allocator;
  }

  *out_device = (iree_hal_device_t*)device;
  return iree_ok_status();
}

static void
strela_driver_destroy(iree_hal_driver_t *base_driver) {
  strela_driver_t *driver = (strela_driver_t *)base_driver;
  iree_allocator_t host_allocator = driver->host_allocator;

  // Perform any additional driver teardown here if necessary

  iree_allocator_free(host_allocator, driver);
}

static iree_status_t
strela_driver_query_available_devices(
  iree_hal_driver_t *base_driver,
  iree_allocator_t host_allocator,
  iree_host_size_t *out_device_info_count,
  iree_hal_device_info_t **out_device_infos
) {
  printf("%s\n", __func__);

  // We only have 1 FPGA device to expose
  *out_device_info_count = 1;

  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *device_infos * 1, (void**)&device_infos));

  iree_string_view_t name = iree_string_view_literal("strela-fpga-0");
  // Populate the dummy device info
  device_infos[0].device_id = 0; // Just use ID 0 for the first device
  device_infos[0].name = name;

  *out_device_infos = device_infos;
  return iree_ok_status();
}

static iree_status_t
strela_driver_dump_device_info(
  iree_hal_driver_t *base_driver,
  iree_hal_device_id_t device_id,
  iree_string_builder_t* builder
) {
  printf("%s\n", __func__);

  return iree_string_builder_append_cstring(builder, "STRELA Custom FPGA Accelerator\n");
}

static iree_status_t
strela_driver_create_device_by_path(
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
  return strela_driver_create_device_by_id(
   base_driver, 0, param_count, params,
   NULL, host_allocator, out_device
  );
}

static const iree_hal_driver_vtable_t
strela_driver_vtable = {
  .destroy = strela_driver_destroy,
  .query_available_devices = strela_driver_query_available_devices,
  .dump_device_info = strela_driver_dump_device_info,
  .create_device_by_id = strela_driver_create_device_by_id,
  .create_device_by_path = strela_driver_create_device_by_path,
};
