#include "iree/async/util/proactor_pool.h"

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

    *out_driver = (iree_hal_driver_t *)driver;
  }

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

  iree_status_t res = iree_ok_status();

  unsigned count = 0;
  if (strela_device_count(&count) == -1) {
    res = iree_status_from_code(IREE_STATUS_NOT_FOUND);
  }

  if (iree_status_is_ok(res)) {
    // TODO: is count != 0 we should return all the available devices.
    static const iree_hal_device_info_t device_infos[1] = {
      {
        .device_id = 0,
        .name = iree_string_view_literal("default_strela"),
      },
    };
    res = iree_allocator_clone(
      host_allocator,
      iree_make_const_byte_span(device_infos, sizeof device_infos),
      (void **)out_device_infos
    );
    *out_device_info_count = IREE_ARRAYSIZE(device_infos);
  }

  return res;
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

  iree_hal_strela_driver_t* driver = iree_hal_strela_driver_cast(base_driver);

  (void)driver;

  iree_hal_strela_device_t *device = NULL;
  // TODO: implement iree_hal_strela_device_create
  {
    iree_host_size_t total_size = sizeof(*device) + driver->identifier.size;
    IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device)
    );
    iree_hal_resource_initialize(&iree_hal_strela_device_vtable, &device->resource);
    iree_string_view_append_to_buffer(
      driver->identifier, &device->identifier,
      (char*)device + total_size - driver->identifier.size
    );
    device->host_allocator = host_allocator;
    device->proactor_pool = device_create_params->proactor_pool;
    iree_async_proactor_pool_retain(device->proactor_pool);
    iree_atomic_store(&device->epoch, 0, iree_memory_order_relaxed);
    iree_status_t status = iree_async_proactor_pool_get(
      device->proactor_pool, 0, &device->proactor
    );
    // FIXME: here we can leak device...
    IREE_RETURN_IF_ERROR(status);
  }

  iree_hal_strela_allocator_t *allocator = NULL;
  // TODO: implement iree_hal_strela_allocator_create
  {
    // FIXME: here we can leak device...
    IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof *allocator, (void **)&allocator)
    );
    iree_hal_resource_initialize(&iree_hal_strela_allocator_vtable, &allocator->resource);

    allocator->host_allocator = host_allocator;
  }

  device->device_allocator = (iree_hal_allocator_t *)allocator;

  *out_device = (iree_hal_device_t*)device;
  return iree_ok_status();
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
   NULL, host_allocator, out_device
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
