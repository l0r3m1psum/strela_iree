#include "strela.h"
#include "iree/hal/api.h"

#include "buffer.c"
#include "allocator.c"
#include "command_buffer.c"
#include "semaphore.c"
#include "executable.c"
#include "executable_cache.c"
#include "device.c"
#include "driver.c"

static iree_status_t
strela_driver_factory_enumerate(
  void* self,
  iree_host_size_t* out_driver_info_count,
  const iree_hal_driver_info_t** out_driver_infos
) {
  printf("%s\n", __func__);

  static const iree_hal_driver_info_t driver_info = {
    .driver_name = iree_string_view_literal("strela"),
    .full_name = iree_string_view_literal("STRELA FPGA SoC Accelerator"),
  };
  *out_driver_info_count = 1;
  *out_driver_infos = (iree_hal_driver_info_t*)&driver_info;
  return iree_ok_status();
}

static iree_status_t
strela_driver_factory_try_create(
  void* self,
  iree_string_view_t driver_name,
  iree_allocator_t host_allocator,
  iree_hal_driver_t** out_driver
) {
  printf("%s\n", __func__);

  iree_string_view_t strela_name = iree_string_view_literal("strela");
  if (!iree_string_view_equal(driver_name, strela_name)) {
    return iree_make_status(
      IREE_STATUS_UNAVAILABLE, "no driver '%.*s' found",
      (int)driver_name.size, driver_name.data
    );
  }

  strela_driver_t *driver = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof *driver, (void**)&driver));
  memset(driver, 0, sizeof *driver);
  iree_hal_resource_initialize(&strela_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  driver->identifier = strela_name;

  *out_driver = (iree_hal_driver_t *)driver;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_my_driver_module_register(iree_hal_driver_registry_t *registry) {
  printf("%s\n", __func__);

  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = strela_driver_factory_enumerate,
      .try_create = strela_driver_factory_try_create,
  };

  return iree_hal_driver_registry_register_factory(registry, &factory);
}
