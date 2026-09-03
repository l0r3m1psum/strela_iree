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
#include "factory.c"

IREE_API_EXPORT iree_status_t
iree_hal_my_driver_module_register(iree_hal_driver_registry_t *registry) {
  printf("%s\n", __func__);

  return iree_hal_driver_registry_register_factory(registry, &factory);
}
