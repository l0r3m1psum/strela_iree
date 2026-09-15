#include "strela.h"
#include "iree/hal/api.h"

static bool
is_all_zero(const void *ptr, size_t size) {
  const unsigned char *buf = (const unsigned char *)ptr;

  if (size == 0) return true;

  // checks if every byte matches its preceding byte
  return buf[0] == 0 && memcmp(buf, buf + 1, size - 1) == 0;
}

#define TRACE_FUNC do { printf("%s\n", __func__); } while(0)

#include "buffer.c"
#include "allocator.c"
#include "executable.c"
#include "command_buffer.c"
#include "semaphore.c"
#include "executable_cache.c"
#include "device.c"
#include "driver.c"
#include "factory.c"

IREE_API_EXPORT iree_status_t
iree_hal_strela_driver_module_register(iree_hal_driver_registry_t *registry) {
  TRACE_FUNC;

  static const iree_hal_driver_factory_t factory = {
    .self = NULL,
    .enumerate = iree_hal_strela_driver_factory_enumerate,
    .try_create = iree_hal_strela_driver_factory_try_create,
  };

  return iree_hal_driver_registry_register_factory(registry, &factory);
}
