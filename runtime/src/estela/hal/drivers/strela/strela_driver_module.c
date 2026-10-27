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

// libstrela keeps one sticky error per device. Turns it into a status and
// clears it, so the device stays usable after a failed call.
static iree_status_t
iree_hal_strela_status_from_dev(strela_dev *dev, const char *what) {
  if (strela_dev_ok(dev)) {
    return iree_ok_status();
  }
  int errnum = strela_dev_get_err(dev).errnum;
  strela_dev_reset_err(dev);
  if (errnum > 0) {
    return iree_make_status(
      iree_status_code_from_errno(errnum), "%s: %s", what, strerror(errnum)
    );
  }
  switch (-errnum) {
  case STRELA_ERR_NO_MEM:
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED, "%s: out of STRELA memory", what);
  case STRELA_ERR_BAD_ARG:
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "%s: rejected by libstrela", what);
  default:
    return iree_make_status(IREE_STATUS_INTERNAL, "%s: libstrela error %d", what, errnum);
  }
}

enum {
  // libstrela syncs whole cache lines, and the driver checks against this.
  IREE_HAL_STRELA_CACHE_LINE = 64,
};

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
