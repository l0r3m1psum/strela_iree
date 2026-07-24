#include "strela.h"
#include "iree/hal/api.h"

typedef struct {
  // 1. Base resource (must be first for IREE's refcounting system)
  iree_hal_resource_t resource;
  // 2. String identifier for debugging and profiling
  iree_string_view_t identifier;
  // 3. The host allocator used to create this device (needed for teardown)
  iree_allocator_t host_allocator;
  // 4. Your custom HAL allocator implementation
  iree_hal_allocator_t *strela_allocator;
  // 5. The actual STRELA hardware context from your driver API
  strela_dev *dev;
} strela_device_t;

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  strela_dev* dev;
} strela_allocator_t;

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_allocator_allocate_buffer(
  iree_hal_allocator_t *base_allocator,
  const iree_hal_buffer_params_t* params,
  iree_device_size_t allocation_size,
  iree_hal_buffer_t **out_buffer
) {

  strela_allocator_t *allocator = (strela_allocator_t *)base_allocator;

  // Allocate contiguous hardware memory using your existing STRELA API.
  strela_buffer s_buf = strela_buffer_alloc(allocator->dev, allocation_size);

  // Retrieve the mapped host-visible pointer.
  [[maybe_unused]] void *host_ptr = strela_buffer_to_ptr(allocator->dev, s_buf);

  // Wrap the physical strela_buffer and host_ptr inside a custom iree_hal_buffer_t struct
  // to return to the IREE Virtual Machine.
  // ...

  return iree_ok_status();
}

static void
strela_allocator_destroy(iree_hal_allocator_t *base_allocator) {
  [[maybe_unused]] strela_allocator_t *allocator = (strela_allocator_t*)base_allocator;

  // NOTE: Assuming you add `iree_allocator_t host_allocator` to strela_allocator_t
  // iree_allocator_free(allocator->host_allocator, allocator);
}

static iree_allocator_t
strela_allocator_host_allocator(
  const iree_hal_allocator_t *base_allocator
) {
  return ((strela_allocator_t *)base_allocator)->host_allocator;
}

static iree_status_t strela_allocator_trim(
  iree_hal_allocator_t *base_allocator
) {
  return iree_ok_status(); // No-op
}

static void
strela_allocator_deallocate_buffer(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_t *base_buffer
) {
  // We will implement this later when you handle actual memory freeing
}

static void
strela_allocator_query_statistics(
  iree_hal_allocator_t *base_allocator,
  iree_hal_allocator_statistics_t *out_statistics
) {
  memset(out_statistics, 0, sizeof(*out_statistics));
}

static iree_status_t
strela_allocator_query_memory_heaps(
  iree_hal_allocator_t *base_allocator,
  iree_host_size_t capacity,
  iree_hal_allocator_memory_heap_t *heaps,
  iree_host_size_t *out_count
) {
  *out_count = 1;
  if (capacity > 0 && heaps != NULL) {
    heaps[0].type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL | IREE_HAL_MEMORY_TYPE_HOST_VISIBLE;
    heaps[0].allowed_usage = IREE_HAL_BUFFER_USAGE_DEFAULT;
    heaps[0].max_allocation_size = 1024 * 1024 * 256; // 256 MB max allocation size
    heaps[0].min_alignment = 64;
  }
  return iree_ok_status();
}

static iree_hal_buffer_compatibility_t
strela_allocator_query_buffer_compatibility(
  iree_hal_allocator_t *base_allocator,
  iree_hal_buffer_params_t *params,
  iree_device_size_t *allocation_size
) {
  return IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE |
         IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
}

static iree_status_t strela_allocator_import_buffer(
  iree_hal_allocator_t* IREE_RESTRICT allocator,
  const iree_hal_buffer_params_t* IREE_RESTRICT params,
  iree_hal_external_buffer_t* IREE_RESTRICT external_buffer,
  iree_hal_buffer_release_callback_t release_callback,
  iree_hal_buffer_t** IREE_RESTRICT out_buffer
) {
  return iree_make_status(
    IREE_STATUS_UNIMPLEMENTED,
    "strela allocator does not support importing external buffers"
  );
}

static iree_status_t strela_allocator_export_buffer(
  iree_hal_allocator_t* IREE_RESTRICT allocator,
  iree_hal_buffer_t* IREE_RESTRICT buffer,
  iree_hal_external_buffer_type_t requested_type,
  iree_hal_external_buffer_flags_t requested_flags,
  iree_hal_external_buffer_t* IREE_RESTRICT out_external_buffer
) {
  return iree_make_status(
    IREE_STATUS_UNIMPLEMENTED,
    "strela allocator does not support exporting external buffers"
  );
}

static const iree_hal_allocator_vtable_t
strela_allocator_vtable = {
  .destroy                    = strela_allocator_destroy,
  .host_allocator             = strela_allocator_host_allocator,
  .trim                       = strela_allocator_trim,
  .query_statistics           = strela_allocator_query_statistics,
  .query_memory_heaps         = strela_allocator_query_memory_heaps,
  .query_buffer_compatibility = strela_allocator_query_buffer_compatibility,
  .allocate_buffer            = strela_allocator_allocate_buffer,
  .deallocate_buffer          = strela_allocator_deallocate_buffer,
  .import_buffer              = strela_allocator_import_buffer,
  .export_buffer              = strela_allocator_export_buffer,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
typedef struct {
  iree_hal_resource_t resource;
  // Internal ring buffer or array to store recorded STRELA commands
} strela_command_buffer_t;

static iree_status_t
strela_command_buffer_dispatch(
  iree_hal_command_buffer_t *base_command_buffer,
  iree_hal_executable_t *executable,
  iree_hal_executable_function_t executable_function,
  const iree_hal_dispatch_config_t dispatch_config,
  iree_const_byte_span_t const_byte_span,
  iree_hal_buffer_ref_list_t buffer_ref_list,
  iree_hal_dispatch_flags_t dispatch_flags
) {

  [[maybe_unused]] strela_command_buffer_t *cmd = (strela_command_buffer_t*)base_command_buffer;

  // Initialize the configuration for the accelerator.
  [[maybe_unused]] strela_conf conf = {0};

  // Extract bound buffer offsets from the executable descriptor sets and
  // map them to inp0_offset, inp1_offset, out1_offset, etc.
  // conf.inp0_offset = ...

  // Push this `conf` struct into your command buffer's internal queue.
  // Do NOT execute the hardware here!
  // ...

  return iree_ok_status();
}

static void
strela_command_buffer_destroy(iree_hal_command_buffer_t* base_command_buffer) {
  [[maybe_unused]] strela_command_buffer_t* cmd = (strela_command_buffer_t*)base_command_buffer;

  // NOTE: Assuming you add `iree_allocator_t host_allocator` to strela_command_buffer_t
  // iree_allocator_free(cmd->host_allocator, cmd);
}

static const iree_hal_command_buffer_vtable_t
strela_command_buffer_vtable = {
  .destroy = strela_command_buffer_destroy,
  .dispatch = strela_command_buffer_dispatch,
  // Add other required command buffer vtable methods here (begin, end, etc.)
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_command_buffer_create(
  iree_allocator_t host_allocator,
  strela_dev *dev,
  iree_hal_command_buffer_t** out_command_buffer
) {

  strela_command_buffer_t* cmd = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*cmd), (void**)&cmd));

  // Initialize the resource tracking and map the vtable
  iree_hal_resource_initialize(&strela_command_buffer_vtable, &cmd->resource);

  // cmd->host_allocator = host_allocator; // Store this if added to the struct

  *out_command_buffer = (iree_hal_command_buffer_t*)cmd;
  return iree_ok_status();
}

static iree_status_t
strela_device_create_command_buffer(
  iree_hal_device_t *base_device,
  iree_hal_command_buffer_mode_t mode,
  iree_hal_command_category_t command_categories,
  iree_hal_queue_affinity_t queue_affinity,
  iree_host_size_t binding_capacity,
  iree_hal_command_buffer_t **out_command_buffer
) {
  printf("%s\n", __func__);

  strela_device_t *device = (strela_device_t*)base_device;

  // Allocate and return your custom strela_command_buffer_t
  return strela_command_buffer_create(
    device->host_allocator,
    device->dev,
    out_command_buffer
  );
}

static void
strela_device_destroy(iree_hal_device_t *base_device) {
  strela_device_t *device = (strela_device_t *)base_device;
  iree_allocator_t host_allocator = device->host_allocator;

  // 1. Release the custom allocator (this decrements the refcount)
  if (device->strela_allocator) {
    iree_hal_allocator_release(device->strela_allocator);
  }

  // 2. Teardown the hardware handle
  if (device->dev) {
    // strela_dev_destroy(device->dev); // Replace with your actual hardware free function
  }

  // 3. Free the device memory
  iree_allocator_free(host_allocator, device);
}

static iree_status_t
strela_device_queue_execute(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  const iree_hal_semaphore_list_t wait_semaphore_list,
  const iree_hal_semaphore_list_t signal_semaphore_list,
  iree_hal_command_buffer_t *command_buffer,
  iree_hal_buffer_binding_table_t buffer_binding_table,
  iree_hal_execute_flags_t execute_flags
) {

  [[maybe_unused]] strela_device_t *device = (strela_device_t *)base_device;

  // 1. Wait on the provided wait_semaphore_list.
  // 2. Iterate over the recorded command_buffers.

  // For each recorded task, apply the configuration to the hardware.
  // strela_config(device->dev, kernel, &recorded_conf);

  // Trigger the accelerator execution.
  // strela_execute(device->dev);

  // Verify that the accelerator hasn't encountered a hardware fault.
  // if (!strela_dev_ok(device->dev)) {
  //     return iree_make_status(IREE_STATUS_INTERNAL, "STRELA execution failed");
  // }

  // 3. Signal the provided signal_semaphore_list to notify the VM of completion.

  return iree_ok_status();
}

static iree_hal_allocator_t *
strela_device_allocator(iree_hal_device_t *base_device) {
  strela_device_t* device = (strela_device_t*)base_device;
  return device->strela_allocator;
}

static iree_string_view_t
strela_device_id(iree_hal_device_t *base_device) {
  return ((strela_device_t *)base_device)->identifier;
}

static iree_allocator_t
strela_device_host_allocator(iree_hal_device_t *base_device) {
  return ((strela_device_t *)base_device)->host_allocator;
}

static iree_status_t
strela_device_query_i64(
  iree_hal_device_t *base_device,
  iree_string_view_t category,
  iree_string_view_t key,
  int64_t *out_value
) {
  printf("%s\n", __func__);

  // Tell the VM we have basic default capabilities for now
  *out_value = 0;
  return iree_ok_status();
}

static iree_status_t
strela_device_create_semaphore(
  iree_hal_device_t *base_device,
  iree_hal_queue_affinity_t queue_affinity,
  uint64_t initial_value,
  iree_hal_semaphore_flags_t flags,
  iree_hal_semaphore_t **out_semaphore
) {
  return iree_make_status(
    IREE_STATUS_UNIMPLEMENTED,
    "strela_device_create_semaphore not implemented yet"
  );
}

static iree_status_t
strela_device_create_executable_cache(
  iree_hal_device_t* base_device,
  iree_string_view_t identifier,
  iree_hal_executable_cache_t** out_executable_cache
) {
  // iree-run-module will call this to load your .vmfb file
  return iree_make_status(
    IREE_STATUS_UNIMPLEMENTED,
    "strela_device_create_executable_cache not implemented yet"
  );
}

static const iree_hal_device_vtable_t
strela_device_vtable = {
  .destroy                 = strela_device_destroy,
  .id                      = strela_device_id,
  .host_allocator          = strela_device_host_allocator,
  .device_allocator        = strela_device_allocator,
  .query_i64               = strela_device_query_i64,
  .create_semaphore        = strela_device_create_semaphore,
  .create_executable_cache = strela_device_create_executable_cache,
  .create_command_buffer   = strela_device_create_command_buffer,
  .queue_execute           = strela_device_queue_execute,
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_allocator_create(
  iree_allocator_t host_allocator,
  strela_dev* dev,
  iree_hal_allocator_t** out_allocator
) {

  strela_allocator_t *allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator), (void**)&allocator));
  memset(allocator, 0, sizeof (*allocator));

  // Initialize the resource tracking and map the vtable
  iree_hal_resource_initialize(&strela_allocator_vtable, &allocator->resource);

  allocator->dev = dev;
  // allocator->host_allocator = host_allocator; // Store this if added to the struct

  *out_allocator = (iree_hal_allocator_t*)allocator;
  return iree_ok_status();
}

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
  iree_hal_driver_t* base_driver,
  iree_hal_device_id_t device_id,
  iree_host_size_t param_count,
  const iree_string_pair_t* params,
  const iree_hal_device_create_params_t *device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t** out_device
) {
  printf("%s\n", __func__);

  [[maybe_unused]] strela_driver_t* driver = (strela_driver_t*)base_driver;

  // 1. Allocate your custom strela_device_t
  strela_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof (*device), (void **)&device));
  memset(device, 0, sizeof(*device));

  // 2. Initialize the basics
  iree_hal_resource_initialize(&strela_device_vtable, &device->resource);
  iree_string_view_t identifier = iree_string_view_literal("strela-fpga-0");
  device->host_allocator = host_allocator;
  device->identifier = identifier;

  // 3. Initialize your custom allocator HERE. The device owns it.
  IREE_RETURN_IF_ERROR(strela_allocator_create(host_allocator, device->dev, &device->strela_allocator));

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

  // We only have 1 FPGA device to expose
  *out_device_info_count = 1;

  iree_hal_device_info_t* device_infos = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*device_infos) * 1, (void**)&device_infos));

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
  return iree_string_builder_append_cstring(builder, "STRELA Custom FPGA Accelerator\n");
}

static iree_status_t
strela_driver_create_device_by_path(
  iree_hal_driver_t *base_driver,
  iree_string_view_t driver_name,
  iree_string_view_t device_path,
  iree_host_size_t param_count,
  const iree_string_pair_t *params,
  const iree_hal_device_create_params_t * device_create_params,
  iree_allocator_t host_allocator,
  iree_hal_device_t **out_device
) {

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
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
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
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof (*driver), (void**)&driver));
  memset(driver, 0, sizeof(*driver));
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

////////////////////////////////////////////////////////////////////////////////
