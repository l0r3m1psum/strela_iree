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

static const iree_hal_allocator_vtable_t
strela_allocator_vtable = {
  .destroy = strela_allocator_destroy,
  .allocate_buffer = strela_allocator_allocate_buffer,
  // Add other required allocator vtable methods here (free_buffer, map, etc.)
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

// The device vtable exposes your allocator and command buffer factory to the VM.
static const iree_hal_device_vtable_t
strela_device_vtable = {
  .destroy = strela_device_destroy,
  .device_allocator = strela_device_allocator, // Returns device->strela_allocator
  .create_command_buffer = strela_device_create_command_buffer,
  .queue_execute = strela_device_queue_execute,
  // ...
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_allocator_create(
  iree_allocator_t host_allocator,
  strela_dev* dev,
  iree_hal_allocator_t** out_allocator
) {

  strela_allocator_t* allocator = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*allocator), (void**)&allocator));

  // Initialize the resource tracking and map the vtable
  iree_hal_resource_initialize(&strela_allocator_vtable, &allocator->resource);

  allocator->dev = dev;
  // allocator->host_allocator = host_allocator; // Store this if added to the struct

  *out_allocator = (iree_hal_allocator_t*)allocator;
  return iree_ok_status();
}

typedef struct {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
} strela_driver_t;

// This is called when the IREE runtime wants to instantiate your FPGA device.
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

  [[maybe_unused]] strela_driver_t* driver = (strela_driver_t*)base_driver;

  // 1. Allocate your custom strela_device_t
  strela_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*device), (void**)&device));

  // 2. Initialize the device vtable (which includes queue_execute)
  iree_hal_resource_initialize(&strela_device_vtable, &device->resource);
  device->host_allocator = host_allocator;

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

static const iree_hal_driver_vtable_t
strela_driver_vtable = {
  .destroy = strela_driver_destroy,
  .create_device_by_id = strela_driver_create_device_by_id,
  // ... other vtable methods (query_available_devices, etc.)
};
////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////
static iree_status_t
strela_driver_factory_enumerate(
  void* self,
  iree_host_size_t* out_driver_info_count,
  const iree_hal_driver_info_t** out_driver_infos
) {

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

  iree_string_view_t strela_name = iree_string_view_literal("strela");
  if (!iree_string_view_equal(driver_name, strela_name)) {
    return iree_make_status(
      IREE_STATUS_UNAVAILABLE, "no driver '%.*s' found",
      (int)driver_name.size, driver_name.data
    );
  }

  // Allocate and initialize your strela_driver_t here
  strela_driver_t* driver = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(host_allocator, sizeof(*driver), (void**)&driver));
  iree_hal_resource_initialize(&strela_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;

  *out_driver = (iree_hal_driver_t*)driver;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_hal_strela_driver_register(
  iree_hal_driver_registry_t* registry, iree_allocator_t host_allocator
) {

  static const iree_hal_driver_factory_t factory = {
      .self = NULL,
      .enumerate = strela_driver_factory_enumerate,
      .try_create = strela_driver_factory_try_create,
  };

  return iree_hal_driver_registry_register_factory(registry, &factory);
}
////////////////////////////////////////////////////////////////////////////////
