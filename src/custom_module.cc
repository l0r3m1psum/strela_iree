#include <memory>
#include <iostream>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/dynamic/api.h"
#include "iree/vm/native_module_cc.h"

using namespace iree;

class CustomModuleState final {
 public:
  CustomModuleState(iree_allocator_t allocator) : allocator_(allocator) {}

  StatusOr<vm::ref<iree_hal_buffer_view_t>>
  MyCenteredGemm(
    int32_t kernel_handle,
    vm::ref<iree_hal_buffer_view_t> x,
    int32_t z_x,
    vm::ref<iree_hal_buffer_view_t> w,
    int32_t z_w,
    vm::ref<iree_hal_buffer_view_t> y
  ) {

    /*
     * The STRELA kernel multiplies 3 rows of A with one column of B in single
     * execution. As shown below. Of course doing A B' is more efficient because
     * we an load from B with "unit" stride.
     *
     *          A              B
     * [ --------------- ] [ |     ]
     * [ --------------- ] [ |     ]
     * [ --------------- ] [ |     ]
     * [                 ] [ |     ]
     * [                 ] [ |     ]
     */

    // TODO: verify that the delays and zero points are correct. The delay
    // should be the number of rows in B

    // strela_config(dev, kernel, &conf);
    // The previous line must be looped.

    std::cerr << "handle " << kernel_handle << ' ';

    iree_hal_buffer_view_t* x_view = x.get();
    iree_host_size_t rank = iree_hal_buffer_view_shape_rank(x_view);

    std::cerr << "x";
    for (iree_host_size_t i = 0; i < rank; ++i) {
      iree_hal_dim_t dim = iree_hal_buffer_view_shape_dim(x_view, i);
      std::cerr << ' ' << dim;
    }
    std::cerr << '\n';

    return y;
  }

  StatusOr<int32_t>
  InitAccelerator(
    vm::ref<iree_hal_buffer_view_t> kernel
  ) {

    // strela_kernel kernel = strela_kernel_alloc(dev);
    // strela_kernel_set(dev, kernel, bypass_kernel);
    // TODO: this function should return the handle from STRELA or an error.
    static int32_t i = 0;
    std::cerr << "init\n";

    return i++;
  }

 private:
  iree_allocator_t allocator_;
};

static const vm::NativeFunction<CustomModuleState> kCustomModuleFunctions[] = {
  vm::MakeNativeFunction("my_centered_gemm", &CustomModuleState::MyCenteredGemm),
  vm::MakeNativeFunction("init_accelerator", &CustomModuleState::InitAccelerator),
};

class CustomModule final : public vm::NativeModule<CustomModuleState> {
 public:
  using vm::NativeModule<CustomModuleState>::NativeModule;

  StatusOr<std::unique_ptr<CustomModuleState>>
  CreateState(iree_allocator_t allocator) override {
    return std::make_unique<CustomModuleState>(allocator);
  }

  StatusOr<std::unique_ptr<CustomModuleState>>
  ForkState(CustomModuleState* parent_state, iree_allocator_t allocator) override {
    return CreateState(allocator);
  }
};

IREE_API_EXPORT iree_status_t
iree_vm_dynamic_module_create(
  uint32_t max_version,
  iree_vm_instance_t* instance,
  iree_host_size_t param_count,
  const iree_string_pair_t* params,
  iree_allocator_t allocator,
  iree_vm_module_t** out_module
) {

  if (max_version != IREE_VM_DYNAMIC_MODULE_VERSION_LATEST) {
    return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "unsupported runtime version %u, module compiled with version %u",
      max_version, IREE_VM_DYNAMIC_MODULE_VERSION_LATEST);
  }

  IREE_RETURN_IF_ERROR(iree_hal_module_resolve_all_types(instance));

  auto module = std::make_unique<CustomModule>(
    "custom", /*version=*/0, instance, allocator,
    iree::span<const vm::NativeFunction<CustomModuleState>>(kCustomModuleFunctions)
  );

  *out_module = module.release()->interface();
  return iree_ok_status();
}

// Hackity hack
#include <cstdint>
extern "C" {
  void* backtrace_create_state(const char *filename, int threaded,
                               void (*error_callback) (void *data, const char *msg, int errnum),
                               void *data) {
    return nullptr;
  }

  int backtrace_syminfo(void *state, uintptr_t addr,
                        void (*callback) (void *data, uintptr_t pc, const char *symname, uintptr_t symval, uintptr_t symsize),
                        void (*error_callback) (void *data, const char *msg, int errnum),
                        void *data) {
    return 0;
  }

  int backtrace_pcinfo(void *state, uintptr_t addr,
                       int (*callback) (void *data, uintptr_t pc, const char *filename, int lineno, const char *function),
                       void (*error_callback) (void *data, const char *msg, int errnum),
                       void *data) {
    return 0;
  }
}
