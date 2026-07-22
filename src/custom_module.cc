#include <memory>
#include <iostream>

#include "iree/hal/api.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/api.h"
#include "iree/vm/dynamic/api.h"
#include "iree/vm/native_module_cc.h"

#include "strela.h"

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
     * The STRELA kernel multiplies 1 row of X (were the number of rows is the
     * batch size), with 3 column of W, to produce 3 consecutive elements a row
     * of Y in single execution.
     *
     *          X               W              Y
     * [ --------------- ] [ | | |     ]   [ * * *    ]
     * [                 ] [ | | |     ]   [          ]
     * [                 ] [ | | |     ] = [          ]
     * [                 ] [ | | |     ]   [          ]
     * [                 ] [ | | |     ]   [          ]
     *
     * Of course doing X W' is more efficient because we an load from W with
     * "unit" stride.
     * P.S. right now we only support X W'.
     */

    iree_hal_buffer_view_t* x_view = x.get();
    iree_hal_buffer_view_t* w_view = w.get();
    iree_hal_buffer_view_t* y_view = y.get();

    // TODO: verify that the delays and zero points are correct.
    if (
      iree_hal_buffer_view_shape_rank(x_view) != 2
      || iree_hal_buffer_view_shape_rank(w_view) != 2
      || iree_hal_buffer_view_shape_rank(y_view) != 2
    ) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Expected 2D matrices for GEMM"
      );
    }

    if (
      iree_hal_buffer_view_element_type(x_view) != IREE_HAL_ELEMENT_TYPE_INT_8
      || iree_hal_buffer_view_element_type(w_view) != IREE_HAL_ELEMENT_TYPE_INT_8
      || iree_hal_buffer_view_element_type(y_view) != IREE_HAL_ELEMENT_TYPE_INT_32
    ) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Expected SINT_8 inputs and SINT_32 output"
      );
    }

    // X Matrix [M x K] & W' Matrix [N x K]
    iree_host_size_t M = iree_hal_buffer_view_shape_dim(x_view, 0);
    iree_host_size_t K = iree_hal_buffer_view_shape_dim(x_view, 1);
    iree_host_size_t N = iree_hal_buffer_view_shape_dim(w_view, 0);

    if (K != iree_hal_buffer_view_shape_dim(w_view, 1)) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Expected the input matrices to have compatible shapes"
      );
    }

    struct ScopedMap {
      iree_hal_buffer_mapping_t mapping = {0};
      bool is_mapped = false;

      iree_status_t map(iree_hal_buffer_view_t* view, iree_hal_memory_access_t access) {
        iree_status_t status = iree_hal_buffer_map_range(
          iree_hal_buffer_view_buffer(view),
          IREE_HAL_MAPPING_MODE_SCOPED, access, 0,
          iree_hal_buffer_view_byte_length(view), &mapping);
        is_mapped = iree_status_is_ok(status);
        return status;
      }

      ~ScopedMap() { if (is_mapped) iree_hal_buffer_unmap_range(&mapping); }
    };

    ScopedMap map_x, map_w, map_y;
    IREE_RETURN_IF_ERROR(map_x.map(x_view, IREE_HAL_MEMORY_ACCESS_READ));
    IREE_RETURN_IF_ERROR(map_w.map(w_view, IREE_HAL_MEMORY_ACCESS_READ));
    IREE_RETURN_IF_ERROR(map_y.map(y_view, IREE_HAL_MEMORY_ACCESS_READ|IREE_HAL_MEMORY_ACCESS_WRITE));

    strela_dev *dev = strela_dev_init(0);

    strela_buffer sx = strela_buffer_alloc(dev, M * K);
    strela_buffer sw = strela_buffer_alloc(dev, N * K);
    strela_buffer sy = strela_buffer_alloc(dev, M * N);

    if (!strela_dev_ok(dev)) {
      return iree_make_status(
        IREE_STATUS_INTERNAL,
        "STRELA something during initialization or allocation.");
    }

    int32_t *sx_ptr = (int32_t *) strela_buffer_to_ptr(dev, sx);
    int32_t *sw_ptr = (int32_t *) strela_buffer_to_ptr(dev, sw);
    int32_t *sy_ptr = (int32_t *) strela_buffer_to_ptr(dev, sy);

    const int8_t *x_in = (const int8_t *) map_x.mapping.contents.data;
    for (size_t i = 0; i < M * K; ++i) { sx_ptr[i] = x_in[i]; }

    const int8_t *w_in = (const int8_t *) map_w.mapping.contents.data;
    for (size_t i = 0; i < N * K; ++i) { sw_ptr[i] = w_in[i]; }

    strela_kernel kernel = {true, (unsigned) kernel_handle};

#if 1
   for (size_t i = 0; i < M; i += 1) {
      for (size_t j = 0; j < N; j += 3) {
        size_t j0 = j + 0, j1 = j + 1, j2 = j + 2;

        strela_conf conf = {0};

        // STRELA columns are disabled to avoid out-of-bounds r/w.
        size_t K0 = K, K1 = j1 < N ? K : 0, K2 = j2 < N ? K : 0;
        size_t O0 = 1, O1 = j1 < N ? 1 : 0, O2 = j2 < N ? 1 : 0;

        // Input: 1 row of X [M x K]
        conf.inp0_offset = sx.offset_words_from_base + i * K; conf.inp0_count = K; conf.inp0_stride = 1;

        // Inputs: 3 rows of W' [N x K]
        conf.inp1_offset = sw.offset_words_from_base + j0 * K; conf.inp1_count = K0; conf.inp1_stride = 1;
        conf.inp2_offset = sw.offset_words_from_base + j1 * K; conf.inp2_count = K1; conf.inp2_stride = 1;
        conf.inp3_offset = sw.offset_words_from_base + j2 * K; conf.inp3_count = K2; conf.inp3_stride = 1;

        // Outputs: mapped to 3 adjacent elements in the SAME row (i) of Y
        conf.out1_offset = sy.offset_words_from_base + i * N + j0; conf.out1_count = O0;
        conf.out2_offset = sy.offset_words_from_base + i * N + j1; conf.out2_count = O1;
        conf.out3_offset = sy.offset_words_from_base + i * N + j2; conf.out3_count = O2;

        // TODO configure just at the first and last iterations of the loop.
        strela_config(dev, kernel, &conf);
        strela_execute(dev);
      }
    }
#else
  for (size_t i = 0; i < M; ++i) {
    for (size_t j = 0; j < N; ++j) {
      int32_t acc = 0;
      for (size_t k = 0; k < K; ++k) {
        int32_t x_val = x_in[i * K + k] - z_x;
        int32_t w_val = w_in[j * K + k] - z_w;
        acc += x_val * w_val;
      }
      sy_ptr[i * N + j] = acc;
    }
  }
#endif

    if (!strela_dev_ok(dev)) {
      // Here we leak buffers but it does not matter since we halt the execution of the VM.
      return iree_make_status(
        IREE_STATUS_INTERNAL,
        "STRELA something went wrong during the execution");
    }

    int32_t *y_out = (int32_t *) map_y.mapping.contents.data;
    // This += is necessary to implement the accumulation behavior of linalg.matmul
    for (size_t i = 0; i < M * N; ++i) { y_out[i] += sy_ptr[i]; }

    strela_buffer_free(dev, sx);
    strela_buffer_free(dev, sw);
    strela_buffer_free(dev, sy);

    return y;
  }

  StatusOr<int32_t>
  InitAccelerator(
    vm::ref<iree_hal_buffer_view_t> kernel_bitstream
  ) {
    iree_hal_buffer_view_t *kernel_buf_view = kernel_bitstream.get();
    iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(kernel_buf_view);

    iree_hal_element_type_t element_type = iree_hal_buffer_view_element_type(kernel_buf_view);
    // TODO: even if it does not really matter this should be a UINT_32 but we
    // need to change the type emitted by the compiler plugin.
    if (element_type != IREE_HAL_ELEMENT_TYPE_INT_32) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Expected kernel_bitstream to be UINT_32"
      );
    }

    iree_device_size_t byte_length = iree_hal_buffer_view_byte_length(kernel_buf_view);
    if (byte_length != sizeof (uint32_t) * STRELA_KERNEL_SIZE) {
      return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Expected kernel_bitstream to have the correct length"
      );
    }

    iree_hal_buffer_mapping_t mapping = {0};
    iree_status_t status = iree_hal_buffer_map_range(
      buffer,
      IREE_HAL_MAPPING_MODE_SCOPED,
      IREE_HAL_MEMORY_ACCESS_READ,
      0,
      byte_length,
      &mapping
    );

    if (!iree_status_is_ok(status)) {
      return status;
    }

    const uint32_t *data = (uint32_t*)mapping.contents.data;

    strela_dev *dev = strela_dev_init(0);
    strela_kernel kernel = strela_kernel_alloc(dev);
    strela_kernel_set(dev, kernel, data);

    if (!strela_dev_ok(dev)) {
      iree_hal_buffer_unmap_range(&mapping);
      return iree_make_status(
        IREE_STATUS_INTERNAL,
        "strela_dev failed to initialize or set kernel"
      );
    }

    iree_hal_buffer_unmap_range(&mapping);

    return kernel.handle;
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
