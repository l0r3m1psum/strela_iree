// NOTE: Shapes are static on purpose: with dynamic shapes the TOSA lowering of
// the elementwise ops emits a runtime "is this dim 1?" broadcast check as its
// own dispatch, which is not what these tests are about.
//
// RUN: iree-compile --iree-plugin=estela --split-input-file \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=strela \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=executable-targets %s | FileCheck %s
//
// RUN: iree-compile --iree-plugin=estela --split-input-file \
// RUN:   --iree-scheduling-optimize-bindings=false \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=strela \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=executable-targets %s | FileCheck %s --check-prefix=UNFUSED

// The work meant for device_a is a sub and a minimum, which STRELA has no kernel
// for; the matvec runs on device_b. Not valid TOSA (no i32 matmul), but it is
// what reaches STRELA.

// Variant 1: the topology says the two devices share memory, so the transfers
// need no staging buffers, each operand keeps its own binding and every subspan
// starts at byte zero.
//
// The job reads the row (C = 16 words) from binding 0, the weights (C words
// F = 9 apart, 4 bytes between columns) from binding 1, and writes the result
// to binding 2. Only the binding, offset and stride of each column are checked
// here; materialize_bindings.mlir checks the rest.
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     strela.job
// CHECK-SAME:  inputs([#strela.inp_col<address = <binding = 0, offset = 0 : i32, delta = 0,{{[^#]*}}words = 16, stride = 1>,
// CHECK-SAME:          #strela.inp_col<address = <binding = 1, offset = 0 : i32, delta = 0,{{[^#]*}}words = 16, stride = 9>,
// CHECK-SAME:          #strela.inp_col<address = <binding = 1, offset = 0 : i32, delta = 4,{{[^#]*}}words = 16, stride = 9>,
// CHECK-SAME:          #strela.inp_col<address = <binding = 1, offset = 0 : i32, delta = 8,{{[^#]*}}words = 16, stride = 9>])
// CHECK-SAME:  outputs([#strela.out_col<{{[^#]*}}words = 0>,
// CHECK-SAME:           #strela.out_col<address = <binding = 2, offset = 0 : i32, delta = 0,{{[^#]*}}words = 1>,
// CHECK-SAME:           #strela.out_col<address = <binding = 2, offset = 0 : i32, delta = 4,{{[^#]*}}words = 1>,
// CHECK-SAME:           #strela.out_col<address = <binding = 2, offset = 0 : i32, delta = 8,{{[^#]*}}words = 1>])

// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     strela.job
// UNFUSED-SAME:  inputs([#strela.inp_col<address = <binding = 0, offset = 0 : i32
// UNFUSED-SAME:          #strela.inp_col<address = <binding = 1, offset = 0 : i32
// UNFUSED-SAME:  outputs([#strela.out_col<{{[^#]*}}words = 0>,
// UNFUSED-SAME:           #strela.out_col<address = <binding = 2, offset = 0 : i32

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true, transparent_access = true}),
    (@device_b -> @device_a = {unified_memory = true, transparent_access = true})
  ]>
} {
  func.func @main(
    %i0: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %x: tensor<1x8x16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %w: tensor<1x16x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %zp = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %c0 = arith.constant 0 : index
    %mat_shape = tosa.const_shape values(dense<[8, 9]> : tensor<2xindex>) : () -> !tosa.shape<2>
    %new_shape = tosa.const_shape values(dense<[-1]> : tensor<1xindex>) : () -> !tosa.shape<1>

    %x_b = flow.tensor.transfer %x : tensor<1x8x16xi32> to #hal.device.promise<@device_b>
    %w_b = flow.tensor.transfer %w : tensor<1x16x9xi32> to #hal.device.promise<@device_b>

    %a0 = tosa.sub %i0, %i1 : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %y_b = tosa.matmul %x_b, %w_b, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>

    %r0 = tosa.minimum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    // The transfer back to device_a is a fusion boundary, as is the transfer of
    // %x/%w above, so nothing on the CPU fuses into the matvec.
    %y3 = flow.tensor.transfer %y_b : tensor<1x8x9xi32> to #hal.device.promise<@device_a>
    %y = tosa.reshape %y3, %mat_shape : (tensor<1x8x9xi32>, !tosa.shape<2>) -> tensor<8x9xi32>

    %a2 = tosa.sub %r0, %y : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened axis(0) : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}

// -----

// Variant 2: the same program without the topology. The compiler does not know
// the devices share memory, so it stages the transfers through extra buffers;
// FuseDispatchBindings then packs both matvec operands into one binding and
// the second one gets a compile-time byte offset: 512 is 8 x 16 x i32, the
// bytes of %x, already a multiple of the 64 byte alignment.
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     strela.job
// CHECK-SAME:  inputs([#strela.inp_col<address = <binding = 0, offset = 0 : i32
// CHECK-SAME:          #strela.inp_col<address = <binding = 0, offset = 512 : i32, delta = 0,
// CHECK-SAME:          #strela.inp_col<address = <binding = 0, offset = 512 : i32, delta = 4,
// CHECK-SAME:          #strela.inp_col<address = <binding = 0, offset = 512 : i32, delta = 8,
// CHECK-SAME:  outputs([#strela.out_col<{{[^#]*}}words = 0>,
// CHECK-SAME:           #strela.out_col<address = <binding = 1, offset = 0 : i32

// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     strela.job
// UNFUSED-SAME:  inputs([#strela.inp_col<address = <binding = 0, offset = 0 : i32
// UNFUSED-SAME:          #strela.inp_col<address = <binding = 1, offset = 0 : i32
// UNFUSED-SAME:  outputs([#strela.out_col<{{[^#]*}}words = 0>,
// UNFUSED-SAME:           #strela.out_col<address = <binding = 2, offset = 0 : i32

module {
  func.func @main(
    %i0: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %x: tensor<1x8x16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %w: tensor<1x16x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %zp = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %c0 = arith.constant 0 : index
    %mat_shape = tosa.const_shape values(dense<[8, 9]> : tensor<2xindex>) : () -> !tosa.shape<2>
    %new_shape = tosa.const_shape values(dense<[-1]> : tensor<1xindex>) : () -> !tosa.shape<1>

    %x_b = flow.tensor.transfer %x : tensor<1x8x16xi32> to #hal.device.promise<@device_b>
    %w_b = flow.tensor.transfer %w : tensor<1x16x9xi32> to #hal.device.promise<@device_b>

    %a0 = tosa.sub %i0, %i1 : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %y_b = tosa.matmul %x_b, %w_b, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>

    %r0 = tosa.minimum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    %y3 = flow.tensor.transfer %y_b : tensor<1x8x9xi32> to #hal.device.promise<@device_a>
    %y = tosa.reshape %y3, %mat_shape : (tensor<1x8x9xi32>, !tosa.shape<2>) -> tensor<8x9xi32>

    %a2 = tosa.sub %r0, %y : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened axis(0) : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}

// -----

// Variant 3: TransferWrapPass places both matvecs on device_b and they share
// one executable. FuseDispatchBindings packs the operands of the two dispatch
// sites at different offsets, so the offsets reach the kernel as push
// constants, each a single i32 because Stream proved it fits in 32 bits. Each
// column adds its own compile-time delta to the push constant offset of its
// tensor: 4k bytes for weight and result column k.
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     strela.job
// CHECK-SAME:  inputs([#strela.inp_col<address = <binding = 0, offset = #strela.push_constant<0>, delta = 0,
// CHECK-SAME:          #strela.inp_col<address = <binding = 0, offset = #strela.push_constant<1>, delta = 0,
// CHECK-SAME:          #strela.inp_col<address = <binding = 0, offset = #strela.push_constant<1>, delta = 4,
// CHECK-SAME:          #strela.inp_col<address = <binding = 0, offset = #strela.push_constant<1>, delta = 8,
// CHECK-SAME:  outputs([#strela.out_col<{{[^#]*}}words = 0>,
// CHECK-SAME:           #strela.out_col<address = <binding = 1, offset = #strela.push_constant<2>, delta = 0,
// CHECK-SAME:           #strela.out_col<address = <binding = 1, offset = #strela.push_constant<2>, delta = 4,
// CHECK-SAME:           #strela.out_col<address = <binding = 1, offset = #strela.push_constant<2>, delta = 8,

// Without fused bindings every operand has a binding of its own at byte 0.
//
// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     strela.job
// UNFUSED-SAME:  inputs([#strela.inp_col<address = <binding = 0, offset = 0 : i32
// UNFUSED-SAME:          #strela.inp_col<address = <binding = 1, offset = 0 : i32
// UNFUSED-SAME:  outputs([#strela.out_col<{{[^#]*}}words = 0>,
// UNFUSED-SAME:           #strela.out_col<address = <binding = 2, offset = 0 : i32

module {
  func.func @main(
    %x0: tensor<1x8x16xi32>, %w0: tensor<1x16x9xi32>,
    %x1: tensor<1x8x16xi32>, %w1: tensor<1x16x9xi32>
  ) -> tensor<1x8x9xi32> {
    %zp = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %y0 = tosa.matmul %x0, %w0, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>
    %y1 = tosa.matmul %x1, %w1, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>
    %d = tosa.sub %y0, %y1 : (tensor<1x8x9xi32>, tensor<1x8x9xi32>) -> tensor<1x8x9xi32>
    return %d : tensor<1x8x9xi32>
  }
}
