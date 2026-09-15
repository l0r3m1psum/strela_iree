// NOTE: Shapes are static on purpose: with tensor<?x9xi32> the TOSA lowering of
// tosa.add emits a runtime "is this dim 1?" broadcast check as its own dispatch,
// which is placed on device_b and has no STRELA equivalent.
// TODO: test it this can be somehow supported (maybe a new strela operation for
// this kind of checks which is executed at dispatch time by the runtime)
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

// Variant 1: the topology says the two devices share memory, so the transfers
// need no staging buffers, each operand keeps its own binding and every subspan
// starts at byte zero.
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     %[[LHS:.+]] = strela.binding_load binding(0) : tensor<72xi32>
// CHECK:     %[[RHS:.+]] = strela.binding_load binding(1) : tensor<72xi32>
// CHECK:     %[[SUM:.+]] = strela.add %[[LHS]], %[[RHS]] : tensor<72xi32>
// CHECK:     strela.binding_store %[[SUM]], binding(2) : tensor<72xi32>
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     %[[IN:.+]] = strela.binding_load binding(0) : tensor<72xi32>
// CHECK:     %[[OUT:.+]] = strela.relu %[[IN]] : tensor<72xi32>
// CHECK:     strela.binding_store %[[OUT]], binding(1) : tensor<72xi32>

// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     %[[U_LHS:.+]] = strela.binding_load binding(0) : tensor<72xi32>
// UNFUSED:     %[[U_RHS:.+]] = strela.binding_load binding(1) : tensor<72xi32>
// UNFUSED:     %[[U_SUM:.+]] = strela.add %[[U_LHS]], %[[U_RHS]] : tensor<72xi32>
// UNFUSED:     strela.binding_store %[[U_SUM]], binding(2) : tensor<72xi32>

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true, transparent_access = true}),
    (@device_b -> @device_a = {unified_memory = true, transparent_access = true})
  ]>
} {
  func.func @main(
    %i0: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i2: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i3: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %c0 = arith.constant 0 : index
    %new_shape = tosa.const_shape {values = dense<[-1]> : tensor<1xindex>} : () -> !tosa.shape<1>

    %i2_b = flow.tensor.transfer %i2 : tensor<8x9xi32> to #hal.device.promise<@device_b>
    %i3_b = flow.tensor.transfer %i3 : tensor<8x9xi32> to #hal.device.promise<@device_b>

    %a0 = "tosa.add"(%i0, %i1) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %a1_b = "tosa.add"(%i2_b, %i3_b) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %a1_barrier = util.optimization_barrier %a1_b : tensor<8x9xi32>

    %r0 = tosa.maximum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>
    %r1_b = tosa.maximum %a1_barrier, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    // No barrier needed here: the transfer back to device_a is already a fusion
    // boundary, as is the transfer of %i2/%i3 above.
    %r1 = flow.tensor.transfer %r1_b : tensor<8x9xi32> to #hal.device.promise<@device_a>

    %a2 = "tosa.add"(%r0, %r1) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened {axis = 0 : i32} : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}

// -----

// Variant 2: the same program without the topology. The compiler does not know
// the devices share memory, so it stages the transfers through extra buffers;
// FuseDispatchBindings then packs both add operands into one binding and the
// second one gets a compile-time byte offset. 320 is 72 x i32 = 288 bytes
// rounded up to the 64 byte alignment. Note the byte_offset present.
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     %[[P_LHS:.+]] = strela.binding_load binding(0) : tensor<72xi32>
// CHECK:     %[[P_RHS:.+]] = strela.binding_load binding(0) byte_offset 320 : tensor<72xi32>
// CHECK:     %[[P_SUM:.+]] = strela.add %[[P_LHS]], %[[P_RHS]] : tensor<72xi32>
// CHECK:     strela.binding_store %[[P_SUM]], binding(1) : tensor<72xi32>

// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     %[[V_LHS:.+]] = strela.binding_load binding(0) : tensor<72xi32>
// UNFUSED:     %[[V_RHS:.+]] = strela.binding_load binding(1) : tensor<72xi32>
// UNFUSED:     %[[V_SUM:.+]] = strela.add %[[V_LHS]], %[[V_RHS]] : tensor<72xi32>
// UNFUSED:     strela.binding_store %[[V_SUM]], binding(2) : tensor<72xi32>

module {
  func.func @main(
    %i0: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i2: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i3: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %c0 = arith.constant 0 : index
    %new_shape = tosa.const_shape {values = dense<[-1]> : tensor<1xindex>} : () -> !tosa.shape<1>

    %i2_b = flow.tensor.transfer %i2 : tensor<8x9xi32> to #hal.device.promise<@device_b>
    %i3_b = flow.tensor.transfer %i3 : tensor<8x9xi32> to #hal.device.promise<@device_b>

    %a0 = "tosa.add"(%i0, %i1) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %a1_b = "tosa.add"(%i2_b, %i3_b) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %a1_barrier = util.optimization_barrier %a1_b : tensor<8x9xi32>

    %r0 = tosa.maximum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>
    %r1_b = tosa.maximum %a1_barrier, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    %r1 = flow.tensor.transfer %r1_b : tensor<8x9xi32> to #hal.device.promise<@device_a>

    %a2 = "tosa.add"(%r0, %r1) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened {axis = 0 : i32} : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}
