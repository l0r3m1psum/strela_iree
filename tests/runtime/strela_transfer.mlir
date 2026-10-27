// The program of tests/transfer.mlir, with the topology STRELA really has, for
// running on the board: an add and a ReLU on STRELA between CPU work.
//
//   iree-run-module --module=strela_transfer.vmfb --function=main \
//     --device=local-sync --device=strela --device_lead_allocator=1 \
//     --input="8x9xi32=1" --input="8x9xi32=2" --input="8x9xi32=3" --input="8x9xi32=-10"
//
// gives min(relu(1 + 2) + relu(3 - 10)) = 3. With --input="8x9xi32=20" in third
// place it is 3 + relu(20 - 10) = 13. A bypass in place of either STRELA kernel
// changes at least one of the two results (add: 6 and 23, relu: -4 and 13).
//
// transfer.mlir declares unified memory, which binds the CPU buffers themselves
// to the STRELA dispatch; the HAL rejects that. See strela_ops.mlir for the
// links used here and for --device_lead_allocator.
//
// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=strela \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=executable-targets %s | FileCheck %s
//
// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=strela \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   %s -o /dev/null

// CHECK-DAG: strela.add %{{.+}}, %{{.+}} : tensor<72xi32>
// CHECK-DAG: strela.relu %{{.+}} : tensor<72xi32>

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {}),
    (@device_b -> @device_a = {transparent_access = true})
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
    %new_shape = tosa.const_shape values(dense<[-1]> : tensor<1xindex>) : () -> !tosa.shape<1>

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
    %min_tensor = tosa.reduce_min %flattened axis(0) : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}
