// The program of tests/transfer.mlir, with the topology STRELA really has, for
// running on the board: a matvec on STRELA between CPU work.
//
//   iree-run-module --module=strela_transfer.vmfb --function=main \
//     --device=local-sync --device=strela --device_lead_allocator=1 \
//     --input="8x9xi32=1" --input="8x9xi32=2" \
//     --input="1x8x16xi32=1" --input="1x16x9xi32=-1"
//
// gives min(relu(1 + 2) + 16 * 1 * -1) = 3 - 16 = -13. With
// --input="1x16x9xi32=2" in last place it is 3 + 16 * 2 = 35. A STRELA
// dispatch that leaves its result untouched gives 3 both times.
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

// The add and the ReLU stay on the CPU: STRELA only runs the matvec.
// CHECK:     strela.matvec %{{.+}}, %{{.+}} : tensor<1x8x16xi32>, tensor<1x16x9xi32> -> tensor<1x8x9xi32>
// CHECK-NOT: strela.matvec

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {}),
    (@device_b -> @device_a = {transparent_access = true})
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

    %a0 = "tosa.add"(%i0, %i1) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %y_b = tosa.matmul %x_b, %w_b, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>

    %r0 = tosa.maximum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    // The transfer back to device_a is a fusion boundary, as is the transfer of
    // %x/%w above.
    %y3 = flow.tensor.transfer %y_b : tensor<1x8x9xi32> to #hal.device.promise<@device_a>
    %y = tosa.reshape %y3, %mat_shape : (tensor<1x8x9xi32>, !tosa.shape<2>) -> tensor<8x9xi32>

    %a2 = "tosa.add"(%r0, %y) : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened axis(0) : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}
