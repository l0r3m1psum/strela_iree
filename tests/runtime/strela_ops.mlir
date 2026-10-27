// One STRELA operation per function, for running on the board:
//
//   iree-run-module --module=strela_ops.vmfb --device=local-sync --device=strela \
//     --device_lead_allocator=1 --function=add \
//     --input="16xi32=..." --input="16xi32=..."
//
// The topology says what STRELA can and cannot do: the CPU reads STRELA memory
// in place, STRELA cannot reach the CPU's, so the operands are copied into STRELA
// buffers and the result is read back directly.
//
// Note the direction: ElideAsyncCopies elides a transfer from A to B when
// hasTransparentAccess(A, B), so the link that keeps the copies into device_b is
// (@device_a -> @device_b = {}).
//
// --device_lead_allocator=1 makes hal.allocator.select pick STRELA's allocator
// for buffers both devices touch; with the default (the CPU) the result buffer
// of a STRELA dispatch would be memory the CGRA cannot write.
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

// CHECK-DAG: strela.add %{{.+}}, %{{.+}} : tensor<16xi32>
// CHECK-DAG: strela.relu %{{.+}} : tensor<16xi32>

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {}),
    (@device_b -> @device_a = {transparent_access = true})
  ]>
} {
  func.func @add(
    %lhs: tensor<16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %rhs: tensor<16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (tensor<16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %lhs_b = flow.tensor.transfer %lhs : tensor<16xi32> to #hal.device.promise<@device_b>
    %rhs_b = flow.tensor.transfer %rhs : tensor<16xi32> to #hal.device.promise<@device_b>
    %sum_b = "tosa.add"(%lhs_b, %rhs_b) : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
    %sum = flow.tensor.transfer %sum_b : tensor<16xi32> to #hal.device.promise<@device_a>
    return %sum : tensor<16xi32>
  }

  func.func @relu(
    %input: tensor<16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (tensor<16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %input_b = flow.tensor.transfer %input : tensor<16xi32> to #hal.device.promise<@device_b>
    %relu_b = tosa.maximum %input_b, %zero : (tensor<16xi32>, tensor<1xi32>) -> tensor<16xi32>
    %relu = flow.tensor.transfer %relu_b : tensor<16xi32> to #hal.device.promise<@device_a>
    return %relu : tensor<16xi32>
  }
}
