// The smallest end-to-end check of the STRELA HAL: one addition on the CGRA.
//
// On the board:
//
//   iree-run-module --module=strela_add.vmfb --function=add \
//     --device=local-sync --device=strela --device_lead_allocator=1 \
//     --input="16xi32=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" \
//     --input="16xi32=10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160" \
//     --expected_output="16xi32=11 22 33 44 55 66 77 88 99 110 121 132 143 154 165 176"
//
// Operands are copied into STRELA memory (the topology says STRELA cannot read
// the CPU's), the dispatch runs the add bitstream with them on input columns 0
// and 1, and the result comes back from output column 1. See strela_ops.mlir for
// the topology and --device_lead_allocator.
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

// CHECK: %[[LHS:.+]] = strela.binding_load binding(0) : tensor<16xi32>
// CHECK: %[[RHS:.+]] = strela.binding_load binding(0) byte_offset 64 : tensor<16xi32>
// CHECK: %[[SUM:.+]] = strela.add %[[LHS]], %[[RHS]] : tensor<16xi32>
// CHECK: strela.binding_store %[[SUM]], binding(1) : tensor<16xi32>

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
}
