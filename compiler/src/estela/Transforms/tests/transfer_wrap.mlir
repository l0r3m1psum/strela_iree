// RUN: iree-opt --iree-plugin=estela \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-strela-transfer-wrap))' %s \
// RUN:   | FileCheck %s

// The matvec's N only exists as a dim of its operands, so the transfer back
// needs a tensor.dim. It must be of the init, built before the op: one of the
// result would be rewired to the transfer back and the IR would not verify.
// The zero fill stays on the host; only the matvec's inputs move to STRELA.

module attributes {stream.affinity.default = #hal.device.affinity<@cpu0>} {
  util.global private @cpu0 = #hal.device.target<
    "local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]
  > : !hal.device
  util.global private @strela0 = #hal.device.target<
    "strela", [#hal.executable.target<"strela", "strela_bitstream">]
  > : !hal.device

  util.func public @matvec(%x: tensor<1x?x16xi32>, %w: tensor<1x16x8xi32>) -> tensor<1x?x8xi32> {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %n = tensor.dim %x, %c1 : tensor<1x?x16xi32>
    %empty = tensor.empty(%n) : tensor<1x?x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x?x16xi32>, tensor<1x16x8xi32>)
                             outs(%init : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
    util.return %r : tensor<1x?x8xi32>
  }
}

// CHECK-LABEL: util.func public @matvec(
// CHECK-SAME:    %[[X:[^:]+]]: tensor<1x?x16xi32>, %[[W:[^:]+]]: tensor<1x16x8xi32>)
// CHECK:         %[[INIT:.+]] = linalg.fill {{.+}} -> tensor<1x?x8xi32>
// CHECK:         %[[INIT_N:.+]] = tensor.dim %[[INIT]], %{{.+}} : tensor<1x?x8xi32>
// CHECK:         %[[X_N:.+]] = tensor.dim %[[X]], %{{.+}} : tensor<1x?x16xi32>
// CHECK:         %[[X_S:.+]] = flow.tensor.transfer %[[X]] : tensor<1x?x16xi32>{%[[X_N]]} to #hal.device.affinity<@strela0>
// CHECK:         %[[W_S:.+]] = flow.tensor.transfer %[[W]] : tensor<1x16x8xi32> to #hal.device.affinity<@strela0>
// CHECK:         %[[R:.+]] = linalg.batch_matmul ins(%[[X_S]], %[[W_S]] : tensor<1x?x16xi32>, tensor<1x16x8xi32>) outs(%[[INIT]] : tensor<1x?x8xi32>)
// CHECK:         %[[R_H:.+]] = flow.tensor.transfer %[[R]] : tensor<1x?x8xi32>{%[[INIT_N]]} to #hal.device.affinity<@cpu0>
// CHECK:         util.return %[[R_H]]
