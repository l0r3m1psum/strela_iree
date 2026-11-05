// RUN: iree-opt --iree-plugin=estela \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-strela-transfer-wrap))' %s \
// RUN:   | FileCheck %s

// The relu's size only exists as the dim of its input, so the transfer back
// needs a tensor.dim. It must be of the init, built before the op: one of the
// result would be rewired to the transfer back and the IR would not verify.

#map = affine_map<(d0) -> (d0)>
module attributes {stream.affinity.default = #hal.device.affinity<@cpu0>} {
  util.global private @cpu0 = #hal.device.target<
    "local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]
  > : !hal.device
  util.global private @strela0 = #hal.device.target<
    "strela", [#hal.executable.target<"strela", "strela_bitstream">]
  > : !hal.device

  util.func public @relu(%x: tensor<?xi32>) -> tensor<?xi32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %n = tensor.dim %x, %c0 : tensor<?xi32>
    %init = tensor.empty(%n) : tensor<?xi32>
    %r = linalg.generic {
      indexing_maps = [#map, #map], iterator_types = ["parallel"]
    } ins(%x : tensor<?xi32>) outs(%init : tensor<?xi32>) {
    ^bb0(%in: i32, %out: i32):
      %m = arith.maxsi %in, %c0_i32 : i32
      linalg.yield %m : i32
    } -> tensor<?xi32>
    util.return %r : tensor<?xi32>
  }
}

// CHECK-LABEL: util.func public @relu(
// CHECK-SAME:    %[[X:.+]]: tensor<?xi32>)
// CHECK:         %[[INIT:.+]] = tensor.empty(%{{.+}}) : tensor<?xi32>
// CHECK:         %[[INIT_N:.+]] = tensor.dim %[[INIT]], %{{.+}} : tensor<?xi32>
// CHECK:         %[[X_N:.+]] = tensor.dim %[[X]], %{{.+}} : tensor<?xi32>
// CHECK:         %[[X_S:.+]] = flow.tensor.transfer %[[X]] : tensor<?xi32>{%[[X_N]]} to #hal.device.affinity<@strela0>
// CHECK:         %[[R:.+]] = linalg.generic {{.+}} ins(%[[X_S]] : tensor<?xi32>) outs(%[[INIT]] : tensor<?xi32>)
// CHECK:         %[[R_H:.+]] = flow.tensor.transfer %[[R]] : tensor<?xi32>{%[[INIT_N]]} to #hal.device.affinity<@cpu0>
// CHECK:         util.return %[[R_H]]
