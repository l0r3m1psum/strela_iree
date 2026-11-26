// RUN: iree-opt --iree-plugin=estela \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-strela-transfer-wrap))' %s \
// RUN:   | FileCheck %s

// iree-strela-transfer-wrap pins to STRELA the contractions the matvec kernel
// runs whose operands are proven to fit the 16 bit multipliers, by wrapping
// them in flow.tensor.transfer. The operands here are i16 values widened to
// i32 by a linalg.generic with arith.extsi, which is what tosa.cast lowers to.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {stream.affinity.default = #hal.device.affinity<@cpu0>} {
  util.global private @cpu0 = #hal.device.target<
    "local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]
  > : !hal.device
  util.global private @strela0 = #hal.device.target<
    "strela", [#hal.executable.target<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>]
  > : !hal.device

  // The matvec's N only exists as a dim of its operands, so the transfer back
  // needs a tensor.dim. It must be of the init, built before the op: one of
  // the result would be rewired to the transfer back and the IR would not
  // verify. The zero fill and the widening stay on the host; only the matvec's
  // inputs move to STRELA.

  // CHECK-LABEL: util.func public @matvec(
  // CHECK:         %[[X:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<1x?x16xi16>)
  // CHECK:         %[[W:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<1x16x8xi16>)
  // CHECK:         %[[INIT:.+]] = linalg.fill {{.+}} -> tensor<1x?x8xi32>
  // CHECK:         %[[INIT_N:.+]] = tensor.dim %[[INIT]], %{{.+}} : tensor<1x?x8xi32>
  // CHECK:         %[[X_N:.+]] = tensor.dim %[[X]], %{{.+}} : tensor<1x?x16xi32>
  // CHECK:         %[[X_S:.+]] = flow.tensor.transfer %[[X]] : tensor<1x?x16xi32>{%[[X_N]]} to #hal.device.affinity<@strela0>
  // CHECK:         %[[W_S:.+]] = flow.tensor.transfer %[[W]] : tensor<1x16x8xi32> to #hal.device.affinity<@strela0>
  // CHECK:         %[[R:.+]] = linalg.batch_matmul ins(%[[X_S]], %[[W_S]] : tensor<1x?x16xi32>, tensor<1x16x8xi32>) outs(%[[INIT]] : tensor<1x?x8xi32>)
  // CHECK:         %[[R_H:.+]] = flow.tensor.transfer %[[R]] : tensor<1x?x8xi32>{%[[INIT_N]]} to #hal.device.affinity<@cpu0>
  // CHECK:         util.return %[[R_H]]
  util.func public @matvec(%x16: tensor<1x?x16xi16>, %w16: tensor<1x16x8xi16>) -> tensor<1x?x8xi32> {
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    %n = tensor.dim %x16, %c1 : tensor<1x?x16xi16>
    %xe = tensor.empty(%n) : tensor<1x?x16xi32>
    %x = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x16 : tensor<1x?x16xi16>) outs(%xe : tensor<1x?x16xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x?x16xi32>
    %we = tensor.empty() : tensor<1x16x8xi32>
    %w = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%w16 : tensor<1x16x8xi16>) outs(%we : tensor<1x16x8xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x16x8xi32>
    %empty = tensor.empty(%n) : tensor<1x?x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x?x16xi32>, tensor<1x16x8xi32>)
                             outs(%init : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
    util.return %r : tensor<1x?x8xi32>
  }

  // The match is structural: a linalg.generic with the maps and the
  // multiply-accumulate body of linalg.batch_matmul is a matvec too. The
  // weights are a broadcast of a constant within int16, -2^15 and 2^15 - 1
  // included.

  // CHECK-LABEL: util.func public @matvec_generic(
  // CHECK:         %[[X:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<1x2x16xi16>)
  // CHECK:         %[[W:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<1x1x8xi32>)
  // CHECK:         %[[X_S:.+]] = flow.tensor.transfer %[[X]] : tensor<1x2x16xi32> to #hal.device.affinity<@strela0>
  // CHECK:         %[[W_S:.+]] = flow.tensor.transfer %[[W]] : tensor<1x16x8xi32> to #hal.device.affinity<@strela0>
  // CHECK:         %[[R:.+]] = linalg.generic {{.+}} ins(%[[X_S]], %[[W_S]] : tensor<1x2x16xi32>, tensor<1x16x8xi32>)
  // CHECK:         %[[R_H:.+]] = flow.tensor.transfer %[[R]] : tensor<1x2x8xi32> to #hal.device.affinity<@cpu0>
  // CHECK:         util.return %[[R_H]]
  util.func public @matvec_generic(%x16: tensor<1x2x16xi16>) -> tensor<1x2x8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %xe = tensor.empty() : tensor<1x2x16xi32>
    %x = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x16 : tensor<1x2x16xi16>) outs(%xe : tensor<1x2x16xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x2x16xi32>
    %w = arith.constant dense<[[[-32768, 1, 2, 3, 4, 5, 6, 32767]]]> : tensor<1x1x8xi32>
    %wb = tensor.empty() : tensor<1x16x8xi32>
    %wfull = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%w : tensor<1x1x8xi32>) outs(%wb : tensor<1x16x8xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x16x8xi32>
    %empty = tensor.empty() : tensor<1x2x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x2x8xi32>) -> tensor<1x2x8xi32>
    %r = linalg.generic {
      indexing_maps = [affine_map<(b, n, f, c) -> (b, n, c)>,
                       affine_map<(b, n, f, c) -> (b, c, f)>,
                       affine_map<(b, n, f, c) -> (b, n, f)>],
      iterator_types = ["parallel", "parallel", "parallel", "reduction"]
    } ins(%x, %wfull : tensor<1x2x16xi32>, tensor<1x16x8xi32>) outs(%init : tensor<1x2x8xi32>) {
    ^bb0(%a: i32, %b: i32, %acc: i32):
      %mul = arith.muli %a, %b : i32
      %add = arith.addi %acc, %mul : i32
      linalg.yield %add : i32
    } -> tensor<1x2x8xi32>
    util.return %r : tensor<1x2x8xi32>
  }

  // A job computes 3 features, so F = 2 has no valid tiling: the frontend
  // leaves the op on the host instead of pinning something the backend would
  // reject.

  // CHECK-LABEL: util.func public @matvec_f2(
  // CHECK-NOT:     flow.tensor.transfer
  // CHECK:         linalg.batch_matmul
  // CHECK-NOT:     flow.tensor.transfer
  // CHECK:         util.return
  util.func public @matvec_f2(%x16: tensor<1x4x16xi16>, %w16: tensor<1x16x2xi16>) -> tensor<1x4x2xi32> {
    %c0_i32 = arith.constant 0 : i32
    %xe = tensor.empty() : tensor<1x4x16xi32>
    %x = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x16 : tensor<1x4x16xi16>) outs(%xe : tensor<1x4x16xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x4x16xi32>
    %we = tensor.empty() : tensor<1x16x2xi32>
    %w = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%w16 : tensor<1x16x2xi16>) outs(%we : tensor<1x16x2xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x16x2xi32>
    %empty = tensor.empty() : tensor<1x4x2xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x4x2xi32>) -> tensor<1x4x2xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x4x16xi32>, tensor<1x16x2xi32>)
                             outs(%init : tensor<1x4x2xi32>) -> tensor<1x4x2xi32>
    util.return %r : tensor<1x4x2xi32>
  }

  // An i32 operand of unknown range could use the upper bits the multipliers
  // drop: the op stays on the host.

  // CHECK-LABEL: util.func public @unknown_range(
  // CHECK-NOT:     flow.tensor.transfer
  // CHECK:         util.return
  util.func public @unknown_range(%x: tensor<1x4x16xi32>, %w16: tensor<1x16x8xi16>) -> tensor<1x4x8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %we = tensor.empty() : tensor<1x16x8xi32>
    %w = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%w16 : tensor<1x16x8xi16>) outs(%we : tensor<1x16x8xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x16x8xi32>
    %empty = tensor.empty() : tensor<1x4x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x4x8xi32>) -> tensor<1x4x8xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x4x16xi32>, tensor<1x16x8xi32>)
                             outs(%init : tensor<1x4x8xi32>) -> tensor<1x4x8xi32>
    util.return %r : tensor<1x4x8xi32>
  }

  // arith ops on tensors are elementwise and bound the elements as they do
  // scalars: %x is i16 widened to i32 without a linalg op, and %w is an i32 of
  // unknown range shifted right by 16, so within [-2^15, 2^15).

  // CHECK-LABEL: util.func public @tensor_arith(
  // CHECK:         %[[X:.+]] = arith.extsi %{{.+}} : tensor<1x4x16xi16> to tensor<1x4x16xi32>
  // CHECK:         %[[W:.+]] = arith.shrsi %{{.+}}, %{{.+}} : tensor<1x16x8xi32>
  // CHECK:         %[[X_S:.+]] = flow.tensor.transfer %[[X]] : tensor<1x4x16xi32> to #hal.device.affinity<@strela0>
  // CHECK:         %[[W_S:.+]] = flow.tensor.transfer %[[W]] : tensor<1x16x8xi32> to #hal.device.affinity<@strela0>
  // CHECK:         linalg.batch_matmul ins(%[[X_S]], %[[W_S]] : tensor<1x4x16xi32>, tensor<1x16x8xi32>)
  util.func public @tensor_arith(%x16: tensor<1x4x16xi16>, %w32: tensor<1x16x8xi32>) -> tensor<1x4x8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %x = arith.extsi %x16 : tensor<1x4x16xi16> to tensor<1x4x16xi32>
    %c16 = arith.constant dense<16> : tensor<1x16x8xi32>
    %w = arith.shrsi %w32, %c16 : tensor<1x16x8xi32>
    %empty = tensor.empty() : tensor<1x4x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x4x8xi32>) -> tensor<1x4x8xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x4x16xi32>, tensor<1x16x8xi32>)
                             outs(%init : tensor<1x4x8xi32>) -> tensor<1x4x8xi32>
    util.return %r : tensor<1x4x8xi32>
  }

  // So does a constant with a value outside int16 (32768).

  // CHECK-LABEL: util.func public @constant_out_of_range(
  // CHECK-NOT:     flow.tensor.transfer
  // CHECK:         util.return
  util.func public @constant_out_of_range(%x16: tensor<1x4x16xi16>) -> tensor<1x4x3xi32> {
    %c0_i32 = arith.constant 0 : i32
    %xe = tensor.empty() : tensor<1x4x16xi32>
    %x = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x16 : tensor<1x4x16xi16>) outs(%xe : tensor<1x4x16xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x4x16xi32>
    %w = arith.constant dense<32768> : tensor<1x16x3xi32>
    %empty = tensor.empty() : tensor<1x4x3xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x4x3xi32>) -> tensor<1x4x3xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x4x16xi32>, tensor<1x16x3xi32>)
                             outs(%init : tensor<1x4x3xi32>) -> tensor<1x4x3xi32>
    util.return %r : tensor<1x4x3xi32>
  }
}
