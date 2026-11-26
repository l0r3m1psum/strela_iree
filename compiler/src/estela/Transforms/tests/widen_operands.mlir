// RUN: iree-opt --iree-plugin=estela --mlir-print-local-scope --split-input-file \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-strela-widen-operands))' %s \
// RUN:   --verify-diagnostics | FileCheck %s

// iree-strela-widen-operands extends the narrow inputs of the matmuls STRELA
// will run to i32, the only width its DMA moves.

#id2 = affine_map<(d0, d1) -> (d0, d1)>

module attributes {stream.affinity.default = #hal.device.affinity<@cpu0>} {
  util.global private @cpu0 = #hal.device.target<
    "local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]
  > : !hal.device
  util.global private @strela0 = #hal.device.target<
    "strela", [#hal.executable.target<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>]
  > : !hal.device

  // The model's input layer, as the frontend leaves a TFLite fully connected
  // layer: a matmul of i8 activations and constant i8 weights into i32, from
  // a zero fill. The activations have no producer to fuse with, so each input
  // gets an extsi of its own (const-eval folds the weights').

  // CHECK-LABEL: util.func public @input_layer(
  // CHECK-SAME:    %[[X:[^:]+]]: tensor<?x16xi8>
  // CHECK:         %[[X32:.+]] = linalg.generic {{.+}} ins(%[[X]] : tensor<?x16xi8>) outs(%{{.+}} : tensor<?x16xi32>)
  // CHECK:           arith.extsi %{{.+}} : i8 to i32
  // CHECK:         %[[W32:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<16x8xi8>) outs(%{{.+}} : tensor<16x8xi32>)
  // CHECK:           arith.extsi %{{.+}} : i8 to i32
  // CHECK:         linalg.matmul ins(%[[X32]], %[[W32]] : tensor<?x16xi32>, tensor<16x8xi32>) outs(%{{.+}} : tensor<?x8xi32>)
  util.func public @input_layer(%x: tensor<?x16xi8>) -> tensor<?x8xi32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %w = arith.constant dense<3> : tensor<16x8xi8>
    %n = tensor.dim %x, %c0 : tensor<?x16xi8>
    %empty = tensor.empty(%n) : tensor<?x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x8xi32>) -> tensor<?x8xi32>
    %y = linalg.matmul ins(%x, %w : tensor<?x16xi8>, tensor<16x8xi8>) outs(%init : tensor<?x8xi32>) -> tensor<?x8xi32>
    util.return %y : tensor<?x8xi32>
  }

  // A later layer: its activations are the previous layer's rescale, an
  // elementwise truncation to i8 used only by this matmul. The extsi goes into
  // the rescale's body, so the narrow value never becomes a tensor: one op
  // from i32 to i32. (Dispatch formation would not fuse a truncation with an
  // extension, and the widening would cost a dispatch of its own.)

  // CHECK-LABEL: util.func public @hidden_layer(
  // CHECK-SAME:    %[[ACC:[^:]+]]: tensor<?x16xi32>
  // CHECK-NOT:     -> tensor<?x16xi8>
  // CHECK:         %[[X32:.+]] = linalg.generic {{.+}} ins(%[[ACC]] : tensor<?x16xi32>) outs(%{{.+}} : tensor<?x16xi32>)
  // CHECK:           %[[T:.+]] = arith.trunci %{{.+}} : i32 to i8
  // CHECK:           %[[E:.+]] = arith.extsi %[[T]] : i8 to i32
  // CHECK:           linalg.yield %[[E]] : i32
  // CHECK-NOT:     -> tensor<?x16xi8>
  // CHECK:         linalg.matmul ins(%[[X32]], %{{.+}} : tensor<?x16xi32>, tensor<16x8xi32>)
  util.func public @hidden_layer(%acc: tensor<?x16xi32>) -> tensor<?x8xi32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %n = tensor.dim %acc, %c0 : tensor<?x16xi32>
    %xe = tensor.empty(%n) : tensor<?x16xi8>
    %x = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%acc : tensor<?x16xi32>) outs(%xe : tensor<?x16xi8>) {
    ^bb0(%in: i32, %out: i8):
      %t = arith.trunci %in : i32 to i8
      linalg.yield %t : i8
    } -> tensor<?x16xi8>
    %w = arith.constant dense<3> : tensor<16x8xi8>
    %empty = tensor.empty(%n) : tensor<?x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x8xi32>) -> tensor<?x8xi32>
    %y = linalg.matmul ins(%x, %w : tensor<?x16xi8>, tensor<16x8xi8>) outs(%init : tensor<?x8xi32>) -> tensor<?x8xi32>
    util.return %y : tensor<?x8xi32>
  }

  // Two features are less than one job (three): the CPU runs the matmul, on
  // its i8 operands.

  // CHECK-LABEL: util.func public @cpu_matmul(
  // CHECK-NOT:     arith.extsi
  // CHECK:         linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<?x16xi8>, tensor<16x2xi8>)
  util.func public @cpu_matmul(%x: tensor<?x16xi8>) -> tensor<?x2xi32> {
    %c0 = arith.constant 0 : index
    %c0_i32 = arith.constant 0 : i32
    %w = arith.constant dense<3> : tensor<16x2xi8>
    %n = tensor.dim %x, %c0 : tensor<?x16xi8>
    %empty = tensor.empty(%n) : tensor<?x2xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<?x2xi32>) -> tensor<?x2xi32>
    %y = linalg.matmul ins(%x, %w : tensor<?x16xi8>, tensor<16x2xi8>) outs(%init : tensor<?x2xi32>) -> tensor<?x2xi32>
    util.return %y : tensor<?x2xi32>
  }
}

// -----

// Without a STRELA device the pass has nothing to widen for: a CPU-only build
// is compiled without the plugin.

// expected-error @+1 {{no STRELA device}}
util.func public @no_strela(%x: tensor<4x16xi8>) -> tensor<4x16xi8> {
  util.return %x : tensor<4x16xi8>
}
