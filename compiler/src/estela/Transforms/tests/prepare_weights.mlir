// RUN: iree-opt --iree-plugin=estela --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-strela-prepare-weights))' %s \
// RUN:   | FileCheck %s

// iree-strela-prepare-weights transposes the constant C x F weights of a
// matvec STRELA will run to F x C, so that each feature's weights are
// contiguous, and rewrites the contraction to read them that way.

#id3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

module attributes {stream.affinity.default = #hal.device.affinity<@cpu0>} {
  util.global private @cpu0 = #hal.device.target<
    "local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]
  > : !hal.device
  util.global private @strela0 = #hal.device.target<
    "strela", [#hal.executable.target<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>]
  > : !hal.device

  // C x F = 640 x 128 = 81920 words, more than a strided column covers; as
  // F x C a weight column is 640 contiguous words. The weights are computed
  // from constants only (a broadcast), which const-eval folds, transpose
  // included.

  // CHECK-LABEL: util.func public @wide_constant(
  // CHECK:         %[[W:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<1x1x128xi32>)
  // CHECK:         %[[EMPTY:.+]] = tensor.empty() : tensor<1x128x640xi32>
  // CHECK:         %[[WT:.+]] = linalg.transpose ins(%[[W]] : tensor<1x640x128xi32>) outs(%[[EMPTY]] : tensor<1x128x640xi32>) permutation = [0, 2, 1]
  // CHECK:         linalg.generic
  // CHECK-SAME:      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
  // CHECK-SAME:      ins(%{{.+}}, %[[WT]] : tensor<1x2x640xi32>, tensor<1x128x640xi32>)
  // CHECK-NOT:     linalg.batch_matmul
  util.func public @wide_constant(%x16: tensor<1x2x640xi16>) -> tensor<1x2x128xi32> {
    %c0_i32 = arith.constant 0 : i32
    %xe = tensor.empty() : tensor<1x2x640xi32>
    %x = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x16 : tensor<1x2x640xi16>) outs(%xe : tensor<1x2x640xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x2x640xi32>
    %w1 = arith.constant dense<[[[1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8, 9, -9, 10, -10, 11, -11, 12, -12, 13, -13, 14, -14, 15, -15, 16, -16, 17, -17, 18, -18, 19, -19, 20, -20, 21, -21, 22, -22, 23, -23, 24, -24, 25, -25, 26, -26, 27, -27, 28, -28, 29, -29, 30, -30, 31, -31, 32, -32, 33, -33, 34, -34, 35, -35, 36, -36, 37, -37, 38, -38, 39, -39, 40, -40, 41, -41, 42, -42, 43, -43, 44, -44, 45, -45, 46, -46, 47, -47, 48, -48, 49, -49, 50, -50, 51, -51, 52, -52, 53, -53, 54, -54, 55, -55, 56, -56, 57, -57, 58, -58, 59, -59, 60, -60, 61, -61, 62, -62, 63, -63, 64, -64]]]> : tensor<1x1x128xi32>
    %we = tensor.empty() : tensor<1x640x128xi32>
    %w = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, 0, d2)>, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%w1 : tensor<1x1x128xi32>) outs(%we : tensor<1x640x128xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<1x640x128xi32>
    %empty = tensor.empty() : tensor<1x2x128xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x2x128xi32>) -> tensor<1x2x128xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x2x640xi32>, tensor<1x640x128xi32>)
                             outs(%init : tensor<1x2x128xi32>) -> tensor<1x2x128xi32>
    util.return %r : tensor<1x2x128xi32>
  }

  // A TFLite fully connected layer: a quantized 1x1 convolution on i8, with
  // constant weights, an input zero point zx = 5, a weight zero point zw = 0,
  // and the bias as the init. It becomes
  //   y = bias + matmul(extsi(x), w') - zx * sum_c w'[c, f],   w' = extsi(w) - zw
  // where the matmul is from a zero fill (what STRELA runs, its weights then
  // transposed as above) and the last term is computed from constants alone.

  // CHECK-LABEL: util.func public @quantized_fc(
  // CHECK-SAME:    %[[X:[^:]+]]: tensor<?x1x1x16xi8>
  // CHECK:         %[[X2:.+]] = tensor.collapse_shape %[[X]] {{\[}}[0, 1, 2], [3]] : tensor<?x1x1x16xi8> into tensor<?x16xi8>
  // CHECK:         %[[W2:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1, 2], [3]] : tensor<1x1x16x8xi8> into tensor<16x8xi8>
  // CHECK:         %[[X32:.+]] = linalg.generic {{.+}} ins(%[[X2]] : tensor<?x16xi8>)
  // CHECK:           arith.extsi
  // CHECK:         %[[W32:.+]] = linalg.generic {{.+}} ins(%[[W2]] : tensor<16x8xi8>)
  // CHECK:           arith.extsi
  // CHECK:         %[[ACC0:.+]] = linalg.fill ins(%{{.+}} : i32) outs(%{{.+}} : tensor<?x8xi32>)
  // CHECK:         %[[WT:.+]] = linalg.transpose ins(%[[W32]] : tensor<16x8xi32>) {{.+}} permutation = [1, 0]
  // CHECK:         %[[ACC:.+]] = linalg.generic {{.+}} ins(%[[X32]], %[[WT]] : tensor<?x16xi32>, tensor<8x16xi32>) outs(%[[ACC0]] : tensor<?x8xi32>)
  // CHECK:         %[[CORR:.+]] = linalg.generic {{.+}} iterator_types = ["reduction", "parallel"]} ins(%[[W32]] : tensor<16x8xi32>)
  // CHECK:           %[[MZX:.+]] = arith.constant -5 : i32
  // CHECK:           arith.muli %{{.+}}, %[[MZX]]
  // CHECK:         %[[Y:.+]] = linalg.generic {{.+}} ins(%[[ACC]], %{{.+}}, %[[CORR]] : tensor<?x8xi32>, tensor<?x8xi32>, tensor<8xi32>)
  // CHECK:         tensor.expand_shape %[[Y]] {{\[}}[0, 1, 2], [3]] {{.+}} : tensor<?x8xi32> into tensor<?x1x1x8xi32>
  // CHECK-NOT:     linalg.conv_2d_nhwc_hwcf_q
  util.func public @quantized_fc(%x: tensor<?x1x1x16xi8>, %bias: tensor<?x1x1x8xi32>) -> tensor<?x1x1x8xi32> {
    %zx = arith.constant 5 : i32
    %zw = arith.constant 0 : i32
    %w1 = arith.constant dense<[[[[1, -2, 3, -4, 5, -6, 7, -8]]]]> : tensor<1x1x1x8xi8>
    %we = tensor.empty() : tensor<1x1x16x8xi8>
    %w = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%w1 : tensor<1x1x1x8xi8>) outs(%we : tensor<1x1x16x8xi8>) {
    ^bb0(%in: i8, %out: i8):
      linalg.yield %in : i8
    } -> tensor<1x1x16x8xi8>
    %y = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%x, %w, %zx, %zw : tensor<?x1x1x16xi8>, tensor<1x1x16x8xi8>, i32, i32)
        outs(%bias : tensor<?x1x1x8xi32>) -> tensor<?x1x1x8xi32>
    util.return %y : tensor<?x1x1x8xi32>
  }

  // Weights that are an input of the function are not transposed: that would
  // cost a transpose at run time, and they fit as a strided column anyway.

  // CHECK-LABEL: util.func public @input_weights(
  // CHECK-NOT:     linalg.transpose
  // CHECK:         linalg.batch_matmul
  util.func public @input_weights(%x16: tensor<1x2x16xi16>, %w16: tensor<1x16x8xi16>) -> tensor<1x2x8xi32> {
    %c0_i32 = arith.constant 0 : i32
    %xe = tensor.empty() : tensor<1x2x16xi32>
    %x = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%x16 : tensor<1x2x16xi16>) outs(%xe : tensor<1x2x16xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x2x16xi32>
    %we = tensor.empty() : tensor<1x16x8xi32>
    %w = linalg.generic {indexing_maps = [#id3, #id3], iterator_types = ["parallel", "parallel", "parallel"]}
        ins(%w16 : tensor<1x16x8xi16>) outs(%we : tensor<1x16x8xi32>) {
    ^bb0(%in: i16, %out: i32):
      %e = arith.extsi %in : i16 to i32
      linalg.yield %e : i32
    } -> tensor<1x16x8xi32>
    %empty = tensor.empty() : tensor<1x2x8xi32>
    %init = linalg.fill ins(%c0_i32 : i32) outs(%empty : tensor<1x2x8xi32>) -> tensor<1x2x8xi32>
    %r = linalg.batch_matmul ins(%x, %w : tensor<1x2x16xi32>, tensor<1x16x8xi32>)
                             outs(%init : tensor<1x2x8xi32>) -> tensor<1x2x8xi32>
    util.return %r : tensor<1x2x8xi32>
  }
}
