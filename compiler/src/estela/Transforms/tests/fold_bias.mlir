// RUN: iree-opt --iree-plugin=estela --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(util.func(iree-strela-prepare-weights{fold-bias=true}))' %s \
// RUN:   | FileCheck %s

// With fold-bias, iree-strela-prepare-weights puts the constant bias of a
// TFLite fully connected layer into its matmul as extra reduction elements:
//   y = [x - zx, 1, ..., 1] . [w'; b_0; ...; b_{m-1}]
// with b split into m parts that each fit the 16-bit multipliers, and no add
// left on the host.

module attributes {stream.affinity.default = #hal.device.affinity<@cpu0>} {
  util.global private @cpu0 = #hal.device.target<
    "local", [#hal.executable.target<"vmvx", "vmvx-bytecode-fb">]
  > : !hal.device
  util.global private @strela0 = #hal.device.target<
    "strela", [#hal.executable.target<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>]
  > : !hal.device

  // Layer 0 reads the function's argument: x - 5 and its trailing ones are
  // one generic over the longer domain, which reads x with tensor.extract at
  // the index clamped to x and yields 1 past it. Its bias 40000 needs two
  // parts, floor(40000 / 2) and floor(40001 / 2); -3 becomes -2 and -1.
  //
  // Layer 1 reads the rescale of layer 0: x - (-128) is fused into it, and
  // the fused body is what the longer generic runs on each extracted element,
  // with a single one, as its bias fits 16 bits.

  // CHECK-LABEL: util.func public @fold_bias(
  // CHECK-SAME:    %[[X:[^:]+]]: tensor<?x1x1x4xi8>
  // CHECK:         %[[X0:.+]] = linalg.generic {{.+}} outs(%{{.+}} : tensor<?x1x1x6xi32>)
  // CHECK:           %[[N:.+]] = linalg.index 0 : index
  // CHECK:           %[[H:.+]] = linalg.index 1 : index
  // CHECK:           %[[W:.+]] = linalg.index 2 : index
  // CHECK:           %[[K:.+]] = linalg.index 3 : index
  // CHECK:           %[[C4:.+]] = arith.constant 4 : index
  // CHECK:           %[[IN:.+]] = arith.cmpi ult, %[[K]], %[[C4]] : index
  // CHECK:           %[[C3:.+]] = arith.constant 3 : index
  // CHECK:           %[[KX:.+]] = arith.minui %[[K]], %[[C3]] : index
  // CHECK:           %[[V:.+]] = tensor.extract %[[X]][%[[N]], %[[H]], %[[W]], %[[KX]]] : tensor<?x1x1x4xi8>
  // CHECK:           %[[E:.+]] = arith.extsi %[[V]] : i8 to i32
  // CHECK:           %[[S:.+]] = arith.subi %[[E]], %{{.+}} : i32
  // CHECK:           %[[ONE:.+]] = arith.constant 1 : i32
  // CHECK:           arith.select %[[IN]], %[[S]], %[[ONE]] : i32
  // CHECK:         %[[X0C:.+]] = tensor.collapse_shape %[[X0]] {{\[}}[0, 1, 2], [3]] : tensor<?x1x1x6xi32> into tensor<?x6xi32>
  // CHECK:         %[[B0:.+]] = arith.constant dense<{{\[}}[20000, -2, 0], [20000, -1, 0]]> : tensor<2x3xi32>
  // CHECK:         %[[W0:.+]] = tensor.concat dim(0) %{{.+}}, %[[B0]] : (tensor<4x3xi32>, tensor<2x3xi32>) -> tensor<6x3xi32>
  // CHECK:         %[[W0T:.+]] = linalg.transpose ins(%[[W0]] : tensor<6x3xi32>) {{.+}} permutation = [1, 0]
  // CHECK:         %[[Y0:.+]] = linalg.generic {{.+}} ins(%[[X0C]], %[[W0T]] : tensor<?x6xi32>, tensor<3x6xi32>)
  // CHECK:         %[[Y04:.+]] = tensor.expand_shape %[[Y0]] {{\[}}[0, 1, 2], [3]] {{.+}} : tensor<?x3xi32> into tensor<?x1x1x3xi32>
  // CHECK:         %[[X1:.+]] = linalg.generic {{.+}} outs(%{{.+}} : tensor<?x1x1x4xi32>)
  // CHECK:           %[[K1:.+]] = linalg.index 3 : index
  // CHECK:           %[[IN1:.+]] = arith.cmpi ult, %[[K1]], %{{.+}} : index
  // CHECK:           %[[KX1:.+]] = arith.minui %[[K1]], %{{.+}} : index
  // CHECK:           %[[V1:.+]] = tensor.extract %[[Y04]][%{{.+}}, %{{.+}}, %{{.+}}, %[[KX1]]] : tensor<?x1x1x3xi32>
  // CHECK:           %[[T1:.+]] = arith.trunci %[[V1]] : i32 to i8
  // CHECK:           %[[E1:.+]] = arith.extsi %[[T1]] : i8 to i32
  // CHECK:           %[[S1:.+]] = arith.subi %[[E1]], %{{.+}} : i32
  // CHECK:           arith.select %[[IN1]], %[[S1]], %{{.+}} : i32
  // CHECK:         %[[X1C:.+]] = tensor.collapse_shape %[[X1]] {{\[}}[0, 1, 2], [3]] : tensor<?x1x1x4xi32> into tensor<?x4xi32>
  // CHECK:         %[[B1:.+]] = arith.constant dense<{{\[}}[7, 8, 9]]> : tensor<1x3xi32>
  // CHECK:         %[[W1:.+]] = tensor.concat dim(0) %{{.+}}, %[[B1]] : (tensor<3x3xi32>, tensor<1x3xi32>) -> tensor<4x3xi32>
  // CHECK:         %[[W1T:.+]] = linalg.transpose ins(%[[W1]] : tensor<4x3xi32>)
  // CHECK:         %[[Y1:.+]] = linalg.generic {{.+}} ins(%[[X1C]], %[[W1T]] : tensor<?x4xi32>, tensor<3x4xi32>)
  // CHECK:         %[[Y14:.+]] = tensor.expand_shape %[[Y1]]
  // CHECK:         util.return %[[Y14]]
  util.func public @fold_bias(%x: tensor<?x1x1x4xi8>) -> tensor<?x1x1x3xi32> {
    %c0 = arith.constant 0 : index
    %n = tensor.dim %x, %c0 : tensor<?x1x1x4xi8>
    %zx0 = arith.constant 5 : i32
    %zx1 = arith.constant -128 : i32
    %zw = arith.constant 0 : i32
    %w0 = arith.constant dense<[[[[1, -2, 1], [3, -4, 1], [5, -6, 1], [7, -8, 1]]]]> : tensor<1x1x4x3xi8>
    %b0 = arith.constant dense<[40000, -3, 0]> : tensor<3xi32>
    %w1 = arith.constant dense<[[[[1, 2, 3], [-1, -2, -3], [0, 1, 0]]]]> : tensor<1x1x3x3xi8>
    %b1 = arith.constant dense<[7, 8, 9]> : tensor<3xi32>

    %e0 = tensor.empty(%n) : tensor<?x1x1x3xi32>
    %init0 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%b0 : tensor<3xi32>) outs(%e0 : tensor<?x1x1x3xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?x1x1x3xi32>
    %y0 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%x, %w0, %zx0, %zw : tensor<?x1x1x4xi8>, tensor<1x1x4x3xi8>, i32, i32)
        outs(%init0 : tensor<?x1x1x3xi32>) -> tensor<?x1x1x3xi32>

    // A stand-in for the rescale and clamp: i32 -> i8.
    %er = tensor.empty(%n) : tensor<?x1x1x3xi8>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%y0 : tensor<?x1x1x3xi32>) outs(%er : tensor<?x1x1x3xi8>) {
    ^bb0(%in: i32, %out: i8):
      %t = arith.trunci %in : i32 to i8
      linalg.yield %t : i8
    } -> tensor<?x1x1x3xi8>

    %e1 = tensor.empty(%n) : tensor<?x1x1x3xi32>
    %init1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
        ins(%b1 : tensor<3xi32>) outs(%e1 : tensor<?x1x1x3xi32>) {
    ^bb0(%in: i32, %out: i32):
      linalg.yield %in : i32
    } -> tensor<?x1x1x3xi32>
    %y1 = linalg.conv_2d_nhwc_hwcf_q {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
        ins(%r, %w1, %zx1, %zw : tensor<?x1x1x3xi8>, tensor<1x1x3x3xi8>, i32, i32)
        outs(%init1 : tensor<?x1x1x3xi32>) -> tensor<?x1x1x3xi32>
    util.return %y1 : tensor<?x1x1x3xi32>
  }
}
