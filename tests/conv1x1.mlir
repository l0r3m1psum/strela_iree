// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=preprocessing --mlir-print-local-scope \
// RUN:   --mlir-print-ir-after=linalg-fold-unit-extent-dims \
// RUN:   --mlir-print-ir-after=iree-dispatch-creation-collapse-contraction-dimensions \
// RUN:   --mlir-print-ir-after=iree-strela-transfer-wrap \
// RUN:   %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefixes=UNIT,COLLAPSE,WRAP

// A quantized 1x1 tosa.conv2d over a 5x2 image, the layer that does most of
// the work of the depthwise separable CNNs of MLPerf Tiny (KWS, VWW): every
// pixel is a fully connected layer with the same weights. The frontend (see
// frontend.mlir) treats it as it treats a fully connected one, with the
// pixels as the matmul's rows.

func.func @main(%x: tensor<1x5x2x4xi8>) -> tensor<1x5x2x3xi32> {
  %zx = "tosa.const"() {values = dense<5> : tensor<1xi8>} : () -> tensor<1xi8>
  %zw = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %w = "tosa.const"() {values = dense<[[[[1, -2, 3, -4]]], [[[5, 6, -7, 8]]], [[[-9, 10, 11, 12]]]]> : tensor<3x1x1x4xi8>} : () -> tensor<3x1x1x4xi8>
  %b = "tosa.const"() {values = dense<[100, -200, 300]> : tensor<3xi32>} : () -> tensor<3xi32>
  %y = tosa.conv2d %x, %w, %b, %zx, %zw pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x5x2x4xi8>, tensor<3x1x1x4xi8>, tensor<3xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x2x3xi32>
  return %y : tensor<1x5x2x3xi32>
}

// Once FoldUnitExtentDims has dropped the batch and the kernel position, the
// convolution is a contraction with two row dims, h and w: one more than a
// matmul has, so nothing below would take it.

// UNIT-LABEL: IR Dump After LinalgFoldUnitExtentDimsPass
// UNIT:         linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]
// UNIT-SAME:      iterator_types = ["parallel", "parallel", "parallel", "reduction"]}
// UNIT-SAME:      ins(%{{.+}}, %{{.+}} : tensor<5x2x4xi8>, tensor<4x3xi8>) outs(%{{.+}} : tensor<5x2x3xi32>)

// IREE's CollapseContractionDimensions merges h and w, contiguous in both the
// activations and the result: a contraction over (n, f, c) of the 10 pixels,
// between reshapes of its operands and its result.

// COLLAPSE-LABEL: IR Dump After CollapseContractionDimensionsPass
// COLLAPSE:         %[[X:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1], [2]] : tensor<5x2x4xi8> into tensor<10x4xi8>
// COLLAPSE:         %[[Y:.+]] = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// COLLAPSE-SAME:      ins(%[[X]], %{{.+}} : tensor<10x4xi8>, tensor<4x3xi8>) outs(%{{.+}} : tensor<10x3xi32>)
// COLLAPSE:         tensor.expand_shape %[[Y]] {{\[}}[0, 1], [2]] output_shape [5, 2, 3] : tensor<10x3xi32> into tensor<5x2x3xi32>

// From there it is a fully connected layer of 10 rows: a linalg.matmul of
// widened operands with F x C weights, pinned to STRELA.

// WRAP-LABEL: IR Dump After TransferWrapPass
// WRAP:         %[[XS:.+]] = flow.tensor.transfer %{{.+}} : tensor<10x4xi32> to #hal.device.affinity<@strela0>
// WRAP:         %[[WS:.+]] = flow.tensor.transfer %{{.+}} : tensor<3x4xi32> to #hal.device.affinity<@strela0>
// WRAP:         %[[Y:.+]] = linalg.matmul {{.+}} ins(%[[XS]], %[[WS]] : tensor<10x4xi32>, tensor<3x4xi32>) outs(%{{.+}} : tensor<10x3xi32>)
// WRAP:         flow.tensor.transfer %[[Y]] : tensor<10x3xi32> to #hal.device.affinity<@cpu0>
