// What the Estela frontend does to a quantized TFLite model, pass by pass. To
// run it from the command line
// $ ctest --test-dir iree-build -R estela/frontend --output-on-failure
//
// The frontend is the preprocessing pipeline the plugin installs
// (extendPreprocessingPassPipeline in PluginRegistration.cpp). It turns each
// fully connected layer into the one thing STRELA runs, a matmul of i32
// operands from a zero fill, and pins it to STRELA. Almost all of it is IREE's
// and MLIR's own passes; only the widening and the pinning are STRELA's:
//
//   iree-global-opt-quantized-conv-to-conv                 (IREE)
//   iree-global-opt-detach-elementwise-from-named-ops      (IREE)
//   iree-preprocessing-convert-1x1-filter-conv2d-to-matmul (IREE)
//   linalg-fold-unit-extent-dims, canonicalize             (MLIR)
//   linalg-specialize-generic-ops                          (MLIR)
//   iree-strela-widen-operands                             (STRELA)
//   iree-preprocessing-transpose-matmul-pass{input=rhs}    (IREE)
//   iree-strela-transfer-wrap                              (STRELA)
//
// The IR is printed before the first of them and after each one, and every
// stage is checked on the first layer (the second where it differs).

// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=preprocessing --mlir-print-local-scope \
// RUN:   --mlir-print-ir-before=iree-global-opt-quantized-conv-to-conv \
// RUN:   --mlir-print-ir-after=iree-global-opt-quantized-conv-to-conv \
// RUN:   --mlir-print-ir-after=iree-global-opt-detach-elementwise-from-named-ops \
// RUN:   --mlir-print-ir-after=iree-preprocessing-convert-1x1-filter-conv2d-to-matmul \
// RUN:   --mlir-print-ir-after=linalg-fold-unit-extent-dims \
// RUN:   --mlir-print-ir-after=linalg-specialize-generic-ops \
// RUN:   --mlir-print-ir-after=iree-strela-widen-operands \
// RUN:   --mlir-print-ir-after=iree-preprocessing-transpose-matmul-pass \
// RUN:   --mlir-print-ir-after=iree-strela-transfer-wrap \
// RUN:   %s -o /dev/null 2>&1 \
// RUN: | FileCheck %s --check-prefixes=INP,QCONV,DETACH,CONV1X1,UNIT,NAMED,WIDEN,TRANSPOSE,WRAP

// Two fully connected layers as TFLite emits them, the shape of every layer of
// MLPerf Tiny's anomaly detection model, only smaller (4 -> 3 -> 3): a
// quantized 1x1 tosa.conv2d on i8 with an input zero point zx (5, then -128),
// symmetric weights (zw = 0) and an i32 bias, then a rescale back to i8 and a
// clamp.

func.func @main(%x: tensor<?x4xi8>) -> tensor<?x3xi8> {
  %xs = tosa.const_shape values(dense<[-1, 1, 1, 4]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %hs = tosa.const_shape values(dense<[-1, 1, 1, 3]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %h2 = tosa.const_shape values(dense<[-1, 3]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %zx0 = "tosa.const"() {values = dense<5> : tensor<1xi8>} : () -> tensor<1xi8>
  %zx1 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
  %zw = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %w0 = "tosa.const"() {values = dense<[[[[1, -2, 3, -4]]], [[[5, 6, -7, 8]]], [[[-9, 10, 11, 12]]]]> : tensor<3x1x1x4xi8>} : () -> tensor<3x1x1x4xi8>
  %b0 = "tosa.const"() {values = dense<[100, -200, 300]> : tensor<3xi32>} : () -> tensor<3xi32>
  %w1 = "tosa.const"() {values = dense<[[[[1, 2, 3]]], [[[-4, 5, -6]]], [[[7, -8, 9]]]]> : tensor<3x1x1x3xi8>} : () -> tensor<3x1x1x3xi8>
  %b1 = "tosa.const"() {values = dense<[-10, 20, -30]> : tensor<3xi32>} : () -> tensor<3xi32>
  %mul = "tosa.const"() {values = dense<1462485049> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<40> : tensor<1xi8>} : () -> tensor<1xi8>
  %izp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %ozp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>

  %x4 = tosa.reshape %x, %xs : (tensor<?x4xi8>, !tosa.shape<4>) -> tensor<?x1x1x4xi8>
  %a0 = tosa.conv2d %x4, %w0, %b0, %zx0, %zw pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<?x1x1x4xi8>, tensor<3x1x1x4xi8>, tensor<3xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1x1x3xi32>
  %r0 = tosa.rescale %a0, %mul, %shift, %izp, %ozp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<?x1x1x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<?x1x1x3xi8>
  %r0f = tosa.reshape %r0, %h2 : (tensor<?x1x1x3xi8>, !tosa.shape<2>) -> tensor<?x3xi8>
  %c0 = tosa.clamp %r0f min_val(-128 : i8) max_val(127 : i8) : (tensor<?x3xi8>) -> tensor<?x3xi8>
  %c04 = tosa.reshape %c0, %hs : (tensor<?x3xi8>, !tosa.shape<4>) -> tensor<?x1x1x3xi8>
  %a1 = tosa.conv2d %c04, %w1, %b1, %zx1, %zw pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<?x1x1x3xi8>, tensor<3x1x1x3xi8>, tensor<3xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1x1x3xi32>
  %r1 = tosa.rescale %a1, %mul, %shift, %izp, %ozp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<?x1x1x3xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<?x1x1x3xi8>
  %r1f = tosa.reshape %r1, %h2 : (tensor<?x1x1x3xi8>, !tosa.shape<2>) -> tensor<?x3xi8>
  %y = tosa.clamp %r1f min_val(-128 : i8) max_val(127 : i8) : (tensor<?x3xi8>) -> tensor<?x3xi8>
  return %y : tensor<?x3xi8>
}

// Input conversion lowers TOSA to linalg: each layer is a
// linalg.conv_2d_nhwc_hwcf_q of the NHWC activations and the HWCF weights (a
// linalg.transpose of TOSA's OHWI constant), with the zero points as scalar
// operands and the bias broadcast into the accumulator. The rescale is an
// elementwise linalg.generic from i32 to i8 (TFLite's DOUBLE_ROUND rounding),
// and the clamp, a no-op on i8, is gone.
//
//   y[n, f] = b[f] + sum_c{(x[n, c] - zx) * (w[c, f] - zw)}

// INP-LABEL: IR Dump Before LinalgQuantizedConvToConvPass
// INP:         %[[X:.+]] = hal.tensor.import %{{.+}} : !hal.buffer_view -> tensor<?x4xi8>
// INP:         %[[X4:.+]] = tensor.expand_shape %[[X]] {{.+}} into tensor<?x1x1x4xi8>
// INP:         %[[W:.+]] = linalg.transpose ins(%{{.+}} : tensor<3x1x1x4xi8>) {{.+}} permutation = [1, 2, 3, 0]
// INP:         %[[B:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<3xi32>) outs(%{{.+}} : tensor<?x1x1x3xi32>)
// INP:         %[[Y:.+]] = linalg.conv_2d_nhwc_hwcf_q
// INP-SAME:      ins(%[[X4]], %[[W]], %c5_i32, %c0_i32 : tensor<?x1x1x4xi8>, tensor<1x1x4x3xi8>, i32, i32)
// INP-SAME:      outs(%[[B]] : tensor<?x1x1x3xi32>)
// INP:         linalg.generic {{.+}} ins(%[[Y]] : tensor<?x1x1x3xi32>) outs(%{{.+}} : tensor<?x1x1x3xi8>)
// INP:           arith.trunci %{{.+}} : i32 to i8

// IREE's QuantizedConvToConv takes the zero points out. With zw = 0 only the
// zx term is left, and since zx is a scalar it is a per-feature vector:
//
//   y[n, f] = b[f] + sum_c{x[n, c] * w[c, f]} - zx * sum_c{w[c, f]}
//
// The convolution is now a plain one, still of i8 operands (they are extended
// inside its body), and sum_c w is a reduction of a constant, which const-eval
// folds.

// QCONV-LABEL: IR Dump After LinalgQuantizedConvToConvPass
// QCONV:         %[[W:.+]] = linalg.transpose
// QCONV:         %[[B:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<3xi32>)
// QCONV:         %[[Y:.+]] = linalg.conv_2d_nhwc_hwcf
// QCONV-SAME:      ins(%{{.+}}, %[[W]] : tensor<?x1x1x4xi8>, tensor<1x1x4x3xi8>) outs(%[[B]] : tensor<?x1x1x3xi32>)
// QCONV:         %[[S:.+]] = linalg.generic {{.+}} iterator_types = ["parallel", "reduction", "reduction", "reduction"]}
// QCONV-SAME:      ins(%[[W]] : tensor<1x1x4x3xi8>) outs(%{{.+}} : tensor<3xi32>)
// QCONV:         linalg.generic {{.+}} ins(%[[Y]], %[[S]] : tensor<?x1x1x3xi32>, tensor<3xi32>)
// QCONV:           %[[ZXS:.+]] = arith.muli %{{.+}}, %c5_i32 : i32
// QCONV:           arith.subi %{{.+}}, %[[ZXS]] : i32

// STRELA's kernel has no accumulator input, it computes from zero. IREE's
// DetachElementwiseFromNamedOps moves the bias out of the accumulator: the
// convolution starts from a zero fill and the bias is added after it.

// DETACH-LABEL: IR Dump After DetachElementwiseFromNamedOpsPass
// DETACH:         %[[B:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<3xi32>)
// DETACH:         %[[ZERO:.+]] = linalg.fill ins(%c0_i32 : i32) outs(%{{.+}} : tensor<?x1x1x3xi32>)
// DETACH:         %[[Y:.+]] = linalg.conv_2d_nhwc_hwcf {{.+}} outs(%[[ZERO]] : tensor<?x1x1x3xi32>)
// DETACH:         linalg.generic {{.+}} ins(%[[Y]], %[[B]] : tensor<?x1x1x3xi32>, tensor<?x1x1x3xi32>)
// DETACH:           arith.addi

// A 1x1 filter makes the convolution a matrix multiplication: IREE's
// Convert1X1FilterConv2DToMatmul rewrites it into a linalg.generic that reads
// the input at kernel position 0. It still has all seven loops (n, h, w, f,
// kh, kw, c), unit ones included.

// CONV1X1-LABEL: IR Dump After Convert1X1FilterConv2DToMatmulPass
// CONV1X1-NOT:     linalg.conv_2d_nhwc_hwcf
// CONV1X1:         linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d6)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d4, d5, d6, d3)>, affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>]
// CONV1X1-SAME:      iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
// CONV1X1-SAME:      ins(%{{.+}}, %{{.+}} : tensor<?x1x1x4xi8>, tensor<1x1x4x3xi8>)

// MLIR's FoldUnitExtentDims drops them: a contraction over (n, f, c) of the
// ?x4 activations, straight from the function's input, and the 4x3 weights.

// UNIT-LABEL: IR Dump After LinalgFoldUnitExtentDimsPass
// UNIT:         %[[X:.+]] = hal.tensor.import
// UNIT:         %[[W:.+]] = tensor.collapse_shape %{{.+}} {{\[}}[0, 1, 2], [3]] : tensor<1x1x4x3xi8> into tensor<4x3xi8>
// UNIT:         linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// UNIT-SAME:      iterator_types = ["parallel", "parallel", "reduction"]}
// UNIT-SAME:      ins(%[[X]], %[[W]] : tensor<?x4xi8>, tensor<4x3xi8>) outs(%{{.+}} : tensor<?x3xi32>)

// MLIR's SpecializeGenericOps names what it recognizes: the contraction is a
// linalg.matmul (i8 x i8 into i32), the bias a linalg.broadcast and its add a
// linalg.elementwise. IREE's GlobalOptimization turns the latter two back into
// generics later.

// NAMED-LABEL: IR Dump After LinalgSpecializeGenericOpsPass
// NAMED:         %[[B:.+]] = linalg.broadcast ins(%{{.+}} : tensor<3xi32>) outs(%{{.+}} : tensor<?x3xi32>) dimensions = [0]
// NAMED:         %[[ZERO:.+]] = linalg.fill
// NAMED:         %[[Y:.+]] = linalg.matmul ins(%{{.+}}, %{{.+}} : tensor<?x4xi8>, tensor<4x3xi8>) outs(%[[ZERO]] : tensor<?x3xi32>)
// NAMED:         linalg.elementwise <add> ins(%[[Y]], %[[B]] : tensor<?x3xi32>, tensor<?x3xi32>)

// STRELA moves i32 words, so StrelaWidenOperands extends the operands of the
// matmuls it will run to i32 (the CPU would keep i8). The model's input gets
// an extsi of its own, the constant weights one that const-eval folds.

// WIDEN-LABEL: IR Dump After StrelaWidenOperandsPass
// WIDEN:         %[[X:.+]] = hal.tensor.import
// WIDEN:         %[[X32:.+]] = linalg.generic {{.+}} ins(%[[X]] : tensor<?x4xi8>) outs(%{{.+}} : tensor<?x4xi32>)
// WIDEN:           arith.extsi %{{.+}} : i8 to i32
// WIDEN:         %[[W32:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<4x3xi8>) outs(%{{.+}} : tensor<4x3xi32>)
// WIDEN:           arith.extsi %{{.+}} : i8 to i32
// WIDEN:         linalg.matmul ins(%[[X32]], %[[W32]] : tensor<?x4xi32>, tensor<4x3xi32>)

// The second layer's activations are the first layer's rescale, used only by
// this matmul: the extsi goes into the rescale's body, right after its
// truncation, so i8 only exists inside it and the CPU dispatch writes i32
// directly. (IREE never fuses a truncation with an extension, so a separate
// widening would cost a dispatch and an i8 buffer per layer.)

// WIDEN:         %[[X1:.+]] = linalg.generic {{.+}} ins(%{{[a-z0-9_]+}} : tensor<?x3xi32>) outs(%{{[a-z0-9_]+}} : tensor<?x3xi32>)
// WIDEN:           %[[T:.+]] = arith.trunci %{{.+}} : i32 to i8
// WIDEN:           arith.extsi %[[T]] : i8 to i32
// WIDEN:         linalg.matmul ins(%[[X1]], %{{.+}} : tensor<?x3xi32>, tensor<3x3xi32>)

// STRELA reads each feature's weights as one DMA column, which is shortest
// when they are contiguous: IREE's TransposeMatmul transposes the weights of
// every matmul. The weights go through a linalg.transpose to F x C (const-eval
// folds it with the widening), and the matmul reads them through the
// transposed-B map.

// TRANSPOSE-LABEL: IR Dump After TransposeMatmulPass
// TRANSPOSE:         %[[W32:.+]] = linalg.generic {{.+}} ins(%{{.+}} : tensor<4x3xi8>) outs(%{{.+}} : tensor<4x3xi32>)
// TRANSPOSE:         %[[WT:.+]] = linalg.transpose ins(%[[W32]] : tensor<4x3xi32>) outs(%{{.+}} : tensor<3x4xi32>) permutation = [1, 0]
// TRANSPOSE:         linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>]
// TRANSPOSE-SAME:      ins(%{{.+}}, %[[WT]] : tensor<?x4xi32>, tensor<3x4xi32>)

// Finally TransferWrap pins each matmul to STRELA: its operands move to it
// with flow.tensor.transfer and its result moves back. The zero fill stays
// outside (dispatch formation pulls it in), and so do the bias add, the zero
// point correction and the rescale, which the CPU runs. Dispatch creation
// then makes two dispatches per layer, see quantized_fc.mlir.

// WRAP-LABEL: IR Dump After TransferWrapPass
// WRAP:         %[[X32:.+]] = linalg.generic {{.+}} outs(%{{.+}} : tensor<?x4xi32>)
// WRAP:         %[[WT:.+]] = linalg.transpose
// WRAP:         %[[XS:.+]] = flow.tensor.transfer %[[X32]] : tensor<?x4xi32>{%{{.+}}} to #hal.device.affinity<@strela0>
// WRAP:         %[[WS:.+]] = flow.tensor.transfer %[[WT]] : tensor<3x4xi32> to #hal.device.affinity<@strela0>
// WRAP:         %[[YS:.+]] = linalg.matmul {{.+}} ins(%[[XS]], %[[WS]] : tensor<?x4xi32>, tensor<3x4xi32>)
// WRAP:         flow.tensor.transfer %[[YS]] : tensor<?x3xi32>{%{{.+}}} to #hal.device.affinity<@cpu0>
