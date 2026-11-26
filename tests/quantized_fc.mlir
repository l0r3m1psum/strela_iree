// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=flow %s | FileCheck %s

// Two fully connected layers as TFLite emits them (the shape of every AD
// layer): a quantized 1x1 tosa.conv2d on i8 with an input zero point, i32 bias
// and accumulator, then a rescale to i8 and a clamp. The frontend (see
// frontend.mlir) turns each into a matmul of i32 operands, so each layer is two
// dispatches: the matmul on STRELA, and on the CPU the bias, the zero point
// correction (one constant: const-eval merges b - zx * sum_c w), the rescale,
// the clamp and the widening of the next layer's input, fused, writing i32.
// Only the model's input needs a widening of its own.

// CHECK-LABEL: util.func public @main
// CHECK:         %[[X:.+]] = flow.dispatch @{{.+}}_elementwise_{{[a-zA-Z0-9_]+}}_i8xi32[
// CHECK:         flow.tensor.transfer %[[X]] : {{.+}} to #hal.device.affinity<@strela0>
// CHECK:         %[[WS:.+]] = flow.tensor.transfer %{{.+}} : tensor<8x16xi32> to #hal.device.affinity<@strela0>
// CHECK:         %[[Y0:.+]] = flow.dispatch @{{.+}}_matmul_{{.+}}(%{{.+}}, %[[WS]],
// CHECK:         %[[Y0H:.+]] = flow.tensor.transfer %[[Y0]] : {{.+}} to #hal.device.affinity<@cpu0>
// CHECK:         %[[X1:.+]] = flow.dispatch @{{.+}}_elementwise_{{[a-zA-Z0-9_]+}}_i32[{{.+}}](%[[Y0H]],
// CHECK-SAME:      : (tensor<?x8xi32>{{.*}}, tensor<8xi32>, index) -> tensor<?x8xi32>
// CHECK:         %[[X1S:.+]] = flow.tensor.transfer %[[X1]] : {{.+}} to #hal.device.affinity<@strela0>
// CHECK:         %[[W1S:.+]] = flow.tensor.transfer %{{.+}} : tensor<6x8xi32> to #hal.device.affinity<@strela0>
// CHECK:         %[[Y1:.+]] = flow.dispatch @{{.+}}_matmul_{{.+}}(%[[X1S]], %[[W1S]],
// CHECK:         %[[Y1H:.+]] = flow.tensor.transfer %[[Y1]] : {{.+}} to #hal.device.affinity<@cpu0>
// CHECK:         flow.dispatch @{{.+}}_elementwise_{{[a-zA-Z0-9_]+}}xi8[{{.+}}](%[[Y1H]],
// CHECK-NOT:     flow.dispatch
// CHECK:         util.return

func.func @main(%x: tensor<?x16xi8>) -> tensor<?x6xi8> {
  %xs = tosa.const_shape values(dense<[-1, 1, 1, 16]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %hs = tosa.const_shape values(dense<[-1, 1, 1, 8]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %h2 = tosa.const_shape values(dense<[-1, 8]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %ys = tosa.const_shape values(dense<[-1, 6]> : tensor<2xindex>) : () -> !tosa.shape<2>
  %zx0 = "tosa.const"() {values = dense<89> : tensor<1xi8>} : () -> tensor<1xi8>
  %zx1 = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>
  %zw = "tosa.const"() {values = dense<0> : tensor<1xi8>} : () -> tensor<1xi8>
  %w0 = "tosa.const"() {values = dense<"0xD373A7E6278D93530A99DE168F6902B78A97F0EC92BE980EED905411A073BA22211673901416E68D7AB98C0F5CA3CBECA50B9F13CF10512FAF9B151324B1E0990D379111901FB5002F09EE47D1F8166DF5DDCDC04CAF3348BF9514CD07FF61D83BF3CA1C7B939F04ECAB42D8A76FFEEC8B772C94440F134B6152D1D832DA1900"> : tensor<8x1x1x16xi8>} : () -> tensor<8x1x1x16xi8>
  %b0 = "tosa.const"() {values = dense<"0x77010000F00400007CFFFFFF49F9FFFFA0050000AFF9FFFF4D07000081FCFFFF"> : tensor<8xi32>} : () -> tensor<8xi32>
  %w1 = "tosa.const"() {values = dense<"0xFA332B91903C34D026147F2F53F3C938E3642CD98671F7DBAC1D9EFF90B845CAA23EC0E6E56B600095ABF3E70DC863A4"> : tensor<6x1x1x8xi8>} : () -> tensor<6x1x1x8xi8>
  %b1 = "tosa.const"() {values = dense<"0x4B05000013FFFFFF02060000FD000000A4FCFFFF7D030000"> : tensor<6xi32>} : () -> tensor<6xi32>
  %mul = "tosa.const"() {values = dense<1462485049> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<40> : tensor<1xi8>} : () -> tensor<1xi8>
  %izp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %ozp = "tosa.const"() {values = dense<-128> : tensor<1xi8>} : () -> tensor<1xi8>

  %x4 = tosa.reshape %x, %xs : (tensor<?x16xi8>, !tosa.shape<4>) -> tensor<?x1x1x16xi8>
  %a0 = tosa.conv2d %x4, %w0, %b0, %zx0, %zw pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<?x1x1x16xi8>, tensor<8x1x1x16xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1x1x8xi32>
  %r0 = tosa.rescale %a0, %mul, %shift, %izp, %ozp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<?x1x1x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<?x1x1x8xi8>
  %r0f = tosa.reshape %r0, %h2 : (tensor<?x1x1x8xi8>, !tosa.shape<2>) -> tensor<?x8xi8>
  %c0 = tosa.clamp %r0f min_val(-128 : i8) max_val(127 : i8) : (tensor<?x8xi8>) -> tensor<?x8xi8>
  %c04 = tosa.reshape %c0, %hs : (tensor<?x8xi8>, !tosa.shape<4>) -> tensor<?x1x1x8xi8>
  %a1 = tosa.conv2d %c04, %w1, %b1, %zx1, %zw pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<?x1x1x8xi8>, tensor<6x1x1x8xi8>, tensor<6xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1x1x6xi32>
  %r1 = tosa.rescale %a1, %mul, %shift, %izp, %ozp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<?x1x1x6xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<?x1x1x6xi8>
  %r1f = tosa.reshape %r1, %ys : (tensor<?x1x1x6xi8>, !tosa.shape<2>) -> tensor<?x6xi8>
  %y = tosa.clamp %r1f min_val(-128 : i8) max_val(127 : i8) : (tensor<?x6xi8>) -> tensor<?x6xi8>
  return %y : tensor<?x6xi8>
}
