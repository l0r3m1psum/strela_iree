// RUN: iree-compile --split-input-file --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=preprocessing %s | FileCheck %s

// The STRELA multipliers use the low 16 bits of each i32 operand
// (multiplier_width in #strela.target_desc), so a matmul is pinned to STRELA
// only if both operands are proven to fit: what the frontend can see of their
// producers, since the backend only sees bindings.

// i8 activations widened by tosa.cast (a linalg.generic with arith.extsi once
// lowered) and constant weights within int16: pinned. Being constant, the
// weights are also transposed to F x C (StrelaPrepareWeights).

// CHECK-LABEL: util.func public @widened_i8
// CHECK:         %[[X:.+]] = flow.tensor.transfer %{{.+}} : tensor<1x4x16xi32> to #hal.device.affinity<@strela0>
// CHECK:         %[[W:.+]] = flow.tensor.transfer %{{.+}} : tensor<1x3x16xi32> to #hal.device.affinity<@strela0>
// CHECK:         linalg.generic {{.+}} ins(%[[X]], %[[W]]
func.func @widened_i8(%x8: tensor<1x4x16xi8>) -> tensor<1x4x3xi32> {
  %x = tosa.cast %x8 : (tensor<1x4x16xi8>) -> tensor<1x4x16xi32>
  %w = "tosa.const"() {values = dense<[[[-32768, 0, 32767]]]> : tensor<1x1x3xi32>} : () -> tensor<1x1x3xi32>
  %multiples = tosa.const_shape values(dense<[1, 16, 1]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %wt = tosa.tile %w, %multiples : (tensor<1x1x3xi32>, !tosa.shape<3>) -> tensor<1x16x3xi32>
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %y = tosa.matmul %x, %wt, %zp, %zp : (tensor<1x4x16xi32>, tensor<1x16x3xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x4x3xi32>
  return %y : tensor<1x4x3xi32>
}

// -----

// i32 activations of unknown range: the CPU runs the matmul.

// CHECK-LABEL: util.func public @unknown_i32
// CHECK-NOT:     flow.tensor.transfer
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     flow.tensor.transfer
func.func @unknown_i32(%x: tensor<1x4x16xi32>) -> tensor<1x4x3xi32> {
  %w = "tosa.const"() {values = dense<1> : tensor<1x16x3xi32>} : () -> tensor<1x16x3xi32>
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %y = tosa.matmul %x, %w, %zp, %zp : (tensor<1x4x16xi32>, tensor<1x16x3xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x4x3xi32>
  return %y : tensor<1x4x3xi32>
}

// -----

// A constant weight of 2^15 needs a 17th bit: the CPU runs the matmul.

// CHECK-LABEL: util.func public @constant_too_wide
// CHECK-NOT:     flow.tensor.transfer
// CHECK:         linalg.batch_matmul
// CHECK-NOT:     flow.tensor.transfer
func.func @constant_too_wide(%x8: tensor<1x4x16xi8>) -> tensor<1x4x3xi32> {
  %x = tosa.cast %x8 : (tensor<1x4x16xi8>) -> tensor<1x4x16xi32>
  %w = "tosa.const"() {values = dense<32768> : tensor<1x16x3xi32>} : () -> tensor<1x16x3xi32>
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %y = tosa.matmul %x, %w, %zp, %zp : (tensor<1x4x16xi32>, tensor<1x16x3xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x4x3xi32>
  return %y : tensor<1x4x3xi32>
}
