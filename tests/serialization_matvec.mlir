// The matvec export, decoded: how each CGRA column finds its data for
// workgroup (x, y) of the (ceildiv(F, 3), N, 1) grid. serialization.mlir
// explains the format; this only follows the numbers.

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --iree-hal-dump-executable-binaries-to=%t %s -o %t/matvec.vmfb
// RUN: estela-dump-executable %t/module_matvec_dispatch_0_strela_bitstream.str1 \
// RUN:   | FileCheck %s

// N rows of C = 16 words times a C x F = 16 x 8 matrix. Not valid TOSA (no i32
// matmul), but it is what reaches STRELA.
func.func @matvec(%x: tensor<1x?x16xi32>, %w: tensor<1x16x8xi32>) -> tensor<1x?x8xi32> {
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = tosa.matmul %x, %w, %zp, %zp : (tensor<1x?x16xi32>, tensor<1x16x8xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x?x8xi32>
  return %0 : tensor<1x?x8xi32>
}

// Workgroup (x, y) computes result[y, f0 .. f0 + 2], f0 = min(3x, F - 3). The
// grid is (ceildiv(8, 3), N, 1) = (3, N, 1), so x = 2 would start at feature 6
// and run past F; the clamp moves it back to feature 5, F - 3, and it rewrites
// result[y, 5] with the same value the job before it wrote.
// CHECK:      "name": "matvec_dispatch_0_batch_matmul_1xDx8x16_i32",

// The matvec bitstream with the accumulation length of PEs 9-11 patched to C:
// (16 << 16) | 0x80 = 1048704.
// CHECK:      "bitstream": [
// CHECK:        1048704,

// Column 0 reads row y of %x: C = 16 contiguous words, 4C = 64 bytes further
// for each y. FuseDispatchBindings packed both operands into binding 0, %x
// after the 512 bytes of %w.
// CHECK:      "inputs": [{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 0,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 512,
// CHECK:          "workgroup_x": {
// CHECK-NEXT:       "step": {
// CHECK-NEXT:         "value": 0,
// CHECK:          "workgroup_y": {
// CHECK-NEXT:       "step": {
// CHECK-NEXT:         "value": 64,
// CHECK:        "word_count": {
// CHECK-NEXT:     "value": 16,
// CHECK:        "word_stride": {
// CHECK-NEXT:     "value": 1,

// Columns 1-3 read columns f0, f0 + 1 and f0 + 2 of %w: C words, one every
// F = 8, starting 0, 4 and 8 bytes in. Each x moves them 3 features, 12 bytes,
// up to the clamp of 4(F - 3) = 20 bytes; y does not move them.
// CHECK:      },{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 0,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 0,
// CHECK:          "workgroup_x": {
// CHECK-NEXT:       "step": {
// CHECK-NEXT:         "value": 12,
// CHECK:            "limit": {
// CHECK-NEXT:         "value": 20,
// CHECK:          "workgroup_y": {
// CHECK-NEXT:       "step": {
// CHECK-NEXT:         "value": 0,
// CHECK:        "word_count": {
// CHECK-NEXT:     "value": 16,
// CHECK:        "word_stride": {
// CHECK-NEXT:     "value": 8,
// CHECK:      },{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 0,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 4,
// CHECK:      },{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 0,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 8,

// Output column 0 is unused. Columns 1-3 each write one word, result[y, f0 + k]
// in binding 1: 12 bytes per x with the same clamp, a row of 4F = 32 bytes
// per y.
// CHECK:      "outputs": [{
// CHECK:        "word_count": {
// CHECK-NEXT:     "value": 0,
// CHECK:      },{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 1,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 0,
// CHECK:          "workgroup_x": {
// CHECK-NEXT:       "step": {
// CHECK-NEXT:         "value": 12,
// CHECK:            "limit": {
// CHECK-NEXT:         "value": 20,
// CHECK:          "workgroup_y": {
// CHECK-NEXT:       "step": {
// CHECK-NEXT:         "value": 32,
// CHECK:        "word_count": {
// CHECK-NEXT:     "value": 1,
// CHECK:      },{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 1,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 4,
// CHECK:      },{
// CHECK-NEXT:   "address": {
// CHECK-NEXT:     "binding": 1,
// CHECK-NEXT:     "offset": {
// CHECK-NEXT:       "value": 8,

// Two push constants, the low and high halves of N, which only the workgroup
// count uses; no column needs one.
// CHECK:      "constant_count": 2,
// CHECK-NEXT: "binding_count": 2
