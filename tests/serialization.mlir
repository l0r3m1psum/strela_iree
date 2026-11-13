// What the STRELA backend writes into the VMFB, decoded. Each STRELA
// executable is dumped on its own with --iree-hal-dump-executable-binaries-to;
// the file is an IREE header followed by a FlatBuffer whose schema is
// runtime/src/estela/schemas/strela_executable_def.fbs, and
// estela-dump-executable prints both as JSON.

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --iree-hal-dump-executable-binaries-to=%t %s -o %t/matvec.vmfb
// RUN: estela-dump-executable --header %t/module_matvec_dispatch_0_strela_bitstream.str1 \
// RUN:   | FileCheck %s

// The smallest matvec: N = 2 rows of C = 16 words times a C x F = 16 x 3
// matrix, every shape static. Not valid TOSA (no i32 matmul), but it is what
// reaches STRELA. serialization_matvec.mlir follows the numbers of a bigger,
// dynamic one.
func.func @matvec(%x: tensor<1x2x16xi32>, %w: tensor<1x16x3xi32>) -> tensor<1x2x3xi32> {
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %0 = tosa.matmul %x, %w, %zp, %zp : (tensor<1x2x16xi32>, tensor<1x16x3xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x2x3xi32>
  return %0 : tensor<1x2x3xi32>
}

// The IREE header (iree_flatbuffer_file_header_t), 64 bytes: magic "STR1" (the
// schema's file_identifier), version 0, the size of the FlatBuffer that
// follows, and six reserved 64 bit words. The runtime checks it before handing
// the rest to the FlatBuffer verifier.
// CHECK:      "header": {
// CHECK-NEXT:   "magic": "STR1",
// CHECK-NEXT:   "version": 0,
// CHECK-NEXT:   "content_size": {{[0-9]+}},
// CHECK-NEXT:   "reserved": [0, 0, 0, 0, 0, 0]
// CHECK-NEXT: },

// One ExportDef per dispatch STRELA runs, named as hal.executable.export.
// CHECK-NEXT: "executable": {
// CHECK-NEXT:   "exports": [{
// CHECK-NEXT:       "name": "matvec_dispatch_0_batch_matmul_1x2x3x16_i32",

// The CGRA configuration, STRELA_KERNEL_SIZE (80) words; not checked here.
// CHECK-NEXT:       "bitstream": [

// Four inputs and four outputs, one per CGRA column. Every number is a ValueDef:
// an immediate, or with is_constant the ordinal of the push constant holding
// it. Addresses are in bytes, counts and strides in words.
//
// The grid is (ceildiv(F, 3), N, 1) = (1, 2, 1): one job per row, each
// computing all three features. A column's address is offset plus, for each
// workgroup id, min(id * step, limit).
//
// The first column, row y of %x, is spelled out in full. It starts at byte 0
// of binding 0 and moves one row of C words, 64 bytes, per y; x and z do not
// move it (step 0, no clamp: limit 2^32 - 1).
// CHECK:            "inputs": [{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 0,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 0,
// CHECK-NEXT:               "is_constant": false
// CHECK-NEXT:             },
// CHECK-NEXT:             "workgroup_x": {
// CHECK-NEXT:               "step": {
// CHECK-NEXT:                 "value": 0,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               },
// CHECK-NEXT:               "limit": {
// CHECK-NEXT:                 "value": 4294967295,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               }
// CHECK-NEXT:             },
// CHECK-NEXT:             "workgroup_y": {
// CHECK-NEXT:               "step": {
// CHECK-NEXT:                 "value": 64,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               },
// CHECK-NEXT:               "limit": {
// CHECK-NEXT:                 "value": 4294967295,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               }
// CHECK-NEXT:             },
// CHECK-NEXT:             "workgroup_z": {
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 16,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           },
// CHECK-NEXT:           "word_stride": {
// CHECK-NEXT:             "value": 1,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }

// Columns 1-3 read the three columns of %w: C words, one every F = 3.
// FuseDispatchBindings put both operands in binding 0, %w right after the 128
// bytes of %x, a constant offset so no push constants are involved. With
// F = 3 there is a single x, so the x clamp is 4(F - 3) = 0.
// CHECK-NEXT:         },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 0,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 128,
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 16,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           },
// CHECK-NEXT:           "word_stride": {
// CHECK-NEXT:             "value": 3,
// CHECK:              },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 0,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 132,
// CHECK:              },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 0,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 136,

// A word count of 0 leaves a column unused; its other fields are still
// present. The matvec does not use output column 0. Outputs have no
// word_stride.
// CHECK:            "outputs": [{
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 0,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }

// Output columns 1-3 each write one word of row y into binding 1, a row of
// 4F = 12 bytes further per y.
// CHECK-NEXT:         },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 1,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 0,
// CHECK:                  "workgroup_y": {
// CHECK-NEXT:               "step": {
// CHECK-NEXT:                 "value": 12,
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 1,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }
// CHECK-NEXT:         },{

// The pipeline layout: no push constants (every offset and size is static) and
// two bindings, the packed inputs and the result.
// CHECK:            "constant_count": 0,
// CHECK-NEXT:       "binding_count": 2
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: }
