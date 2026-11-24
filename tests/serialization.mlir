// What the STRELA backend writes into the VMFB, decoded. Each STRELA
// executable is dumped on its own with --iree-hal-dump-executable-binaries-to;
// the file is an IREE header followed by a FlatBuffer whose schema is
// runtime/src/estela/schemas/strela_executable_def.fbs, and
// estela-dump-executable prints both as JSON.

// RUN: rm -rf %t && mkdir -p %t
// RUN: iree-compile --iree-plugin=estela --iree-strela-multiplier-width=32 \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --iree-hal-dump-executable-binaries-to=%t %s -o %t/matvec.vmfb
// RUN: estela-dump-executable --header %t/module_matvec_dispatch_0_strela_bitstream.str1 \
// RUN:   | FileCheck %s
//
// The inputs are i32 of unknown range, which the operand-width check keeps on
// the host: --iree-strela-multiplier-width=32 lifts it (operand_width.mlir
// tests it).

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
// One job per row, each computing all three features. With F = 3 the feature
// loop of IREE's tiling has a single trip and goes away, so the rows are the
// only distributed dimension and run on workgroup x: the grid is (2, 1, 1). A column's address is offset plus delta plus,
// for each workgroup id, min(id * step, limit). Without push constants the
// whole constant is the offset and the delta is 0; with one, the push constant
// is the offset and the constant the delta.
//
// The first column, row x of %x, is spelled out in full. It starts at byte 0
// of binding 0 and moves one row of C words, 64 bytes, per x (no clamp:
// limit 2^32 - 1); y does not move it (step 0).
// CHECK:            "inputs": [{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 0,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 0,
// CHECK-NEXT:               "is_constant": false
// CHECK-NEXT:             },
// CHECK-NEXT:             "delta": 0,
// CHECK-NEXT:             "wg_x_params": {
// CHECK-NEXT:               "step": {
// CHECK-NEXT:                 "value": 64,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               },
// CHECK-NEXT:               "limit": {
// CHECK-NEXT:                 "value": 4294967295,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               }
// CHECK-NEXT:             },
// CHECK-NEXT:             "wg_y_params": {
// CHECK-NEXT:               "step": {
// CHECK-NEXT:                 "value": 0,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               },
// CHECK-NEXT:               "limit": {
// CHECK-NEXT:                 "value": 4294967295,
// CHECK-NEXT:                 "is_constant": false
// CHECK-NEXT:               }
// CHECK-NEXT:             }
// CHECK-NEXT:           },
// CHECK-NEXT:           "word_count": {
// CHECK-NEXT:             "value": 16,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           },
// CHECK-NEXT:           "word_stride": {
// CHECK-NEXT:             "value": 1,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }

// Columns 1-3 read the three columns of %w: C words, one every F = 3,
// starting 0, 4 and 8 bytes into %w.
// %w has a binding of its own, binding 1, as %x has binding 0: each device
// works on the other's buffers in place, so nothing packs them into one.
// Every job reads the same three columns.
// CHECK-NEXT:         },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 1,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 0,
// CHECK-NEXT:               "is_constant": false
// CHECK-NEXT:             },
// CHECK-NEXT:             "delta": 0,
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 16,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           },
// CHECK-NEXT:           "word_stride": {
// CHECK-NEXT:             "value": 3,
// CHECK:              },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 1,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 4,
// CHECK-NEXT:               "is_constant": false
// CHECK-NEXT:             },
// CHECK-NEXT:             "delta": 0,
// CHECK:              },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 1,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 8,
// CHECK-NEXT:               "is_constant": false
// CHECK-NEXT:             },
// CHECK-NEXT:             "delta": 0,

// A word count of 0 leaves a column unused; its other fields are still
// present. The matvec does not use output column 0. Outputs have no
// word_stride.
// CHECK:            "outputs": [{
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 0,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }

// Output columns 1-3 each write one word of row x into binding 2, a row of
// 4F = 12 bytes further per x.
// CHECK-NEXT:         },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 2,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 0,
// CHECK-NEXT:               "is_constant": false
// CHECK-NEXT:             },
// CHECK-NEXT:             "delta": 0,
// CHECK-NEXT:             "wg_x_params": {
// CHECK-NEXT:               "step": {
// CHECK-NEXT:                 "value": 12,
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 1,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }
// CHECK-NEXT:         },{

// The pipeline layout: no push constants (every offset and size is static) and
// three bindings, %x, %w and the result.
// CHECK:            "constant_count": 0,
// CHECK-NEXT:       "binding_count": 3
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: }
