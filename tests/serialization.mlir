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
// RUN:   --iree-hal-dump-executable-binaries-to=%t %s -o %t/add.vmfb
// RUN: estela-dump-executable --header %t/module_add_dispatch_0_strela_bitstream.str1 \
// RUN:   | FileCheck %s

func.func @add(%a: tensor<16xi32>, %b: tensor<16xi32>) -> tensor<16xi32> {
  %0 = tosa.add %a, %b : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  return %0 : tensor<16xi32>
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
// CHECK-NEXT:       "name": "add_dispatch_0_elementwise_16_i32",

// The CGRA configuration, STRELA_KERNEL_SIZE (80) words; not checked here.
// CHECK-NEXT:       "bitstream": [

// Four inputs and four outputs, one per CGRA column. Every number is a ValueDef:
// an immediate, or with is_constant the ordinal of the push constant holding
// it. Addresses are in bytes, counts and strides in words.
//
// The first column is spelled out in full. %a starts at byte 0 of binding 0.
// The add runs as a single workgroup, so every workgroup term is step 0 with
// no clamp (limit 2^32 - 1).
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
// CHECK:                  "workgroup_z": {
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 16,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           },
// CHECK-NEXT:           "word_stride": {
// CHECK-NEXT:             "value": 1,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }

// FuseDispatchBindings put both operands in binding 0: %b starts 64 bytes after
// %a (16 words rounded up to the 64 byte min_buffer_offset_alignment), a
// constant offset so no push constants are involved. The add kernel reads
// columns 0 and 1.
// CHECK-NEXT:         },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 0,
// CHECK-NEXT:             "offset": {
// CHECK-NEXT:               "value": 64,
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 16,

// A word count of 0 leaves a column unused; its other fields are still present.
// CHECK:              },{
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 0,
// CHECK:              },{
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 0,

// The sum leaves by column 1 into binding 1. Outputs have no word_stride.
// CHECK:            "outputs": [{
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 0,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }
// CHECK-NEXT:         },{
// CHECK-NEXT:           "address": {
// CHECK-NEXT:             "binding": 1,
// CHECK:                "word_count": {
// CHECK-NEXT:             "value": 16,
// CHECK-NEXT:             "is_constant": false
// CHECK-NEXT:           }
// CHECK-NEXT:         },{

// The pipeline layout: no push constants (every offset is static) and two
// bindings, the packed inputs and the result.
// CHECK:            "constant_count": 0,
// CHECK-NEXT:       "binding_count": 2
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
// CHECK-NEXT: }
