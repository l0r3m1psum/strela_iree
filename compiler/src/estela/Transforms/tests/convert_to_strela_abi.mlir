// RUN: iree-opt --iree-plugin=estela --split-input-file --verify-diagnostics --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-strela-convert-to-strela-abi)))))' \
// RUN:   %s | FileCheck %s

// iree-strela-convert-to-strela-abi turns the hal.interface.* ops into their
// strela.* counterparts, then replaces a strela.matvec on one job's tiles, the
// subviews of the bindings bufferization leaves, with one strela.job: the
// matvec bitstream and the DMA columns, in hardware column order. A column
// address is SSA index arithmetic: the byte offset of its binding, plus where
// the tile starts (on the workgroup ids) and the column in it, times the
// binding's strides; the job verifier folds it.

#layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// C = 16, F = 8, as the tiling leaves it: workgroup (x, y) computes row y and
// features f0 = min(3x, 5) .. f0 + 2. The activations and the weights are at
// push constant offsets in binding 0, as FuseDispatchBindings leaves them when
// an executable has several dispatch sites, and the result 64 bytes into
// binding 1.
//   - Input 0, the row: C words at pc0 + 4C·y.
//   - Inputs 1-3, weight column k: C words F apart, at pc1 + 4k + 4·f0.
//   - Output 0 is unused. Outputs 1-3: one word at 64 + 4F·y + 4k + 4·f0.

// CHECK-LABEL: func.func @offsets
// CHECK-NEXT:    %[[PC0:[^ ]+]] = strela.push_constant 0 : i32
// CHECK-NEXT:    %[[X:[^ ]+]] = arith.index_castui %[[PC0]] : i32 to index
// CHECK-NEXT:    %[[PC1:[^ ]+]] = strela.push_constant 1 : i32
// CHECK-NEXT:    %[[W:[^ ]+]] = arith.index_castui %[[PC1]] : i32 to index
// CHECK-NEXT:    %[[WG_Y:[^ ]+]] = strela.workgroup_id[1] : index
// CHECK-NEXT:    %[[WG_X:[^ ]+]] = strela.workgroup_id[0] : index
// CHECK-NEXT:    %[[F0:[^ ]+]] = affine.min affine_map<()[s0] -> (5, s0 * 3)>()[%[[WG_X]]]
// CHECK-NEXT:    %[[A0:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 64)>()[%[[X]], %[[WG_Y]]]
// CHECK-NEXT:    %[[I0:[^ ]+]] = strela.inp_column binding(0) address(%[[A0]]) words(16) stride(1)
// CHECK-NEXT:    %[[A1:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4)>()[%[[W]], %[[F0]]]
// CHECK-NEXT:    %[[I1:[^ ]+]] = strela.inp_column binding(0) address(%[[A1]]) words(16) stride(8)
// CHECK-NEXT:    %[[A2:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4 + 4)>()[%[[W]], %[[F0]]]
// CHECK-NEXT:    %[[I2:[^ ]+]] = strela.inp_column binding(0) address(%[[A2]]) words(16) stride(8)
// CHECK-NEXT:    %[[A3:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 4 + 8)>()[%[[W]], %[[F0]]]
// CHECK-NEXT:    %[[I3:[^ ]+]] = strela.inp_column binding(0) address(%[[A3]]) words(16) stride(8)
// CHECK-NEXT:    %[[O0:[^ ]+]] = strela.out_column unused
// CHECK-NEXT:    %[[R1:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4 + 64)>()[%[[WG_Y]], %[[F0]]]
// CHECK-NEXT:    %[[O1:[^ ]+]] = strela.out_column binding(1) address(%[[R1]]) words(1)
// CHECK-NEXT:    %[[R2:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4 + 68)>()[%[[WG_Y]], %[[F0]]]
// CHECK-NEXT:    %[[O2:[^ ]+]] = strela.out_column binding(1) address(%[[R2]]) words(1)
// CHECK-NEXT:    %[[R3:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 32 + s1 * 4 + 72)>()[%[[WG_Y]], %[[F0]]]
// CHECK-NEXT:    %[[O3:[^ ]+]] = strela.out_column binding(1) address(%[[R3]]) words(1)
// CHECK-NEXT:    strela.job bitstream([{{[-0-9, ]+}}]) inputs(%[[I0]], %[[I1]], %[[I2]], %[[I3]]) outputs(%[[O0]], %[[O1]], %[[O2]], %[[O3]])
// CHECK-NEXT:    return
hal.executable private @offsets {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    hal.executable.export public @offsets ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      hal.return %c3, %c2, %c1 : index, index, index
    }
    builtin.module {
      func.func @offsets() {
        %c64 = arith.constant 64 : index
        %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.constant.load layout(#layout) ordinal(1) : i32
        %3 = arith.index_castui %2 : i32 to index
        %4 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect") : memref<2x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %a4 = memref.assume_alignment %4, 64 : memref<2x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %5 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%3) flags("ReadOnly|Indirect") : memref<16x8xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %a5 = memref.assume_alignment %5, 64 : memref<16x8xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %6 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c64) flags(Indirect) : memref<2x8xi32, strided<[8, 1], offset: 16>, #hal.descriptor_type<storage_buffer>>
        %a6 = memref.assume_alignment %6, 64 : memref<2x8xi32, strided<[8, 1], offset: 16>, #hal.descriptor_type<storage_buffer>>
        %wg_y = hal.interface.workgroup.id[1] : index
        %wg_x = hal.interface.workgroup.id[0] : index
        %f0 = affine.min affine_map<()[s0] -> (5, s0 * 3)>()[%wg_x]
        %7 = memref.subview %a4[%wg_y, 0] [1, 16] [1, 1] : memref<2x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %8 = memref.subview %a5[0, %f0] [16, 3] [1, 1] : memref<16x8xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16x3xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %9 = memref.subview %a6[%wg_y, %f0] [1, 3] [1, 1] : memref<2x8xi32, strided<[8, 1], offset: 16>, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        strela.matvec ins(%7, %8 : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16x3xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%9 : memref<1x3xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>)
        return
      }
    }
  }
}

// -----

// F x C weights (IREE's TransposeMatmul transposed them): weight column k reads
// row f0 + k, C = 16 contiguous words, at 4C·(f0 + k) = 64·f0 + 64k bytes.

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, ReadOnly>,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// CHECK-LABEL: func.func @transposed_weights
// CHECK:         %[[F0:[^ ]+]] = affine.min affine_map<()[s0] -> (5, s0 * 3)>
// CHECK:         strela.inp_column binding(0) address(%{{.+}}) words(16) stride(1)
// CHECK-NEXT:    %[[A1:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%[[F0]]]
// CHECK-NEXT:    strela.inp_column binding(1) address(%[[A1]]) words(16) stride(1)
// CHECK-NEXT:    %[[A2:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 64 + 64)>()[%[[F0]]]
// CHECK-NEXT:    strela.inp_column binding(1) address(%[[A2]]) words(16) stride(1)
// CHECK-NEXT:    %[[A3:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 64 + 128)>()[%[[F0]]]
// CHECK-NEXT:    strela.inp_column binding(1) address(%[[A3]]) words(16) stride(1)
hal.executable private @transposed_weights {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @transposed_weights() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<2x16xi32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(ReadOnly) : memref<8x16xi32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<2x8xi32, #hal.descriptor_type<storage_buffer>>
        %wg_y = hal.interface.workgroup.id[1] : index
        %wg_x = hal.interface.workgroup.id[0] : index
        %f0 = affine.min affine_map<()[s0] -> (5, s0 * 3)>()[%wg_x]
        %3 = memref.subview %0[%wg_y, 0] [1, 16] [1, 1] : memref<2x16xi32, #hal.descriptor_type<storage_buffer>> to memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %4 = memref.subview %1[%f0, 0] [3, 16] [1, 1] : memref<8x16xi32, #hal.descriptor_type<storage_buffer>> to memref<3x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %5 = memref.subview %2[%wg_y, %f0] [1, 3] [1, 1] : memref<2x8xi32, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        strela.matvec ins(%3, %4 : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<3x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%5 : memref<1x3xi32, strided<[8, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) transpose_rhs
        return
      }
    }
  }
}

// -----

// Offsets the fold sees through, which the old syntactic match refused:
//   - the row's, as Stream writes it before cleanup: a 64 bit value whose high
//     half is the constant 0, (0 << 32) | pc0;
//   - the weights', a push constant plus 64 bytes: the 64 joins the 4k of each
//     weight column.
// The job verifies, so every column folded.

#layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// CHECK-LABEL: func.func @offset_forms
// CHECK:         %[[ROW:[^ ]+]] = arith.index_castui %{{.+}} : i64 to index
// CHECK:         strela.inp_column binding(0) address(%[[ROW]])
// CHECK:         strela.job
hal.executable private @offset_forms {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    hal.executable.export public @offset_forms ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %c1 = arith.constant 1 : index
      hal.return %c1, %c1, %c1 : index, index, index
    }
    builtin.module {
      func.func @offset_forms() {
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %c0_i32 = arith.constant 0 : i32
        %c32_i64 = arith.constant 32 : i64
        %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %1 = arith.extui %0 : i32 to i64
        %2 = arith.extui %c0_i32 : i32 to i64
        %3 = arith.shli %2, %c32_i64 : i64
        %4 = arith.ori %1, %3 : i64
        %5 = arith.index_castui %4 : i64 to index
        %6 = hal.interface.constant.load layout(#layout) ordinal(1) : i32
        %7 = arith.index_castui %6 : i32 to index
        %8 = arith.addi %7, %c64 : index
        %9 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%5) flags("ReadOnly|Indirect") : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %10 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%8) flags("ReadOnly|Indirect") : memref<16x3xi32, strided<[3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %11 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : memref<1x3xi32, #hal.descriptor_type<storage_buffer>>
        %12 = memref.subview %9[0, 0] [1, 16] [1, 1] : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %13 = memref.subview %10[0, 0] [16, 3] [1, 1] : memref<16x3xi32, strided<[3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<16x3xi32, strided<[3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %14 = memref.subview %11[0, 0] [1, 3] [1, 1] : memref<1x3xi32, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        strela.matvec ins(%12, %13 : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16x3xi32, strided<[3, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>) outs(%14 : memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>)
        return
      }
    }
  }
}

// -----

// A 64 bit offset whose high half is only known at dispatch time is two push
// constants, which a column address can't hold. This is what Stream emits by
// default; --iree-stream-resource-index-bits=32 makes it a single constant.
// The column carries the location of the subview it came from.

#layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

hal.executable private @split_offset {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @split_offset() {
        %c0 = arith.constant 0 : index
        %c32_i64 = arith.constant 32 : i64
        %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %1 = hal.interface.constant.load layout(#layout) ordinal(1) : i32
        %2 = arith.extui %0 : i32 to i64
        %3 = arith.extui %1 : i32 to i64
        %4 = arith.shli %3, %c32_i64 : i64
        %5 = arith.ori %2, %4 : i64
        %6 = arith.index_castui %5 : i64 to index
        %7 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%6) flags("ReadOnly|Indirect") : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %8 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x3xi32, #hal.descriptor_type<storage_buffer>>
        %9 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<1x3xi32, #hal.descriptor_type<storage_buffer>>
        // expected-error @+1 {{is a 64 bit value split over two push constants; STRELA takes 32 bit offsets, retry with --iree-stream-resource-index-bits=32}}
        %10 = memref.subview %7[0, 0] [1, 16] [1, 1] : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>> to memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>
        %11 = memref.subview %8[0, 0] [16, 3] [1, 1] : memref<16x3xi32, #hal.descriptor_type<storage_buffer>> to memref<16x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        %12 = memref.subview %9[0, 0] [1, 3] [1, 1] : memref<1x3xi32, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        strela.matvec ins(%10, %11 : memref<1x16xi32, strided<[16, 1], offset: ?>, #hal.descriptor_type<storage_buffer>>, memref<16x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>) outs(%12 : memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>)
        return
      }
    }
  }
}

// -----

// Every operand must be a tile of a binding: the kernel's columns read and
// write nothing else. A buffer of the dispatch's own, here the output, would
// be memory the CGRA cannot reach; allocating it is already refused.

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">
], flags = Indirect>

hal.executable private @local_output {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @local_output() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1x16xi32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x3xi32, #hal.descriptor_type<storage_buffer>>
        %2 = memref.subview %0[0, 0] [1, 16] [1, 1] : memref<1x16xi32, #hal.descriptor_type<storage_buffer>> to memref<1x16xi32, strided<[16, 1]>, #hal.descriptor_type<storage_buffer>>
        %3 = memref.subview %1[0, 0] [16, 3] [1, 1] : memref<16x3xi32, #hal.descriptor_type<storage_buffer>> to memref<16x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        // expected-error @+1 {{writes memory besides the STRELA kernel}}
        %4 = memref.alloca() : memref<1x3xi32>
        strela.matvec ins(%2, %3 : memref<1x16xi32, strided<[16, 1]>, #hal.descriptor_type<storage_buffer>>, memref<16x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>) outs(%4 : memref<1x3xi32>)
        return
      }
    }
  }
}

// -----

// The job is all a STRELA dispatch runs. A copy bufferization could not
// remove, here into the output tile, would be dropped without a word.

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

hal.executable private @leftover_copy {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @leftover_copy() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<1x16xi32, #hal.descriptor_type<storage_buffer>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : memref<16x3xi32, #hal.descriptor_type<storage_buffer>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : memref<1x3xi32, #hal.descriptor_type<storage_buffer>>
        %3 = memref.subview %0[0, 0] [1, 16] [1, 1] : memref<1x16xi32, #hal.descriptor_type<storage_buffer>> to memref<1x16xi32, strided<[16, 1]>, #hal.descriptor_type<storage_buffer>>
        %4 = memref.subview %1[0, 0] [16, 3] [1, 1] : memref<16x3xi32, #hal.descriptor_type<storage_buffer>> to memref<16x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        %5 = memref.subview %2[0, 0] [1, 3] [1, 1] : memref<1x3xi32, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        %6 = memref.subview %0[0, 0] [1, 3] [1, 1] : memref<1x16xi32, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[16, 1]>, #hal.descriptor_type<storage_buffer>>
        strela.matvec ins(%3, %4 : memref<1x16xi32, strided<[16, 1]>, #hal.descriptor_type<storage_buffer>>, memref<16x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>) outs(%5 : memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>)
        // expected-error @+1 {{writes memory besides the STRELA kernel}}
        memref.copy %6, %5 : memref<1x3xi32, strided<[16, 1]>, #hal.descriptor_type<storage_buffer>> to memref<1x3xi32, strided<[3, 1]>, #hal.descriptor_type<storage_buffer>>
        return
      }
    }
  }
}

// -----

// After the conversion the HAL dialect is illegal: an interface op without a
// strela counterpart is reported where it is.

hal.executable private @workgroup_count {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @workgroup_count() {
        // expected-error @+1 {{failed to legalize operation 'hal.interface.workgroup.count'}}
        %0 = hal.interface.workgroup.count[0] : index
        return
      }
    }
  }
}
