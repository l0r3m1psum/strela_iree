// RUN: iree-opt --iree-plugin=estela --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-strela-materialize-bindings)))))' \
// RUN:   %s | FileCheck %s

// iree-strela-materialize-bindings replaces a strela.matvec and the dispatch
// interface around it with one strela.job: the matvec bitstream and the DMA
// columns, in hardware column order. Each column address is a binding, the
// byte offset of the tensor in it, known at compile time (an i32) or held by a
// single 32 bit push constant, the byte stride of the column in the tensor,
// and the workgroup terms of the (ceildiv(F, 3), N, 1) grid.

#layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// The activations and the weights are at push constant offsets in binding 0,
// as FuseDispatchBindings leaves them when an executable has several dispatch
// sites, and the result 64 bytes into binding 1. C = 16, F = 8.
//   - Input 0, the row: C words, the next row (y) 4C = 64 bytes further.
//   - Inputs 1-3, weight columns k = 0..2: stride 4k, C words F apart; the
//     next 3 features (x) 12 bytes further, clamped at 4(F - 3) = 20 so the
//     last job overlaps the one before it.
//   - Output 0 is unused. Outputs 1-3: one word at stride 4k, the same x term,
//     the next row 4F = 32 bytes further.

// CHECK-LABEL: func.func @offsets
// CHECK-NEXT:    strela.job bitstream([{{[-0-9, ]+}}])
// CHECK-SAME:      inputs([#strela.inp_col<address = <binding = 0, offset = #strela.push_constant<0>, delta = 0, wg_x_params = <0, 4294967295>, wg_y_params = <64, 4294967295>>, words = 16, stride = 1>,
// CHECK-SAME:              #strela.inp_col<address = <binding = 0, offset = #strela.push_constant<1>, delta = 0, wg_x_params = <12, 20>, wg_y_params = <0, 4294967295>>, words = 16, stride = 8>,
// CHECK-SAME:              #strela.inp_col<address = <binding = 0, offset = #strela.push_constant<1>, delta = 4, wg_x_params = <12, 20>, wg_y_params = <0, 4294967295>>, words = 16, stride = 8>,
// CHECK-SAME:              #strela.inp_col<address = <binding = 0, offset = #strela.push_constant<1>, delta = 8, wg_x_params = <12, 20>, wg_y_params = <0, 4294967295>>, words = 16, stride = 8>])
// CHECK-SAME:      outputs([#strela.out_col<address = <binding = 0, offset = 0 : i32, delta = 0, wg_x_params = <0, 4294967295>, wg_y_params = <0, 4294967295>>, words = 0>,
// CHECK-SAME:               #strela.out_col<address = <binding = 1, offset = 64 : i32, delta = 0, wg_x_params = <12, 20>, wg_y_params = <32, 4294967295>>, words = 1>,
// CHECK-SAME:               #strela.out_col<address = <binding = 1, offset = 64 : i32, delta = 4, wg_x_params = <12, 20>, wg_y_params = <32, 4294967295>>, words = 1>,
// CHECK-SAME:               #strela.out_col<address = <binding = 1, offset = 64 : i32, delta = 8, wg_x_params = <12, 20>, wg_y_params = <32, 4294967295>>, words = 1>])
// CHECK-NEXT:    return
hal.executable private @offsets {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @offsets() {
        %c64 = arith.constant 64 : index
        %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = hal.interface.constant.load layout(#layout) ordinal(1) : i32
        %3 = arith.index_castui %2 : i32 to index
        %4 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16xi32>>
        %5 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%3) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8xi32>>
        %6 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c64) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x8xi32>>
        %7 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0], sizes = [2, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16xi32>> -> tensor<2x16xi32>
        %8 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
        %9 = strela.matvec %7, %8 : tensor<2x16xi32>, tensor<16x8xi32> -> tensor<2x8xi32>
        iree_tensor_ext.dispatch.tensor.store %9, %6, offsets = [0, 0], sizes = [2, 8], strides = [1, 1] : tensor<2x8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x8xi32>>
        return
      }
    }
  }
}

// -----

// A 64 bit offset arrives as two push constants whose high half is only known
// at dispatch time, so it is refused. This is what Stream emits by default;
// --iree-stream-resource-index-bits=32 makes it a single constant.

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
        %7 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%6) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16xi32>>
        %8 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8xi32>>
        %9 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x8xi32>>
        // expected-error @+1 {{byte offset is a 64 bit value split over two push constants; STRELA takes 32 bit offsets, retry with --iree-stream-resource-index-bits=32}}
        %10 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0, 0], sizes = [2, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16xi32>> -> tensor<2x16xi32>
        %11 = iree_tensor_ext.dispatch.tensor.load %8, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
        %12 = strela.matvec %10, %11 : tensor<2x16xi32>, tensor<16x8xi32> -> tensor<2x8xi32>
        iree_tensor_ext.dispatch.tensor.store %12, %9, offsets = [0, 0], sizes = [2, 8], strides = [1, 1] : tensor<2x8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x8xi32>>
        return
      }
    }
  }
}

// -----

// The strela.job verifier: one input and one output column per fabric column.

#addr = #strela.col_addr<binding = 0, offset = 0 : i32, delta = 0, wg_x_params = <0, 4294967295>, wg_y_params = <0, 4294967295>>
#in = #strela.inp_col<address = #addr, words = 16, stride = 1>
#out = #strela.out_col<address = #addr, words = 1>

hal.executable private @column_count {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @column_count() {
        // expected-error @+1 {{has 3 input and 4 output columns, the fabric has 4}}
        strela.job bitstream([0]) inputs([#in, #in, #in]) outputs([#out, #out, #out, #out])
        return
      }
    }
  }
}
