// RUN: iree-opt --iree-plugin=estela --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-strela-select-lowering-strategy)))))' \
// RUN:   %s | FileCheck %s

// iree-strela-select-lowering-strategy rewrites the contraction STRELA runs
// into the N x C times C x F linalg.matmul of the matvec kernel, between
// reshapes of unit dims (FoldReshapeIntoInterfaceTensor folds them into the
// bindings), and tiles it by one row and three features: the shape of one job.

// The unit batch dim of tosa.matmul's linalg.batch_matmul goes; N stays
// dynamic and comes from the init.

// CHECK-LABEL: func.func @batch_matmul()
// CHECK-SAME:    translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline>
// CHECK:         %[[X:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<1x?x16xi32>
// CHECK:         %[[W:.+]] = iree_tensor_ext.dispatch.tensor.load {{.+}} -> tensor<1x16x8xi32>
// CHECK:         %[[X2:.+]] = tensor.collapse_shape %[[X]] {{\[}}[0, 1], [2]] : tensor<1x?x16xi32> into tensor<?x16xi32>
// CHECK:         %[[W2:.+]] = tensor.collapse_shape %[[W]] {{\[}}[0, 1], [2]] : tensor<1x16x8xi32> into tensor<16x8xi32>
// CHECK:         %[[EMPTY:.+]] = tensor.empty(%{{.+}}) : tensor<?x8xi32>
// CHECK:         %[[INIT:.+]] = linalg.fill ins(%{{.+}} : i32) outs(%[[EMPTY]] : tensor<?x8xi32>)
// CHECK:         %[[Y:.+]] = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 3, 0]]>} ins(%[[X2]], %[[W2]] : tensor<?x16xi32>, tensor<16x8xi32>) outs(%[[INIT]] : tensor<?x8xi32>)
// CHECK:         %[[Y3:.+]] = tensor.expand_shape %[[Y]] {{\[}}[0, 1], [2]] {{.+}} : tensor<?x8xi32> into tensor<1x?x8xi32>
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[Y3]]
#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
hal.executable private @batch_matmul {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @batch_matmul() {
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = iree_tensor_ext.dispatch.workload.ordinal %1, 0 : index
        %3 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x8xi32>>
        %4 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c512) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x16xi32>>{%2}
        %5 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x8xi32>>{%2}
        %6 = iree_tensor_ext.dispatch.tensor.load %4, offsets = [0, 0, 0], sizes = [1, %2, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x16xi32>>{%2} -> tensor<1x?x16xi32>
        %7 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0, 0], sizes = [1, 16, 8], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x8xi32>> -> tensor<1x16x8xi32>
        %8 = tensor.empty(%2) : tensor<1x?x8xi32>
        %9 = linalg.fill ins(%c0_i32 : i32) outs(%8 : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
        %10 = linalg.batch_matmul ins(%6, %7 : tensor<1x?x16xi32>, tensor<1x16x8xi32>) outs(%9 : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
        iree_tensor_ext.dispatch.tensor.store %10, %5, offsets = [0, 0, 0], sizes = [1, %2, 8], strides = [1, 1, 1] : tensor<1x?x8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x8xi32>>{%2}
        return
      }
    }
  }
}

// -----

// A vecmat has no row dim: it becomes a matmul of one row.

// CHECK-LABEL: func.func @vecmat()
// CHECK:         %[[X:.+]] = tensor.expand_shape %{{.+}} {{\[}}[0, 1]] output_shape [1, 16] : tensor<16xi32> into tensor<1x16xi32>
// CHECK:         %[[Y:.+]] = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = {{\[}}[1, 3, 0]]>} ins(%[[X]], %{{.+}} : tensor<1x16xi32>, tensor<16x9xi32>) outs(%{{.+}} : tensor<1x9xi32>)
// CHECK:         tensor.collapse_shape %[[Y]] {{\[}}[0, 1]] : tensor<1x9xi32> into tensor<9xi32>
#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
hal.executable private @vecmat {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @vecmat() {
        %c0_i32 = arith.constant 0 : i32
        %c0 = arith.constant 0 : index
        %c64 = arith.constant 64 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c64) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x9xi32>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<9xi32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [16], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x9xi32>> -> tensor<16x9xi32>
        %5 = tensor.empty() : tensor<9xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<9xi32>) -> tensor<9xi32>
        %7 = linalg.vecmat ins(%3, %4 : tensor<16xi32>, tensor<16x9xi32>) outs(%6 : tensor<9xi32>) -> tensor<9xi32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0], sizes = [9], strides = [1] : tensor<9xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<9xi32>>
        return
      }
    }
  }
}
