// RUN: iree-opt --iree-plugin=estela --split-input-file --mlir-print-local-scope \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-codegen-tile-and-distribute-to-workgroups-using-forall-op, iree-strela-store-workgroup-tiles), iree-codegen-reconcile-translation-info{distribute-along=y}, iree-codegen-resolve-workgroup-count-hints, func.func(canonicalize, cse, iree-strela-overlap-last-tile, canonicalize, cse)), iree-codegen-propagate-dispatch-config)))' \
// RUN:   %s | FileCheck %s

// The tiling half of the STRELA translation pipeline, on what
// iree-strela-select-lowering-strategy leaves: an N x C times C x F
// linalg.matmul tiled by one row and three features. IREE distributes the
// tiles to workgroups; iree-strela-store-workgroup-tiles stores each tile
// (and loads each input tile) from inside the forall, so that
// ReconcileTranslationInfo can turn it into workgroup ids while the dispatch
// is on tensors; iree-strela-overlap-last-tile makes every tile whole.
//
// C = 16, F = 8, N dynamic: the grid is (ceildiv(8, 3), N, 1) = (3, N, 1).
// Workgroup (x, y) computes row y, features f0 = 3x clamped to F - 3 = 5: the
// last job, x = 2, computes features 5..7 and rewrites feature 5.

// CHECK-LABEL: hal.executable.export public @tail
// CHECK-SAME:    count(%{{.+}}: !hal.device, %[[N:.+]]: index)
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         hal.return %[[C3]], %[[N]], %[[C1]] : index, index, index
// CHECK:       func.func @tail()
// CHECK-DAG:     %[[WG_Y:.+]] = hal.interface.workgroup.id[1] : index
// CHECK-DAG:     %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK:         %[[ROW:.+]] = iree_tensor_ext.dispatch.tensor.load %{{.+}}, offsets = [%[[WG_Y]], 0], sizes = [1, 16], strides = [1, 1] {{.+}} -> tensor<1x16xi32>
// CHECK:         %[[F0:.+]] = affine.min affine_map<()[s0] -> (5, s0 * 3)>()[%[[WG_X]]]
// CHECK:         %[[W:.+]] = iree_tensor_ext.dispatch.tensor.load %{{.+}}, offsets = [0, %[[F0]]], sizes = [16, 3], strides = [1, 1] {{.+}} -> tensor<16x3xi32>
// CHECK:         %[[Y:.+]] = linalg.matmul ins(%[[ROW]], %[[W]] : tensor<1x16xi32>, tensor<16x3xi32>)
// CHECK:         iree_tensor_ext.dispatch.tensor.store %[[Y]], %{{.+}}, offsets = [%[[WG_Y]], %[[F0]]], sizes = [1, 3], strides = [1, 1]
#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
hal.executable private @tail {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    hal.executable.export public @tail ordinal(0) layout(#layout) count(%device: !hal.device, %n: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%n)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @tail() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline>} {
        %c512 = arith.constant 512 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %1 = arith.index_castui %0 : i32 to index
        %2 = util.assume.int %1<umin = 0, umax = 9007199254740991> : index
        %3 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8xi32>>
        %4 = iree_tensor_ext.dispatch.workload.ordinal %2, 0 : index
        %5 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c512) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xi32>>{%4}
        %6 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x8xi32>>{%4}
        %7 = iree_tensor_ext.dispatch.tensor.load %5, offsets = [0, 0], sizes = [%4, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xi32>>{%4} -> tensor<?x16xi32>
        %8 = iree_tensor_ext.dispatch.tensor.load %3, offsets = [0, 0], sizes = [16, 8], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x8xi32>> -> tensor<16x8xi32>
        %9 = tensor.empty(%4) : tensor<?x8xi32>
        %10 = linalg.fill ins(%c0_i32 : i32) outs(%9 : tensor<?x8xi32>) -> tensor<?x8xi32>
        %11 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 3, 0]]>} ins(%7, %8 : tensor<?x16xi32>, tensor<16x8xi32>) outs(%10 : tensor<?x8xi32>) -> tensor<?x8xi32>
        iree_tensor_ext.dispatch.tensor.store %11, %6, offsets = [0, 0], sizes = [%4, 8], strides = [1, 1] : tensor<?x8xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?x8xi32>>{%4}
        return
      }
      iree_codegen.dispatch_config @tail {
      ^bb0(%device: !hal.device, %n: index):
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%n)
        iree_codegen.yield %x, %y, %z : index, index, index
      }
    }
  }
}

// -----

// F = 9: 3 divides F, there is no tail and the offsets stay 3x. N = 2 is
// static, and so is the whole grid, (3, 2, 1).

// CHECK-LABEL: hal.executable.export public @no_tail
// CHECK-DAG:     %[[C1:.+]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.+]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.+]] = arith.constant 3 : index
// CHECK:         hal.return %[[C3]], %[[C2]], %[[C1]] : index, index, index
// CHECK:       func.func @no_tail()
// CHECK-DAG:     %[[WG_X:.+]] = hal.interface.workgroup.id[0] : index
// CHECK-NOT:     affine.min
// CHECK:         %[[F0:.+]] = affine.apply affine_map<()[s0] -> (s0 * 3)>()[%[[WG_X]]]
// CHECK:         iree_tensor_ext.dispatch.tensor.load %{{.+}}, offsets = [0, %[[F0]]], sizes = [16, 3]
// CHECK:         iree_tensor_ext.dispatch.tensor.store %{{.+}}, offsets = [%{{.+}}, %[[F0]]], sizes = [1, 3]
#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>
hal.executable private @no_tail {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    hal.executable.export public @no_tail ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @no_tail() attributes {translation_info = #iree_codegen.translation_info<pipeline = #iree_codegen.no_pipeline>} {
        %c128 = arith.constant 128 : index
        %c0 = arith.constant 0 : index
        %c0_i32 = arith.constant 0 : i32
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16xi32>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c128) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x9xi32>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x9xi32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0], sizes = [2, 16], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<2x16xi32>> -> tensor<2x16xi32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0], sizes = [16, 9], strides = [1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16x9xi32>> -> tensor<16x9xi32>
        %5 = tensor.empty() : tensor<2x9xi32>
        %6 = linalg.fill ins(%c0_i32 : i32) outs(%5 : tensor<2x9xi32>) -> tensor<2x9xi32>
        %7 = linalg.matmul {lowering_config = #iree_codegen.lowering_config<tile_sizes = [[1, 3, 0]]>} ins(%3, %4 : tensor<2x16xi32>, tensor<16x9xi32>) outs(%6 : tensor<2x9xi32>) -> tensor<2x9xi32>
        iree_tensor_ext.dispatch.tensor.store %7, %2, offsets = [0, 0], sizes = [2, 9], strides = [1, 1] : tensor<2x9xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<2x9xi32>>
        return
      }
      iree_codegen.dispatch_config @no_tail {
      ^bb0(%device: !hal.device):
        %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
        iree_codegen.yield %x, %y, %z : index, index, index
      }
    }
  }
}
