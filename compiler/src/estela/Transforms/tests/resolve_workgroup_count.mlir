// RUN: iree-opt --iree-plugin=estela --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-strela-resolve-workgroup-count)))' \
// RUN:   %s | FileCheck %s

// iree-strela-resolve-workgroup-count replaces the
// workgroup_count_from_slice placeholder dispatch creation leaves in every
// export with the grid STRELA runs the kernel on: one job for an elementwise
// kernel, one job per activation row and triple of output features for a
// matvec. It runs before MaterializeBindings, so the dispatch body still has
// the loads that say which workload value is N.

#layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// An elementwise kernel always runs as a single job.

// CHECK-LABEL: hal.executable private @add
// CHECK:         hal.executable.export public @add ordinal(0) {{.+}} count(%{{.+}}: !hal.device)
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-NOT:       workgroup_count_from_slice
// CHECK:           hal.return %[[C1]], %[[C1]], %[[C1]] : index, index, index
hal.executable private @add {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream">) {
    hal.executable.export public @add ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @add() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0], sizes = [16], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0], sizes = [16], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
        %5 = strela.add %3, %4 : tensor<16xi32>
        iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0], sizes = [16], strides = [1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
        return
      }
    }
  }
}

// -----

// A matvec with a static number of rows: x is ceildiv(F, 3) = ceildiv(128, 3)
// = 43, the last job overlapping the one before it, and y is N = 4.

#layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// CHECK-LABEL: hal.executable private @matvec_static
// CHECK:         hal.executable.export public @matvec_static {{.+}} count(%{{.+}}: !hal.device)
// CHECK-DAG:       %[[C43:.+]] = arith.constant 43 : index
// CHECK-DAG:       %[[C4:.+]] = arith.constant 4 : index
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-NOT:       workgroup_count_from_slice
// CHECK:           hal.return %[[C43]], %[[C4]], %[[C1]] : index, index, index
hal.executable private @matvec_static {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream">) {
    hal.executable.export public @matvec_static ordinal(0) layout(#layout) count(%device: !hal.device) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice()
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_static() {
        %c0 = arith.constant 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x128xi32>>
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128x128xi32>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x128xi32>>
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, 4, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x4x128xi32>> -> tensor<1x4x128xi32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 128, 128], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x128x128xi32>> -> tensor<1x128x128xi32>
        %5 = strela.matvec %3, %4 : tensor<1x4x128xi32>, tensor<1x128x128xi32> -> tensor<1x4x128xi32>
        iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0, 0, 0], sizes = [1, 4, 128], strides = [1, 1, 1] : tensor<1x4x128xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x4x128xi32>>
        return
      }
    }
  }
}

// -----

// The matmul_bias case: N is only known at run time. The body reads it
// through workload ordinal 0, so y is the count region's first workload
// argument (%n here; argument 0 is the device). F = 9 is a multiple of 3, so
// x is exactly 3 with no overlapping job.

#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// CHECK-LABEL: hal.executable private @matvec_dynamic
// CHECK:         hal.executable.export public @matvec_dynamic {{.+}} count(%{{.+}}: !hal.device, %[[N:[^:]+]]: index)
// CHECK-DAG:       %[[C3:.+]] = arith.constant 3 : index
// CHECK-DAG:       %[[C1:.+]] = arith.constant 1 : index
// CHECK-NOT:       workgroup_count_from_slice
// CHECK:           hal.return %[[C3]], %[[N]], %[[C1]] : index, index, index
hal.executable private @matvec_dynamic {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream">) {
    hal.executable.export public @matvec_dynamic ordinal(0) layout(#layout) count(%device: !hal.device, %n: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%n)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_dynamic() {
        %c0 = arith.constant 0 : index
        %cst = hal.interface.constant.load layout(#layout) ordinal(0) : i32
        %idx = arith.index_castui %cst : i32 to index
        %n = iree_tensor_ext.dispatch.workload.ordinal %idx, 0 : index
        %0 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x16xi32>>{%n}
        %1 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x9xi32>>
        %2 = hal.interface.binding.subspan layout(#layout) binding(2) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x9xi32>>{%n}
        %3 = iree_tensor_ext.dispatch.tensor.load %0, offsets = [0, 0, 0], sizes = [1, %n, 16], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x16xi32>>{%n} -> tensor<1x?x16xi32>
        %4 = iree_tensor_ext.dispatch.tensor.load %1, offsets = [0, 0, 0], sizes = [1, 16, 9], strides = [1, 1, 1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x9xi32>> -> tensor<1x16x9xi32>
        %5 = strela.matvec %3, %4 : tensor<1x?x16xi32>, tensor<1x16x9xi32> -> tensor<1x?x9xi32>
        iree_tensor_ext.dispatch.tensor.store %5, %2, offsets = [0, 0, 0], sizes = [1, %n, 9], strides = [1, 1, 1] : tensor<1x?x9xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x?x9xi32>>{%n}
        return
      }
    }
  }
}

// -----

// After MaterializeBindings the link from N to the workload is gone, so the
// pass refuses to guess rather than launching the wrong number of jobs.

#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

hal.executable private @matvec_materialized {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream">) {
    hal.executable.export public @matvec_materialized ordinal(0) layout(#layout) count(%device: !hal.device, %n: index) -> (index, index, index) {
      %x, %y, %z = iree_tensor_ext.dispatch.workgroup_count_from_slice(%n)
      hal.return %x, %y, %z : index, index, index
    }
    builtin.module {
      func.func @matvec_materialized() {
        %0 = strela.binding_load binding(0) : tensor<1x?x16xi32>
        %1 = strela.binding_load binding(1) : tensor<1x16x9xi32>
        // expected-error @+1 {{lhs is not a dispatch tensor load}}
        %2 = strela.matvec %0, %1 : tensor<1x?x16xi32>, tensor<1x16x9xi32> -> tensor<1x?x9xi32>
        strela.binding_store %2, binding(2) : tensor<1x?x9xi32>
        return
      }
    }
  }
}
