// RUN: iree-opt --iree-plugin=estela --split-input-file --verify-diagnostics \
// RUN:   --pass-pipeline='builtin.module(func.func(iree-strela-materialize-bindings))' \
// RUN:   %s | FileCheck %s

// iree-strela-materialize-bindings restates where a STRELA kernel reads and
// writes as strela.binding_load/store. A byte offset is either known at compile
// time (byte_offset) or held by a single 32 bit push constant (offset_ordinal),
// the two forms the executable format can describe.

#layout = #hal.pipeline.layout<constants = 1, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

// CHECK-LABEL: func.func @offsets
// CHECK-NEXT:    %[[IN:.+]] = strela.binding_load binding(0) offset_ordinal 0 : tensor<16xi32>
// CHECK-NEXT:    %[[OUT:.+]] = strela.relu %[[IN]] : tensor<16xi32>
// CHECK-NEXT:    strela.binding_store %[[OUT]], binding(1) byte_offset 64 : tensor<16xi32>
// CHECK-NEXT:    return
func.func @offsets() {
  %c64 = arith.constant 64 : index
  %0 = hal.interface.constant.load layout(#layout) ordinal(0) : i32
  %1 = arith.index_castui %0 : i32 to index
  %2 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%1) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
  %3 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c64) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  %4 = iree_tensor_ext.dispatch.tensor.load %2, offsets = [0], sizes = [16], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
  %5 = strela.relu %4 : tensor<16xi32>
  iree_tensor_ext.dispatch.tensor.store %5, %3, offsets = [0], sizes = [16], strides = [1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  return
}

// -----

// A 64 bit offset arrives as two push constants whose high half is only known
// at dispatch time, so it is refused. This is what Stream emits by default;
// --iree-stream-resource-index-bits=32 makes it a single constant.

#layout = #hal.pipeline.layout<constants = 2, bindings = [
  #hal.pipeline.binding<storage_buffer, "ReadOnly|Indirect">,
  #hal.pipeline.binding<storage_buffer, Indirect>
], flags = Indirect>

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
  %7 = hal.interface.binding.subspan layout(#layout) binding(0) alignment(64) offset(%6) flags("ReadOnly|Indirect") : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>
  %8 = hal.interface.binding.subspan layout(#layout) binding(1) alignment(64) offset(%c0) flags(Indirect) : !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  // expected-error @+1 {{byte offset is a 64 bit value split over two push constants; STRELA takes 32 bit offsets, retry with --iree-stream-resource-index-bits=32}}
  %9 = iree_tensor_ext.dispatch.tensor.load %7, offsets = [0], sizes = [16], strides = [1] : !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>> -> tensor<16xi32>
  %10 = strela.relu %9 : tensor<16xi32>
  iree_tensor_ext.dispatch.tensor.store %10, %8, offsets = [0], sizes = [16], strides = [1] : tensor<16xi32> -> !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
  return
}
