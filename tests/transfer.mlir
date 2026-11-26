// NOTE: Shapes are static on purpose: with dynamic shapes the TOSA lowering of
// the elementwise ops emits a runtime "is this dim 1?" broadcast check as its
// own dispatch, which is not what these tests are about.
//
// RUN: iree-compile --iree-plugin=estela --iree-strela-multiplier-width=32 --split-input-file \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=strela \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --mlir-print-local-scope --compile-to=executable-targets %s | FileCheck %s
//
// RUN: iree-compile --iree-plugin=estela --iree-strela-multiplier-width=32 --split-input-file \
// RUN:   --iree-scheduling-optimize-bindings=false \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=strela \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --mlir-print-local-scope --compile-to=executable-targets %s | FileCheck %s --check-prefix=UNFUSED
//
// The inputs are i32 of unknown range, which the operand-width check keeps on
// the host: --iree-strela-multiplier-width=32 lifts it (operand_width.mlir
// tests it).

// The work meant for device_a is a sub and a minimum, which STRELA has no kernel
// for; the matvec runs on device_b. Not valid TOSA (no i32 matmul), but it is
// what reaches STRELA.
//
// Only %x is transferred by hand. The frontend transposes the weights of every
// matmul to F x C (IREE's TransposeMatmul), and %w is an input, so the
// transpose runs at run time. Left on device_a it is a CPU dispatch, and
// TransferWrap moves its result to device_b. Transferred by hand to device_b,
// %w would be transposed there, in a dispatch of its own that STRELA has no
// kernel for: compilation fails with "no STRELA op in the dispatch body".
// Weights already on STRELA are only supported when they are constant, as in
// MLPerf Tiny, where const-eval folds the transpose.

// Variant 1: the topology says the two devices share memory, so the transfers
// need no staging buffers, each operand keeps its own binding and every subspan
// starts at byte zero.
//
// The job reads the row (C = 16 words) from binding 0, the transposed weights
// (C contiguous words, 4C = 64 bytes between columns, 3 rows = 192 bytes per
// job) from binding 1, and writes the result to binding 2. Only the binding,
// address, words and stride of each column are checked here;
// Transforms/tests/convert_to_strela_abi.mlir checks the rest.
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     %[[WG_Y:[^ ]+]] = strela.workgroup_id[1] : index
// CHECK:     %[[WG_X:[^ ]+]] = strela.workgroup_id[0] : index
// CHECK:     %[[A0:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%[[WG_Y]]]
// CHECK:     strela.inp_column binding(0) address(%[[A0]]) words(16) stride(1)
// CHECK:     %[[A1:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 192)>()[%[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[A1]]) words(16) stride(1)
// CHECK:     %[[A2:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 192 + 64)>()[%[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[A2]]) words(16) stride(1)
// CHECK:     %[[A3:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 192 + 128)>()[%[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[A3]]) words(16) stride(1)
// CHECK:     strela.out_column unused
// CHECK:     %[[R1:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 36 + s1 * 12)>()[%[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R1]]) words(1)
// CHECK:     %[[R2:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 36 + s1 * 12 + 4)>()[%[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R2]]) words(1)
// CHECK:     %[[R3:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 36 + s1 * 12 + 8)>()[%[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R3]]) words(1)
// CHECK:     strela.job

// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     strela.inp_column binding(0) address(
// UNFUSED:     strela.inp_column binding(1) address(
// UNFUSED:     strela.out_column unused
// UNFUSED:     strela.out_column binding(2) address(

module attributes {
  stream.topology = #hal.device.topology<links = [
    (@device_a -> @device_b = {unified_memory = true, transparent_access = true}),
    (@device_b -> @device_a = {unified_memory = true, transparent_access = true})
  ]>
} {
  func.func @main(
    %i0: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %x: tensor<1x8x16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %w: tensor<1x16x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %zp = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %c0 = arith.constant 0 : index
    %mat_shape = tosa.const_shape values(dense<[8, 9]> : tensor<2xindex>) : () -> !tosa.shape<2>
    %new_shape = tosa.const_shape values(dense<[-1]> : tensor<1xindex>) : () -> !tosa.shape<1>

    %x_b = flow.tensor.transfer %x : tensor<1x8x16xi32> to #hal.device.promise<@device_b>

    %a0 = tosa.sub %i0, %i1 : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %y_b = tosa.matmul %x_b, %w, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>

    %r0 = tosa.minimum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    // The transfer back to device_a is a fusion boundary, as is the transfer of
    // %x above and the one TransferWrap adds for the transposed %w, so nothing
    // on the CPU fuses into the matvec.
    %y3 = flow.tensor.transfer %y_b : tensor<1x8x9xi32> to #hal.device.promise<@device_a>
    %y = tosa.reshape %y3, %mat_shape : (tensor<1x8x9xi32>, !tosa.shape<2>) -> tensor<8x9xi32>

    %a2 = tosa.sub %r0, %y : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened axis(0) : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}

// -----

// Variant 2: the same program without a topology attribute. AssignTopology
// gives it STRELA's: each device works on the other's buffers in place, so no
// transfer is staged and, as in variant 1, each operand keeps its own binding
// and every subspan starts at byte zero. (An older topology made the CPU's
// data go through a staging buffer, where FuseDispatchBindings packed both
// operands at compile-time offsets.)
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     %[[WG_Y:[^ ]+]] = strela.workgroup_id[1] : index
// CHECK:     %[[WG_X:[^ ]+]] = strela.workgroup_id[0] : index
// CHECK:     %[[A0:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 64)>()[%[[WG_Y]]]
// CHECK:     strela.inp_column binding(0) address(%[[A0]]) words(16) stride(1)
// CHECK:     %[[A1:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 192)>()[%[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[A1]]) words(16) stride(1)
// CHECK:     %[[A2:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 192 + 64)>()[%[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[A2]]) words(16) stride(1)
// CHECK:     %[[A3:[^ ]+]] = affine.apply affine_map<()[s0] -> (s0 * 192 + 128)>()[%[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[A3]]) words(16) stride(1)
// CHECK:     strela.out_column unused
// CHECK:     %[[R1:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 36 + s1 * 12)>()[%[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R1]]) words(1)
// CHECK:     %[[R2:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 36 + s1 * 12 + 4)>()[%[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R2]]) words(1)
// CHECK:     %[[R3:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 * 36 + s1 * 12 + 8)>()[%[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R3]]) words(1)
// CHECK:     strela.job

// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     strela.inp_column binding(0) address(
// UNFUSED:     strela.inp_column binding(1) address(
// UNFUSED:     strela.out_column unused
// UNFUSED:     strela.out_column binding(2) address(

module {
  func.func @main(
    %i0: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<8x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %x: tensor<1x8x16xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %w: tensor<1x16x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %zp = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %c0 = arith.constant 0 : index
    %mat_shape = tosa.const_shape values(dense<[8, 9]> : tensor<2xindex>) : () -> !tosa.shape<2>
    %new_shape = tosa.const_shape values(dense<[-1]> : tensor<1xindex>) : () -> !tosa.shape<1>

    %x_b = flow.tensor.transfer %x : tensor<1x8x16xi32> to #hal.device.promise<@device_b>

    %a0 = tosa.sub %i0, %i1 : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>
    %y_b = tosa.matmul %x_b, %w, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>

    %r0 = tosa.minimum %a0, %zero : (tensor<8x9xi32>, tensor<1x1xi32>) -> tensor<8x9xi32>

    %y3 = flow.tensor.transfer %y_b : tensor<1x8x9xi32> to #hal.device.promise<@device_a>
    %y = tosa.reshape %y3, %mat_shape : (tensor<1x8x9xi32>, !tosa.shape<2>) -> tensor<8x9xi32>

    %a2 = tosa.sub %r0, %y : (tensor<8x9xi32>, tensor<8x9xi32>) -> tensor<8x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<8x9xi32>, !tosa.shape<1>) -> tensor<72xi32>
    %min_tensor = tosa.reduce_min %flattened axis(0) : (tensor<72xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}

// -----

// Variant 3: TransferWrapPass places both matvecs on device_b and they share
// one executable. Each dispatch site reads its own %x, bound where it is. The
// weights are inputs, so each is transposed at run time on the CPU, and the
// two transposed copies go into one buffer, as do both results:
// FuseDispatchBindings packs the copies at 0 and 576 (4 * 9 * 16) and the
// results at 0 and 320, so both offsets reach the kernel as push constants, a
// single i32 each because Stream proved they fit in 32 bits. Each column adds
// its own compile-time bytes to them (seen through util.assume.int).
//
// CHECK: hal.executable.variant public @strela_bitstream
// CHECK:   func.func @main_dispatch_{{[0-9]+}}
// CHECK:     %[[PC0:[^ ]+]] = strela.push_constant 0 : i32
// CHECK:     %[[PC1:[^ ]+]] = strela.push_constant 1 : i32
// CHECK:     %[[W:[^ ]+]] = arith.index_castui %[[PC0]] : i32 to index
// CHECK:     %[[Y:[^ ]+]] = arith.index_castui %[[PC1]] : i32 to index
// CHECK:     %[[A:[^ ]+]]:2 = util.assume.int
// CHECK-NEXT:  %[[W]][<umin = 0, umax = 0>, <umin = 576, umax = 576, udiv = 576>],
// CHECK-NEXT:  %[[Y]][<umin = 0, umax = 0>, <umin = 320, umax = 320, udiv = 320>]
// CHECK:     %[[WG_Y:[^ ]+]] = strela.workgroup_id[1] : index
// CHECK:     %[[WG_X:[^ ]+]] = strela.workgroup_id[0] : index
// CHECK:     strela.inp_column binding(0) address(%{{.+}}) words(16) stride(1)
// CHECK:     %[[W1:[^ ]+]] = affine.apply affine_map<()[s0, s1] -> (s0 + s1 * 192)>()[%[[A]]#0, %[[WG_X]]]
// CHECK:     strela.inp_column binding(1) address(%[[W1]]) words(16) stride(1)
// CHECK:     strela.out_column unused
// CHECK:     %[[R1:[^ ]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 36 + s2 * 12)>()[%[[A]]#1, %[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R1]])
// CHECK:     %[[R2:[^ ]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 36 + s2 * 12 + 4)>()[%[[A]]#1, %[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R2]])
// CHECK:     %[[R3:[^ ]+]] = affine.apply affine_map<()[s0, s1, s2] -> (s0 + s1 * 36 + s2 * 12 + 8)>()[%[[A]]#1, %[[WG_Y]], %[[WG_X]]]
// CHECK:     strela.out_column binding(2) address(%[[R3]])
// CHECK:     strela.job

// Without fused bindings every operand has a binding of its own at byte 0.
//
// UNFUSED: hal.executable.variant public @strela_bitstream
// UNFUSED:   func.func @main_dispatch_{{[0-9]+}}
// UNFUSED:     strela.inp_column binding(0) address(
// UNFUSED:     strela.inp_column binding(1) address(
// UNFUSED:     strela.out_column unused
// UNFUSED:     strela.out_column binding(2) address(

module {
  func.func @main(
    %x0: tensor<1x8x16xi32>, %w0: tensor<1x16x9xi32>,
    %x1: tensor<1x8x16xi32>, %w1: tensor<1x16x9xi32>
  ) -> tensor<1x8x9xi32> {
    %zp = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
    %y0 = tosa.matmul %x0, %w0, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>
    %y1 = tosa.matmul %x1, %w1, %zp, %zp : (tensor<1x8x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x8x9xi32>
    %d = tosa.sub %y0, %y1 : (tensor<1x8x9xi32>, tensor<1x8x9xi32>) -> tensor<1x8x9xi32>
    return %d : tensor<1x8x9xi32>
  }
}
