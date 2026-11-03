// Some tests meant to explain the Estela compilation flow. To run it from the
// command line
// $ ctest --test-dir iree-build -R estela/end2end --rerun-failed --output-on-failure

// We run the compilation flow up to various points that we use in the test to
// illustrate what is going on.

// RUN: iree-compile --split-input-file --iree-plugin=estela --compile-to=input %s \
// RUN:   | FileCheck %s --enable-var-scope --check-prefix=INP

// RUN: iree-compile --split-input-file --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=preprocessing %s | FileCheck %s --enable-var-scope --check-prefix=PRE

// RUN: iree-compile --split-input-file --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=flow %s | FileCheck %s --enable-var-scope --check-prefix=FLW

// RUN: iree-compile --split-input-file --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=stream %s | FileCheck %s --enable-var-scope --check-prefix=STM

// RUN: iree-compile --split-input-file --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=executable-targets %s | FileCheck %s --enable-var-scope --check-prefix=EXC

// RUNFIXME: iree-compile --split-input-file --iree-plugin=estela \
// RUNFIXME:   --iree-hal-target-device=cpu0=local[0] \
// RUNFIXME:   --iree-hal-target-device=strela0=strela \
// RUNFIXME:   --iree-hal-default-device=cpu0 \
// RUNFIXME:   --iree-hal-local-target-device-backends=vmvx \
// RUNFIXME:   --compile-to=vm %s %s | FileCheck %s --enable-var-scope --check-prefix=VMX

// We start with arguably the simplest non-trivial function.

func.func @add(%lhs: tensor<?xi32>, %rhs: tensor<?xi32>) -> (tensor<?xi32>) {
  %sum = "tosa.add"(%lhs, %rhs) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %sum : tensor<?xi32>
}

// The first phase of compilation converts TOSA to linalg (generic). The fact
// that we are using dynamic shapes makes makes it so the compilers makes some
// shenanigans to support broadcast semantics.

// INP-LABEL: module attributes {
// INP:         util.func public @add
// INP-SAME:    (%[[LHS:.+]]: tensor<?xi32>, %[[RHS:.+]]: tensor<?xi32>)
// INP:           %[[BCAST_LHS:.+]] = scf.if %[[LHS:.+]] -> (tensor<?xi32>) {
// INP:           %[[BCAST_RHS:.+]] = scf.if %[[RHS:.+]] -> (tensor<?xi32>) {
// INP:           %[[SUM:.+]] = linalg.generic {{.*}} ins(%[[BCAST_LHS]], %[[BCAST_RHS]] : tensor<?xi32>, tensor<?xi32>)
// INP:             arith.addi
// INP:           util.return %[[SUM]]

// After the input has been lowered to linalg the preprocessing phase follows.
// We now have to specify devices for executing the various operations, this
// will become global variables in the module. After this Estela adds the device
// topology i.e. how STRELA and the CPU can communicate. Then operations
// supported by STRELA are wrapped in flow.tensor.transfer. For now broadcasting
// is left outside the "STRELA region".

// PRE-LABEL: module attributes {
// PRE-SAME:  stream.affinity.default = #hal.device.affinity<@[[CPU:[a-z0-9_]+]]>
// PRE-SAME:  stream.topology = #hal.device.topology<links = [
// PRE-SAME:    (@[[CPU]] -> @[[STRELA:[a-z0-9_]+]] = {}),
// PRE-SAME:    (@[[STRELA]] -> @[[CPU]] = {transparent_access = true})]>
// PRE-SAME:  }
// PRE:       util.global private @[[CPU]] = #hal.device.target<"local"
// PRE:       util.global private @[[STRELA]] = #hal.device.target<"strela"
// PRE:       util.func public @add
// PRE-SAME:  (%[[LHS:.+]]: !hal.buffer_view, %[[RHS:.+]]: !hal.buffer_view)
// PRE:         %{{.+}} = hal.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[LHS]]
// PRE:         %{{.+}} = hal.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[RHS]]
// PRE:         %[[BCAST_LHS:.+]] = scf.if %{{.+}} -> (tensor<?xi32>) {
// PRE:         %[[BCAST_RHS:.+]] = scf.if %{{.+}} -> (tensor<?xi32>) {
// PRE:         %[[LHS_S:.+]] = flow.tensor.transfer %[[BCAST_LHS]] : tensor<?xi32>{%{{.+}}} to #hal.device.affinity<@[[STRELA]]>
// PRE:         %[[RHS_S:.+]] = flow.tensor.transfer %[[BCAST_RHS]] : tensor<?xi32>{%{{.+}}} to #hal.device.affinity<@[[STRELA]]>
// PRE:         %[[SUM_S:.+]] = linalg.generic
// PRE-SAME:      ins(%[[LHS_S]], %[[RHS_S]] : tensor<?xi32>, tensor<?xi32>)
// PRE:           arith.addi
// PRE:         %[[SUM:.+]] = flow.tensor.transfer %[[SUM_S]] : tensor<?xi32>{%{{.+}}} to #hal.device.affinity<@[[CPU]]>
// PRE:         %{{.+}} = hal.tensor.export on(#hal.device.affinity<@[[CPU]]>) %[[SUM]]

// After preprocessing data flow information is injected in the IR using the
// flow dialect and linalg computations are gruupped in flow.executable. For the
// moment we do not go in more detail here...

// FLW-LABEL: module attributes {
// TODO: flow test
// TODO: talk about the flow.dispatch.tie_shape

// After the data flow phase the parallelism in the dataflow graph is made
// explicit with the stream dialect. The broadcast are independent and can run
// in parallel on the CPU. Notice how the data is copied from the CPU to STRELA
// but not the opposite because the CPU can read directly from STRELA buffers.
// The stream dialect also handles allocation of buffers in our case we show the
// allocation of the output buffer and a transient one that its used to copy
// data from the CPU visible memory (two buffers are combined in one) to the
// STRELA visible one.

// STM-LABEL: module attributes {
// STM:         util.global private @[[CPU:.+]] = #hal.device.target<"local"
// STM:         util.global private @[[STRELA:.+]] = #hal.device.target<"strela"
// STM:         stream.executable private @[[BCAST_DISPATCH:.+]] {
// STM:           builtin.module {
// STM:             func.func @[[BCAST_DISPATCH_FUNC:.+]]({{.+}}) {
// STM:               %{{.+}} = linalg.generic {{{.+}}} ins(%{{.+}} : tensor<?xi32>) outs(%{{.+}} : tensor<?xi32>) {
// STM:                 linalg.yield %{{.+}} : i32
// STM:         stream.executable private @[[ADD_DISPATCH:.+]] {
// STM:           builtin.module {
// STM:             func.func @[[ADD_DISPATCH_FUNC:.+]]({{.+}}) {
// STM:               %{{.+}} = linalg.generic {{{.+}}} ins(%{{.+}}, %{{.+}} : tensor<?xi32>, tensor<?xi32>) outs(%{{.+}} : tensor<?xi32>) {
// STM:                 %{{.+}} = arith.addi %{{.+}}, %{{.+}} : i32
// STM:         util.func public @add(%[[LHS:.+]]: !hal.buffer_view, %[[RHS:.+]]: !hal.buffer_view)
// STM:           %{{.+}} = stream.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[LHS]] : !hal.buffer_view -> tensor<?xi32>{%{{.+}}}
// STM:           %{{.+}} = stream.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[RHS]] : !hal.buffer_view -> tensor<?xi32>{%{{.+}}}
// STM:           %{{.+}} = stream.cmd.execute on(#hal.device.affinity<@[[CPU]]>)
// STM:             stream.cmd.dispatch @[[BCAST_DISPATCH]]::@[[BCAST_DISPATCH_FUNC]]
// STM:           %{{.+}} = stream.cmd.execute on(#hal.device.affinity<@[[CPU]]>)
// STM:             stream.cmd.dispatch @[[BCAST_DISPATCH]]::@[[BCAST_DISPATCH_FUNC]]
// STM:           %[[BCAST_TIMEPOINT:.+]] = stream.timepoint.join max(
// STM:           %{{.+}}, %[[OUT_TP:.+]] = stream.resource.alloca {{.*}} await(%[[BCAST_TIMEPOINT]])
// STM:           %{{.+}}, %[[TRANSIENT_TP:.+]] = stream.resource.alloca {{.*}} await(%[[BCAST_TIMEPOINT]])
// STM:           %[[ALLOC_TP:.+]] = stream.timepoint.join max(%[[OUT_TP]], %[[TRANSIENT_TP]])
// STM:           %{{.+}} = stream.cmd.execute on(#hal.device.affinity<@[[STRELA]]>) await(%[[ALLOC_TP]])
// STM:             stream.cmd.concurrent {
// STM:               stream.cmd.invalidate from(#hal.device.affinity<@[[CPU]]>)
// STM:               stream.cmd.copy
// STM:               stream.cmd.invalidate from(#hal.device.affinity<@[[CPU]]>)
// STM:               stream.cmd.copy
// STM:             stream.cmd.dispatch @[[ADD_DISPATCH]]::@[[ADD_DISPATCH_FUNC]]
// STM-NOT:           stream.cmd.copy
// STM:           %{{.+}} = stream.tensor.export on(#hal.device.affinity<@[[CPU]]>) %{{.+}} : tensor<?xi32>{%{{.+}}}

// stream.executable are converted to hal.executable via executable-{sources,configurations,targets}
// the function body is still left as at the stream level.
// TODO: make a sensible thing here...

// EXC-LABEL: module attributes {stream.affinity.default
// EXC:         hal.executable private @{{.+}} {
// EXC:         hal.executable private @{{.+}} {
// EXC:           hal.executable.variant public @{{.+}} target(<"strela", "strela_bitstream">) {
// EXC:             builtin.module {
// EXC:               func.func @{{.+}}() {
// EXC:                 %[[LHS:.+]] = strela.binding_load binding(0) : tensor<?xi32>
// EXC:                 %[[RHS:.+]] = strela.binding_load binding(0) offsets [0, 1] : tensor<?xi32>
// EXC:                 %[[SUM:.+]] = strela.add %[[LHS]], %[[RHS]] : tensor<?xi32>
// EXC:                 strela.binding_store %[[SUM]], binding(1) : tensor<?xi32>

// Finally compilation should conclude without problems...

// TODO: VMX test. Here we need to support dynamic sizes at compile time.

// On the board:
//
//   iree-run-module --module=strela_add.vmfb --function=add \
//     --device=local-sync --device=strela --device_lead_allocator=1 \
//     --input="16xi32=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" \
//     --input="16xi32=10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160" \
//     --expected_output="16xi32=11 22 33 44 55 66 77 88 99 110 121 132 143 154 165 176"
//
// Operands are copied into STRELA memory (the topology says STRELA cannot read
// the CPU's), the dispatch runs the add bitstream with them on input columns 0
// and 1, and the result comes back from output column 1. See strela_ops.mlir for
// the topology and --device_lead_allocator.

// -----

// This is the trivial case and it just returns the buffer view without touching
// it.

func.func @bypass(%arg: tensor<?xi32>) -> (tensor<?xi32>) {
  return %arg : tensor<?xi32>
}

// INP-LABEL: module {

// PRE-LABEL: module attributes {
// PRE-NOT:     flow.tensor.transfer

// FLW-LABEL: module attributes {

// STM-LABEL: module attributes {
// STM-NOT:     stream.cmd.execute

// EXC-LABEL: module attributes {stream.affinity.default
// EXC-NOT:     hal.executable private
// EXC:         util.func public @bypass(%[[ARG:.+]]: !hal.buffer_view) -> !hal.buffer_view
// EXC:           util.return %[[ARG]] : !hal.buffer_view

// TODO: VMX test

// -----

// TODO(Diego): make this more meaningful.
// TODO(Diego): show how constants are handled. Small constants can be baked in
// the kernel bitstream while big constants (weights) need to be moved to a
// STRELA addressable memory.

func.func @relu(%input: tensor<?xi32>) -> (tensor<?xi32>) {
  %zero = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %relu = tosa.maximum %input, %zero : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %relu : tensor<?xi32>
}

// With compile time constants there are no bradcasting shenanigans.

// INP-LABEL: module attributes {
// INP:         util.func public @relu(%[[ARG:.+]]: tensor<?xi32>) -> tensor<?xi32> {
// INP:           %[[CST:.+]] = arith.constant dense<0> : tensor<1xi32>
// INP-NOT:       %{{.+}} = scf.if %{{.+}} -> (tensor<?xi32>) {
// INP:           %[[RES:.+]] = linalg.generic {{.*}} ins(%[[ARG]], %[[CST]] : tensor<?xi32>, tensor<1xi32>)
// INP:           util.return %[[RES]]

// Compile time constants do not need to be "imported" from hal.buffer_view
// since they are already in a standard MLIR type. Also scalar constants are not
// transferred since it will be baked in the bitstream as immediate values.

// PRE-LABEL: module attributes {
// PRE:         util.func public @relu(%{{.+}}: !hal.buffer_view) -> !hal.buffer_view
// PRE:           %[[CST:.+]] = arith.constant dense<0> : tensor<1xi32>
// PRE-NOT:       %{{.+}} = hal.tensor.import on(#hal.device.affinity<@{{.+}}>) %[[CST]]
// PRE-NOT:       %{{.+}} = flow.tensor.transfer %[[CST]] : tensor<1xi32>{%{{.+}}} to #hal.device.affinity<@{{.+}}>

// During dispatch creation (--compile-to=dispatch-creation) the constant is
// copied inside the executable.

// FLW-LABEL: module attributes {
// FLW:         flow.executable private @[[DISPATCH:.+]] {
// FLW:           builtin.module {
// FLW:             func.func @[[DISPATCH_FUNC:.+]](
// FLW-SAME:        %{{.+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<?xi32>>,
// FLW-SAME:        %{{.+}}: index,
// FLW-SAME:        %{{.+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<?xi32>>
// FLW-SAME:        ) {
// FLW:               %[[CONST:.+]] = arith.constant 0 : i32
// FLW:               %{{.+}} = linalg.generic
// FLW:                 %{{.+}} = arith.maxsi %{{.+}}, %[[CONST]] : i32
// FLW:             util.func public @relu(%{{.+}}: !hal.buffer_view) {{.+}} {

// STM-LABEL: module attributes {

// EXC-LABEL: module attributes {stream.affinity.default
// EXC:             %{{.+}} = strela.relu %{{.+}} : tensor<?xi32>

// -----

// STRELA can have immediate scalar values (int32) in its bitstream but bigger
// constants need to be moved to STRELA addressable memory.

func.func @add_big_const(%input: tensor<16xi32>) -> (tensor<16xi32>) {
  %big_const = "tosa.const"() {
    values = dense<[0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,
                    0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF]> : tensor<16xi32>
  } : () -> tensor<16xi32>
  %add = "tosa.add"(%big_const, %input) : (tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
  return %add : tensor<16xi32>
}

// FLW-LABEL: module attributes {
// FLW:         flow.executable private @[[DISPATCH:.+]] {
// FLW:           builtin.module {
// FLW:             func.func @[[DISPATCH_FUNC:.+]](
// FLW-SAME:        %{{[a-z0-9]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>,
// FLW-SAME:        %{{[a-z0-9]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<16xi32>>,
// FLW-SAME:        %{{[a-z0-9]+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<16xi32>>
// FLW-SAME:        ) {
// FLW:             util.global private @[[CONST_GLOB:.+]] <{{.+}}> {{{.+}}} = dense<[{{.+}}]> : tensor<16xi32>
// FLW:             util.func public @add_big_const(%{{.+}}: !hal.buffer_view) {{.+}} {
// FLW:               %[[CONST_LOC:.+]] = util.global.load immutable @[[CONST_GLOB]] : tensor<16xi32>
// FLW:               %{{.+}} = flow.tensor.transfer %[[CONST_LOC]] : tensor<16xi32> to #hal.device.affinity

// -----

func.func @qmatmul(%act: tensor<?x128xi8>, %wgt: tensor<128x128xi8>) -> (tensor<?x128xi8>) {
  %act_shape = tosa.const_shape values(dense<[-1, 1, 1, 128]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %wgt_shape = tosa.const_shape values(dense<[128, 1, 1, 128]> : tensor<4xindex>) : () -> !tosa.shape<4>
  %out_shape = tosa.const_shape values(dense<[-1, 128]> : tensor<2xindex>) : () -> !tosa.shape<2>

  %act4 = tosa.reshape %act, %act_shape : (tensor<?x128xi8>, !tosa.shape<4>) -> tensor<?x1x1x128xi8>
  %wgt4 = tosa.reshape %wgt, %wgt_shape : (tensor<128x128xi8>, !tosa.shape<4>) -> tensor<128x1x1x128xi8>
  %bias = "tosa.const"() {values = dense<777> : tensor<128xi32>} : () -> tensor<128xi32>

  %zp_i8 = "tosa.const"() {values = dense<111> : tensor<1xi8>} : () -> tensor<1xi8>
  %zp_i32 = "tosa.const"() {values = dense<222> : tensor<1xi32>} : () -> tensor<1xi32>
  %multiplier = "tosa.const"() {values = dense<1073741824> : tensor<1xi32>} : () -> tensor<1xi32>
  %shift = "tosa.const"() {values = dense<38> : tensor<1xi8>} : () -> tensor<1xi8>

  %acc4 = tosa.conv2d %act4, %wgt4, %bias, %zp_i8, %zp_i8 pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<?x1x1x128xi8>, tensor<128x1x1x128xi8>, tensor<128xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1x1x128xi32>
  %out4 = tosa.rescale %acc4, %multiplier, %shift, %zp_i32, %zp_i8 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<?x1x1x128xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<?x1x1x128xi8>

  %out = tosa.reshape %out4, %out_shape : (tensor<?x1x1x128xi8>, !tosa.shape<2>) -> tensor<?x128xi8>
  return %out : tensor<?x128xi8>
}

// The two reshape get converted to to tensor.expand_shape since are not a
// general reshape. The conv2d is converted to a conv2d with a specific layout
// preceded by a transpose. This is equivalent to the transposed matrix
// multiplication that is more efficient (for cache access) in row-major layout.
// To support broadcasting rescale is wrapped in a scf.for. For the way in which
// linalg does things the bias is bradcased with linalg.generic to a buffer in
// which linalg.conv_2d_nhwc_hwcf accumulates into.

// INP-LABEL: module attributes {
// INP:         util.func public @qmatmul
// INP-SAME:    (%[[ACT:.+]]: tensor<?x128xi8>, %[[WGT:.+]]: tensor<128x128xi8>)
// INP:           %[[CONST:.+]] = arith.constant dense<777> : tensor<128xi32>
// INP:           %[[ACT_EXP:.+]] = tensor.expand_shape %[[ACT]]
// INP:           %[[WGT_EXP:.+]] = tensor.expand_shape %[[WGT]]
// INP:           %[[WGT_EXP_T:.+]] = linalg.transpose ins(%[[WGT_EXP]] : tensor<128x1x1x128xi8>)
// INP:           %[[ACC:.+]] = linalg.generic {{{.+}}} ins(%[[CONST]] : tensor<128xi32>)
// INP:           %{{.+}} = linalg.conv_2d_nhwc_hwcf {{{.+}}}
// INP-SAME:      ins(%[[ACT_EXP]], %[[WGT_EXP_T]] : tensor<?x1x1x128xi8>, tensor<1x1x128x128xi8>)
// INP-SAME:      outs(%[[ACC]] : tensor<?x1x1x128xi32>)
// INP:           %[[RSC:.+]] = scf.for
// INP:           %[[OUT:.+]] = tensor.collapse_shape %[[RSC]]
// INP:           util.return %[[OUT]]
