// The whole Estela flow on the simplest (non trivial) program. The idea is that
// starting from a function that uses the TOSA dialect Estela can automatically
// partition the graph to execute on STRELA the supported operations and on CPU
// the rest.
// ctest --test-dir iree-build -R estela/end2end

func.func @add(%lhs: tensor<?xi32>, %rhs: tensor<?xi32>) -> (tensor<?xi32>) {
  %sum = "tosa.add"(%lhs, %rhs) : (tensor<?xi32>, tensor<?xi32>) -> tensor<?xi32>
  return %sum : tensor<?xi32>
}

// The first phase of compilation converts TOSA to linalg (generic). The fact
// that we are using dynamic shapes makes makes it so the compilers makes some
// shenanigans to support broadcast semantics.

// RUN: iree-compile --iree-plugin=estela --compile-to=input %s \
// RUN:   | FileCheck %s --check-prefix=INP

// INP-LABEL: util.func public @add
// INP-SAME:  (%[[LHS:.+]]: tensor<?xi32>, %[[RHS:.+]]: tensor<?xi32>)
// INP:         %[[BCAST_LHS:.+]] = scf.if %[[LHS:.+]] -> (tensor<?xi32>) {
// INP:         %[[BCAST_RHS:.+]] = scf.if %[[RHS:.+]] -> (tensor<?xi32>) {
// INP:         %[[SUM:.+]] = linalg.generic {{.*}} ins(%[[BCAST_LHS]], %[[BCAST_RHS]] : tensor<?xi32>, tensor<?xi32>)
// INP:           arith.addi
// INP:         util.return %[[SUM]]

// After the input has been lowered to linalg the preprocessing phase follows.
// We now have to specify devices for executing the various operations, this
// will become global variables in the module. After this Estela adds the device
// topology i.e. how STRELA and the CPU can communicate. Then operations
// supported by STRELA are wrapped in flow.tensor.transfer. For now broadcasting
// is left outside the "STRELA region".

// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=preprocessing %s | FileCheck %s --check-prefix=PRE

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

// TODO: FLW:

// After the data flow phase the parallelism in the dataflow graph is made
// explicit with the stream dialect. The broadcast are independent and can run
// in parallel on the CPU. Notice how the data is copied from the CPU to STRELA
// but not the opposite because the CPU can read directly from STRELA buffers.
// The stream dialect also handles allocation of buffers in our case we show the
// allocation of the output buffer and a transient one that its used to copy
// data from the CPU visible memory (two buffers are combined in one) to the
// STRELA visible one.

// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=stream %s | FileCheck %s --check-prefix=STM

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

// RUN: iree-compile --iree-plugin=estela \
// RUN:   --iree-hal-target-device=cpu0=local[0] \
// RUN:   --iree-hal-target-device=strela0=strela \
// RUN:   --iree-hal-default-device=cpu0 \
// RUN:   --iree-hal-local-target-device-backends=vmvx \
// RUN:   --compile-to=executable-targets %s | FileCheck %s --check-prefix=EXC

// EXC:      %[[LHS:.+]] = strela.binding_load binding(0) : tensor<?xi32>
// EXC:      %[[RHS:.+]] = strela.binding_load binding(0) offsets [0, 1] : tensor<?xi32>
// EXC:      %[[SUM:.+]] = strela.add %[[LHS]], %[[RHS]] : tensor<?xi32>
// EXC:      strela.binding_store %[[SUM]], binding(1) : tensor<?xi32>

// Finally compilation should conclude without problems...

// RUNFIXME: iree-compile --iree-plugin=estela \
// RUNFIXME:   --iree-hal-target-device=cpu0=local[0] \
// RUNFIXME:   --iree-hal-target-device=strela0=strela \
// RUNFIXME:   --iree-hal-default-device=cpu0 \
// RUNFIXME:   --iree-hal-local-target-device-backends=vmvx \
// RUNFIXME:   %s -o /dev/null

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

// TODO(Diego): describe what happens in the trivial case.

func.func @bypass(%arg: tensor<?xi32>) -> (tensor<?xi32>) {
  return %arg : tensor<?xi32>
}

// -----

// TODO: check that strela.relu %{{.+}} : tensor<?xi32> is produced

func.func @relu(%input: tensor<?xi32>) -> (tensor<?xi32>) {
  %zero = "tosa.const"() { values = dense<0> : tensor<1xi32> } : () -> tensor<1xi32>
  %relu = tosa.maximum %input_b, %zero : (tensor<?xi32>, tensor<1xi32>) -> tensor<?xi32>
  return %relu : tensor<?xi32>
}
