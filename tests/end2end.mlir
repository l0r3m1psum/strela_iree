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
// RUN:   --iree-stream-resource-index-bits=32 \
// RUN:   --compile-to=executable-targets %s | FileCheck %s --enable-var-scope --check-prefix=EXC

// RUNFIXME: iree-compile --split-input-file --iree-plugin=estela \
// RUNFIXME:   --iree-hal-target-device=cpu0=local[0] \
// RUNFIXME:   --iree-hal-target-device=strela0=strela \
// RUNFIXME:   --iree-hal-default-device=cpu0 \
// RUNFIXME:   --iree-hal-local-target-device-backends=vmvx \
// RUNFIXME:   --compile-to=hal %s %s | FileCheck %s --enable-var-scope --check-prefix=HAL

// TODO: --compile-to=hal shows how the work is divided between workgroups
// hal.command_buffer.dispatch<%cmd : !hal.command_buffer> {{.+}} workgroups([%c3, %n, %c1])

// We start with the one thing STRELA runs: N rows of C = 16 words times a
// C x F = 16 x 9 matrix. This is, strictly speaking not valid TOSA (because
// tosa.matmul does not support int32), and TOSA matmul is necessarily a batched
// matrix multiplication, hence the leading 1.

func.func @matvec(%x: tensor<1x?x16xi32>, %w: tensor<1x16x9xi32>) -> tensor<1x?x9xi32> {
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %y = tosa.matmul %x, %w, %zp, %zp : (tensor<1x?x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x?x9xi32>
  return %y : tensor<1x?x9xi32>
}

// The first phase of compilation converts TOSA to linalg. The zero points are
// zero, so they disappear; the linalg.fill initializes the accumulator of
// linalg.batch_matmul (this is required by the linalg semantics).

// INP-LABEL: module attributes {
// INP:         util.func public @matvec
// INP-SAME:    (%[[X:[^:]+]]: tensor<1x?x16xi32>, %[[W:[^:]+]]: tensor<1x16x9xi32>)
// INP:           %[[ZERO:.+]] = linalg.fill
// INP:           %[[Y:.+]] = linalg.batch_matmul ins(%[[X]], %[[W]] : tensor<1x?x16xi32>, tensor<1x16x9xi32>) outs(%[[ZERO]] : tensor<1x?x9xi32>)
// INP:           util.return %[[Y]]

// After the input has been lowered to linalg the preprocessing phase follows.
// We now have to specify devices for executing the various operations, this
// will become global variables in the module. After this Estela adds the device
// topology i.e. how STRELA and the CPU can communicate. Then operations
// supported by STRELA are wrapped in flow.tensor.transfer. The zero fill is
// left outside the "STRELA region": the kernel computes from zero anyway.

// PRE-LABEL: module attributes {
// PRE-SAME:  stream.affinity.default = #hal.device.affinity<@[[CPU:[a-z0-9_]+]]>
// PRE-SAME:  stream.topology = #hal.device.topology<links = [
// PRE-SAME:    (@[[CPU]] -> @[[STRELA:[a-z0-9_]+]] = {}),
// PRE-SAME:    (@[[STRELA]] -> @[[CPU]] = {transparent_access = true})]>
// PRE-SAME:  }
// PRE:       util.global private @[[CPU]] = #hal.device.target<"local"
// PRE:       util.global private @[[STRELA]] = #hal.device.target<"strela"
// PRE:       util.func public @matvec
// PRE-SAME:  (%[[X:[^:]+]]: !hal.buffer_view, %[[W:[^:]+]]: !hal.buffer_view)
// PRE:         %[[X_H:.+]] = hal.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[X]]
// PRE:         %[[W_H:.+]] = hal.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[W]]
// PRE:         %[[ZERO:.+]] = linalg.fill
// PRE:         %[[X_S:.+]] = flow.tensor.transfer %[[X_H]] : tensor<1x?x16xi32>{%{{.+}}} to #hal.device.affinity<@[[STRELA]]>
// PRE:         %[[W_S:.+]] = flow.tensor.transfer %[[W_H]] : tensor<1x16x9xi32> to #hal.device.affinity<@[[STRELA]]>
// PRE:         %[[Y_S:.+]] = linalg.batch_matmul
// PRE-SAME:      ins(%[[X_S]], %[[W_S]] : tensor<1x?x16xi32>, tensor<1x16x9xi32>)
// PRE-SAME:      outs(%[[ZERO]] : tensor<1x?x9xi32>)
// PRE:         %[[Y:.+]] = flow.tensor.transfer %[[Y_S]] : tensor<1x?x9xi32>{%{{.+}}} to #hal.device.affinity<@[[CPU]]>
// PRE:         %{{.+}} = hal.tensor.export on(#hal.device.affinity<@[[CPU]]>) %[[Y]]

// After preprocessing data flow information is injected in the IR using the
// flow dialect and linalg computations are grouped in flow.executable. Also the
// concept work groups appear at this point. Dispatch region formation pulls the
// zero fill into the dispatch, so there is a single dispatch and nothing runs
// on the CPU. N reaches the dispatch as workload ordinal 0.

// FLW-LABEL: module attributes {
// FLW:         util.global private @cpu0 = #hal.device.target<"local", {{.+}}> : !hal.device
// FLW:         util.global private @strela0 = #hal.device.target<"strela", {{.+}}> : !hal.device
// FLW:         flow.executable private @[[DISPATCH:[a-z0-9_]+]] {
// FLW:           flow.executable.export public @[[DISPATCH_FUNC:[a-zA-Z0-9_]+]] workgroups(%{{.+}}: index) -> (index, index, index) {
// FLW:             %{{.+}}, %{{.+}}, %{{.+}} = iree_tensor_ext.dispatch.workgroup_count_from_slice
// FLW:           builtin.module {
// FLW:             func.func @[[DISPATCH_FUNC]](
// FLW:               %{{.+}} = iree_tensor_ext.dispatch.workload.ordinal %{{.+}}, 0 : index
// FLW:               %[[ZERO:.+]] = linalg.fill
// FLW:               %{{.+}} = linalg.batch_matmul
// FLW-SAME:          outs(%[[ZERO]] : tensor<1x?x9xi32>)
// FLW:         util.func public @matvec(
// FLW:           %[[X_S:.+]] = flow.tensor.transfer %{{.+}} to #hal.device.affinity<@strela0>
// FLW:           %[[W_S:.+]] = flow.tensor.transfer %{{.+}} to #hal.device.affinity<@strela0>
// FLW:           %[[Y_S:.+]] = flow.dispatch @[[DISPATCH]]::@[[DISPATCH_FUNC]][%{{.+}}](%[[X_S]], %[[W_S]], %{{.+}})
// FLW:           %{{.+}} = flow.tensor.transfer %[[Y_S]] : {{.+}} to #hal.device.affinity<@cpu0>
// TODO: talk about the flow.dispatch.tie_shape

// After the data flow phase the parallelism in the dataflow graph is made
// explicit with the stream dialect. Notice how the data is copied from the CPU
// to STRELA but not the opposite because the CPU can read directly from STRELA
// buffers. The stream dialect also handles allocation of buffers: we show the
// allocation of the output buffer and a transient one that is used to copy data
// from the CPU visible memory to the STRELA visible one. Both operands are
// packed into it, %w at byte 0 and %x after its 16 x 9 x 4 = 576 bytes, so the
// dispatch has a single input binding. N reaches it as two i32 operands, the
// halves of a 64 bit index.

// STM-LABEL: module attributes {
// STM:         util.global private @[[CPU:[a-z0-9_]+]] = #hal.device.target<"local"
// STM:         util.global private @[[STRELA:[a-z0-9_]+]] = #hal.device.target<"strela"
// STM:         stream.executable private @[[DISPATCH:[a-z0-9_]+]] {
// STM:           builtin.module {
// STM:             func.func @[[DISPATCH_FUNC:[a-zA-Z0-9_]+]](%{{.+}}: !stream.binding {{.+}}, %{{.+}}: !stream.binding {{.+}}, %{{.+}}: i32, %{{.+}}: i32) {
// STM:               stream.binding.subspan %{{.+}}[%c0] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x9xi32>>
// STM:               stream.binding.subspan %{{.+}}[%c576] : !stream.binding -> !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x?x16xi32>>
// STM:               %{{.+}} = linalg.batch_matmul
// STM:         util.func public @matvec(%[[X:[^:]+]]: !hal.buffer_view, %[[W:[^:]+]]: !hal.buffer_view)
// STM:           %{{.+}} = stream.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[X]] : !hal.buffer_view -> tensor<1x?x16xi32>{%{{.+}}}
// STM:           %{{.+}} = stream.tensor.import on(#hal.device.affinity<@[[CPU]]>) %[[W]] : !hal.buffer_view -> tensor<1x16x9xi32>
// STM:           %{{.+}}, %{{.+}} = stream.resource.alloca uninitialized on(#hal.device.optimal<[#hal.device.affinity<@[[CPU]]>, #hal.device.affinity<@[[STRELA]]>]>) : !stream.resource<external>
// STM:           %{{.+}}, %{{.+}} = stream.resource.alloca uninitialized on(#hal.device.affinity<@[[STRELA]]>) : !stream.resource<transient>
// STM:           %{{.+}} = stream.cmd.execute on(#hal.device.affinity<@[[STRELA]]>)
// STM:             stream.cmd.concurrent {
// STM:               stream.cmd.invalidate from(#hal.device.affinity<@[[CPU]]>)
// STM:               stream.cmd.copy %{{.+}}[%c0], %{{.+}}[%c576]
// STM:               stream.cmd.invalidate from(#hal.device.affinity<@[[CPU]]>)
// STM:               stream.cmd.copy %{{.+}}[%c0], %{{.+}}[%c0], %c576
// STM:             stream.cmd.dispatch @[[DISPATCH]]::@[[DISPATCH_FUNC]][%{{.+}}](%{{.+}}, %{{.+}} : i32, i32) {
// STM-NEXT:          ro %{{.+}} : !stream.resource<transient>
// STM-NEXT:          wo %{{.+}} : !stream.resource<external>
// STM-NOT:           stream.cmd.copy
// STM:           %{{.+}} = stream.tensor.export on(#hal.device.affinity<@[[CPU]]>) %{{.+}} : tensor<1x?x9xi32>{%{{.+}}}

// stream.executable are converted to hal.executable via
// executable-{sources,configurations,targets}. The STRELA backend sets the
// workgroup count, (ceildiv(F, 3), N, 1) = (3, N, 1), one job per row and
// triple of features, and restates the dispatch interface as
// strela.binding_load/store. With --iree-stream-resource-index-bits=32 N is a
// single push constant, which only the count uses.

// EXC-LABEL: module attributes {stream.affinity.default
// EXC:         hal.executable private @{{.+}} {
// EXC:           hal.executable.variant public @{{.+}} target(<"strela", "strela_bitstream", {{{.+}}}>) {
// EXC:             hal.executable.export public @{{.+}} layout(#hal.pipeline.layout<constants = 1, {{.+}}>) count(%{{[^:]+}}: !hal.device, %[[N:[^:]+]]: index)
// EXC:               %[[C3:.+]] = arith.constant 3 : index
// EXC:               hal.return %[[C3]], %[[N]], %{{.+}} : index, index, index
// EXC:             builtin.module {
// EXC:               func.func @{{.+}}() {
// EXC:                 %[[X:.+]] = strela.binding_load binding(0) byte_offset 576 : tensor<1x?x16xi32>
// EXC:                 %[[W:.+]] = strela.binding_load binding(0) : tensor<1x16x9xi32>
// EXC:                 %[[Y:.+]] = strela.matvec %[[X]], %[[W]] : tensor<1x?x16xi32>, tensor<1x16x9xi32> -> tensor<1x?x9xi32>
// EXC:                 strela.binding_store %[[Y]], binding(1) : tensor<1x?x9xi32>

// Finally compilation should conclude without problems...

// TODO: HAL test. Here we need to support dynamic sizes at compile time.

// On the board:
//
//   iree-run-module --module=strela_matvec.vmfb --function=matvec \
//     --device=local-sync --device=strela --device_lead_allocator=1 \
//     --input="1x1x16xi32=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" \
//     --input="1x16x9xi32=1" \
//     --expected_output="1x1x9xi32=136"
//
// Operands are copied into STRELA memory (the topology says STRELA cannot read
// the CPU's), the dispatch runs the matvec bitstream as 3 jobs, each with the
// row on input column 0 and three columns of %w on inputs 1-3, and the results
// come back from output columns 1-3. See strela_ops.mlir for the topology and
// --device_lead_allocator.

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

// TODO: HAL test

// -----

// STRELA only runs the matvec: the ReLU after it stays on the CPU, which reads
// the matvec's result straight from STRELA memory.

func.func @matvec_relu(%x: tensor<1x?x16xi32>, %w: tensor<1x16x9xi32>) -> tensor<1x?x9xi32> {
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %zero = "tosa.const"() { values = dense<0> : tensor<1x1x1xi32> } : () -> tensor<1x1x1xi32>
  %y = tosa.matmul %x, %w, %zp, %zp : (tensor<1x?x16xi32>, tensor<1x16x9xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x?x9xi32>
  %relu = tosa.maximum %y, %zero : (tensor<1x?x9xi32>, tensor<1x1x1xi32>) -> tensor<1x?x9xi32>
  return %relu : tensor<1x?x9xi32>
}

// With compile time constants there are no broadcasting shenanigans.

// INP-LABEL: module attributes {
// INP:         util.func public @matvec_relu(%[[X:[^:]+]]: tensor<1x?x16xi32>, %[[W:[^:]+]]: tensor<1x16x9xi32>) -> tensor<1x?x9xi32> {
// INP:           %[[CST:.+]] = arith.constant dense<0> : tensor<1x1x1xi32>
// INP-NOT:       scf.if
// INP:           %[[Y:.+]] = linalg.batch_matmul ins(%[[X]], %[[W]]
// INP:           %[[RES:.+]] = linalg.generic {{.*}} ins(%[[Y]], %[[CST]] : tensor<1x?x9xi32>, tensor<1x1x1xi32>)
// INP:             arith.maxsi
// INP:           util.return %[[RES]]

// Compile time constants do not need to be "imported" from hal.buffer_view
// since they are already in a standard MLIR type. Only the matvec is wrapped in
// transfers; the ReLU takes the result after it has come back to the CPU.

// PRE-LABEL: module attributes {
// PRE:         util.func public @matvec_relu(%{{.+}}: !hal.buffer_view, %{{.+}}: !hal.buffer_view) -> !hal.buffer_view
// PRE:           %[[CST:.+]] = arith.constant dense<0> : tensor<1x1x1xi32>
// PRE-NOT:       hal.tensor.import {{.+}} %[[CST]]
// PRE:           %[[Y_S:.+]] = linalg.batch_matmul
// PRE:           %[[Y:.+]] = flow.tensor.transfer %[[Y_S]] : {{.+}} to #hal.device.affinity<@cpu0>
// PRE-NOT:       flow.tensor.transfer
// PRE:           linalg.generic {{.+}} ins(%[[Y]], %[[CST]] : tensor<1x?x9xi32>, tensor<1x1x1xi32>)
// PRE:             arith.maxsi

// During dispatch creation (--compile-to=dispatch-creation) the ReLU gets a
// dispatch of its own, and its scalar zero is copied inside the executable.

// FLW-LABEL: module attributes {
// FLW:         flow.executable private @[[MATVEC:[a-z0-9_]+]] {
// FLW:             func.func @{{[a-z0-9_]+}}_batch_matmul_{{[a-zA-Z0-9_]+}}(
// FLW:               linalg.batch_matmul
// FLW:         flow.executable private @[[RELU:[a-z0-9_]+]] {
// FLW:             func.func @{{[a-z0-9_]+}}_elementwise_{{[a-zA-Z0-9_]+}}(
// FLW:               %[[CONST:.+]] = arith.constant 0 : i32
// FLW:               %{{.+}} = linalg.generic
// FLW:                 %{{.+}} = arith.maxsi %{{.+}}, %[[CONST]] : i32
// FLW:         util.func public @matvec_relu(%{{.+}}: !hal.buffer_view, %{{.+}}: !hal.buffer_view) {{.+}} {
// FLW:           %[[Y_S:.+]] = flow.dispatch @[[MATVEC]]::
// FLW:           %{{.+}} = flow.tensor.transfer %[[Y_S]] : {{.+}} to #hal.device.affinity<@cpu0>
// FLW:           %{{.+}} = flow.dispatch @[[RELU]]::

// STM-LABEL: module attributes {

// The matvec is a STRELA executable, the ReLU a VMVX one.

// EXC-LABEL: module attributes {stream.affinity.default
// EXC:         hal.executable private @{{.+}} {
// EXC:           hal.executable.variant public @strela_bitstream
// EXC:                 %{{.+}} = strela.matvec %{{.+}}, %{{.+}} : tensor<1x?x16xi32>, tensor<1x16x9xi32> -> tensor<1x?x9xi32>
// EXC:         hal.executable private @{{.+}} {
// EXC:           hal.executable.variant public @vmvx_bytecode_fb

// -----

// STRELA can have immediate scalar values (int32) in its bitstream but bigger
// constants, like the weights of a layer, need to be moved to STRELA
// addressable memory.

func.func @matvec_const_weights(%x: tensor<1x1x16xi32>) -> tensor<1x1x3xi32> {
  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %w = "tosa.const"() {values = dense<[[
    [0, 1, 2],    [3, 4, 5],    [6, 7, 8],    [9, 10, 11],
    [12, 13, 14], [15, 16, 17], [18, 19, 20], [21, 22, 23],
    [24, 25, 26], [27, 28, 29], [30, 31, 32], [33, 34, 35],
    [36, 37, 38], [39, 40, 41], [42, 43, 44], [45, 46, 47]
  ]]> : tensor<1x16x3xi32>} : () -> tensor<1x16x3xi32>
  %y = tosa.matmul %x, %w, %zp, %zp : (tensor<1x1x16xi32>, tensor<1x16x3xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x3xi32>
  return %y : tensor<1x1x3xi32>
}

// INP-LABEL: module attributes {
// INP:         util.func public @matvec_const_weights(%[[X:[^:]+]]: tensor<1x1x16xi32>)
// INP:           %[[W:.+]] = arith.constant dense<{{.+}}> : tensor<1x16x3xi32>
// INP:           %{{.+}} = linalg.batch_matmul ins(%[[X]], %[[W]] : tensor<1x1x16xi32>, tensor<1x16x3xi32>)

// The constant becomes a global, which is loaded and transferred to STRELA like
// any other operand.

// FLW-LABEL: module attributes {
// FLW:         flow.executable private @{{[a-z0-9_]+}} {
// FLW:           builtin.module {
// FLW:             func.func @{{[a-zA-Z0-9_]+}}(
// FLW-SAME:        %{{[a-z0-9]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x1x16xi32>>,
// FLW-SAME:        %{{[a-z0-9]+}}: !iree_tensor_ext.dispatch.tensor<readonly:tensor<1x16x3xi32>>,
// FLW-SAME:        %{{[a-z0-9]+}}: !iree_tensor_ext.dispatch.tensor<writeonly:tensor<1x1x3xi32>>
// FLW-SAME:        ) {
// FLW:         util.global private @[[CONST_GLOB:[a-z0-9_]+]] <{{.+}}> {{{.+}}} = dense<[{{.+}}]> : tensor<1x16x3xi32>
// FLW:         util.func public @matvec_const_weights(%{{.+}}: !hal.buffer_view) {{.+}} {
// FLW:           %[[CONST_LOC:.+]] = util.global.load immutable @[[CONST_GLOB]] : tensor<1x16x3xi32>
// FLW:           %{{.+}} = flow.tensor.transfer %[[CONST_LOC]] : tensor<1x16x3xi32> to #hal.device.affinity<@strela0>

// The weights get a binding of their own, not Indirect: constants live in a
// buffer of their own, bound once.

// EXC-LABEL: module attributes {stream.affinity.default
// EXC:         hal.executable.export public @{{.+}} layout(#hal.pipeline.layout<bindings = [{{.+}}, #hal.pipeline.binding<storage_buffer, ReadOnly>, {{.+}}]
// EXC:                 %[[X:.+]] = strela.binding_load binding(0) : tensor<1x1x16xi32>
// EXC:                 %[[W:.+]] = strela.binding_load binding(1) : tensor<1x16x3xi32>
// EXC:                 %{{.+}} = strela.matvec %[[X]], %[[W]] : tensor<1x1x16xi32>, tensor<1x16x3xi32> -> tensor<1x1x3xi32>

// -----

// This is, strictly speaking not valid TOSA (because tosa.matmul does not
// support int32). TOSA matmul is also necessarily a batched matrix
// multiplication hence the reshape.

func.func @matmul_bias(%act: tensor<?x128xi32>, %wgt: tensor<128x128xi32>) -> (tensor<?x128xi32>) {
  %act_shape = tosa.const_shape values(dense<[1, -1, 128]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %wgt_shape = tosa.const_shape values(dense<[1, 128, 128]> : tensor<3xindex>) : () -> !tosa.shape<3>
  %out_shape = tosa.const_shape values(dense<[-1, 128]> : tensor<2xindex>) : () -> !tosa.shape<2>

  %zp = "tosa.const"() {values = dense<0> : tensor<1xi32>} : () -> tensor<1xi32>
  %bias = "tosa.const"() {values = dense<[[[
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF,
    0x0,0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,0x9,0xA,0xB,0xC,0xD,0xE,0xF
  ]]]> : tensor<1x1x128xi32>} : () -> tensor<1x1x128xi32>

  %act3 = tosa.reshape %act, %act_shape : (tensor<?x128xi32>, !tosa.shape<3>) -> tensor<1x?x128xi32>
  %wgt3 = tosa.reshape %wgt, %wgt_shape : (tensor<128x128xi32>, !tosa.shape<3>) -> tensor<1x128x128xi32>

  %wgt3_t = tosa.transpose %wgt3 perms([0, 2, 1]) : (tensor<1x128x128xi32>) -> tensor<1x128x128xi32>
  %acc3 = tosa.matmul %act3, %wgt3_t, %zp, %zp : (tensor<1x?x128xi32>, tensor<1x128x128xi32>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x?x128xi32>
  %out3 = tosa.add %acc3, %bias : (tensor<1x?x128xi32>, tensor<1x1x128xi32>) -> tensor<1x?x128xi32>

  %out = tosa.reshape %out3, %out_shape : (tensor<1x?x128xi32>, !tosa.shape<2>) -> tensor<?x128xi32>
  return %out : tensor<?x128xi32>
}

// The three reshape get converted to to tensor.{expand_shape,collapse_shape}
// since are not a general reshapes. The linalg.fill is used to initialize the
// accumulator of linalg.batch_matmul (this is required by the linalg semantics)
// other operations like convolution initialize the accumlator with the bias.

// INP-LABEL: module attributes {
// INP:         util.func public @matmul_bias
// INP-SAME:    (%[[ACT:.+]]: tensor<?x128xi32>, %[[WGT:.+]]: tensor<128x128xi32>)
// INP:           %[[BIAS:.+]] = arith.constant dense<{{.+}}> : tensor<1x1x128xi32>
// INP:           %[[ACT_EXP:.+]] = tensor.expand_shape %[[ACT]]
// INP:           %[[WGT_EXP:.+]] = tensor.expand_shape %[[WGT]]
// INP:           %[[WGT_T:.+]] = linalg.transpose ins(%[[WGT_EXP]] : tensor<1x128x128xi32>) {{.+}} permutation = [0, 2, 1]
// INP:           %[[ZERO:.+]] = linalg.fill
// INP:           %[[MM:.+]] = linalg.batch_matmul ins(%[[ACT_EXP]], %[[WGT_T]] : tensor<1x?x128xi32>, tensor<1x128x128xi32>) outs(%[[ZERO]] : tensor<1x?x128xi32>)
// INP:           %[[ADD:.+]] = linalg.generic {{.+}} affine_map<(d0, d1, d2) -> (d0, 0, d2)>{{.+}} ins(%[[MM]], %[[BIAS]] : tensor<1x?x128xi32>, tensor<1x1x128xi32>)
// INP:             arith.addi
// INP:           %[[OUT:.+]] = tensor.collapse_shape %[[ADD]]
// INP:           util.return %[[OUT]]

// linalg.batch_matmul is wrapped in transfers except for the accumulator
// (linalg.fill) which is left untouched.

// PRE-LABEL: module attributes {
// PRE:          util.func public @matmul_bias
// PRE:            %[[ACC:.+]] = linalg.fill
// PRE:            %[[LHS:.+]] = flow.tensor.transfer
// PRE:            %[[WGT:.+]] = flow.tensor.transfer
// PRE:            %[[OUT:.+]] = linalg.batch_matmul
// PRE-SAME:       ins(%[[LHS]], %[[WGT]] : tensor<1x?x128xi32>, tensor<1x128x128xi32>)
// PRE-SAME:       outs(%[[ACC]] : tensor<1x?x128xi32>) -> tensor<1x?x128xi32>
// PRE:            %{{.+}} = flow.tensor.transfer %[[OUT]]

// FLW-LABEL: module attributes {
// FLW:         flow.executable private @[[TRANS_DISPATCH:.+]] {
// FLW:         flow.executable private @[[MATMUL_DISPATCH:.+]] {
// FLW:           builtin.module {
// FLW:             func.func @[[MATMUL_FUNC:.+]]
// FLW:               %[[ACC:.+]] = linalg.fill
// FLW:               %{{.+}} = linalg.batch_matmul
// FLW:               ins(%{{.+}}, %{{.+}} : tensor<1x?x128xi32>, tensor<1x128x128xi32>)
// FLW:               outs(%[[ACC]] : tensor<1x?x128xi32>) -> tensor<1x?x128xi32>
// FLW:         flow.executable private @[[BIAS_DISPATCH:.+]] {
// FLW:         util.global private @{{.+}}

// EXC-LABEL: module attributes {stream.affinity.default
// EXC:               %{{.+}} = strela.matvec %{{.+}}, %{{.+}} :
