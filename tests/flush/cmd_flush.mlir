// XFAIL: *
// stream.cmd.flush is still erased on upstream main; the lowering is in
// iree-org/iree#24962. Remove the XFAIL once it lands.
//
// Stream-level module containing stream.cmd.flush, used to exercise the
// Stream -> HAL -> VM lowering in isolation (no need to run the whole pipeline).
//
// stream.cmd.flush is lowered to hal.command_buffer.flush_buffer and from there
// to a call to the hal.command_buffer.flush_buffer VM import.
//
// RUN: iree-opt --split-input-file --iree-hal-conversion --cse \
// RUN:   --iree-hal-indirect-command-buffers=true %s | FileCheck %s --check-prefix=HAL
//
// RUN: iree-opt --split-input-file --iree-hal-conversion --cse \
// RUN:   --iree-hal-indirect-command-buffers=true --iree-vm-conversion %s | FileCheck %s --check-prefix=VM

// Copies followed by a flush are recorded in order into one command buffer.
// This is the shape ScheduleAllocation produces for a device_a -> device_b
// transfer executed on device_a.

util.global private @device_a : !hal.device

// HAL-LABEL: @flushInCommandBuffer
// HAL-SAME: (%[[SRC:.+]]: !hal.buffer, %[[DST:.+]]: !hal.buffer, %[[SIZE:.+]]: index)
// HAL: %[[CMD:.+]] = hal.command_buffer.create
// HAL: hal.command_buffer.copy_buffer<%[[CMD]]
// HAL-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
// HAL-NEXT: hal.command_buffer.copy_buffer<%[[CMD]]
// HAL-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
// HAL-NEXT: hal.command_buffer.flush_buffer<%[[CMD]] : !hal.command_buffer> target(%[[DST]] : !hal.buffer)[%c0, %[[SIZE]]]
// HAL-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
// HAL-NEXT: hal.command_buffer.finalize<%[[CMD]]

// VM-LABEL: vm.func private @flushInCommandBuffer
// VM-SAME: (%[[SRC:.+]]: !vm.ref<!hal.buffer>, %[[DST:.+]]: !vm.ref<!hal.buffer>, %[[SIZE:.+]]: i64)
// VM: vm.call @hal.command_buffer.create
// VM: vm.call @hal.command_buffer.copy_buffer
// VM: vm.call @hal.command_buffer.copy_buffer
// VM: vm.call @hal.command_buffer.flush_buffer(%{{.+}}, %{{.+}}, %[[DST]], %{{.+}}, %[[SIZE]])
// VM: vm.call @hal.command_buffer.finalize
util.func public @flushInCommandBuffer(%src: !stream.resource<transient>, %dst: !stream.resource<transient>, %size: index) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device_a>) with(%src as %src_capture: !stream.resource<transient>{%size}, %dst as %dst_capture: !stream.resource<transient>{%size}) {
    stream.cmd.copy %src_capture[%c0], %dst_capture[%c0], %c16 : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.cmd.copy %src_capture[%c16], %dst_capture[%c16], %c16 : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.cmd.flush to(#hal.device.affinity<@device_b>) %dst_capture[%c0 for %size] : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return %signal : !stream.timepoint
}

// -----

// A single copy followed by a flush used to become hal.device.queue.copy with
// the flush dropped. There is no queue flush so it is now a command buffer.

util.global private @device_a : !hal.device

// HAL-LABEL: @flushAfterSingleCopy
// HAL-NOT: hal.device.queue.copy
// HAL: %[[CMD:.+]] = hal.command_buffer.create
// HAL: hal.command_buffer.copy_buffer<%[[CMD]]
// HAL-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
// HAL-NEXT: hal.command_buffer.flush_buffer<%[[CMD]]
// HAL-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
// HAL-NEXT: hal.command_buffer.finalize<%[[CMD]]
// HAL: hal.device.queue.execute

// VM-LABEL: vm.func private @flushAfterSingleCopy
// VM-NOT: vm.call @hal.device.queue.copy
// VM: vm.call @hal.command_buffer.copy_buffer
// VM: vm.call @hal.command_buffer.flush_buffer
// VM: vm.call @hal.device.queue.execute
util.func public @flushAfterSingleCopy(%src: !stream.resource<transient>, %dst: !stream.resource<transient>, %size: index, %wait: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device_a>) await(%wait) => with(%src as %src_capture: !stream.resource<transient>{%size}, %dst as %dst_capture: !stream.resource<transient>{%size}) {
    stream.cmd.copy %src_capture[%c0], %dst_capture[%c0], %size : !stream.resource<transient>{%size} -> !stream.resource<transient>{%size}
    stream.cmd.flush to(#hal.device.affinity<@device_b>) %dst_capture[%c0 for %size] : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return %signal : !stream.timepoint
}

// -----

// A region with only a flush: the zero-copy shape (CPU writes -> flush ->
// accelerator reads).

util.global private @device_a : !hal.device

// HAL-LABEL: @flushOnly
// HAL-SAME: (%[[DST:.+]]: !hal.buffer, %[[SIZE:.+]]: index
// HAL: %[[CMD:.+]] = hal.command_buffer.create
// HAL-NEXT: hal.command_buffer.flush_buffer<%[[CMD]] : !hal.command_buffer> target(%[[DST]] : !hal.buffer)[%c0, %[[SIZE]]]
// HAL-NEXT: hal.command_buffer.execution_barrier<%[[CMD]]
// HAL-NEXT: hal.command_buffer.finalize<%[[CMD]]

// VM-LABEL: vm.func private @flushOnly
// VM: vm.call @hal.command_buffer.flush_buffer
util.func public @flushOnly(%dst: !stream.resource<transient>, %size: index, %wait: !stream.timepoint) -> !stream.timepoint {
  %c0 = arith.constant 0 : index
  %signal = stream.cmd.execute once on(#hal.device.affinity<@device_a>) await(%wait) => with(%dst as %dst_capture: !stream.resource<transient>{%size}) {
    stream.cmd.flush to(#hal.device.affinity<@device_b>) %dst_capture[%c0 for %size] : !stream.resource<transient>{%size}
  } => !stream.timepoint
  util.return %signal : !stream.timepoint
}
