// Minimal program that produces a stream.cmd.flush.
//
// ScheduleAllocation lowers stream.async.transfer into:
//   stream.cmd.invalidate  if the source lives on another device (transfer in)
//   stream.cmd.copy
//   stream.cmd.flush       if the result lives on another device (transfer out)
//
// Which device executes the transfer is decided by
// AsyncTransferOp::getAffinityAttr(): an external source (e.g. a function
// argument) makes it run on the consumer, which gives an invalidate. A
// transient source (a dispatch result) makes it run on the producer, which is
// what we want here: device_a computes %sum, copies it into a buffer for
// device_b and flushes it.
//
// RUN: iree-compile --compile-to=stream \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=local[1] \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=STREAM
//
// RUN: iree-compile --compile-to=hal \
// RUN:   --iree-execution-model=async-external \
// RUN:   --iree-hal-target-device=device_a=local[0] \
// RUN:   --iree-hal-target-device=device_b=local[1] \
// RUN:   --iree-hal-default-device=device_a \
// RUN:   --iree-hal-local-target-device-backends=vmvx %s | FileCheck %s --check-prefix=HAL

// STREAM: stream.cmd.execute on(#hal.device.affinity<@device_a>)
// STREAM:   stream.cmd.dispatch
// STREAM:   stream.cmd.copy %{{.+}}[%c0], %[[DST:.+]][%c0], %c16
// STREAM:   stream.cmd.flush to(#hal.device.affinity<@device_b>) %[[DST]][%c0 for %c16]
// STREAM: stream.cmd.execute on(#hal.device.affinity<@device_b>)

// The flush is recorded on device_a right after the copy.
// HAL: hal.command_buffer.create device(%device_a
// HAL:   hal.command_buffer.copy_buffer
// HAL-NEXT: hal.command_buffer.execution_barrier
// HAL-NEXT: hal.command_buffer.flush_buffer
// HAL-NEXT: hal.command_buffer.execution_barrier
// HAL-NEXT: hal.command_buffer.finalize
func.func @main(
  %arg0: tensor<4xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
) -> (tensor<4xi32> {iree.abi.affinity = #hal.device.promise<@device_b>}) {
  %sum = arith.addi %arg0, %arg0 : tensor<4xi32>
  %sum_b = flow.tensor.transfer %sum : tensor<4xi32> to #hal.device.promise<@device_b>
  %prod = arith.muli %sum_b, %sum_b : tensor<4xi32>
  return %prod : tensor<4xi32>
}
