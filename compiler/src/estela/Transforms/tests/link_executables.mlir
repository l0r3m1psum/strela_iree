// RUN: iree-opt --iree-plugin=estela --iree-strela-link-executables %s | FileCheck %s

// Two STRELA executables become one with two exports, ordinals 0 and 1, and the
// references to them follow. The vmvx executable is another target's business.

#strela_target = #hal.executable.target<"strela", "strela_bitstream">
#vmvx_target = #hal.executable.target<"vmvx", "vmvx-bytecode-fb">
#pipeline_layout = #hal.pipeline.layout<bindings = [
  #hal.pipeline.binding<storage_buffer>,
  #hal.pipeline.binding<storage_buffer>
]>

module @ops {
  hal.executable private @add_dispatch_0 {
    hal.executable.variant public @strela_bitstream target(#strela_target) {
      hal.executable.export public @add_dispatch_0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @add_dispatch_0() {
          return
        }
      }
    }
  }
  hal.executable private @relu_dispatch_0 {
    hal.executable.variant public @strela_bitstream target(#strela_target) {
      hal.executable.export public @relu_dispatch_0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
        func.func @relu_dispatch_0() {
          return
        }
      }
    }
  }
  hal.executable private @sub_dispatch_0 {
    hal.executable.variant public @vmvx target(#vmvx_target) {
      hal.executable.export public @sub_dispatch_0 ordinal(0) layout(#pipeline_layout) count(%device: !hal.device) -> (index, index, index) {
        %c1 = arith.constant 1 : index
        hal.return %c1, %c1, %c1 : index, index, index
      }
      builtin.module {
      }
    }
  }
  func.func @dispatch_sites() attributes {
    testing.add = @add_dispatch_0::@strela_bitstream::@add_dispatch_0,
    testing.relu = @relu_dispatch_0::@strela_bitstream::@relu_dispatch_0,
    testing.sub = @sub_dispatch_0::@vmvx::@sub_dispatch_0
  } {
    return
  }
}

// CHECK-NOT:   hal.executable private @add_dispatch_0
// CHECK-NOT:   hal.executable private @relu_dispatch_0
// CHECK:       hal.executable private @ops_linked_strela {
// CHECK-NEXT:    hal.executable.variant public @strela_bitstream_0 target(#executable_target_strela_bitstream) {
// CHECK:           hal.executable.export public @add_dispatch_0 ordinal(0)
// CHECK:           hal.executable.export public @relu_dispatch_0 ordinal(1)
// CHECK:           builtin.module {
// CHECK-NEXT:        func.func @add_dispatch_0()
// CHECK:             func.func @relu_dispatch_0()
// CHECK:       hal.executable private @sub_dispatch_0 {
// CHECK-NEXT:    hal.executable.variant public @vmvx
// CHECK:       func.func @dispatch_sites()
// CHECK-SAME:    testing.add = @ops_linked_strela::@strela_bitstream_0::@add_dispatch_0
// CHECK-SAME:    testing.relu = @ops_linked_strela::@strela_bitstream_0::@relu_dispatch_0
// CHECK-SAME:    testing.sub = @sub_dispatch_0::@vmvx::@sub_dispatch_0
