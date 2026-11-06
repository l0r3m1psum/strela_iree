#ifndef ESTELA_TRANSFORMS_PASSES
#define ESTELA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Input and global passes
//===----------------------------------------------------------------------===//

def NormalizeRescalePass
    : Pass<"iree-strela-normalize-rescale", "mlir::func::FuncOp"> {
  let summary = "Normalizes tosa.rescale before input conversion";
  let description = [{
    Rewrites tosa.rescale so the rest of the pipeline sees a shape it can
    handle: splits double-rounding rescales and rewrites the dynamic batch
    dimension.
  }];
}

def FuseConv2DPass : Pass<"iree-strela-fuse-conv2d", "mlir::func::FuncOp"> {
  let summary = "Fuses centering subtractions into a GEMM ukernel";
}

def AssignTopologyPass : Pass<"iree-strela-assign-topology", "mlir::ModuleOp"> {
  let summary = "Assigns the topology of the STRELA device";
}

def TransferWrapPass
    : Pass<"iree-strela-transfer-wrap", "mlir::iree_compiler::IREE::Util::FuncOp"> {
  let summary =
      "Wraps the supported operations in flow.tensor.transfer operations";
  let dependentDialects = ["mlir::iree_compiler::IREE::Flow::FlowDialect"];
}

//===----------------------------------------------------------------------===//
// STRELA executable translation
//===----------------------------------------------------------------------===//

def ConvertLinalgToStrelaPass
    : Pass<"iree-strela-convert-linalg", "mlir::func::FuncOp"> {
  let summary = "Converts linalg ops to strela backend ops";
  let description = [{
    Replaces the linalg ops STRELA can run with strela.add, strela.relu and
    strela.matvec.
  }];
  let dependentDialects = ["mlir::strela::StrelaDialect"];
}

def MaterializeBindingsPass
    : Pass<"iree-strela-materialize-bindings", "mlir::func::FuncOp"> {
  let summary =
      "Replaces the STRELA dispatch interface with strela.binding_load/store";
  let description = [{
    Rewrites the dispatch interface into strela.binding_load and
    strela.binding_store, deleting the hal.interface.* ops and the arithmetic
    that reassembled the byte offsets from the push constants.
  }];
  let dependentDialects = ["mlir::strela::StrelaDialect"];
}

def ResolveWorkgroupCountPass
    : Pass<"iree-strela-resolve-workgroup-count",
           "mlir::iree_compiler::IREE::HAL::ExecutableVariantOp"> {
  let summary = "Replaces STRELA export workgroup count regions with the kernel's grid";
  let description = [{
    Replaces the iree_tensor_ext.dispatch.workgroup_count_from_slice
    placeholder of every export with the grid STRELA runs the kernel on:
    (1, 1, 1) for elementwise kernels, (ceildiv(F, 3), N, 1) for a matvec.
    Must run before MaterializeBindings, which erases the workload ordinals
    that say which count region argument is N.
  }];
}

//===----------------------------------------------------------------------===//
// STRELA executable linking
//===----------------------------------------------------------------------===//

def LinkExecutablesPass
    : Pass<"iree-strela-link-executables", "mlir::ModuleOp"> {
  let summary = "Links all STRELA executables into one with several exports";
  let description = [{
    Moves the exports of every STRELA executable into a single
    `<module>_linked_strela` executable, renumbering their ordinals and
    rewriting the dispatch sites, so the runtime loads one executable instead
    of one per dispatch. Lazily loaded executables are left alone.
  }];
}

#endif // ESTELA_TRANSFORMS_PASSES
