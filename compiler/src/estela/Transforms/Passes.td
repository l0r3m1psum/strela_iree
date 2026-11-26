#ifndef ESTELA_TRANSFORMS_PASSES
#define ESTELA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Input and global passes
//===----------------------------------------------------------------------===//

def AssignTopologyPass : Pass<"iree-strela-assign-topology", "mlir::ModuleOp"> {
  let summary = "Assigns the topology of the STRELA device";
}

def StrelaWidenOperandsPass
    : Pass<"iree-strela-widen-operands", "mlir::iree_compiler::IREE::Util::FuncOp"> {
  let summary = "Extends the narrow operands of the matmuls STRELA will run to i32";
  let description = [{
    STRELA moves i32 words. Each linalg.matmul with integer inputs narrower
    than i32 that TransferWrap will pin once they are widened gets them
    through an elementwise arith.extsi to i32, and becomes a matmul of i32.
    When the input comes from an elementwise linalg.generic used only by the
    matmul (the rescale of the layer before), the extsi is fused into it with
    linalg::fuseElementwiseOps: the narrow value only exists inside its body,
    and its dispatch writes i32 directly. (Dispatch formation never fuses a
    truncation with an extension.) The extsi stays visible, so the
    operand-width check can still prove the range. Constant weights are
    widened by const-eval. Fails without a STRELA device.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

def TransferWrapPass
    : Pass<"iree-strela-transfer-wrap", "mlir::iree_compiler::IREE::Util::FuncOp"> {
  let summary =
      "Wraps the supported operations in flow.tensor.transfer operations";
  let dependentDialects = ["mlir::iree_compiler::IREE::Flow::FlowDialect"];
}

//===----------------------------------------------------------------------===//
// STRELA executable configuration
//===----------------------------------------------------------------------===//

def StrelaSelectLoweringStrategyPass
    : Pass<"iree-strela-select-lowering-strategy", "mlir::func::FuncOp"> {
  let summary = "Picks how a STRELA dispatch is tiled into matvec jobs";
  let description = [{
    Rewrites the contraction STRELA can run into the N x C times C x F
    linalg.matmul of the matvec kernel, between reshapes of unit dims that
    FoldReshapeIntoInterfaceTensor folds into the bindings, and puts on it a
    lowering_config tiling N by 1 and F by 3 (C untiled), the shape of one job,
    with a no_pipeline translation_info on the function. IREE's workgroup
    tiling then makes the (ceildiv(F, 3), N, 1) grid of jobs.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::iree_compiler::IREE::Codegen::IREECodegenDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
  ];
}

//===----------------------------------------------------------------------===//
// STRELA executable translation
//===----------------------------------------------------------------------===//

def StrelaStoreWorkgroupTilesPass
    : Pass<"iree-strela-store-workgroup-tiles", "mlir::func::FuncOp"> {
  let summary = "Stores and loads the tiles of the workgroup forall one by one";
  let description = [{
    After TileAndDistributeToWorkgroupsUsingForallOp the workgroup scf.forall
    returns the tiles and the dispatch stores them whole. Each tile is stored
    from inside the forall instead (a dispatch.tensor.store where its
    tensor.parallel_insert_slice wrote), so that ReconcileTranslationInfo,
    which resolves only foralls without results, can turn the forall into
    workgroup ids while the dispatch is still on tensors. Likewise every tile
    cut from a whole-tensor load becomes a load of just the tile.
  }];
  let dependentDialects = [
    "mlir::iree_compiler::IREE::TensorExt::IREETensorExtDialect",
    "mlir::scf::SCFDialect",
  ];
}

def StrelaOverlapLastTilePass
    : Pass<"iree-strela-overlap-last-tile", "mlir::func::FuncOp"> {
  let summary = "Shifts the smaller last tile back so every job is full";
  let description = [{
    IREE tiles F by 3 with offset 3·id and size min(3, F - 3·id); the matvec
    bitstream only runs whole tiles. Every dispatch.tensor.load/store whose
    size along a dimension is that clamp gets offset min(3·id, F - 3) and size
    3 instead: when 3 does not divide F the last job overlaps the one before
    it, rewriting the same values. The clamp is recognized by its arithmetic
    (size + offset = F), not by its spelling. Runs after
    ReconcileTranslationInfo, on plain stores.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
  ];
}

def ConvertLinalgToStrelaPass
    : Pass<"iree-strela-convert-linalg", "mlir::func::FuncOp"> {
  let summary = "Converts linalg ops to strela backend ops";
  let description = [{
    Replaces the linalg ops STRELA can run with strela.matvec.
  }];
  let dependentDialects = ["mlir::strela::StrelaDialect"];
}

def ConvertToStrelaABIPass
    : Pass<"iree-strela-convert-to-strela-abi", "mlir::func::FuncOp"> {
  let summary =
      "Replaces the STRELA dispatch interface and kernel with a strela.job";
  let description = [{
    Converts the hal.interface.* ops of a dispatch to their strela.*
    counterparts one at a time (the HAL dialect is illegal afterwards), then
    replaces the strela.matvec and the loads and store around it with one
    strela.job: the matvec bitstream and its DMA columns, placed by the kernel's
    column table. Each column address is SSA index arithmetic, an affine.apply
    of the binding's byte offset and the slice offsets; the job verifier and the
    serializer fold it. The workgroup terms of the grid
    ResolveWorkgroupCountPass set are still built by hand, as wg_x / wg_y on
    the columns.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect",
    "mlir::arith::ArithDialect",
    "mlir::strela::StrelaDialect",
  ];
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
