#ifndef ESTELA_TRANSFORMS_PASSES
#define ESTELA_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Input and global passes
//===----------------------------------------------------------------------===//

def AssignTopologyPass : Pass<"iree-strela-assign-topology", "mlir::ModuleOp"> {
  let summary = "Assigns the topology of the STRELA device";
}

def StrelaPrepareWeightsPass
    : Pass<"iree-strela-prepare-weights", "mlir::iree_compiler::IREE::Util::FuncOp"> {
  let summary = "Folds zero points and transposes the constant weights of STRELA matvecs";
  let description = [{
    First, a quantized 1x1 convolution with constant weights and zero points
    (a TFLite fully connected layer) becomes an i32 matmul of the widened
    activations and the weights minus their zero point, from a zero fill,
    plus an elementwise add of its init and of the constant
    -zx * sum_c w'[c, f] on the host: what the matvec kernel runs, with the
    zero points folded into constants. With fold-bias and a constant bias,
    the bias goes into the matmul instead, as extra reduction elements: the
    activations x - zx get trailing ones and the weights trailing rows holding
    the bias, split into parts that fit the multipliers. No host add is left.

    Then, for each contraction TransferWrap will pin to STRELA whose weights
    are a constant C x F tensor (TOSA's layout), transposes them to F x C with
    a linalg.transpose that const-eval folds, and rewrites the contraction into
    a linalg.generic reading them that way. Each feature's C weights are then
    contiguous, so a weight column spans C words instead of C * F.
  }];
  let options = [
    Option<"foldBias", "fold-bias", "bool", /*default=*/"false",
           "Fold the constant bias of a quantized fully connected layer into "
           "its matmul as extra reduction elements, instead of adding it on "
           "the host.">
  ];
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
