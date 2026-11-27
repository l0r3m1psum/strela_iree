// RUN: iree-opt --iree-plugin=estela --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-strela-convert-linalg)))))' \
// RUN:   %s | FileCheck %s

// iree-strela-convert-linalg runs after bufferization, on one job's tile: the
// lowering strategy made the contraction an N x C times C x F (or F x C)
// matmul, so the tiles are already the kernel's operands. The zero fill of the
// output goes, the kernel computes from zero.

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %[[X:[^:]+]]: memref<1x16xi32>, %[[W:[^:]+]]: memref<16x3xi32>, %[[Y:[^:]+]]: memref<1x3xi32>
// CHECK-NEXT:    strela.matvec ins(%[[X]], %[[W]] : memref<1x16xi32>, memref<16x3xi32>) outs(%[[Y]] : memref<1x3xi32>)
// CHECK-NOT:     transpose_rhs
// CHECK-NEXT:    return
hal.executable private @matmul {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @matmul(%x: memref<1x16xi32>, %w: memref<16x3xi32>, %y: memref<1x3xi32>) {
        %c0 = arith.constant 0 : i32
        linalg.fill ins(%c0 : i32) outs(%y : memref<1x3xi32>)
        linalg.matmul ins(%x, %w : memref<1x16xi32>, memref<16x3xi32>) outs(%y : memref<1x3xi32>)
        return
      }
    }
  }
}

// -----

// F x C weights, as TransposeMatmul leaves them, and the lowering strategy
// spells as a linalg.generic.

// CHECK-LABEL: func.func @transposed(
// CHECK-SAME:    %[[X:[^:]+]]: memref<1x16xi32>, %[[W:[^:]+]]: memref<3x16xi32>, %[[Y:[^:]+]]: memref<1x3xi32>
// CHECK-NEXT:    strela.matvec ins(%[[X]], %[[W]] : memref<1x16xi32>, memref<3x16xi32>) outs(%[[Y]] : memref<1x3xi32>) transpose_rhs
// CHECK-NEXT:    return
hal.executable private @transposed {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @transposed(%x: memref<1x16xi32>, %w: memref<3x16xi32>, %y: memref<1x3xi32>) {
        %c0 = arith.constant 0 : i32
        linalg.fill ins(%c0 : i32) outs(%y : memref<1x3xi32>)
        linalg.generic {
          indexing_maps = [affine_map<(n, f, c) -> (n, c)>,
                           affine_map<(n, f, c) -> (f, c)>,
                           affine_map<(n, f, c) -> (n, f)>],
          iterator_types = ["parallel", "parallel", "reduction"]
        } ins(%x, %w : memref<1x16xi32>, memref<3x16xi32>) outs(%y : memref<1x3xi32>) {
        ^bb0(%a: i32, %b: i32, %acc: i32):
          %m = arith.muli %a, %b : i32
          %s = arith.addi %acc, %m : i32
          linalg.yield %s : i32
        }
        return
      }
    }
  }
}

// -----

// On buffers the zero init is the fill right before, on the same memref. Here
// something writes the output in between, and in the second function the fill
// is not zero: the matmul accumulates onto a value the kernel cannot start
// from, so neither is converted.

// CHECK-LABEL: func.func @written_in_between(
// CHECK:         linalg.fill
// CHECK:         memref.store
// CHECK:         linalg.matmul
// CHECK-NOT:     strela.matvec
// CHECK-LABEL: func.func @nonzero_fill(
// CHECK:         linalg.fill
// CHECK:         linalg.matmul
// CHECK-NOT:     strela.matvec
hal.executable private @not_from_zero {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @written_in_between(%x: memref<1x16xi32>, %w: memref<16x3xi32>, %y: memref<1x3xi32>) {
        %c0 = arith.constant 0 : i32
        %c1 = arith.constant 1 : i32
        %i0 = arith.constant 0 : index
        linalg.fill ins(%c0 : i32) outs(%y : memref<1x3xi32>)
        memref.store %c1, %y[%i0, %i0] : memref<1x3xi32>
        linalg.matmul ins(%x, %w : memref<1x16xi32>, memref<16x3xi32>) outs(%y : memref<1x3xi32>)
        return
      }
      func.func @nonzero_fill(%x: memref<1x16xi32>, %w: memref<16x3xi32>, %y: memref<1x3xi32>) {
        %c1 = arith.constant 1 : i32
        linalg.fill ins(%c1 : i32) outs(%y : memref<1x3xi32>)
        linalg.matmul ins(%x, %w : memref<1x16xi32>, memref<16x3xi32>) outs(%y : memref<1x3xi32>)
        return
      }
    }
  }
}
