// RUN: iree-opt --iree-plugin=estela --split-input-file \
// RUN:   --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(builtin.module(func.func(iree-strela-convert-linalg)))))' \
// RUN:   %s | FileCheck %s

// iree-strela-convert-linalg rewrites every form of the matvec into the rank 2
// strela.matvec, N x C times C x F: unit batch dims are collapsed away, an
// empty M becomes a unit row, and the result is reshaped back to the op's own
// type. The zero fill is dropped, the kernel computes from zero.

// CHECK-LABEL: func.func @batch_matmul(
// CHECK-SAME:    %[[X:[^:]+]]: tensor<1x?x16xi32>, %[[W:[^:]+]]: tensor<1x16x8xi32>
// CHECK:         %[[X2:.+]] = tensor.collapse_shape %[[X]] {{\[}}[0, 1], [2]] : tensor<1x?x16xi32> into tensor<?x16xi32>
// CHECK:         %[[W2:.+]] = tensor.collapse_shape %[[W]] {{\[}}[0, 1], [2]] : tensor<1x16x8xi32> into tensor<16x8xi32>
// CHECK:         %[[Y2:.+]] = strela.matvec %[[X2]], %[[W2]] : tensor<?x16xi32>, tensor<16x8xi32> -> tensor<?x8xi32>
// CHECK:         %[[Y:.+]] = tensor.expand_shape %[[Y2]] {{\[}}[0, 1], [2]] {{.+}} : tensor<?x8xi32> into tensor<1x?x8xi32>
// CHECK-NOT:     linalg.
// CHECK:         return %[[Y]]
hal.executable private @batch_matmul {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @batch_matmul(%x: tensor<1x?x16xi32>, %w: tensor<1x16x8xi32>, %n: index) -> tensor<1x?x8xi32> {
        %c0 = arith.constant 0 : i32
        %e = tensor.empty(%n) : tensor<1x?x8xi32>
        %i = linalg.fill ins(%c0 : i32) outs(%e : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
        %r = linalg.batch_matmul ins(%x, %w : tensor<1x?x16xi32>, tensor<1x16x8xi32>) outs(%i : tensor<1x?x8xi32>) -> tensor<1x?x8xi32>
        return %r : tensor<1x?x8xi32>
      }
    }
  }
}

// -----

// Already N x C times C x F: no reshapes.

// CHECK-LABEL: func.func @matmul(
// CHECK-SAME:    %[[X:[^:]+]]: tensor<?x16xi32>, %[[W:[^:]+]]: tensor<16x8xi32>
// CHECK-NEXT:    %[[Y:.+]] = strela.matvec %[[X]], %[[W]] : tensor<?x16xi32>, tensor<16x8xi32> -> tensor<?x8xi32>
// CHECK-NEXT:    return %[[Y]]
hal.executable private @matmul {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @matmul(%x: tensor<?x16xi32>, %w: tensor<16x8xi32>, %n: index) -> tensor<?x8xi32> {
        %c0 = arith.constant 0 : i32
        %e = tensor.empty(%n) : tensor<?x8xi32>
        %i = linalg.fill ins(%c0 : i32) outs(%e : tensor<?x8xi32>) -> tensor<?x8xi32>
        %r = linalg.matmul ins(%x, %w : tensor<?x16xi32>, tensor<16x8xi32>) outs(%i : tensor<?x8xi32>) -> tensor<?x8xi32>
        return %r : tensor<?x8xi32>
      }
    }
  }
}

// -----

// x[C] . W[C, F], the form AD takes at batch 1: M is empty, so x becomes a
// single row and the 1 x F result is collapsed back to F.

// CHECK-LABEL: func.func @vecmat(
// CHECK-SAME:    %[[X:[^:]+]]: tensor<16xi32>, %[[W:[^:]+]]: tensor<16x8xi32>
// CHECK:         %[[X2:.+]] = tensor.expand_shape %[[X]] {{\[}}[0, 1]] {{.+}} : tensor<16xi32> into tensor<1x16xi32>
// CHECK:         %[[Y2:.+]] = strela.matvec %[[X2]], %[[W]] : tensor<1x16xi32>, tensor<16x8xi32> -> tensor<1x8xi32>
// CHECK:         %[[Y:.+]] = tensor.collapse_shape %[[Y2]] {{\[}}[0, 1]] : tensor<1x8xi32> into tensor<8xi32>
// CHECK:         return %[[Y]]
hal.executable private @vecmat {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @vecmat(%x: tensor<16xi32>, %w: tensor<16x8xi32>) -> tensor<8xi32> {
        %c0 = arith.constant 0 : i32
        %e = tensor.empty() : tensor<8xi32>
        %i = linalg.fill ins(%c0 : i32) outs(%e : tensor<8xi32>) -> tensor<8xi32>
        %r = linalg.vecmat ins(%x, %w : tensor<16xi32>, tensor<16x8xi32>) outs(%i : tensor<8xi32>) -> tensor<8xi32>
        return %r : tensor<8xi32>
      }
    }
  }
}

// -----

// The unit batch dim after C in x and after F in the result: x is 16x1 where
// strela.matvec wants 1x16, the same rank with the unit dim elsewhere. No
// single reshape does that, so x goes through the shape without unit dims,
// and so does the result on the way back.

// CHECK-LABEL: func.func @trailing_batch(
// CHECK-SAME:    %[[X:[^:]+]]: tensor<16x1xi32>, %[[W:[^:]+]]: tensor<1x16x8xi32>
// CHECK:         %[[X1:.+]] = tensor.collapse_shape %[[X]] {{\[}}[0, 1]] : tensor<16x1xi32> into tensor<16xi32>
// CHECK:         %[[X2:.+]] = tensor.expand_shape %[[X1]] {{\[}}[0, 1]] {{.+}} : tensor<16xi32> into tensor<1x16xi32>
// CHECK:         %[[W2:.+]] = tensor.collapse_shape %[[W]] {{\[}}[0, 1], [2]] : tensor<1x16x8xi32> into tensor<16x8xi32>
// CHECK:         %[[Y2:.+]] = strela.matvec %[[X2]], %[[W2]] : tensor<1x16xi32>, tensor<16x8xi32> -> tensor<1x8xi32>
// CHECK:         %[[Y1:.+]] = tensor.collapse_shape %[[Y2]] {{\[}}[0, 1]] : tensor<1x8xi32> into tensor<8xi32>
// CHECK:         %[[Y:.+]] = tensor.expand_shape %[[Y1]] {{\[}}[0, 1]] {{.+}} : tensor<8xi32> into tensor<8x1xi32>
// CHECK:         return %[[Y]]
hal.executable private @trailing_batch {
  hal.executable.variant public @strela_bitstream target(<"strela", "strela_bitstream", {strela.target_desc = #strela.target_desc<rows = 4, columns = 4, max_column_words = 16383>}>) {
    builtin.module {
      func.func @trailing_batch(%x: tensor<16x1xi32>, %w: tensor<1x16x8xi32>) -> tensor<8x1xi32> {
        %c0 = arith.constant 0 : i32
        %e = tensor.empty() : tensor<8x1xi32>
        %i = linalg.fill ins(%c0 : i32) outs(%e : tensor<8x1xi32>) -> tensor<8x1xi32>
        %r = linalg.generic {
          indexing_maps = [affine_map<(b, f, c) -> (c, b)>,
                           affine_map<(b, f, c) -> (b, c, f)>,
                           affine_map<(b, f, c) -> (f, b)>],
          iterator_types = ["parallel", "parallel", "reduction"]
        } ins(%x, %w : tensor<16x1xi32>, tensor<1x16x8xi32>) outs(%i : tensor<8x1xi32>) {
        ^bb0(%a: i32, %b: i32, %acc: i32):
          %m = arith.muli %a, %b : i32
          %s = arith.addi %acc, %m : i32
          linalg.yield %s : i32
        } -> tensor<8x1xi32>
        return %r : tensor<8x1xi32>
      }
    }
  }
}
