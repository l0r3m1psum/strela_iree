module {
  func.func @main(
    %i0: tensor<?x9xi32>,
    %i1: tensor<?x9xi32>,
    %i2: tensor<?x9xi32>,
    %i3: tensor<?x9xi32>
  ) -> i32 {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %c0 = arith.constant 0 : index
    %new_shape = tosa.const_shape {values = dense<[-1]> : tensor<1xindex>} : () -> !tosa.shape<1>

    %a0 = "tosa.add"(%i0, %i1) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    %a1 = "tosa.add"(%i2, %i3) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    %r0 = tosa.maximum %a0, %zero : (tensor<?x9xi32>, tensor<1x1xi32>) -> tensor<?x9xi32>
    %r1 = tosa.maximum %a1, %zero : (tensor<?x9xi32>, tensor<1x1xi32>) -> tensor<?x9xi32>
    %a2 = "tosa.add"(%r0, %r1) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<?x9xi32>, !tosa.shape<1>) -> tensor<?xi32>
    %min_tensor = tosa.reduce_min %flattened {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}
