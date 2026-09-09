module {
  func.func @main(
    %i0: tensor<?x9xi32>,
    %i1: tensor<?x9xi32>,
    %i2: tensor<?x9xi32>,
    %i3: tensor<?x9xi32>
  ) -> tensor<?x9xi32> {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>

    %a0 = "tosa.add"(%i0, %i1) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    %a1 = "tosa.add"(%i2, %i3) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    %r0 = tosa.maximum %a0, %zero : (tensor<?x9xi32>, tensor<1x1xi32>) -> tensor<?x9xi32>
    %r1 = tosa.maximum %a1, %zero : (tensor<?x9xi32>, tensor<1x1xi32>) -> tensor<?x9xi32>
    %a2 = "tosa.add"(%r0, %r1) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    return %a2 : tensor<?x9xi32>
  }
}
