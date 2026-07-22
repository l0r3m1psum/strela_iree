module {
  func.func @main(
    %input: tensor<?x1x1x9xi8>, // NHWC
    %weight: tensor<9x1x1x9xi8> // FHWC
  ) -> tensor<?x1x1x9xi32> {
    %bias = "tosa.const"() {values = dense<1> : tensor<9xi32>} : () -> tensor<9xi32>
    %input_zp = "tosa.const"() {values = dense<1> : tensor<1xi8>} : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() {values = dense<1> : tensor<1xi8>} : () -> tensor<1xi8>

    %output = tosa.conv2d %input, %weight, %bias, %input_zp, %weight_zp {
      acc_type = i32,
      dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 0>,
      stride = array<i64: 1, 1>
    } : (tensor<?x1x1x9xi8>, tensor<9x1x1x9xi8>, tensor<9xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<?x1x1x9xi32>

    return %output : tensor<?x1x1x9xi32>
  }
}
