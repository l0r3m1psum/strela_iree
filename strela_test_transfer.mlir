module {
  func.func @main(
    %i0: tensor<?x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i1: tensor<?x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i2: tensor<?x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>},
    %i3: tensor<?x9xi32> {iree.abi.affinity = #hal.device.promise<@device_a>}
  ) -> (i32 {iree.abi.affinity = #hal.device.promise<@device_a>}) {
    %zero = "tosa.const"() { values = dense<0> : tensor<1x1xi32> } : () -> tensor<1x1xi32>
    %c0 = arith.constant 0 : index
    %new_shape = tosa.const_shape {values = dense<[-1]> : tensor<1xindex>} : () -> !tosa.shape<1>

    %d2 = tensor.dim %i2, %c0 : tensor<?x9xi32>
    %d3 = tensor.dim %i3, %c0 : tensor<?x9xi32>
    %i2_b = flow.tensor.transfer %i2 : tensor<?x9xi32>{%d2} to #hal.device.promise<@device_b>
    %i3_b = flow.tensor.transfer %i3 : tensor<?x9xi32>{%d3} to #hal.device.promise<@device_b>

    %a0 = "tosa.add"(%i0, %i1) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    %a1_b = "tosa.add"(%i2_b, %i3_b) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>
    %r0 = tosa.maximum %a0, %zero : (tensor<?x9xi32>, tensor<1x1xi32>) -> tensor<?x9xi32>
    %r1_b = tosa.maximum %a1_b, %zero : (tensor<?x9xi32>, tensor<1x1xi32>) -> tensor<?x9xi32>

    %dr = tensor.dim %r1_b, %c0 : tensor<?x9xi32>
    %r1 = flow.tensor.transfer %r1_b : tensor<?x9xi32>{%dr} to #hal.device.promise<@device_a>

    %a2 = "tosa.add"(%r0, %r1) : (tensor<?x9xi32>, tensor<?x9xi32>) -> tensor<?x9xi32>

    %flattened = tosa.reshape %a2, %new_shape : (tensor<?x9xi32>, !tosa.shape<1>) -> tensor<?xi32>
    %min_tensor = tosa.reduce_min %flattened {axis = 0 : i32} : (tensor<?xi32>) -> tensor<1xi32>
    %min_scalar = tensor.extract %min_tensor[%c0] : tensor<1xi32>
    return %min_scalar : i32
  }
}