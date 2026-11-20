#ifndef STRELA_DIALECT_STRELA_STRELAOPS
#define STRELA_DIALECT_STRELA_STRELAOPS

include "StrelaBase.td"
include "StrelaAttrs.td"

include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure

def MatvecOp : Strela_Op<"matvec", [Pure]> {
  let summary = "Matrix multiplication with a zero accumulator";
  let description = [{
    `result[n, f] = sum_c lhs[n, c] * rhs[c, f]`, i.e. a `linalg.matmul` whose
    init is a zero fill.

    ```mlir
    %0 = strela.matvec %x, %w : tensor<?x16xi32>, tensor<16x8xi32> -> tensor<?x8xi32>
    ```
  }];
  let arguments = (ins I32Tensor:$lhs, I32Tensor:$rhs);
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

def JobOp : Strela_Op<"job"> {
  let summary = "One STRELA job: a bitstream and its DMA columns";
  let description = [{
    What the hardware runs for each workgroup, and exactly what the serializer
    writes as an `ExportDef`: the bitstream words, then one input and one output
    column per fabric column, in hardware column order. Not `Pure`: it is the
    only op of a dispatch that writes memory.
  }];
  let arguments = (ins
    DenseI32ArrayAttr:$bitstream,
    TypedArrayAttrBase<Strela_InpColAttr, "input columns">:$inputs,
    TypedArrayAttrBase<Strela_OutColAttr, "output columns">:$outputs
  );
  let assemblyFormat = [{
    `bitstream` `(` $bitstream `)` `inputs` `(` $inputs `)` `outputs` `(` $outputs `)` attr-dict
  }];
  let hasVerifier = 1;
}

#endif // STRELA_DIALECT_STRELA_STRELAOPS
