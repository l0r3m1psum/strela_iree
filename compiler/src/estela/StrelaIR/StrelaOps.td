#ifndef STRELA_DIALECT_STRELA_STRELAOPS
#define STRELA_DIALECT_STRELA_STRELAOPS

include "StrelaBase.td"
include "StrelaAttrs.td"

include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure

def MatvecOp : Strela_Op<"matvec", [Pure]> {
  let summary = "Batched matrix multiplication with a zero accumulator";
  let description = [{
    `result[0, n, f] = sum_c lhs[0, n, c] * rhs[0, c, f]`, i.e. a
    `linalg.batch_matmul` with a batch of one whose init is a zero fill. The
    kernel has no accumulator input, so a bias has to be added afterwards.

    ```mlir
    %0 = strela.matvec %x, %w : tensor<1x?x128xi32>, tensor<1x128x128xi32> -> tensor<1x?x128xi32>
    ```
  }];
  let arguments = (ins I32Tensor:$lhs, I32Tensor:$rhs);
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

// Common documentation for the two ops below.
//
// `binding` indexes the dispatch binding list, i.e. the `hal.pipeline.layout`
// binding ordinal. The byte offset of the subspan within that binding takes one
// of three forms, and at most one of the two optional attributes is ever set:
//
//   * neither attribute: the subspan starts at byte zero.
//   * `byte_offset`: a compile-time constant offset. Stream's
//     FuseDispatchBindings produces these when it merges several operands into
//     one buffer and every dispatch site of the executable agrees on where each
//     one sits, so the offset folds to an `arith.constant`.
//   * `offset_ordinal`: the push constant holding an offset that differs per
//     dispatch site. One 32 bit constant, which is what the executable format
//     can describe.
class Strela_BindingOp<string mnemonic, list<Trait> traits = []> :
    Strela_Op<mnemonic, traits>;

def BindingLoadOp : Strela_BindingOp<"binding_load", [Pure]> {
  let summary = "Reads a STRELA kernel operand from a dispatch binding";
  let description = [{
    Replaces the `hal.interface.binding.subspan` plus
    `iree_tensor_ext.dispatch.tensor.load` pair that fed a STRELA kernel, along
    with the push constant loads and the arithmetic that reassembled the byte
    offset from them. `iree-strela-materialize-bindings` resolves all of that
    once and leaves this op in its place, so the dispatch interface is stated
    directly instead of having to be recovered from the surrounding IR.

    ```mlir
    %0 = strela.binding_load binding(0) offset_ordinal 1 : tensor<?x9xi32>
    %1 = strela.binding_load binding(0) byte_offset 320 : tensor<72xi32>
    ```
  }];
  let arguments = (ins
    I64Attr:$binding,
    OptionalAttr<I64Attr>:$offset_ordinal,
    OptionalAttr<I64Attr>:$byte_offset
  );
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    `binding` `(` $binding `)` (`offset_ordinal` $offset_ordinal^)?
    (`byte_offset` $byte_offset^)? attr-dict `:` type($result)
  }];
  let hasVerifier = 1;
}

def BindingStoreOp : Strela_BindingOp<"binding_store"> {
  let summary = "Writes a STRELA kernel result to a dispatch binding";
  let description = [{
    The counterpart of `strela.binding_load`, replacing the
    `iree_tensor_ext.dispatch.tensor.store` that consumed a STRELA kernel result.
    Not `Pure`: it is the only writing op left in the dispatch body and must not
    be folded away.

    ```mlir
    strela.binding_store %0, binding(1) offset_ordinal 2 : tensor<?x9xi32>
    strela.binding_store %1, binding(1) byte_offset 320 : tensor<72xi32>
    ```
  }];
  let arguments = (ins
    I32Tensor:$value,
    I64Attr:$binding,
    OptionalAttr<I64Attr>:$offset_ordinal,
    OptionalAttr<I64Attr>:$byte_offset
  );
  let assemblyFormat = [{
    $value `,` `binding` `(` $binding `)` (`offset_ordinal` $offset_ordinal^)?
    (`byte_offset` $byte_offset^)? attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

#endif // STRELA_DIALECT_STRELA_STRELAOPS
