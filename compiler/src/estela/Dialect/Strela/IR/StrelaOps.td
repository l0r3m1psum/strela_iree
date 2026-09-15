#ifndef STRELA_DIALECT_STRELA_STRELAOPS
#define STRELA_DIALECT_STRELA_STRELAOPS

include "StrelaBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"  // SameOperandsAndResultType
include "mlir/Interfaces/SideEffectInterfaces.td"  // Pure

def AddOp : Strela_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "element-wise addition operation";
  let arguments = (ins I32Tensor:$lhs, I32Tensor:$rhs);
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs)
  }];
}

def ReluOp : Strela_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Rectified Linear Unit (ReLU) operation";
  let arguments = (ins I32Tensor:$input);
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

// Common documentation for the two ops below.
//
// `binding` indexes the dispatch binding list, i.e. the `hal.pipeline.layout`
// binding ordinal. `offsets` is absent when the subspan starts at byte zero and
// otherwise holds the two push constant ordinals carrying the low and high
// halves of the 64 bit byte offset; those two are not necessarily adjacent
// because IREE dedupes the constant loads.
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
    %0 = strela.binding_load binding(0) offsets [0, 1] : tensor<?x9xi32>
    ```
  }];
  let arguments = (ins
    I64Attr:$binding,
    OptionalAttr<DenseI64ArrayAttr>:$offsets
  );
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    `binding` `(` $binding `)` (`offsets` $offsets^)? attr-dict `:` type($result)
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
    strela.binding_store %0, binding(1) offsets [0, 2] : tensor<?x9xi32>
    ```
  }];
  let arguments = (ins
    I32Tensor:$value,
    I64Attr:$binding,
    OptionalAttr<DenseI64ArrayAttr>:$offsets
  );
  let assemblyFormat = [{
    $value `,` `binding` `(` $binding `)` (`offsets` $offsets^)?
    attr-dict `:` type($value)
  }];
  let hasVerifier = 1;
}

#endif // STRELA_DIALECT_STRELA_STRELAOPS
