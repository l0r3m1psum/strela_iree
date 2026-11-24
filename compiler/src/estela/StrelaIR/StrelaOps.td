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
    init is a zero fill. With `transpose_rhs` the weights are F x C,
    `result[n, f] = sum_c lhs[n, c] * rhs[f, c]`, so each feature's weights
    are contiguous.

    ```mlir
    %0 = strela.matvec %x, %w : tensor<?x16xi32>, tensor<16x8xi32> -> tensor<?x8xi32>
    %1 = strela.matvec %x, %wt transpose_rhs : tensor<?x16xi32>, tensor<8x16xi32> -> tensor<?x8xi32>
    ```
  }];
  let arguments = (ins I32Tensor:$lhs, I32Tensor:$rhs, UnitAttr:$transpose_rhs);
  let results = (outs I32Tensor:$result);
  let assemblyFormat = [{
    $lhs `,` $rhs (`transpose_rhs` $transpose_rhs^)? attr-dict `:` type($lhs) `,`
    type($rhs) `->` type($result)
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// The dispatch interface
//===----------------------------------------------------------------------===//
//
// The STRELA counterparts of the hal.interface.* ops, produced one op at a time
// by ConvertToStrelaABI. Each has the result type of its HAL op, so the
// conversion replaces one value with another and their users don't change.

def PushConstantOp : Strela_Op<"push_constant", [Pure]> {
  let summary = "A push constant, read at dispatch time";
  let description = [{
    The counterpart of `hal.interface.constant.load`, and like it an `i32`: its
    users are the `arith.index_castui` of a 32 bit offset, or the `arith.extui`
    of the two halves of a 64 bit one.

    ```mlir
    %0 = strela.push_constant 1 : i32
    ```
  }];
  let arguments = (ins ConfinedAttr<I32Attr, [IntNonNegative]>:$ordinal);
  let results = (outs I32:$result);
  let assemblyFormat = "$ordinal attr-dict `:` type($result)";
}

def WorkgroupIdOp : Strela_Op<"workgroup_id", [Pure]> {
  let summary = "The id of the workgroup, that is the job, along a dimension";
  let description = [{
    The counterpart of `hal.interface.workgroup.id`. The tile offsets IREE
    computes from it reach the column addresses, where the fold turns each
    dimension into a `min(id * step, limit)` term.

    ```mlir
    %0 = strela.workgroup_id[0] : index
    ```
  }];
  let arguments = (ins
    ConfinedAttr<I32Attr, [IntNonNegative, IntMaxValue<2>]>:$dimension
  );
  let results = (outs Index:$result);
  let assemblyFormat = "`[` $dimension `]` attr-dict `:` type($result)";
}

def BindingOp : Strela_Op<"binding", [Pure, AttrSizedOperandSegments]> {
  let summary = "A tensor in a dispatch binding";
  let description = [{
    The counterpart of `hal.interface.binding.subspan`: the tensor starting
    `byte_offset` bytes into binding `ordinal`. It keeps the subspan's result
    type, so the `dispatch.tensor.load/store`s on it work unchanged until
    ConvertToStrelaABI turns them into columns, whose addresses start from
    `byte_offset`. After that it is dead.

    ```mlir
    %0 = strela.binding 0 offset(%off) : !iree_tensor_ext.dispatch.tensor<readonly:tensor<?x16xi32>>{%n}
    ```
  }];
  let arguments = (ins
    ConfinedAttr<I32Attr, [IntNonNegative]>:$ordinal,
    Optional<Index>:$byte_offset,
    Variadic<Index>:$dynamic_dims
  );
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    $ordinal (`offset` `(` $byte_offset^ `)`)? attr-dict `:` type($result)
    (`{` $dynamic_dims^ `}`)?
  }];
}

//===----------------------------------------------------------------------===//
// Jobs
//===----------------------------------------------------------------------===//

def InpColumnOp : Strela_Op<"inp_column", [Pure]> {
  let summary = "One input DMA column of a job";
  let description = [{
    `words` words, `stride` words apart, read from `address` bytes into binding
    `binding`. The address is the index arithmetic that computes it, from the
    binding's offset, the tile's offsets and the workgroup ids: the serializer
    folds it into an `offset + delta` and one `min(wg * step, limit)` term per
    workgroup dimension (`foldColumnAddress`), and the job verifier checks
    that it folds.

    ```mlir
    %c1 = strela.inp_column binding(0) address(%a) words(16) stride(8)
    ```
  }];
  let arguments = (ins
    ConfinedAttr<I32Attr, [IntNonNegative]>:$binding,
    Index:$address,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$words,
    ConfinedAttr<I32Attr, [IntNonNegative]>:$stride
  );
  let results = (outs Strela_ColumnType:$result);
  let assemblyFormat = [{
    `binding` `(` $binding `)` `address` `(` $address `)` `words` `(` $words `)`
    `stride` `(` $stride `)` attr-dict
  }];
}

def OutColumnOp : Strela_Op<"out_column", [Pure]> {
  let summary = "One output DMA column of a job, or an unused one";
  let description = [{
    `words` consecutive words written to `address` bytes into binding
    `binding`, as for `strela.inp_column`. A column the kernel doesn't use
    has none of them:

    ```mlir
    %o0 = strela.out_column unused
    %o1 = strela.out_column binding(1) address(%a) words(1)
    ```
  }];
  let arguments = (ins
    OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$binding,
    Optional<Index>:$address,
    OptionalAttr<ConfinedAttr<I32Attr, [IntNonNegative]>>:$words
  );
  let results = (outs Strela_ColumnType:$result);
  let assemblyFormat = [{
    (`binding` `(` $binding^ `)` `address` `(` $address `)` `words` `(` $words `)`):(`unused`)?
    attr-dict
  }];
  let extraClassDeclaration = [{
    bool isUnused() { return !getAddress(); }
  }];
  let hasVerifier = 1;
}

def JobOp : Strela_Op<"job", [AttrSizedOperandSegments]> {
  let summary = "One STRELA job: a bitstream and its DMA columns";
  let description = [{
    What the hardware runs for each workgroup, and exactly what the serializer
    writes as an `ExportDef`: the bitstream words, then one input and one output
    column per fabric column, in hardware column order. Not `Pure`: it is the
    only op of a dispatch that writes memory.

    ```mlir
    strela.job bitstream([...]) inputs(%c0, %c1, %c2, %c3) outputs(%o0, %o1, %o2, %o3)
    ```
  }];
  let arguments = (ins
    DenseI32ArrayAttr:$bitstream,
    Variadic<Strela_ColumnType>:$inputs,
    Variadic<Strela_ColumnType>:$outputs
  );
  let assemblyFormat = [{
    `bitstream` `(` $bitstream `)` `inputs` `(` $inputs `)` `outputs` `(` $outputs `)` attr-dict
  }];
  let hasVerifier = 1;
}

#endif // STRELA_DIALECT_STRELA_STRELAOPS
