#ifndef STRELA_DIALECT_STRELA_STRELAATTRS
#define STRELA_DIALECT_STRELA_STRELAATTRS

include "StrelaBase.td"

def Strela_TargetDescAttr : Strela_Attr<"TargetDesc", "target_desc"> {
  let summary = "The STRELA hardware facts the compiler depends on";
  let parameters = (ins
    "unsigned":$rows,
    "unsigned":$columns,
    "unsigned":$max_column_words
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Strela_PushConstantAttr : Strela_Attr<"PushConstant", "push_constant"> {
  let summary = "A value read from a push constant at dispatch time";
  let parameters = (ins "unsigned":$ordinal);
  let assemblyFormat = "`<` $ordinal `>`";
}

// NOTE(Diego): WgDimParams maybe is a better name?
def Strela_WgAxisParamsAttr : Strela_Attr<"WgAxisParams", "wg_axis_params"> {
  let summary = "One workgroup axis parameters of a column address, min(wg_d * step, limit) bytes, where d is x or y";
  let parameters = (ins "uint32_t":$step, "uint32_t":$limit);
  let assemblyFormat = "`<` $step `,` $limit `>`";
}

// The byte address of a column in its binding, as the runtime computes it:
//   offset + delta + min(wg.x * x_step, x_limit) + min(wg.y * y_step, y_limit)
// offset is where the tensor starts in the binding, delta where the column
// starts in the tensor (see ColumnAddressDef in strela_executable_def.fbs).
def Strela_ColAddrAttr : Strela_Attr<"ColAddr", "col_addr"> {
  let parameters = (ins
    "uint32_t":$binding,
    // Known at compile time (an i32 IntegerAttr) or read from a push constant
    // at dispatch time (#strela.push_constant), like the flatbuffer's ValueDef.
    // An attribute constraint can't be a parameter, so verify checks it.
    "::mlir::Attribute":$offset,
    "uint32_t":$delta,
    Strela_WgAxisParamsAttr:$wg_x_params,
    Strela_WgAxisParamsAttr:$wg_y_params
  );
  let assemblyFormat = "`<` struct(params) `>`";
  let genVerifyDecl = 1;
}

def Strela_InpColAttr : Strela_Attr<"InpCol", "inp_col"> {
  let parameters = (ins
    Strela_ColAddrAttr:$address,
    "uint32_t":$words,
    "uint32_t":$stride
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

def Strela_OutColAttr : Strela_Attr<"OutCol", "out_col"> {
  let parameters = (ins
    Strela_ColAddrAttr:$address,
    "uint32_t":$words
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // STRELA_DIALECT_STRELA_STRELAATTRS
