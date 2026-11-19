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

def Strela_WgAxisParamsAttr : Strela_Attr<"WgAxisParams", "wg_axis_params"> {
  let summary = "One workgroup axis parameters of a column address, min(wg_d * step, limit) bytes, where d is x or y";
  let parameters = (ins "uint32_t":$step, "uint32_t":$limit);
  let assemblyFormat = "`<` $step `,` $limit `>`";
}

// The offset in the binding is calculated with the following expression
//   offset = base + min(wg.x*x_step, x_limit) + min(wg.y*y_step, y_limit)
// base is used if multiple buffers are grouped in the same offset; the other
// parameters , as the name implies are needed by workgroups. The offset is
// calculated via the runtime.
def Strela_ColAddrAttr : Strela_Attr<"ColAddr", "col_addr"> {
  let parameters = (ins
    "uint32_t":$binding,
    // Known at compile time (an i32 IntegerAttr) or read from a push constant
    // at dispatch time (#strela.push_constant), like the flatbuffer's ValueDef.
    // An attribute constraint can't be a parameter, so verify checks it.
    "::mlir::Attribute":$offset,
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
