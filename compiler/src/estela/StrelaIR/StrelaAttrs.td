#ifndef STRELA_DIALECT_STRELA_STRELAATTRS
#define STRELA_DIALECT_STRELA_STRELAATTRS

include "StrelaBase.td"

def Strela_TargetDescAttr : Strela_Attr<"TargetDesc", "target_desc"> {
  let summary = "The STRELA hardware facts the compiler depends on";
  let parameters = (ins
    "unsigned":$rows,
    "unsigned":$columns,
    "unsigned":$max_column_words,
    // The signed bits of each operand the multipliers use (multiplier.sv keeps
    // the low DATA_WIDTH / 2). Elided when it is the default.
    DefaultValuedParameter<"unsigned", "16">:$multiplier_width
  );
  let assemblyFormat = "`<` struct(params) `>`";
}

#endif // STRELA_DIALECT_STRELA_STRELAATTRS
