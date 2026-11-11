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

#endif // STRELA_DIALECT_STRELA_STRELAATTRS
