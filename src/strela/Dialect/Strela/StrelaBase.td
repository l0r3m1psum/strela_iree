#ifndef STRELA_DIALECT_STRELA_STRELABASE
#define STRELA_DIALECT_STRELA_STRELABASE

include "mlir/IR/OpBase.td"

def Strela_Dialect : Dialect {
  let name = "strela";
  let cppNamespace = "::mlir::strela";
}

class Strela_Op<string mnemonic, list<Trait> traits = []> :
    Op<Strela_Dialect, mnemonic, traits>;

#endif // STRELA_DIALECT_STRELA_STRELABASE
