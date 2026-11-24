#ifndef STRELA_DIALECT_STRELA_STRELABASE
#define STRELA_DIALECT_STRELA_STRELABASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

def Strela_Dialect : Dialect {
  let name = "strela";
  let cppNamespace = "::mlir::strela";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerAttributes();
    void registerTypes();
  }];
}

class Strela_Op<string mnemonic, list<Trait> traits = []>
  : Op<Strela_Dialect, mnemonic, traits>;

class Strela_Attr<string name, string attrMnemonic, list<Trait> traits = []>
  : AttrDef<Strela_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

class Strela_Type<string name, string typeMnemonic, list<Trait> traits = []>
  : TypeDef<Strela_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Strela_ColumnType : Strela_Type<"Column", "column"> {
  let summary = "One DMA column of a strela.job";
}

#endif // STRELA_DIALECT_STRELA_STRELABASE
