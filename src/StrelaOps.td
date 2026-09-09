#ifndef STRELA_OPS
#define STRELA_OPS

include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

def Strela_Dialect : Dialect {
  let name = "strela";
  let cppNamespace = "::mlir::strela";
}

class Strela_Op<string mnemonic, list<Trait> traits = []> :
    Op<Strela_Dialect, mnemonic, traits>;

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

#endif // STRELA_OPS
