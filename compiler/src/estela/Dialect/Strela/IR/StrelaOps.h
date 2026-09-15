#ifndef ESTELA_DIALECT_STRELA_IR_STRELAOPS_H_
#define ESTELA_DIALECT_STRELA_IR_STRELAOPS_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "estela/Dialect/Strela/IR/StrelaDialect.h"

// Generated op declarations. The matching definitions live in StrelaOps.cpp so
// they are compiled exactly once rather than in every translation unit that
// wants to name a strela op.
#define GET_OP_CLASSES
#include "StrelaOps.h.inc"

#endif // ESTELA_DIALECT_STRELA_IR_STRELAOPS_H_
