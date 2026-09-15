// Conversions into the strela dialect.

#ifndef ESTELA_DIALECT_STRELA_CONVERSION_PASSES_H_
#define ESTELA_DIALECT_STRELA_CONVERSION_PASSES_H_

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::estela {

// Replaces the linalg.generic ops STRELA can run with strela.add / strela.relu.
// Runs on a func.
std::unique_ptr<Pass> createConvertLinalgToStrelaPass();

} // namespace mlir::estela

#endif // ESTELA_DIALECT_STRELA_CONVERSION_PASSES_H_
