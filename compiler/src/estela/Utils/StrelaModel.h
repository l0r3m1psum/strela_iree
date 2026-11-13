#ifndef ESTELA_UTILS_STRELAMODEL_H_
#define ESTELA_UTILS_STRELAMODEL_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "estela/StrelaIR/StrelaAttrs.h"

namespace mlir::estela {

enum class StrelaOpKind { Unsupported, Elementwise, Contraction };

// What the frontend and the backend need to know about a linalg op to decide
// whether STRELA runs it. Built from structure only, never from the op's name.
struct StrelaOpSummary {
  StrelaOpKind kind = StrelaOpKind::Unsupported;
  SmallVector<OperationName> bodyOps;   // in order, without the yield
  unsigned depth = 0;                   // longest dependency chain in the body
  SmallVector<Value> inputs;            // non-splat tensor inputs, deduplicated
  unsigned numResults = 0;
};

StrelaOpSummary summarize(linalg::LinalgOp op);

bool isLegalOnStrela(
  const StrelaOpSummary &summary, linalg::LinalgOp op, strela::TargetDescAttr desc
);

} // namespace mlir::estela

#endif // ESTELA_UTILS_STRELAMODEL_H_
