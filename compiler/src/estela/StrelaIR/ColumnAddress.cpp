#include "estela/StrelaIR/ColumnAddress.h"

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"

#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/AffineExpr.h"

#include "estela/StrelaIR/StrelaOps.h"

namespace mlir::strela {

using namespace mlir::iree_compiler;

namespace {

// The workgroup dimensions an id can come from; the schema has terms for the
// first two.
constexpr unsigned numWorkgroupDims = 3;

// Builds one AffineExpr from the SSA chain computing an address. Dim d is
// strela.workgroup_id[d]. Symbols are push constants, and affine.min results:
// a min can't be part of an AffineExpr, so it stands in as a symbol and its
// results are classified apart. The leaves are told apart by their defining
// op, not by their position in an affine.apply, which passes workgroup ids as
// symbols too.
class AddressExprBuilder {
public:
  struct Symbol {
    std::optional<uint32_t> pushConstant;
    // For a min: its results, built by this builder.
    SmallVector<AffineExpr> minResults;
    Value minValue;
  };

  AddressExprBuilder(
    MLIRContext *context, function_ref<InFlightDiagnostic()> emitError
  ) : context(context), emitError(emitError) {}

  FailureOr<AffineExpr> build(Value value);

  ArrayRef<Symbol> getSymbols() const { return symbols; }

private:
  AffineExpr pushConstantSymbol(uint32_t ordinal) {
    for (auto [i, symbol] : llvm::enumerate(symbols)) {
      if (symbol.pushConstant == ordinal) return getAffineSymbolExpr(i, context);
    }
    symbols.push_back({ordinal, {}, Value()});
    return getAffineSymbolExpr(symbols.size() - 1, context);
  }

  // The symbol standing in for the min `value` computes, of the results
  // `buildResults` builds.
  FailureOr<AffineExpr>
  minSymbol(Value value,
            function_ref<FailureOr<SmallVector<AffineExpr>>()> buildResults);

  // a | b equals a + b when `lo` has no bit at or above where `hi` starts:
  // `lo` is zero-extended from N bits and `hi` is shifted left by at least N.
  static bool disjointBits(Value lo, Value hi) {
    auto shlOp = hi.getDefiningOp<arith::ShLIOp>();
    auto extOp = lo.getDefiningOp<arith::ExtUIOp>();
    if (!shlOp || !extOp) return false;
    std::optional<int64_t> shift = getConstantIntValue(shlOp.getRhs());
    return shift && extOp.getIn().getType().getIntOrFloatBitWidth() <= *shift;
  }

  // The results of `map` applied to `operands`.
  FailureOr<SmallVector<AffineExpr>> applyMap(AffineMap map, ValueRange operands);

  MLIRContext *context;
  function_ref<InFlightDiagnostic()> emitError;
  SmallVector<Symbol> symbols;
};

FailureOr<SmallVector<AffineExpr>>
AddressExprBuilder::applyMap(AffineMap map, ValueRange operands) {
  SmallVector<AffineExpr> exprs;
  for (Value operand : operands) {
    FailureOr<AffineExpr> expr = build(operand);
    if (failed(expr)) return failure();
    exprs.push_back(*expr);
  }
  ArrayRef<AffineExpr> all(exprs);
  SmallVector<AffineExpr> results;
  for (AffineExpr result : map.getResults()) {
    results.push_back(result.replaceDimsAndSymbols(
      all.take_front(map.getNumDims()), all.drop_front(map.getNumDims())
    ));
  }
  return results;
}

FailureOr<AffineExpr>
AddressExprBuilder::minSymbol(
  Value value, function_ref<FailureOr<SmallVector<AffineExpr>>()> buildResults
) {
  for (auto [i, symbol] : llvm::enumerate(symbols)) {
    if (symbol.minValue == value) return getAffineSymbolExpr(i, context);
  }
  FailureOr<SmallVector<AffineExpr>> results = buildResults();
  if (failed(results)) return failure();
  symbols.push_back({std::nullopt, *results, value});
  return getAffineSymbolExpr(symbols.size() - 1, context);
}

FailureOr<AffineExpr>
AddressExprBuilder::build(Value value) {
  if (std::optional<int64_t> constant = getConstantIntValue(value)) {
    return getAffineConstantExpr(*constant, context);
  }
  Operation *op = value.getDefiningOp();
  if (!op) {
    return emitError() << "column address depends on a block argument";
  }

  if (auto pushConstantOp = dyn_cast<PushConstantOp>(op)) {
    return pushConstantSymbol(pushConstantOp.getOrdinal());
  }
  if (auto workgroupIdOp = dyn_cast<WorkgroupIdOp>(op)) {
    return getAffineDimExpr(workgroupIdOp.getDimension(), context);
  }
  // Offsets are unsigned: zero extension doesn't change the value.
  if (isa<arith::IndexCastUIOp, arith::ExtUIOp>(op)) {
    return build(op->getOperand(0));
  }
  if (auto assumeOp = dyn_cast<IREE::Util::AssumeIntOp>(op)) {
    return build(assumeOp->getOperand(cast<OpResult>(value).getResultNumber()));
  }

  if (isa<arith::AddIOp, arith::MulIOp, arith::ShLIOp, arith::OrIOp>(op)) {
    FailureOr<AffineExpr> lhs = build(op->getOperand(0));
    if (failed(lhs)) return failure();
    if (auto shlOp = dyn_cast<arith::ShLIOp>(op)) {
      std::optional<int64_t> shift = getConstantIntValue(shlOp.getRhs());
      if (!shift || *shift < 0 || *shift > 62) {
        return emitError() << "column address shifts by something other than "
                              "a constant below 63";
      }
      return *lhs * (int64_t(1) << *shift);
    }
    FailureOr<AffineExpr> rhs = build(op->getOperand(1));
    if (failed(rhs)) return failure();
    if (isa<arith::AddIOp>(op)) return *lhs + *rhs;
    if (isa<arith::MulIOp>(op)) {
      if (!isa<AffineConstantExpr>(*lhs) && !isa<AffineConstantExpr>(*rhs)) {
        return emitError() << "column address multiplies two values known "
                              "only at dispatch time";
      }
      return *lhs * *rhs;
    }
    // arith.ori
    auto isZero = [](AffineExpr e) {
      auto c = dyn_cast<AffineConstantExpr>(e);
      return c && c.getValue() == 0;
    };
    if (isZero(*lhs)) return *rhs;
    if (isZero(*rhs)) return *lhs;
    Value a = op->getOperand(0), b = op->getOperand(1);
    if (disjointBits(a, b) || disjointBits(b, a)) return *lhs + *rhs;
    return emitError() << "column address ORs two values whose bits may "
                          "overlap";
  }

  if (auto applyOp = dyn_cast<affine::AffineApplyOp>(op)) {
    FailureOr<SmallVector<AffineExpr>> results =
      applyMap(applyOp.getAffineMap(), applyOp.getMapOperands());
    if (failed(results)) return failure();
    return results->front();
  }
  if (auto minOp = dyn_cast<affine::AffineMinOp>(op)) {
    return minSymbol(value, [&] {
      return applyMap(minOp.getAffineMap(), minOp.getOperands());
    });
  }
  // What affine.min becomes once the affine ops are lowered, which happens to
  // executables between translation and serialization.
  if (isa<arith::MinSIOp, arith::MinUIOp>(op)) {
    return minSymbol(value, [&]() -> FailureOr<SmallVector<AffineExpr>> {
      FailureOr<AffineExpr> a = build(op->getOperand(0));
      if (failed(a)) return failure();
      FailureOr<AffineExpr> b = build(op->getOperand(1));
      if (failed(b)) return failure();
      return SmallVector<AffineExpr>{*a, *b};
    });
  }

  return emitError() << "column address goes through '" << op->getName()
                     << "', which it cannot be folded through";
}

// [coefficient of each dim, of each symbol, constant], or failure when `expr`
// is not linear (products of non-constants, floordiv, mod).
static FailureOr<SmallVector<int64_t>>
flatten(AffineExpr expr, unsigned numSymbols) {
  SmallVector<int64_t> flat;
  if (failed(getFlattenedAffineExpr(expr, numWorkgroupDims, numSymbols, &flat)) ||
      flat.size() != numWorkgroupDims + numSymbols + 1) {
    return failure();
  }
  return flat;
}

} // namespace

FailureOr<ColumnAddress>
foldColumnAddress(Value address, function_ref<InFlightDiagnostic()> emitError) {
  AddressExprBuilder builder(address.getContext(), emitError);
  FailureOr<AffineExpr> built = builder.build(address);
  if (failed(built)) return failure();

  ArrayRef<AddressExprBuilder::Symbol> symbols = builder.getSymbols();
  unsigned numSymbols = symbols.size();
  AffineExpr expr = simplifyAffineExpr(*built, numWorkgroupDims, numSymbols);
  std::string printed;
  llvm::raw_string_ostream(printed) << expr;
  auto describe = [&](InFlightDiagnostic diag) -> InFlightDiagnostic {
    diag << "column address " << printed;
    for (auto [i, symbol] : llvm::enumerate(symbols)) {
      diag << (i ? ", " : " (") << "s" << i << " = ";
      if (symbol.pushConstant) {
        diag << "push constant " << *symbol.pushConstant;
      } else {
        diag << "an affine.min";
      }
    }
    if (numSymbols) diag << ")";
    return diag;
  };

  FailureOr<SmallVector<int64_t>> flat = flatten(expr, numSymbols);
  if (failed(flat)) {
    return describe(emitError())
           << " is not linear in the workgroup ids and push constants";
  }

  ColumnAddress result;
  std::array<bool, 2> hasTerm = {false, false};
  SmallVector<std::pair<uint32_t, int64_t>> pushConstants;
  for (unsigned i = 0; i < numSymbols; ++i) {
    int64_t coefficient = (*flat)[numWorkgroupDims + i];
    if (coefficient == 0) continue;
    const AddressExprBuilder::Symbol &symbol = symbols[i];
    if (symbol.pushConstant) {
      pushConstants.push_back({*symbol.pushConstant, coefficient});
      continue;
    }

    // coefficient * min(k * wg_d, limit) = min(coefficient * k * wg_d,
    // coefficient * limit) for a positive coefficient.
    std::optional<int64_t> limit, step;
    std::optional<unsigned> dim;
    for (AffineExpr minResult : symbol.minResults) {
      FailureOr<SmallVector<int64_t>> terms = flatten(
        simplifyAffineExpr(minResult, numWorkgroupDims, numSymbols), numSymbols
      );
      if (failed(terms)) break;
      unsigned nonZero = llvm::count_if(
        ArrayRef(*terms).drop_back(), [](int64_t t) { return t != 0; }
      );
      if (nonZero == 0) {
        limit = terms->back();
      } else if (nonZero == 1 && terms->back() == 0) {
        auto it = llvm::find_if(*terms, [](int64_t t) { return t != 0; });
        unsigned position = it - terms->begin();
        if (position < numWorkgroupDims) {
          dim = position;
          step = *it;
        }
      }
    }
    if (symbol.minResults.size() != 2 || !limit || !step || coefficient < 0 ||
        *step <= 0) {
      return describe(emitError())
             << " has an affine.min that is not min(id * step, limit) of one "
                "workgroup id";
    }
    if (*dim >= 2 || hasTerm[*dim]) {
      return describe(emitError())
             << " needs a second term for workgroup dimension " << *dim
             << "; a column address has one each for x and y";
    }
    int64_t scaledStep = coefficient * *step, scaledLimit = coefficient * *limit;
    if (scaledStep > UINT32_MAX || scaledLimit < 0 || scaledLimit > UINT32_MAX) {
      return describe(emitError())
             << " has a workgroup term that does not fit in 32 bits";
    }
    hasTerm[*dim] = true;
    result.wg[*dim] = {static_cast<uint32_t>(scaledStep),
                       static_cast<uint32_t>(scaledLimit)};
  }

  for (unsigned d = 0; d < numWorkgroupDims; ++d) {
    int64_t coefficient = (*flat)[d];
    if (coefficient == 0) continue;
    if (d >= 2 || hasTerm[d] || coefficient < 0 || coefficient > UINT32_MAX) {
      return describe(emitError())
             << " has a term in workgroup dimension " << d
             << " a column address can't hold (one non-negative term each for "
                "x and y)";
    }
    hasTerm[d] = true;
    result.wg[d] = {static_cast<uint32_t>(coefficient), UINT32_MAX};
  }

  if (pushConstants.size() == 2) {
    int64_t lo = std::min(pushConstants[0].second, pushConstants[1].second);
    int64_t hi = std::max(pushConstants[0].second, pushConstants[1].second);
    if (lo == 1 && hi == (int64_t(1) << 32)) {
      return describe(emitError())
             << " is a 64 bit value split over two push constants; STRELA "
                "takes 32 bit offsets, retry with "
                "--iree-stream-resource-index-bits=32";
    }
  }
  if (pushConstants.size() > 1 ||
      (pushConstants.size() == 1 && pushConstants[0].second != 1)) {
    return describe(emitError())
           << " is not a constant plus at most one push constant";
  }

  int64_t constant = flat->back();
  if (constant < 0 || constant > UINT32_MAX) {
    return describe(emitError())
           << " has a constant part that does not fit in 32 bits";
  }
  result.constant = static_cast<uint32_t>(constant);
  if (!pushConstants.empty()) result.pushConstant = pushConstants[0].first;
  return result;
}

FailureOr<ColumnAddress>
getColumnAddress(Operation *columnOp) {
  Value address;
  if (auto inpOp = dyn_cast<InpColumnOp>(columnOp)) {
    address = inpOp.getAddress();
  } else {
    address = cast<OutColumnOp>(columnOp).getAddress();
  }
  auto emitError = [columnOp] { return columnOp->emitOpError(); };
  return foldColumnAddress(address, emitError);
}

} // namespace mlir::strela
