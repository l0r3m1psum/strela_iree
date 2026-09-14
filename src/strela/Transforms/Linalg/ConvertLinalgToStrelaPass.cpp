#include "strela/Transforms/Linalg/Passes.h"

#include <mutex>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "strela/Dialect/Strela/StrelaDialect.h"
#include "strela/Dialect/Strela/StrelaOps.h"
#include "strela/Transforms/Linalg/Patterns.h"

static std::mutex printMutex;

static void
print(mlir::ModuleOp moduleOp) {
  std::lock_guard<std::mutex> lock(printMutex);
  mlir::OpPrintingFlags flags;
  flags.elideLargeElementsAttrs(16);
  // Required, not cosmetic: without it AsmPrinter's findParent() walks past this
  // module all the way to the root and numbers SSA values over the *whole*
  // top-level module. Executable variants are translated in parallel, so that
  // reads IR another thread is busy rewriting. useLocalScope() stops the walk at
  // the first IsolatedFromAbove ancestor, i.e. this module.
  flags.useLocalScope();
  moduleOp.print(llvm::errs(), flags);
  llvm::errs() << '\n';
}

static void
print(mlir::func::FuncOp funcOp) {
  print(funcOp->getParentOfType<mlir::ModuleOp>());
}

static mlir::LogicalResult
allInputsInteger32(mlir::linalg::GenericOp genericOp) {
  for (mlir::Value input : genericOp.getDpsInputs()) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType || !inputType.getElementType().isInteger(32)) {
      return mlir::failure();
    }
  }
  return mlir::success();
}

mlir::LogicalResult
mlir::strela::isStrelaLinalgAdd(mlir::linalg::GenericOp genericOp) {
  if (genericOp.getNumDpsInputs() != 2 || genericOp.getNumDpsInits() != 1) {
    return mlir::failure();
  }

  if (failed(allInputsInteger32(genericOp))) {
    return mlir::failure();
  }

  if (!genericOp.hasPureTensorSemantics()) {
    return mlir::failure();
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(), mlir::linalg::isParallelIterator)) {
    return mlir::failure();
  }

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return mlir::failure();
  }

  auto addOp = mlir::dyn_cast<mlir::arith::AddIOp>(&body->front());
  if (!addOp) {
    return mlir::failure();
  }

  auto yieldOp = mlir::dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != addOp.getResult()) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult
mlir::strela::isStrelaLinalgRelu(mlir::linalg::GenericOp genericOp) {
  unsigned numInputs = genericOp.getNumDpsInputs();
  if ((numInputs != 1 && numInputs != 2) || genericOp.getNumDpsInits() != 1) {
    return mlir::failure();
  }

  if (failed(allInputsInteger32(genericOp))) {
    return mlir::failure();
  }

  if (!genericOp.hasPureTensorSemantics()) {
    return mlir::failure();
  }

  if (!llvm::all_of(genericOp.getIteratorTypesArray(), mlir::linalg::isParallelIterator)) {
    return mlir::failure();
  }

  mlir::Block *body = genericOp.getBlock();
  if (std::distance(body->begin(), body->end()) != 2) {
    return mlir::failure();
  }

  auto maxOp = mlir::dyn_cast<mlir::arith::MaxSIOp>(&body->front());
  if (!maxOp) {
    return mlir::failure();
  }

  if (maxOp.getLhs() != body->getArgument(0)) {
    return mlir::failure();
  }

  if (numInputs == 2) {
    if (maxOp.getRhs() != body->getArgument(1)) {
      return mlir::failure();
    }
    mlir::DenseIntElementsAttr zeroAttr;
    if (
      !mlir::matchPattern(genericOp.getDpsInputs()[1], mlir::m_Constant(&zeroAttr))
      || !zeroAttr.isSplat()
      || !zeroAttr.getSplatValue<llvm::APInt>().isZero()
    ) {
      return mlir::failure();
    }
  } else {
    llvm::APInt zeroValue;
    if (!mlir::matchPattern(maxOp.getRhs(), mlir::m_ConstantInt(&zeroValue)) ||
        !zeroValue.isZero()) {
      return mlir::failure();
    }
  }

  auto yieldOp = mlir::dyn_cast<mlir::linalg::YieldOp>(body->back());
  if (!yieldOp || yieldOp.getNumOperands() != 1 || yieldOp.getOperand(0) != maxOp.getResult()) {
    return mlir::failure();
  }

  return mlir::success();
}

bool
mlir::strela::isSupportedByStrela(mlir::linalg::GenericOp genericOp) {
  if (mlir::succeeded(isStrelaLinalgAdd(genericOp))) {
    return true;
  }

  if (mlir::succeeded(isStrelaLinalgRelu(genericOp))) {
    return true;
  }

  return false;
}

using namespace mlir;
using mlir::strela::isStrelaLinalgAdd;
using mlir::strela::isStrelaLinalgRelu;

namespace {

struct ConvertLinalgAddToStrela
  : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(
    linalg::GenericOp genericOp, PatternRewriter &rewriter
  ) const override {
    LogicalResult result = isStrelaLinalgAdd(genericOp);

    if (succeeded(result)) {
      rewriter.replaceOpWithNewOp<strela::AddOp>(
        genericOp,
        genericOp.getResultTypes(),
        genericOp.getDpsInputs()[0],
        genericOp.getDpsInputs()[1]
      );
    }

    return result;
  }
};

struct ConvertLinalgReluToStrela
  : public OpRewritePattern<linalg::GenericOp> {
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(
    linalg::GenericOp genericOp, PatternRewriter &rewriter
  ) const override {
    LogicalResult result = isStrelaLinalgRelu(genericOp);

    if (succeeded(result)) {
      rewriter.replaceOpWithNewOp<strela::ReluOp>(
        genericOp,
        genericOp.getResultTypes(),
        genericOp.getDpsInputs()[0]
      );
    }

    return result;
  }
};

struct LinalgToStrelaPass
  : public PassWrapper<LinalgToStrelaPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LinalgToStrelaPass)

  StringRef getArgument() const override { return "iree-strela-convert-linalg"; }
  StringRef getDescription() const override {
    return "Converts linalg ops to strela backend ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<strela::StrelaDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);
    patterns.add<ConvertLinalgAddToStrela, ConvertLinalgReluToStrela>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }

    print(funcOp);
  }
};

} // namespace

namespace mlir::strela {

std::unique_ptr<Pass> createConvertLinalgToStrelaPass() {
  return std::make_unique<LinalgToStrelaPass>();
}

} // namespace mlir::strela
