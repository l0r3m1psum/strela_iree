#include "strela/Transforms/Tosa/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;

namespace {

struct DoubleRoundRewriter : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult
  matchAndRewrite(
    tosa::RescaleOp rescaleOp, PatternRewriter& rewriter
  ) const override {

    if (rescaleOp.getRoundingMode() != mlir::tosa::RoundingMode::DOUBLE_ROUND) {
      return failure();
    }

    rescaleOp.emitWarning("Converting from DOUBLE_ROUND to SINGLE_ROUND");

    rewriter.modifyOpInPlace(rescaleOp, [&]() {
      rescaleOp->setAttr(
        "rounding_mode",
        mlir::tosa::RoundingModeAttr::get(
          rewriter.getContext(),
          mlir::tosa::RoundingMode::SINGLE_ROUND
        )
      );
    });

    return success();
  }
};

// TODO: does it make sense to support multiple dynamic leading dynamic
// dimensions (i.e. ?x?x1x1x128x)? The idea it would be to map each of them to a
// scf.for loop but I am not sure of the semantic of tosa.rescale in this case.
// Probably a better solution is to wrap this in a tensor.collapse_shape
// (flatten) and tensor.expand_shape (reshape) and then use the same loop as
// now.
struct DynamicBatchRewriter : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    tosa::RescaleOp rescaleOp,
    PatternRewriter& rewriter
  ) const override {

    auto inputType = dyn_cast<RankedTensorType>(rescaleOp.getInput().getType());
    if (!inputType || inputType.getRank() == 0 || !inputType.isDynamicDim(0)) {
      return failure();
    }

    rescaleOp.emitRemark("Lowering dynamic batch to an scf.for loop");

    Location loc = rescaleOp.getLoc();
    int64_t rank = inputType.getRank();
    auto outputType = cast<RankedTensorType>(rescaleOp.getResult().getType());

    Value c0 = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value batchSize = tensor::DimOp::create(rewriter, loc, rescaleOp.getInput(), c0);

    SmallVector<Value> dynSizes;
    for (int64_t i = 0; i < rank; ++i) {
      if (outputType.isDynamicDim(i)) {
        Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        Value dynSize = tensor::DimOp::create(rewriter, loc, rescaleOp.getInput(), dimIdx);
        dynSizes.push_back(dynSize);
      }
    }
    Value initTensor = tensor::EmptyOp::create(
      rewriter, loc, outputType.getShape(), outputType.getElementType(), dynSizes
    );

    Value lowerBound = c0;
    Value upperBound = batchSize;
    Value step = arith::ConstantIndexOp::create(rewriter, loc, 1);

    auto forOp = scf::ForOp::create(
      rewriter, loc, lowerBound, upperBound, step, ValueRange{initTensor},
      [&](OpBuilder& builder, Location loc, Value iv, ValueRange iterArgs) {
        // currentIterArg holds the accumulated tensor for this iteration
        Value currentIterArg = iterArgs.front();

        SmallVector<OpFoldResult> offsets(rank, builder.getIndexAttr(0));
        SmallVector<OpFoldResult> sizes(rank, builder.getIndexAttr(1));
        SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));

        offsets[0] = iv;
        sizes[0] = builder.getIndexAttr(1);

        for (int64_t i = 1; i < rank; ++i) {
          if (inputType.isDynamicDim(i)) {
             Value dimIdx = arith::ConstantIndexOp::create(builder, loc, i);
             sizes[i] = tensor::DimOp::create(builder, loc, rescaleOp.getInput(), dimIdx)->getResult(0);
          } else {
             sizes[i] = builder.getIndexAttr(inputType.getDimSize(i));
          }
        }

        SmallVector<int64_t> sliceInputShape(inputType.getShape());
        sliceInputShape[0] = 1;
        auto sliceInputType = RankedTensorType::get(sliceInputShape, inputType.getElementType());

        Value sliceInput = tensor::ExtractSliceOp::create(
          builder, loc, sliceInputType, rescaleOp.getInput(), offsets, sizes, strides
        );

        SmallVector<int64_t> sliceOutputShape(outputType.getShape());
        sliceOutputShape[0] = 1;
        auto sliceOutputType = RankedTensorType::get(sliceOutputShape, outputType.getElementType());

        SmallVector<Value> newOperands(rescaleOp->getOperands().begin(), rescaleOp->getOperands().end());
        newOperands[0] = sliceInput;

        OperationState state(loc, tosa::RescaleOp::getOperationName());
        state.addOperands(newOperands);
        state.addTypes(sliceOutputType);
        state.addAttributes(rescaleOp->getAttrs());

        Operation *newRescale = builder.create(state);
        Value rescaleResult = newRescale->getResult(0);

        Value insertedSlice = tensor::InsertSliceOp::create(
          builder, loc, rescaleResult, currentIterArg, offsets, sizes, strides
        );

        scf::YieldOp::create(builder, loc, insertedSlice);
      }
    );

    rewriter.replaceOp(rescaleOp, forOp.getResult(0));

    return success();
  }
};


struct MyRewritePass : public PassWrapper<MyRewritePass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyRewritePass)

  StringRef getArgument() const override { return "iree-example2-rewrite-pass"; }
  StringRef getDescription() const override { return "Rewrites sruff"; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    llvm::errs() << "DEBUG: MyRewritePass is running on operation: "
               << funcOp.getName() << "\n";
    RewritePatternSet patterns(context);
    patterns.add<DoubleRoundRewriter, DynamicBatchRewriter>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::strela {

std::unique_ptr<Pass> createNormalizeRescalePass() {
  return std::make_unique<MyRewritePass>();
}

} // namespace mlir::strela
