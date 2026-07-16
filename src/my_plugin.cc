#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/PluginAPI/Client.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

static LogicalResult
isMatmulEquivalent(linalg::Conv2DNhwcHwcfQOp convOp) {
  Value x = convOp.getDpsInputOperand(0)->get();
  Value w = convOp.getDpsInputOperand(1)->get();
  Value z_x = convOp.getDpsInputOperand(2)->get();
  Value z_w = convOp.getDpsInputOperand(3)->get();
  Value y = convOp.getDpsInits()[0];

  if (!z_x.getType().isInteger(32) || !z_w.getType().isInteger(32)) {
    return failure();
  }

  auto isAllOnes = [](DenseIntElementsAttr attr) {
    if (!attr) return false;
    return llvm::all_of(attr.getValues<int64_t>(), [](int64_t v) { return v == 1; });
  };
  if (!isAllOnes(convOp.getStrides()) || !isAllOnes(convOp.getDilations())) {
    return failure();
  }

  // Check that the Filter (W) spatial dimensions are 1x1
  // Layout is HWCF: index 0 (H), index 1 (W)
  auto wType = dyn_cast<RankedTensorType>(w.getType());
  if (!wType.getElementType().isInteger(8)) return failure();
  if (wType.getDimSize(0) != 1 || wType.getDimSize(1) != 1) {
    return failure();
  }

  // Check that the Input (X) spatial dimensions are 1x1
  // Layout is NHWC: index 1 (H), index 2 (W)
  auto xType = dyn_cast<RankedTensorType>(x.getType());
  if (!xType.getElementType().isInteger(8)) return failure();
  if (xType.getDimSize(1) != 1 || xType.getDimSize(2) != 1) {
    return failure();
  }

  auto yType = dyn_cast<RankedTensorType>(y.getType());
  if (!yType.getElementType().isInteger(32)) return failure();

  return success();
}

static Type
getDynamicTensorType(Type t) {
  if (auto rankedType = dyn_cast<RankedTensorType>(t)) {
    SmallVector<int64_t> dynShape(rankedType.getRank(), ShapedType::kDynamic);
    return RankedTensorType::get(dynShape, rankedType.getElementType());
  }
  // TODO: assert that it is a scalar...
  return t;
}

struct Conv2DOffload : public OpRewritePattern<linalg::Conv2DNhwcHwcfQOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfQOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    linalg::Conv2DNhwcHwcfQOp convOp,
    PatternRewriter& rewriter
  ) const override {

    if (failed(isMatmulEquivalent(convOp))) {
      return failure();
    }

    convOp.emitRemark("Offloading Conv2D equivalent to Matmul");

    Value x = convOp.getDpsInputOperand(0)->get();
    Value w = convOp.getDpsInputOperand(1)->get();
    Value z_x = convOp.getDpsInputOperand(2)->get();
    Value z_w = convOp.getDpsInputOperand(3)->get();
    Value y = convOp.getDpsInits()[0];

    ModuleOp moduleOp = convOp->getParentOfType<ModuleOp>();
    Location loc = convOp.getLoc();

    {
      static int next_kernel_id = 0; // Simple global ID counter
      int current_kernel_id = next_kernel_id++;

      std::vector<uint8_t> myKernelBinary = {0,1,2,3};

      auto kernelTensorType = RankedTensorType::get(
          {static_cast<int64_t>(myKernelBinary.size())},
          rewriter.getI8Type()
      );

      StringRef initFuncName = "custom.init_accelerator";
      auto initFuncDecl = moduleOp.lookupSymbol<func::FuncOp>(initFuncName);
      if (!initFuncDecl) {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToStart(moduleOp.getBody());

          auto unrankedTensorType = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getI8Type());
          auto initFuncType = rewriter.getFunctionType(
              {rewriter.getI32Type(), unrankedTensorType}, {}
          );
          initFuncDecl = func::FuncOp::create(rewriter, loc, initFuncName, initFuncType);
          initFuncDecl.setPrivate();
      }

      // TODO: the parameters for a kernel are z_x and z_w together with the
      // delay to accumulate a resulting vector.
      {
          // Save insertion point so we don't mess up the convOp replacement
          OpBuilder::InsertionGuard guard(rewriter);
          // Initializers usually go at the end of the module
          rewriter.setInsertionPointToEnd(moduleOp.getBody());

          auto initOp = IREE::Util::InitializerOp::create(rewriter, loc);
          // createBlock automatically moves the insertion point inside the block
          rewriter.createBlock(&initOp.getBody());

          Value idVal = arith::ConstantIntOp::create(rewriter, loc, current_kernel_id, 32);

          auto denseAttr = DenseElementsAttr::get(kernelTensorType, ArrayRef(myKernelBinary));
          Value kernelBlob = arith::ConstantOp::create(rewriter, loc, kernelTensorType, denseAttr);

          auto unrankedTensorType = RankedTensorType::get({ShapedType::kDynamic}, rewriter.getI8Type());
          Value castedBlob = tensor::CastOp::create(rewriter, loc, unrankedTensorType, kernelBlob);
          func::CallOp::create(rewriter, loc, initFuncDecl, ValueRange{idVal, castedBlob});
          IREE::Util::ReturnOp::create(rewriter, loc);
      }

    }

    // TODO: this function should take the kernel ID as input or some more type-safe thing...
    StringRef gemmFuncName = "custom.my_centered_gemm";

    auto funcDecl = moduleOp.lookupSymbol<func::FuncOp>(gemmFuncName);
    SmallVector<Type> inputTypes = {
        x.getType(), z_x.getType(), w.getType(), z_w.getType(), y.getType()
    };
    TypeRange resultTypes = convOp.getResultTypes();

    SmallVector<Type> dynamicInputTypes;
    for (Type t : inputTypes) dynamicInputTypes.push_back(getDynamicTensorType(t));

    SmallVector<Type> dynamicResultTypes;
    for (Type t : resultTypes) dynamicResultTypes.push_back(getDynamicTensorType(t));

    if (!funcDecl) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(moduleOp.getBody());

      auto funcType = rewriter.getFunctionType(dynamicInputTypes, dynamicResultTypes);
      funcDecl = func::FuncOp::create(rewriter, loc, gemmFuncName, funcType);
      // Must be set as private to signify it's an externally resolved symbol
      funcDecl.setPrivate();
    }

    SmallVector<Value> originalOperands = {x, z_x, w, z_w, y};
    SmallVector<Value> castedOperands;
    for (size_t i = 0; i < originalOperands.size(); ++i) {
      if (originalOperands[i].getType() != dynamicInputTypes[i]) {
        castedOperands.push_back(tensor::CastOp::create(
          rewriter, convOp.getLoc(), dynamicInputTypes[i], originalOperands[i])
        );
      } else {
        castedOperands.push_back(originalOperands[i]);
      }
    }

    auto callOp = func::CallOp::create(rewriter, loc, funcDecl, castedOperands);

    SmallVector<Value> castedResults;
    for (size_t i = 0; i < callOp.getNumResults(); ++i) {
      if (callOp.getResult(i).getType() != resultTypes[i]) {
        castedResults.push_back(tensor::CastOp::create(
          rewriter, convOp.getLoc(), resultTypes[i], callOp.getResult(i))
        );
      } else {
        castedResults.push_back(callOp.getResult(i));
      }
    }

    rewriter.replaceOp(convOp, castedResults);

    return success();
  }
};

// NOTE: should OperationPass<T> be OperationPass<ModuleOp>?
struct MyFusionPass : public PassWrapper<MyFusionPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MyFusionPass)

  StringRef getArgument() const override { return "iree-example2-fusion-pass"; }
  StringRef getDescription() const override { return "Fuses centering subtractions into a GEMM ukernel"; }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    llvm::errs() << "DEBUG: MyFusionPass is running on operation: "
               << funcOp.getName() << "\n";
    RewritePatternSet patterns(context);
    patterns.add<Conv2DOffload>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }

    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(16);
    funcOp->getParentOfType<ModuleOp>().print(llvm::errs(), flags);
  }
};

struct DoubleRoundRewriter : public OpRewritePattern<tosa::RescaleOp> {
  using OpRewritePattern<tosa::RescaleOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(
    tosa::RescaleOp rescaleOp,
    PatternRewriter& rewriter
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
    patterns.add<DoubleRoundRewriter>(context);
    patterns.add<DynamicBatchRewriter>(context);

    GreedyRewriteConfig config;
    if (failed(applyPatternsGreedily(funcOp, std::move(patterns), config))) {
      signalPassFailure();
    }

    mlir::OpPrintingFlags flags;
    flags.elideLargeElementsAttrs(16);
    funcOp->getParentOfType<ModuleOp>().print(llvm::errs(), flags);
  }
};

struct MyOptions {
  bool enable_fusion = false;
  void bindOptions(OptionsBinder& binder) {
    static llvm::cl::OptionCategory category("IREE Example 2 Plugin");
    binder.opt<bool>("iree-example2-fusion", enable_fusion,
                     llvm::cl::desc("Enable the custom centered-gemm microkernel fusion"),
                     llvm::cl::cat(category));
  }
};

struct MySession : public PluginSession<MySession, MyOptions> {

  void extendInputConversionPreprocessingPassPipeline(
    OpPassManager &passManager,
    InputDialectOptions::Type inputType
  ) override {
    passManager.addNestedPass<func::FuncOp>(std::make_unique<MyRewritePass>());
  }

  bool extendCustomInputConversionPassPipeline(
    OpPassManager& pm,
    std::string_view typeMnemonic
  ) override {
      llvm::errs()
        << "Custom input conversion pass pipeline type: "
        << typeMnemonic << "\n";
      pm.addNestedPass<func::FuncOp>(std::make_unique<MyFusionPass>());
      bool extensionsWereMade = true;
      return extensionsWereMade;
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(MyOptions);

extern "C" bool iree_register_compiler_plugin_example2(
    mlir::iree_compiler::PluginRegistrar* registrar) {
  registrar->registerPlugin<MySession>("example2");
  return true;
}
