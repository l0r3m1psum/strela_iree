// How much of each TFLite model in a directory maps to STRELA's matvec kernel.
//
//   estela-find-gemv-usage [--converter=<path>] <directory>
//
// e.g.
//   estela-find-gemv-usage 3rdparty/tiny/benchmark/training
// Every .tflite file under <directory> (recursively) is converted to TOSA by
// tosa-converter-for-tflite, whose bytecode is read from a pipe and parsed.
// For each tosa.conv2d and tosa.depthwise_conv2d the multiply-accumulates at
// batch 1 are counted by class, and printed as a table of one row per model:
//   FC (matvec)    1x1 kernel on a 1x1 output
//   1x1 (matmul)   1x1 kernel, one row per output pixel
//   KxK (conv)     a convolution (im2col or accumulated gemv coul be used)
//   depthwise      not a matmul
// with each class's op count and share of the model's MACs. A last column,
// STRELA gemv, counts the same way the FC and 1x1 ops the compiler gives to
// STRELA's matvec kernel. The converter writes fully connected layers as 1x1
// conv2d ops too. Pooling is not counted.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <cstdio>

// Windows pipes are text unless opened binary; glibc rejects a "b".
#ifdef _WIN32
#define popen _popen
#define pclose _pclose
#define fileno _fileno
#define POPEN_READ "rb"
#else
#define POPEN_READ "r"
#endif

using namespace mlir;

static llvm::cl::opt<std::string> directory(
  llvm::cl::Positional, llvm::cl::Required, llvm::cl::desc("<directory>")
);
static llvm::cl::opt<std::string> converter(
  "converter", llvm::cl::init("tosa-converter-for-tflite"),
  llvm::cl::desc("The TFLite to TOSA converter")
);

// The table's columns after the model's name and MACs: the classes, then the
// gemv ops among them.
enum Column { FullyConnected, Conv1x1, ConvKxK, Depthwise, MatVec, NumColumns };
static const char *const columnNames[NumColumns] = {
  "FC (matvec)", "1x1 (matmul)", "KxK (conv)", "depthwise", "matvec",
};

// The default of the compiler's --iree-strela-max-column-words (in
// StrelaOptions.h): how many words a weight column of C channels can span.
static constexpr int64_t maxColumnWords = 0xFFFF / sizeof(uint32_t);

// What the converter always says on stderr: not worth repeating.
static constexpr llvm::StringLiteral converterChatter =
  "[INFO] TOSA MLIR (bytecode) written to: -";

// The TOSA module of `model`, from the converter's bytecode on a pipe. The
// bytecode is versioned, so a converter built on an older MLIR than ours can
// still be read; its text syntax could not. The converter's stderr goes to
// the file `stderrPath`, and is printed unless it only says it is done.
static OwningOpRef<ModuleOp>
convert(MLIRContext &context, StringRef model, StringRef name,
        StringRef stderrPath) {
  std::string command = converter + " --bytecode -o - \"" + model.str()
                        + "\" 2>\"" + stderrPath.str() + "\"";
  FILE *pipe = popen(command.c_str(), POPEN_READ);
  if (!pipe) return {};
  auto stream = llvm::MemoryBuffer::getOpenFile(
    llvm::sys::fs::convertFDToNativeFile(fileno(pipe)), model,
    /*FileSize=*/-1, /*RequiresNullTerminator=*/false
  );
  int status = pclose(pipe);

  if (auto errors = llvm::MemoryBuffer::getFile(stderrPath)) {
    SmallVector<StringRef> lines;
    llvm::SplitString((*errors)->getBuffer(), lines, "\r\n");
    for (StringRef line : lines) {
      if (line.trim() != converterChatter) llvm::errs() << name << ": " << line << "\n";
    }
  }
  if (status != 0 || !stream) return {};

  // The reader wants the resources in the bytecode aligned, which a stream's
  // buffer is not.
  auto bytecode = llvm::WritableMemoryBuffer::getNewUninitMemBuffer(
    (*stream)->getBufferSize(), model, llvm::Align(64)
  );
  llvm::copy((*stream)->getBuffer(), bytecode->getBufferStart());
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(bytecode), llvm::SMLoc());
  return parseSourceFile<ModuleOp>(sourceMgr, &context);
}

// The class and MACs of a convolution, from its weight and result shapes.
static std::pair<Column, int64_t>
classify(Operation *op, ArrayRef<int64_t> w, ArrayRef<int64_t> o) {
  // A dynamic dim (the batch) counts as 1.
  int64_t outputs = 1;
  for (int64_t d : o) outputs *= ShapedType::isDynamic(d) ? 1 : d;
  if (isa<tosa::DepthwiseConv2DOp>(op)) { // weights KH x KW x C x M
    return {Depthwise, outputs * w[0] * w[1]};
  }
  int64_t macs = outputs * w[1] * w[2] * w[3]; // weights OC x KH x KW x IC
  if (w[1] != 1 || w[2] != 1) return {ConvKxK, macs};
  return {o[1] == 1 && o[2] == 1 ? FullyConnected : Conv1x1, macs};
}

int
main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  std::vector<std::string> models;
  std::error_code ec;
  for (llvm::sys::fs::recursive_directory_iterator it(directory, ec), end;
       it != end && !ec; it.increment(ec)) {
    if (llvm::sys::path::extension(it->path()) == ".tflite") {
      models.push_back(it->path());
    }
  }
  if (ec) {
    llvm::errs() << directory << ": " << ec.message() << "\n";
    return 1;
  }
  llvm::sort(models);

  SmallString<128> stderrPath;
  if (std::error_code ec = llvm::sys::fs::createTemporaryFile(
        "estela-find-gemv-usage", "txt", stderrPath)) {
    llvm::errs() << "cannot create a temporary file: " << ec.message() << "\n";
    return 1;
  }
  llvm::FileRemover removeStderr(stderrPath);

  DialectRegistry registry;
  registry.insert<func::FuncDialect, tosa::TosaDialect>();
  MLIRContext context(registry);

  struct Total { int ops = 0; int64_t macs = 0; };
  struct Row { std::string name; int64_t macs = 0; std::array<Total, NumColumns> totals; };
  std::vector<Row> rows;
  int failures = 0;
  for (StringRef model : models) {
    Row row{llvm::sys::path::filename(model).str()};
    OwningOpRef<ModuleOp> module = convert(context, model, row.name, stderrPath);
    if (!module) {
      llvm::errs() << row.name << ": conversion failed\n";
      ++failures;
      continue;
    }
    module->walk([&](Operation *op) {
      if (!isa<tosa::Conv2DOp, tosa::DepthwiseConv2DOp>(op)) return;
      ArrayRef<int64_t> w =
        cast<ShapedType>(op->getOperand(1).getType()).getShape();
      auto [column, macs] = classify(
        op, w, cast<ShapedType>(op->getResult(0).getType()).getShape()
      );
      auto add = [&, macs = macs](Column c) {
        row.totals[c].ops++;
        row.totals[c].macs += macs;
      };
      add(column);
      row.macs += macs;
      if ((column == FullyConnected || column == Conv1x1)) {
        add(MatVec);
      }
    });
    rows.push_back(std::move(row));
  }

  // A cell is the class's op count and its share of the model's MACs.
  constexpr int cellWidth = 14;
  size_t nameWidth = std::string_view("model").size();
  for (const Row &row : rows) nameWidth = std::max(nameWidth, row.name.size());
  llvm::raw_ostream &os = llvm::outs();
  os << llvm::left_justify("model", nameWidth) << llvm::right_justify("M MACs", 9);
  for (const char *name : columnNames) os << llvm::right_justify(name, cellWidth);
  os << "\n" << std::string(nameWidth + 9 + NumColumns * cellWidth, '-') << "\n";
  for (const Row &row : rows) {
    os << llvm::left_justify(row.name, nameWidth)
       << llvm::format("%9.2f", row.macs / 1e6);
    for (const Total &total : row.totals) {
      if (total.ops == 0) {
        os << llvm::right_justify("-", cellWidth);
      } else {
        os << llvm::format("%7d %5.1f%%", total.ops,
                           row.macs ? 100.0 * total.macs / row.macs : 0.0);
      }
    }
    os << "\n";
  }
  return failures ? 1 : 0;
}
