iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    # One token: IREE's OUTS parser takes the first non-flag as the file name.
    --gen-pass-decls --name=Estela Passes.h.inc
)

iree_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
  SRCS
    "AssignTopologyPass.cpp"
    "ConvertLinalgToStrelaPass.cpp"
    "LinkExecutablesPass.cpp"
    "MaterializeBindingsPass.cpp"
    "NormalizeRescalePass.cpp"
    "ResolveWorkgroupCountPass.cpp"
    "TransferWrapPass.cpp"
  INCLUDES
    # Passes.h.inc lands next to the sources in the build tree and is included
    # by bare name; PUBLIC so consumers of Passes.h inherit the path.
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPS
    ::PassesIncGen
    iree::..::..::compiler::src::estela::defs
    iree::..::..::compiler::src::estela::StrelaIR::IR
    iree::..::..::compiler::src::estela::Utils::Utils
    iree::compiler::Codegen::Utils
    iree::compiler::Dialect::Flow::IR
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::Stream::IR
    iree::compiler::Dialect::TensorExt::IR
    iree::compiler::Dialect::Util::IR
    iree::compiler::Utils
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRLinalgUtils
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
    MLIRTensorDialect
    MLIRTosaDialect
    MLIRTransforms
  PUBLIC
)

add_subdirectory(tests)
