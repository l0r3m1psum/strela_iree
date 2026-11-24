# TODO(Diego): This dialect is spread in too many files. It would be nice to
# reduce this number.

iree_tablegen_library(
  NAME
    StrelaOpsGen
  TD_FILE
    "StrelaOps.td"
  OUTS
    --gen-dialect-decls StrelaDialect.h.inc
    --gen-dialect-defs StrelaDialect.cpp.inc
    --gen-op-decls StrelaOps.h.inc
    --gen-op-defs StrelaOps.cpp.inc
    --gen-attrdef-decls StrelaAttrs.h.inc
    --gen-attrdef-defs StrelaAttrs.cpp.inc
    --gen-typedef-decls StrelaTypes.h.inc
    --gen-typedef-defs StrelaTypes.cpp.inc
)

# The generated op definitions are compiled here and only here; everything that
# needs to name a strela op just depends on this library.
iree_cc_library(
  NAME
    IR
  HDRS
    "StrelaDialect.h"
    "StrelaOps.h"
    "StrelaAttrs.h"
    "ColumnAddress.h"
  SRCS
    "StrelaDialect.cpp"
    "StrelaOps.cpp"
    "StrelaAttrs.cpp"
    # Here rather than in Utils: the job verifier calls it, and Utils already
    # depends on this library.
    "ColumnAddress.cpp"
  INCLUDES
    # The .inc files land next to the sources in the build tree and are included
    # by bare name; PUBLIC so consumers of StrelaOps.h inherit the path.
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPS
    ::StrelaOpsGen
    iree::..::..::compiler::src::estela::defs
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::Util::IR
    MLIRAffineDialect
    MLIRAnalysis
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRSupport
  PUBLIC
)

add_subdirectory(tests)
