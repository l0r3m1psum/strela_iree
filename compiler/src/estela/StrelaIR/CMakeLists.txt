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
)

# The generated op definitions are compiled here and only here; everything that
# needs to name a strela op just depends on this library.
iree_cc_library(
  NAME
    IR
  HDRS
    "StrelaDialect.h"
    "StrelaOps.h"
  SRCS
    "StrelaDialect.cpp"
    "StrelaOps.cpp"
  INCLUDES
    # The .inc files land next to the sources in the build tree and are included
    # by bare name; PUBLIC so consumers of StrelaOps.h inherit the path.
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPS
    ::StrelaOpsGen
    iree::..::..::compiler::src::estela::defs
    MLIRIR
    MLIRSideEffectInterfaces
    MLIRSupport
  PUBLIC
)
