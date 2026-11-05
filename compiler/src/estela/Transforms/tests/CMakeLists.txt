# Lit tests of single passes, run through iree-opt. Rooting the package at
# compiler/src/estela names them "estela/Transforms/tests/<file>.mlir.test".
get_filename_component(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(IREE_PACKAGE_ROOT_PREFIX "estela")

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "transfer_wrap.mlir"
  TOOLS
    FileCheck
    iree-opt
  LABELS
    "hostonly"
)
