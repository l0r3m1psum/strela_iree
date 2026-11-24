# Lit tests of the strela dialect on its own (verifiers), run through iree-opt.
# Rooting the package at compiler/src/estela names them
# "estela/StrelaIR/tests/<file>.mlir.test".
get_filename_component(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(IREE_PACKAGE_ROOT_PREFIX "estela")

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "job.mlir"
  TOOLS
    iree-opt
  LABELS
    "hostonly"
)
