# Lit tests for the Estela compiler plugin. These only need the compiler tools,
# so they are registered from compiler/iree_compiler_plugin.cmake and skipped
# when cross compiling (the "hostonly" label) or when IREE_BUILD_TESTS is off.
#
# Run them with:
#   ctest --test-dir iree-build -R estela -V
#
# This directory sits outside the IREE source tree, so without a package root
# iree_package_ns() names the tests by their path relative to it and they show
# up as "iree/../../tests/...". Rooting the package here names them
# "estela/<file>.mlir.test" instead.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "estela")

iree_lit_test_suite(
  NAME
    lit
  SRCS
    "transfer.mlir"
  TOOLS
    FileCheck
    iree-compile
    iree-opt
    not
  LABELS
    "hostonly"
)
