# Host tools for looking at what the compiler plugin produces.
#
# Rooted here like the schemas, so the binary is
# estela::tools::estela-dump-executable rather than named after its path
# relative to the IREE root.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "estela/tools")

iree_cc_binary(
  NAME
    estela-dump-executable
  SRCS
    "estela-dump-executable-main.c"
  DEPS
    flatcc::runtime
    iree::base::internal::flatcc::debugging
    iree::base::internal::flatcc::parsing
    estela::schemas::strela_executable_def_c_fbs
)
