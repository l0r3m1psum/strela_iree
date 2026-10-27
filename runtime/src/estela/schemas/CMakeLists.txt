# The STRELA executable schema. Built by both trees: the host one for the
# compiler plugin (builder) and the ARM one for the HAL driver (reader and
# verifier), each adding this directory itself.
#
# This directory sits outside the IREE source tree, so root the package here or
# iree_package_ns() names the target after its path relative to the IREE root.
# The library is then estela::schemas::strela_executable_def_c_fbs.
set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(IREE_PACKAGE_ROOT_PREFIX "estela/schemas")

# The HAL driver is installed with the runtime, so its dependencies must be too.
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_EXPORT_SET Runtime)
set(IREE_INSTALL_LIBRARY_TARGETS_DEFAULT_COMPONENT IREEDevLibraries-Runtime)

flatbuffer_c_library(
  NAME
    strela_executable_def_c_fbs
  SRCS
    "strela_executable_def.fbs"
  FLATCC_ARGS
    "--reader"
    "--builder"
    "--verifier"
    "--json"
  PUBLIC
)
