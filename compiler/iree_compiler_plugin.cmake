# This file is include()d from a generated CMakeLists.txt under
# ${IREE_BINARY_DIR}/compiler/plugins, so CMAKE_CURRENT_SOURCE_DIR points there
# rather than at these sources. Use CMAKE_CURRENT_LIST_DIR for source paths and
# give add_subdirectory() an explicit binary directory.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/src/estela"
  "${CMAKE_CURRENT_BINARY_DIR}/estela"
)

# The executable schema shared with the runtime, which the target backend writes.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../runtime/src/estela/schemas"
  "${CMAKE_CURRENT_BINARY_DIR}/estela_schemas"
)

# The plugin session (options, the pass pipelines it installs, the pass and HAL
# target registration) and the STRELA HAL target itself.
iree_cc_library(
  NAME
    registration
  HDRS
    "${CMAKE_CURRENT_LIST_DIR}/src/estela/StrelaTarget.h"
  SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/estela/PluginRegistration.cpp"
    "${CMAKE_CURRENT_LIST_DIR}/src/estela/StrelaTarget.cpp"
  DEPS
    iree::..::..::compiler::src::estela::defs
    iree::..::..::compiler::src::estela::StrelaIR::IR
    iree::..::..::compiler::src::estela::Transforms::Transforms
    iree::..::..::compiler::src::estela::Utils::Utils
    iree::compiler::Dialect::HAL::IR
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Dialect::Stream::IR
    iree::compiler::PluginAPI
    iree::compiler::Utils
    iree::base::internal::flatcc::building
    estela::schemas::strela_executable_def_c_fbs
    MLIRFuncDialect
    MLIRIR
    MLIRPass
    MLIRSupport
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    estela
  TARGET
    ::registration
)

# Host tools for inspecting what the plugin produces. They sit at the top level
# like IREE's tools/, and only need the compiler build (the schemas above).
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../tools"
  "${CMAKE_CURRENT_BINARY_DIR}/estela_tools"
)

# The lit tests drive iree-compile/iree-opt with --iree-plugin=estela, so they
# belong to the compiler plugin rather than the runtime one. iree_lit_test_suite
# is a no-op when IREE_BUILD_TESTS is off, which is what the ARM build uses.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/../tests"
  "${CMAKE_CURRENT_BINARY_DIR}/tests"
)
