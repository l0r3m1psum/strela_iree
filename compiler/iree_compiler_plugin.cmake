# This file is include()d from a generated CMakeLists.txt under
# ${IREE_BINARY_DIR}/compiler/plugins, so CMAKE_CURRENT_SOURCE_DIR points there
# rather than at these sources. Use CMAKE_CURRENT_LIST_DIR for source paths and
# give add_subdirectory() an explicit binary directory.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/src/estela"
  "${CMAKE_CURRENT_BINARY_DIR}/estela"
)

# Nothing but the plugin session: options, the pass pipelines it installs, and
# the HAL target registration.
iree_cc_library(
  NAME
    registration
  SRCS
    "${CMAKE_CURRENT_LIST_DIR}/src/estela/PluginRegistration.cpp"
  DEPS
    iree::..::..::compiler::src::estela::defs
    iree::..::..::compiler::src::estela::Codegen::Strela::StrelaTarget
    iree::..::..::compiler::src::estela::Dialect::Strela::Conversion::StrelaConversion
    iree::..::..::compiler::src::estela::Dialect::Strela::IR::IR
    iree::..::..::compiler::src::estela::Transforms::Linalg::LinalgTransforms
    iree::..::..::compiler::src::estela::Transforms::Tosa::TosaTransforms
    iree::compiler::PluginAPI
    MLIRFuncDialect
    MLIRIR
    MLIRPass
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    estela
  TARGET
    ::registration
)
