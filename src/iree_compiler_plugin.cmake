# This file is include()d from a generated CMakeLists.txt under
# ${IREE_BINARY_DIR}/compiler/plugins, so CMAKE_CURRENT_SOURCE_DIR points there
# rather than at these sources. Use CMAKE_CURRENT_LIST_DIR for source paths and
# give add_subdirectory() an explicit binary directory.
add_subdirectory(
  "${CMAKE_CURRENT_LIST_DIR}/strela"
  "${CMAKE_CURRENT_BINARY_DIR}/strela"
)

# Nothing but the plugin session: options, the pass pipelines it installs, and
# the HAL target registration.
iree_cc_library(
  NAME
    registration2
  SRCS
    "${CMAKE_CURRENT_LIST_DIR}/PluginRegistration.cpp"
  DEPS
    iree::..::..::src::strela::defs
    iree::..::..::src::strela::Dialect::Strela::StrelaDialect
    iree::..::..::src::strela::Target::StrelaTarget
    iree::..::..::src::strela::Transforms::Linalg::LinalgTransforms
    iree::..::..::src::strela::Transforms::Tosa::TosaTransforms
    iree::compiler::PluginAPI
    MLIRFuncDialect
    MLIRIR
    MLIRPass
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    example2
  TARGET
    ::registration2
)
