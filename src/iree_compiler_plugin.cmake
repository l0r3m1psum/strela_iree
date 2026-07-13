iree_cc_library(
  NAME
    registration2
  SRCS
    "${CMAKE_CURRENT_LIST_DIR}/my_plugin.cc"
  DEPS
    MLIRIR
    iree::compiler::PluginAPI
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    example2
  TARGET
    ::registration2
)
