set(LLVM_TARGET_DEFINITIONS "${CMAKE_CURRENT_LIST_DIR}/StrelaOps.td")

set(TD_INCLUDES "")
foreach(dir IN LISTS MLIR_INCLUDE_DIRS)
  list(APPEND TD_INCLUDES "-I${dir}")
endforeach()

mlir_tablegen(StrelaOps.h.inc -gen-op-decls ${TD_INCLUDES})
mlir_tablegen(StrelaOps.cpp.inc -gen-op-defs ${TD_INCLUDES})

mlir_tablegen(StrelaDialect.h.inc -gen-dialect-decls ${TD_INCLUDES})
mlir_tablegen(StrelaDialect.cpp.inc -gen-dialect-defs ${TD_INCLUDES})

add_public_tablegen_target(StrelaOpsIncGen)

iree_cc_library(
  NAME
    registration2
  SRCS
    "${CMAKE_CURRENT_LIST_DIR}/my_plugin.cc"
  INCLUDES
    "${CMAKE_CURRENT_BINARY_DIR}"
  DEPS
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRFuncDialect
    MLIRTosaDialect
    MLIRLinalgDialect
    iree::compiler::PluginAPI
    iree::compiler::Codegen::Dialect::Codegen::IR::IREECodegenDialect
  PUBLIC
)

iree_package_name(_PACKAGE_NAME)
add_dependencies(${_PACKAGE_NAME}_registration2 StrelaOpsIncGen)

iree_compiler_register_plugin(
  PLUGIN_ID
    example2
  TARGET
    ::registration2
)
