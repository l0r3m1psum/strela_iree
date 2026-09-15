# Include root for the plugin sources, so headers are referenced by their full
# path from compiler/src (e.g. "estela/Dialect/Strela/IR/StrelaOps.h"). Every
# library here depends on this.
iree_cc_library(
  NAME
    defs
  INCLUDES
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
  PUBLIC
)

add_subdirectory(Utils)
add_subdirectory(Dialect/Strela/IR)
add_subdirectory(Dialect/Strela/Transforms)
add_subdirectory(Dialect/Strela/Conversion)
add_subdirectory(Transforms/Linalg)
add_subdirectory(Transforms/Tosa)
add_subdirectory(Codegen/Strela)
