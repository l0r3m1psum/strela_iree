# Include root for the plugin sources, so headers are referenced by their full
# path from src/ (e.g. "strela/Dialect/Strela/StrelaOps.h"). Every library here
# depends on this.
iree_cc_library(
  NAME
    defs
  INCLUDES
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
  PUBLIC
)

add_subdirectory(Dialect/Strela)
add_subdirectory(Transforms/Linalg)
add_subdirectory(Transforms/Strela)
add_subdirectory(Transforms/Tosa)
add_subdirectory(Target)
