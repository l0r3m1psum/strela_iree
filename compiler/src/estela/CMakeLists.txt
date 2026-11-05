# Include root for the plugin sources, so headers are referenced by their full
# path from compiler/src (e.g. "estela/StrelaIR/StrelaOps.h"). Every library
# here depends on this.
iree_cc_library(
  NAME
    defs
  INCLUDES
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>"
  PUBLIC
)

add_subdirectory(Utils)
add_subdirectory(StrelaIR)
add_subdirectory(Transforms)
