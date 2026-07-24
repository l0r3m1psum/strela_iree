add_library(strela INTERFACE IMPORTED)
set_target_properties(strela PROPERTIES
  INTERFACE_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/strela_pynq/lib"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../3rdparty/strela_pynq/include"
  INTERFACE_LINK_LIBRARIES "-lstrela"
  SKIP_BUILD_RPATH TRUE
)

iree_cc_library(
  NAME
    my_custom_hal_driver
  SRCS
    "hal_driver.c"
  DEPS
    iree::hal
    strela
  PUBLIC
)
