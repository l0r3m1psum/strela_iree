# libstrela from 3rdparty/strela_pynq. Named strela_pynq so it does not read as
# the HAL driver target, which is also called strela.
add_library(strela_pynq INTERFACE IMPORTED)
set_target_properties(strela_pynq PROPERTIES
  INTERFACE_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../3rdparty/strela_pynq/lib"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../3rdparty/strela_pynq/include"
  INTERFACE_LINK_LIBRARIES "-lstrela"
  SKIP_BUILD_RPATH TRUE
)

# Unity build: strela_driver_module.c #includes the rest of the driver.
iree_cc_library(
  NAME
    strela
  SRCS
    "strela_driver_module.c"
  DEPS
    iree::hal
    iree::hal::utils::deferred_work_queue
    iree::hal::utils::file_transfer
    iree::hal::utils::queue_emulation
    iree::hal::utils::queue_host_call_emulation
    strela_pynq
  PUBLIC
)
