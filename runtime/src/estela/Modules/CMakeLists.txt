cmake_minimum_required(VERSION 3.16)
project(EstelaModules C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

# TODO: this should link the same libbacktrace that IREE uses...
add_library(libbacktrace_libbacktrace INTERFACE)

find_package(IREERuntime REQUIRED)

# NOTE: the library and the VM module it registers are still called
# "custom_module"/"custom" -- that name is part of the module ABI the .mlir
# imports refer to, so renaming it is a separate change from this refactor.
add_library(custom_module SHARED strela_module.cc)

target_link_libraries(custom_module
    PRIVATE
    iree_base_base
    iree_hal_hal
    iree_modules_hal_types
    iree_vm_dynamic_api
    iree_vm_vm
)

string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" TARGET_ARCH)
if(TARGET_ARCH MATCHES "^arm")
    add_library(strela_pynq INTERFACE IMPORTED)
    set_target_properties(strela_pynq PROPERTIES
        INTERFACE_LINK_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/strela_pynq/lib"
    )
    set_target_properties(strela_pynq PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/../../../../3rdparty/strela_pynq/include"
    )
    set_target_properties(strela_pynq PROPERTIES INTERFACE_LINK_LIBRARIES "-lstrela")
    set_target_properties(custom_module PROPERTIES SKIP_BUILD_RPATH TRUE)
    target_link_libraries(custom_module PRIVATE strela_pynq)
endif()
