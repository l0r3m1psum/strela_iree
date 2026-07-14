cmake_minimum_required(VERSION 3.16)
project(CustomIREEModule C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

// TODO: this should link the same libbacktrace that IREE uses...
add_library(libbacktrace_libbacktrace INTERFACE)

find_package(IREERuntime REQUIRED)

add_library(custom_module SHARED custom_module.cc)

target_link_libraries(custom_module
    PRIVATE
    iree_base_base
    iree_hal_hal
    iree_modules_hal_types
    iree_vm_dynamic_api
    iree_vm_vm
)
