# set(CMAKE_SYSTEM_NAME Linux)
# set(CMAKE_SYSTEM_PROCESSOR arm)
#
# set(triple arm-linux-gnueabihf)
#
# set(CMAKE_C_COMPILER clang)
# set(CMAKE_C_COMPILER_TARGET ${triple})
# set(CMAKE_CXX_COMPILER clang++)
# set(CMAKE_CXX_COMPILER_TARGET ${triple})
#
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-c++11-narrowing")
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-shift-count-overflow")

set(CMAKE_SYSTEM_NAME  Linux)
set(CMAKE_SYSTEM_PROCESSOR  arm)
set(CMAKE_C_COMPILER  arm-linux-gnueabihf-gcc-13)
set(CMAKE_CXX_COMPILER  arm-linux-gnueabihf-g++-13)
