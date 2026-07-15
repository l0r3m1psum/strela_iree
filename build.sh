#!/bin/sh

set -e

cmake -G Ninja -B iree-build -S 3rdparty/iree \
    -DIREE_CMAKE_PLUGIN_PATHS="${PWD}/src" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_INSTALL_PREFIX="${PWD}/iree-build/install" \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON

cmake --build iree-build --target install

cmake -G Ninja -B iree-build-arm -S 3rdparty/iree \
    -DCMAKE_TOOLCHAIN_FILE="${PWD}/src/arm_toolchain.cmake" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_HOST_BIN_DIR="${PWD}/iree-build/install/bin" \
    -DIREE_BUILD_COMPILER=OFF

cmake --build iree-build-arm

cmake -G Ninja -S src -B build \
    -DIREERuntime_DIR="${PWD}/iree-build/lib/cmake/IREE"

cmake --build build

(
	cd 3rdparty/tosa-converter-for-tflite
	pip wheel .
	mv "tosa_converter_for_tflite-*.whl" ../..
	pip install "tosa_converter_for_tflite-*.whl"
)
