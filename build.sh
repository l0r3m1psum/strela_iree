#!/bin/sh

set -e

cmake -G Ninja -B iree-build -S 3rdparty/iree \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DIREE_ENABLE_SPLIT_DWARF=ON \
    -DIREE_ENABLE_THIN_ARCHIVES=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON

cmake --build iree-build

(
	cd 3rdparty/tosa-converter-for-tflite
	pip wheel .
	mv *.whl ../..
)
