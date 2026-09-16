#!/bin/sh

set -e

# --compile-to={preprocessing,flow,executable-targets,hal} are interesting
iree-compile --iree-plugin=estela \
--iree-estela-partition \
--iree-hal-target-device=cpu=local \
--iree-hal-local-target-device-backends=llvm-cpu \
--iree-llvmcpu-target-triple=armv7a-none-linux-gnueabihf \
--iree-hal-target-device=strela=strela \
--iree-hal-default-device=cpu \
 --compile-to=hal \
strela_test.mlir

iree-compile --iree-plugin=estela \
--iree-estela-partition \
--iree-hal-target-device=vulkan=vulkan \
--iree-hal-target-device=strela=strela \
--iree-hal-default-device=vulkan \
 --compile-to=executable-targets \
strela_test.mlir

iree-compile 3rdparty/iree/tools/test/iree-run-module-multi.mlir \
--iree-execution-model=async-external \
--iree-hal-target-device=device_a=local[0] \
--iree-hal-target-device=device_b=local[1] \
--iree-hal-local-target-device-backends=vmvx \
--compile-to=flow

iree-compile 3rdparty/iree/tools/test/iree-run-module-multi.mlir \
--iree-execution-model=async-external \
--iree-hal-target-device=device_a=local[0] \
--iree-hal-target-device=device_b=vulkan \
--iree-hal-local-target-device-backends=vmvx \
--compile-to=executable-targets

iree-compile strela_test_transfer.mlir \
--iree-plugin=estela \
--iree-execution-model=async-external \
--iree-hal-target-device=device_a=local[0] \
--iree-hal-target-device=device_b=strela \
--iree-hal-default-device=device_a \
--iree-hal-local-target-device-backends=vmvx \
--compile-to=executable-targets
