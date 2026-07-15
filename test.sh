#!/bin/sh

set -e

if [ -z "$1" ]
then
	echo "Missing path to .tflite file" >&2
	exit 1
fi

model_path="$1"
tosa-converter-for-tflite "$model_path" --text >ad01_int8.mlir

iree-opt --mlir-elide-elementsattrs-if-larger=16 ad01_int8.mlir

# --iree-preprocessing-convert-1x1-filter-conv2d-to-matmul
# --mlir-print-ir-after=iree-vm-conversion \
iree-compile \
	--iree-plugin=example2 \
	--iree-example2-fusion \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-target-cpu=host \
	ad01_int8.mlir -o ad01_int8.vmfb

iree-run-module \
	--module=build/libcustom_module.so \
	--module=ad01_int8.vmfb \
	--function=main \
	--input="2x640xi8=0"
