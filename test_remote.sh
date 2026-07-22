#!/bin/sh

set -e

if [ -z "$1" ]
then
	echo "Missing IP address" >&2
	exit 1
fi
ipaddr="$1"

pynq_flags="--iree-llvmcpu-target-triple=armv7a-none-linux-gnueabihf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-cpu-features=+vfp3,+neon"

iree-compile \
	--iree-hal-target-device=local \
	--iree-hal-local-target-device-backends=vmvx \
	3rdparty/iree/samples/models/simple_abs.mlir -o simple_abs_vmx.vmfb

iree-compile \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	3rdparty/iree/samples/models/simple_abs.mlir -o simple_abs_armv7a.vmfb

iree-compile \
	--iree-plugin=example2 \
	--iree-hal-target-device=local \
	--iree-hal-local-target-device-backends=vmvx \
	ad01_int8.mlir -o ad01_int8_vmx.vmfb

iree-compile \
	--iree-plugin=example2 \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	ad01_int8.mlir -o ad01_int8_armv7a.vmfb

iree-compile \
	--iree-plugin=example2 \
	--iree-example2-fusion \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	ad01_int8.mlir -o ad01_int8_armv7a_strela.vmfb

iree-compile \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	matmul.mlir -o matmul_armv7a.vmfb

iree-compile \
	--iree-plugin=example2 \
	--iree-example2-fusion \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	matmul.mlir -o matmul_armv7a_strela.vmfb

scp iree-build-arm/tools/iree-run-module \
	build-arm/libcustom_module.so \
	simple_abs_vmx.vmfb \
	simple_abs_armv7a.vmfb \
	ad01_int8_vmx.vmfb \
	ad01_int8_armv7a.vmfb \
	ad01_int8_armv7a_strela.vmfb \
	matmul_armv7a.vmfb \
	matmul_armv7a_strela.vmfb \
	"root@${ipaddr}:/root"
ssh -t "root@${ipaddr}" << EOF
	set -xe
	./iree-run-module \
		--module=simple_abs_vmx.vmfb \
		--function=abs \
		--input="f32=-1" \
		--device=local-sync
	./iree-run-module \
		--module=simple_abs_armv7a.vmfb \
		--function=abs \
		--input="f32=-1" \
		--device=local-sync
	./iree-run-module \
		--module=ad01_int8_vmx.vmfb \
		--function=main \
		--input="2x640xi8=0" \
		--device=local-sync
	./iree-run-module \
		--module=ad01_int8_armv7a.vmfb \
		--function=main \
		--input="2x640xi8=0" \
		--device=local-sync
	./iree-run-module \
		--module=./libcustom_module.so \
		--module=ad01_int8_armv7a_strela.vmfb \
		--function=main \
		--input="2x640xi8=0" \
		--device=local-sync
	./iree-run-module \
		--module=matmul_armv7a.vmfb \
		--function=main \
		--input="9x1x1x9xi8=2" \
		--input="9x1x1x9xi8=2" \
		--device=local-sync
	./iree-run-module \
		--module=./libcustom_module.so \
		--module=matmul_armv7a_strela.vmfb \
		--function=main \
		--input="9x1x1x9xi8=2" \
		--input="9x1x1x9xi8=2" \
		--device=local-sync
EOF
