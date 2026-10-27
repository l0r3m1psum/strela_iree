#!/bin/sh

set -e

if [ -z "$1" ]
then
	echo "Missing IP address" >&2
	exit 1
fi
ipaddr="$1"

pynq_flags="--iree-llvmcpu-target-triple=armv7a-none-linux-gnueabihf --iree-llvmcpu-target-cpu=generic --iree-llvmcpu-target-cpu-features=+vfp3,+neon"

tosa-converter-for-tflite \
3rdparty/tiny/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite \
-o ad01_int8.mlirbc
# The converter prints TOSA in the syntax of its own, older, MLIR; bytecode is
# versioned, so let iree-opt print it in the current one.
iree-opt ad01_int8.mlirbc -o ad01_int8.mlir

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
	--iree-plugin=estela \
	--iree-hal-target-device=local \
	--iree-hal-local-target-device-backends=vmvx \
	ad01_int8.mlir -o ad01_int8_vmx.vmfb

iree-compile \
	--iree-plugin=estela \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	ad01_int8.mlir -o ad01_int8_armv7a.vmfb

iree-compile \
	--iree-plugin=estela \
	--iree-estela-fusion \
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
	--iree-plugin=estela \
	--iree-estela-fusion \
	--iree-hal-target-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	matmul.mlir -o matmul_armv7a_strela.vmfb

# tests/transfer.mlir declares unified memory, which binds the CPU's own buffers
# to STRELA dispatches; the HAL now rejects that instead of letting the CGRA DMA
# from memory it cannot reach. tests/runtime/strela_ops.mlir has the topology
# STRELA really has.
iree-compile \
	--iree-plugin=estela \
	--iree-execution-model=async-external \
	--iree-hal-target-device=device_a=local[0] \
	--iree-hal-target-device=device_b=strela \
	--iree-hal-default-device=device_a \
	--iree-hal-local-target-device-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	tests/runtime/strela_ops.mlir -o strela_ops_armv7a.vmfb

iree-compile \
	--iree-plugin=estela \
	--iree-execution-model=async-external \
	--iree-hal-target-device=device_a=local[0] \
	--iree-hal-target-device=device_b=strela \
	--iree-hal-default-device=device_a \
	--iree-hal-local-target-device-backends=llvm-cpu \
	--iree-llvmcpu-link-embedded=false \
	$pynq_flags \
	tests/runtime/strela_transfer.mlir -o strela_transfer_armv7a.vmfb

scp iree-build-arm/tools/iree-run-module \
	3rdparty/strela_pynq/build-pynq/lib/libstrela.so \
	build-arm/libcustom_module.so \
	simple_abs_vmx.vmfb \
	simple_abs_armv7a.vmfb \
	ad01_int8_vmx.vmfb \
	ad01_int8_armv7a.vmfb \
	ad01_int8_armv7a_strela.vmfb \
	matmul_armv7a.vmfb \
	matmul_armv7a_strela.vmfb \
	strela_ops_armv7a.vmfb \
	strela_transfer_armv7a.vmfb \
	"root@${ipaddr}:/root"
ssh -t "root@${ipaddr}" << EOF
	set -xe
	# The board also has an old libstrela in /usr/local/lib, which ldconfig
	# would find first.
	export LD_LIBRARY_PATH=/root
	# TODO(Diego): I guess I have to install there the new libstrela when
	# downloading files to the board in the strela_pynq project.
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
	# ./iree-run-module \
	# 	--module=./libcustom_module.so \
	# 	--module=ad01_int8_armv7a_strela.vmfb \
	# 	--function=main \
	# 	--input="2x640xi8=0" \
	# 	--device=local-sync
	./iree-run-module \
		--module=matmul_armv7a.vmfb \
		--function=main \
		--input="1x1x1x9xi8=2" \
		--input="9x1x1x9xi8=2" \
		--device=local-sync
	# ./iree-run-module \
	# 	--module=./libcustom_module.so \
	# 	--module=matmul_armv7a_strela.vmfb \
	# 	--function=main \
	# 	--input="1x1x1x9xi8=2" \
	# 	--input="9x1x1x9xi8=2" \
	# 	--device=local-sync

	./iree-run-module --list_drivers 2>&1
	./iree-run-module --list_devices=strela 2>&1
	# --device_lead_allocator=1 puts buffers both devices touch in STRELA
	# memory, the only memory the CGRA can reach.
	./iree-run-module \
		--module=strela_ops_armv7a.vmfb \
		--function=add \
		--input="16xi32=1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16" \
		--input="16xi32=10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160" \
		--expected_output="16xi32=11 22 33 44 55 66 77 88 99 110 121 132 143 154 165 176" \
		--device=local-sync \
		--device=strela \
		--device_lead_allocator=1
	./iree-run-module \
		--module=strela_ops_armv7a.vmfb \
		--function=relu \
		--input="16xi32=-1 2 -3 4 -5 6 -7 8 -9 10 -11 12 -13 14 -15 16" \
		--expected_output="16xi32=0 2 0 4 0 6 0 8 0 10 0 12 0 14 0 16" \
		--device=local-sync \
		--device=strela \
		--device_lead_allocator=1
	# Expected result: min(relu(1 + 2) + relu(3 - 10)) = 3. Not checked with
	# --expected_output: iree-run-module fails a scalar i32 against i32 as a
	# "variant types mismatch".
	./iree-run-module \
		--module=strela_transfer_armv7a.vmfb \
		--function=main \
		--input="8x9xi32=1" \
		--input="8x9xi32=2" \
		--input="8x9xi32=3" \
		--input="8x9xi32=-10" \
		--device=local-sync \
		--device=strela \
		--device_lead_allocator=1
EOF
