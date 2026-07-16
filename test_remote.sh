#!/bin/sh

set -e

if [ -z "$1" ]
then
	echo "Missing IP address" >&2
	exit 1
fi
ipaddr="$1"

iree-compile \
	--iree-hal-target-device=local \
	--iree-hal-local-target-device-backends=vmvx \
	3rdparty/iree/samples/models/simple_abs.mlir -o simple_abs.vmfb

iree-compile \
	--iree-plugin=example2 \
	--iree-hal-target-device=local \
	--iree-hal-local-target-device-backends=vmvx \
	ad01_int8.mlir -o ad01_int8.vmfb

scp iree-build-arm/tools/iree-run-module simple_abs.vmfb ad01_int8.vmfb "root@${ipaddr}:/root"
ssh -t "root@${ipaddr}" << EOF
	set -xe
	./iree-run-module \
		--module=simple_abs.vmfb \
		--function=abs \
		--input="f32=-1" \
		--device=local-sync
	./iree-run-module \
		--module=ad01_int8.vmfb \
		--function=main \
		--input="2x640xi8=0" \
		--device=local-sync
EOF
