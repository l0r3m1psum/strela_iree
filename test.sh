#!/bin/sh

set -e

paths=$(cat << 'EOF'
3rdparty/tiny/benchmark/training/anomaly_detection/trained_models/ad01_int8.tflite
3rdparty/tiny/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint.tflite
3rdparty/tiny/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro.tflite
3rdparty/tiny/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro_intio.tflite
3rdparty/tiny/benchmark/training/image_classification/trained_models/pretrainedResnet_large_int8.tflite
3rdparty/tiny/benchmark/training/image_classification/trained_models/pretrainedResnet_quant.tflite
3rdparty/tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model.tflite
3rdparty/tiny/benchmark/training/streaming_wakeword/trained_models/str_ww_ref_model.tflite
3rdparty/tiny/benchmark/training/visual_wake_words/trained_models/vww_96_int8.tflite
3rdparty/tiny/benchmark/training/anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant.tflite
3rdparty/tiny/benchmark/training/keyword_spotting/trained_models/kws_ref_model_float32.tflite
EOF
)

old_ifs=$IFS
IFS='
'
for path in $paths
do
	filename=$(basename "$path")
	stem=${filename%.tflite}

	tosa-converter-for-tflite "$path" --text >"${stem}.mlir"
	iree-opt --mlir-elide-elementsattrs-if-larger=16 "${stem}.mlir" >"${stem}_elided.mlir"

	echo $stem

	case "$stem" in
		*"Resnet"* | *"kws_ref_model"* | *"str_ww_ref_model"* | *"vww_96_int8"*)
			echo "IREE has a bug when dealing with dynamic (batch) dimension for ${stem}.mlir we edit it to make batch = 1" >&2
			ed -s "${stem}.mlir" <<- 'EOF'
			g/?x/s//1x/g
			w
			q
			EOF
			;;
	esac

	iree-compile \
		--iree-plugin=example2 \
		--iree-example2-fusion \
		--iree-hal-target-backends=llvm-cpu \
		--iree-llvmcpu-target-cpu=host \
		"${stem}.mlir" -o "${stem}.vmfb" >/dev/null 2>&1
done
IFS=$old_ifs

# --compile-to={preprocessing,flow,executable-targets,hal} are interesting
iree-compile --iree-plugin=example2 \
	--iree-strela-partition \
	--iree-hal-target-device=cpu=local \
	--iree-hal-local-target-device-backends=llvm-cpu \
	--iree-llvmcpu-target-triple=armv7a-none-linux-gnueabihf \
	--iree-hal-target-device=strela=strela \
	--iree-hal-default-device=cpu \
	 --compile-to=hal \
	strela_test.mlir

iree-compile --iree-plugin=example2 \
	--iree-strela-partition \
	--iree-hal-target-device=vulkan=vulkan \
	--iree-hal-target-device=strela=strela \
	--iree-hal-default-device=vulkan \
	 --compile-to=executable-targets \
	strela_test.mlir
