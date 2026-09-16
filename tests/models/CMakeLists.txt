# Compile-and-run smoke tests for the mlperf-tiny models.
#
# Artifacts land in ESTELA_MODEL_OUTPUT_DIR (default: <build>/estela-models), one
# subdirectory per model.
#
# Run them with:
#   ctest --test-dir iree-build -R estela/models --output-on-failure

if(NOT IREE_BUILD_TESTS)
  return()
endif()

find_program(ESTELA_TOSA_CONVERTER NAMES tosa-converter-for-tflite)
if(NOT ESTELA_TOSA_CONVERTER)
  message(STATUS "Estela model tests ignored -- tosa-converter-for-tflite not found -- remeber to source setup_env.sh")
  return()
endif()

set(ESTELA_MODEL_OUTPUT_DIR "${CMAKE_BINARY_DIR}/estela-models" CACHE PATH
  "Where the model tests write their .mlir, .vmfb and run output")

# <path under 3rdparty/tiny/benchmark/training> | <input shape> | <pin batch to 1>
#
# The models with PIN_BATCH=ON are the ones the converter leaves with a dynamic
# batch dimension that reaches a stack temporary; see run_model.cmake for why
# they cannot be compiled or run as generated.
set(_models
  "anomaly_detection/trained_models/ad01_int8.tflite|1x640xi8|OFF"
  "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant.tflite|1x640xf32|OFF"
  "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint.tflite|1x640xf32|OFF"
  "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro.tflite|1x640xf32|OFF"
  "anomaly_detection/trained_models/ToyCar/baseline_tf23/model/model_ToyCar_quant_fullint_micro_intio.tflite|1x640xi8|OFF"
  "keyword_spotting/trained_models/kws_ref_model_float32.tflite|1x49x10x1xf32|OFF"
  "image_classification/trained_models/pretrainedResnet_large_int8.tflite|1x32x32x3xi8|ON"
  "image_classification/trained_models/pretrainedResnet_quant.tflite|1x32x32x3xi8|ON"
  "keyword_spotting/trained_models/kws_ref_model.tflite|1x49x10x1xi8|ON"
  "streaming_wakeword/trained_models/str_ww_ref_model.tflite|1x30x1x40xi8|ON"
  "visual_wake_words/trained_models/vww_96_int8.tflite|1x96x96x3xi8|ON"
)

set(_tiny "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty/tiny/benchmark/training")

foreach(_entry IN LISTS _models)
  string(REPLACE "|" ";" _fields "${_entry}")
  list(GET _fields 0 _path)
  list(GET _fields 1 _input)
  list(GET _fields 2 _pin)
  get_filename_component(_name "${_path}" NAME_WE)

  set(_test "estela/models/${_name}")
  add_test(
    NAME "${_test}"
    COMMAND "${CMAKE_COMMAND}"
            "-DTFLITE=${_tiny}/${_path}"
            "-DOUTDIR=${ESTELA_MODEL_OUTPUT_DIR}"
            "-DCONVERTER=${ESTELA_TOSA_CONVERTER}"
            "-DIREE_COMPILE=$<TARGET_FILE:iree-compile>"
            "-DIREE_RUN_MODULE=$<TARGET_FILE:iree-run-module>"
            "-DINPUT=${_input}"
            "-DPIN_BATCH=${_pin}"
            -P "${CMAKE_CURRENT_LIST_DIR}/run_model.cmake"
  )
  set_tests_properties("${_test}" PROPERTIES LABELS "estela;models" TIMEOUT 300)
endforeach()
