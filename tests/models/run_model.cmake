# Convert one .tflite model, compile it, and run it with all-zero inputs.
#
# Run in CMake script mode so there is no shell involved and the test works on
# any platform cmake does:
#
#   cmake -DTFLITE=... -DOUTDIR=... -DCONVERTER=... -DIREE_OPT=...
#         -DIREE_COMPILE=... -DIREE_RUN_MODULE=... -DINPUT=1x640xi8
#         [-DPIN_BATCH=ON]
#         -P run_model.cmake
#
# Artifacts go to <OUTDIR>/<model>/ so nothing is written into the source tree:
#
#   <model>.mlirbc       TOSA input from the converter, as bytecode
#   <model>.mlir         the same, printed as text by iree-opt
#   <model>.static.mlir  only when PIN_BATCH is set
#   <model>.vmfb         the module that was run
#   run.txt              iree-run-module output, including the result tensors
#
# To try a different compilation flow (e.g. flow.tensor.transfer onto a STRELA
# device) edit COMPILE_FLAGS / RUN_FLAGS below. Nothing else here depends on how
# the model is compiled.

cmake_minimum_required(VERSION 3.20)

set(COMPILE_FLAGS
  --iree-plugin=estela
  --iree-hal-target-backends=llvm-cpu
  --iree-llvmcpu-target-cpu=host
)
set(RUN_FLAGS --device=local-sync)

# Runs one step and, on failure, prints what the tool actually said so the ctest
# output explains itself instead of just reporting a non-zero exit.
function(estela_run_step description)
  execute_process(
    COMMAND ${ARGN}
    RESULT_VARIABLE _rc
    OUTPUT_VARIABLE _out
    ERROR_VARIABLE _err
  )
  if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "${description} failed (exit ${_rc}):\n${_out}${_err}")
  endif()
endfunction()

get_filename_component(_stem "${TFLITE}" NAME_WE)
set(_dir "${OUTDIR}/${_stem}")
file(MAKE_DIRECTORY "${_dir}")
message(STATUS "artifacts: ${_dir}")

# 1. .tflite -> TOSA MLIR. The converter is built against its own MLIR, whose
#    text syntax for TOSA ops can be older than IREE's. Bytecode is versioned and
#    readable by newer MLIR, so it is printed as text by iree-opt instead.
estela_run_step("conversion"
  "${CONVERTER}" "${TFLITE}" -o "${_dir}/${_stem}.mlirbc")
estela_run_step("printing"
  "${IREE_OPT}" "${_dir}/${_stem}.mlirbc" -o "${_dir}/${_stem}.mlir")
set(_source "${_dir}/${_stem}.mlir")

# 2. The converter leaves the batch dimension dynamic. An unbounded dimension
#    makes the llvm-cpu stack-allocation check compute an absurd size (see the
#    explanation in below), and the resulting module cannot be executed at all
#    even if that check is disabled or vmvx is used instead -- it fails with
#      OUT_OF_RANGE; indirect buffer reference slot 0 is out range of the
#      declared binding capacity of the command buffer
#    Pinning the batch to 1 is correct for these single-inference models and is
#    what makes them runnable. Which models need it is decided in CMakeLists.txt.
if(PIN_BATCH)
  file(READ "${_source}" _ir)
  string(REPLACE "?x" "1x" _ir "${_ir}")
  set(_source "${_dir}/${_stem}.static.mlir")
  file(WRITE "${_source}" "${_ir}")
  message(STATUS "batch dimension pinned to 1")
endif()

# 3. Compile.
estela_run_step("compilation"
  "${IREE_COMPILE}" ${COMPILE_FLAGS} "${_source}" -o "${_dir}/${_stem}.vmfb")

# 4. Run with an all-zero input. A single value is splatted across the buffer.
message(STATUS "running: --input=${INPUT}=0")
execute_process(
  COMMAND "${IREE_RUN_MODULE}" ${RUN_FLAGS}
          "--module=${_dir}/${_stem}.vmfb"
          "--function=main"
          "--input=${INPUT}=0"
  RESULT_VARIABLE _rc
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _err
)
file(WRITE "${_dir}/run.txt" "${_out}${_err}")
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "execution failed (exit ${_rc}), see ${_dir}/run.txt:\n${_out}${_err}")
endif()
message(STATUS "result in ${_dir}/run.txt:\n${_out}")

# EXPLANATION
#
# Some models cannot be compiled for llvm-cpu and fall back to vmvx.
#
# How it fails. LLVMCPUCheckIRBeforeLLVMConversion rejects the dispatch:
#
#     'func.func' op exceeded stack allocation limit of 32768 bytes
#     for function. Got 216172782113783808 bytes
#
# Why. tosa-converter-for-tflite leaves the batch dimension dynamic
# (tensor<?x49x10x1xi8>), and some ops bufferize to a memref.alloca whose
# size depends on it. That pass sums every alloca in a dispatch, asking
# ScalableValueBoundsConstraintSet for an upper bound on each dynamic
# extent. An unbounded batch dimension has no real bound, so the analysis
# returns the top of the index range and the total lands on a multiple of
# 2^56, against a 32 KB max_stack_allocation_size. Nothing would actually
# allocate that much: at run time the alloca is sized by the real
# dimension. The check just cannot prove a bound at compile time.
#
# So the models are fine and the IR is fine. vmvx does not run this check,
# which is why compiling them for vmvx works and is what happens below.
# Note it is not simply "dynamic shapes fail" either -- kws_ref_model_float32
# has dynamic shapes throughout and compiles for llvm-cpu, because none of
# them reach a stack temporary.
#
# Two other ways to fix it, neither used here:
#
#   1. Pass --iree-llvmcpu-fail-on-out-of-bounds-stack-allocation=false.
#      Compiles for llvm-cpu with the model untouched. Costs nothing, but
#      disarms the check for every dispatch in the program, including any
#      where an oversized allocation would be a genuine bug.
#
#   2. Specialize the batch dimension before input conversion: rewrite the
#      entry argument to tensor<1x...> and insert a tensor.cast back to the
#      dynamic type. IREE's own shape propagation carries the static shape
#      through the body, every alloca becomes static, and the check passes.
