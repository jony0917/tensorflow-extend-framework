project(tef)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

execute_process(COMMAND python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TENSORFLOW_CFLAGS)
string(REGEX REPLACE "\n$" "" TENSORFLOW_CFLAGS ${TENSORFLOW_CFLAGS})
message(STATUS "TENSORFLOW_CFLAGS=${TENSORFLOW_CFLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -D GOOGLE_CUDA=1 ${TENSORFLOW_CFLAGS} -O3")


execute_process(COMMAND python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TENSORFLOW_LFLAGS)
string(REGEX REPLACE "\n$" "" TENSORFLOW_LFLAGS ${TENSORFLOW_LFLAGS})
message(STATUS "TENSORFLOW_LFLAGS=${TENSORFLOW_LFLAGS}")

find_package(CUDA)

############################################################################
# Gencode arguments
set(SMS 30 35 37 50 52 60 70)
foreach(sm ${SMS})
    set(GENCODE_FLAGS ${GENCODE_FLAGS} "-gencode arch=compute_${sm},code=sm_${sm}")
endforeach()

set(HIGHEST_SM 70)
set(GENCODE_FLAGS ${GENCODE_FLAGS} "-gencode arch=compute_${HIGHEST_SM},code=compute_${HIGHEST_SM}")
message(STATUS "GENCODE_FLAGS=${GENCODE_FLAGS}")


############################################################################
# compile targets
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} ${GENCODE_FLAGS} "-D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC --expt-relaxed-constexpr -DNDEBUG -std=c++11")
cuda_compile(cuda_op_objects
    "kernels/example_op.cu")
message(STATUS "cuda_op_objects=${cuda_op_objects}")


set(SOURCE_FILES
    "ops/example_ops.cc"
    "ops/ps_ops.cc"
    "kernels/zero_out_op.h"
    "kernels/zero_out_op.cc"
    "kernels/example_op.h"
    "kernels/example_op.cc"
    "kernels/ps_hash_pull_op.h"
    "kernels/ps_hash_pull_op.cc"
    "kernels/ps_hash_push_op.h"
    "kernels/ps_hash_push_op.cc"
    "kernels/ps_pull_op.h"
    "kernels/ps_pull_op.cc"
    "kernels/ps_push_op.h"
    "kernels/ps_push_op.cc"
    "kernels/ps_sparse_pull_op.h"
    "kernels/ps_sparse_pull_op.cc"
    "kernels/ps_sparse_push_op.h"
    "kernels/ps_sparse_push_op.cc"
    "kernels/ps_client/ps_client.h"
    "kernels/ps_client/ps_client_dummy.h"
    "kernels/ps_client/ps_client_dummy.cc"
    "kernels/ps_client/ps_client_factory.h"
    "kernels/ps_client/ps_client_factory.cc")

add_library(tef_core SHARED ${SOURCE_FILES} ${cuda_op_objects})
target_link_libraries(tef_core ${TENSORFLOW_LFLAGS})
add_custom_command(TARGET tef_core
                   POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:tef_core>  "${CMAKE_CURRENT_SOURCE_DIR}/../python/tef/pywrap/"
                   COMMENT "coping output of tef_core task")