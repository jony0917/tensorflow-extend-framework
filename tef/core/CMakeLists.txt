project(tef)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)


execute_process(COMMAND python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_compile_flags()))" OUTPUT_VARIABLE TENSORFLOW_CFLAGS)
string(REGEX REPLACE "\n$" "" TENSORFLOW_CFLAGS ${TENSORFLOW_CFLAGS})
message(STATUS "TENSORFLOW_CFLAGS=${TENSORFLOW_CFLAGS}")

execute_process(COMMAND python -c "import tensorflow as tf; print(\" \".join(tf.sysconfig.get_link_flags()))" OUTPUT_VARIABLE TENSORFLOW_LFLAGS)
string(REGEX REPLACE "\n$" "" TENSORFLOW_LFLAGS ${TENSORFLOW_LFLAGS})
message(STATUS "TENSORFLOW_LFLAGS=${TENSORFLOW_LFLAGS}")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g -lpthread -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 ${TENSORFLOW_CFLAGS}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${TENSORFLOW_LFLAGS}")

set(SOURCE_FILES
    "ops/example_ops.cc"
    "kernels/zero_out_op.h"
    "kernels/zero_out_op.cc")

add_library(tef_core SHARED ${SOURCE_FILES})
