project(tef)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)


include_directories(/usr/local/lib/python2.7/dist-packages/tensorflow_core/include)

link_directories(/usr/local/lib/python2.7/dist-packages/tensorflow_core)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g -lpthread -fPIC  -D_GLIBCXX_USE_CXX11_ABI=0")

set(SOURCE_FILES
    "ops/example_ops.cc"
    "kernels/zero_out_op.h"
    "kernels/zero_out_op.cc")

add_library(tef_core SHARED ${SOURCE_FILES})

target_link_libraries(tef_core libtensorflow_framework.so.2)


