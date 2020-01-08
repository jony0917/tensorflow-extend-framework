project(tef)
cmake_minimum_required(VERSION 3.0.0 FATAL_ERROR)

set(PACKAGE_VERSION "0.0.0.1")

add_sub(tef/core)
add_sub(tef/python)
add_sub(examples)
add_sub(third_party)

