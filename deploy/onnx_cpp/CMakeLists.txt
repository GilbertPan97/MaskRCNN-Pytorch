cmake_minimum_required(VERSION 3.0)
project(OnnxDeploy CXX C)

# option(WITH_OPENCV "Run onnx model with opencv" OFF)
option(WITH_GPU "Inference with cuda" OFF)
option(WITH_ORT "Run onnx model inference with ONNXRuntime" ON)
if(WITH_GPU)
    add_definitions(-DWITH_GPU=1)
else()
    set(ORT_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime-win-x64-1.12.1")
    add_definitions(-DWITH_GPU=0)
endif()

# ProjectDir
set(THIRD_PARTY "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/version_1.0")
SET(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

if (NOT WIN32)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/demo)
else()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/onnx_deploy)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/onnx_deploy)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/onnx_deploy)
    add_definitions(-DONNX_DEPLOY)
endif()

# project
include_directories("${PROJECT_SOURCE_DIR}")

# add build src file
add_subdirectory(src)
