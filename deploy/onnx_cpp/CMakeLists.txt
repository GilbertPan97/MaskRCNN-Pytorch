cmake_minimum_required(VERSION 3.0)
project(OnnxDeploy CXX C)

set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "add a postfix, usually d on windows")
set(CMAKE_RELEASE_POSTFIX "" CACHE STRING "add a postfix, usually empty on windows")
set(CMAKE_RELWITHDEBINFO_POSTFIX "rd" CACHE STRING "add a postfix, usually empty on windows")
set(CMAKE_MINSIZEREL_POSTFIX "s" CACHE STRING "add a postfix, usually empty on windows")

# option(WITH_OPENCV "Run onnx model with opencv" OFF)
option(WITH_GPU "Inference with cuda" OFF)
option(BUILD_SHARED "Build librarys with dll output" OFF)
option(BUILD_EXAMPLE "Build example" ON)

# add different dependences of ORT
set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
if(UNIX)
    set(ORT_DIR "${ORT_DIR}-linux")
    message(STATUS "Onnxruntime running on Linux.")
elseif(WIN32)
    set(ORT_DIR "${ORT_DIR}-win")
    message(STATUS "Onnxruntime running on Windows.")
endif()

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    set(ORT_DIR "${ORT_DIR}-x64")
    message(STATUS "Onnxruntime running on x64 architecture.")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ORT_DIR "${ORT_DIR}-aarch64")
    message(STATUS "Onnxruntime running on aarch64 architecture.")
else()
    message(WARNING "Onnxruntime running on an unknown architecture.")
endif()

if(WITH_GPU)
    set(ORT_DIR "${ORT_DIR}-gpu-1.12.1")
    add_definitions(-DWITH_GPU=1)
else()
    set(ORT_DIR "${ORT_DIR}-1.12.1")
    add_definitions(-DWITH_GPU=0)
endif()
message(STATUS "Onnxruntime dir is: ${ORT_DIR}")

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
