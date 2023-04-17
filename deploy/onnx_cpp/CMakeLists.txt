cmake_minimum_required(VERSION 3.0)
project(OnnxDeploy CXX C)

if (WIN32)
  option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static." ON)
else()
  option(WITH_STATIC_LIB "Compile demo with static/shared library, default use static." OFF)
endif()

# option(WITH_OPENCV "Run onnx model with opencv" OFF)
option(WITH_GPU "Inference with cuda" OFF)
option(WITH_ORT "Run onnx model inference with ONNXRuntime" ON)
add_definitions(-DWITH_GPU=${WITH_GPU})

# Third party dir
set(THIRD_PARTY "./third_party")

# ProjectDir
SET(PROJECT_ROOT_DIR  ${CMAKE_CURRENT_SOURCE_DIR})

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

# project third party
if(WITH_ORT)
  if(WIN32)
    if(WITH_GPU)
      set(ORT_DIRS "${PROJECT_ROOT_DIR}/third_party/onnxruntime-win-x64-gpu-1.12.1")
    else()
      set(ORT_DIRS "${PROJECT_ROOT_DIR}/third_party/onnxruntime-win-x64-1.12.1")
    endif()
    file (GLOB ORT_LIBS ${ORT_DIRS}/lib/*.lib)
  else()
    if(WITH_GPU)
      set(ORT_DIRS "${PROJECT_ROOT_DIR}/third_party/onnxruntime-linux-x64-gpu-1.12.1")
    else()
      set(ORT_DIRS "${PROJECT_ROOT_DIR}/third_party/onnxruntime-linux-x64-1.12.1")
    endif()
    file(GLOB ORT_LIBS ${ORT_DIRS}/lib/*.so)
  endif()
  set(ORT_INCLUDE_DIRS "${ORT_DIRS}/include")
  message("OnnxRuntime_LIBS: ${ORT_LIBS}")
  include_directories(${ORT_INCLUDE_DIRS})
  set(DEPS ${DEPS} ${ORT_LIBS})
endif()

# opencv is necessory
if (WIN32)
  find_package(OpenCV REQUIRED HINTS ${THIRD_PARTY} NO_DEFAULT_PATH)
  # unset(OpenCV_DIR CACHE)
else ()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
    set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    #find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR} NO_DEFAULT_PATH)
  endif()
endif ()
message(STATUS "OpenCV dir: ${OpenCV_DIR}")
message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

set(DEPS ${DEPS} ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

# add build src file
add_subdirectory(src)
