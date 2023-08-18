set(LIBS_NAME ModelPredict)

# opencv is necessory
if (WIN32)
  find_package(OpenCV REQUIRED HINTS ${THIRD_PARTY}/opencv-win* NO_DEFAULT_PATH)
else ()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
    find_package(OpenCV REQUIRED)
    # set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    # file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    find_package(OpenCV REQUIRED HINTS ${THIRD_PARTY}/opencv-linux-4.6.0 NO_DEFAULT_PATH)
  endif()
endif ()
# message(STATUS "OpenCV dir: ${OpenCV_DIR}")
# message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# add ort dependence
if(WIN32)
    file(GLOB ORT_LIBS ${ORT_DIR}/lib/*.lib)
    file(GLOB ORT_DLLS ${ORT_DIR}/lib/*.dll)
else()
    file(GLOB ORT_LIBS ${ORT_DIR}/lib/*)
endif()

set(ORT_INCLUDE_DIRS "${ORT_DIR}/include")

# config ModelPredict lib build
set(SRC_FILES ModelPredict.cpp)

set(HEADER_FILES 
    ModelPredict.h
    ModelPredictGlobal.h
)

source_group("Source Files" FILES ${SRC_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(BUILD_SHARED)
  add_library(${LIBS_NAME} SHARED ${SRC_FILES} ${HEADER_FILES})
  target_compile_definitions(${LIBS_NAME} PRIVATE MP_SHARED_EXPORT)
else()
  add_library(${LIBS_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})
  target_compile_definitions(${LIBS_NAME} PUBLIC MP_STATIC)  
endif()

target_link_libraries(${LIBS_NAME} PUBLIC 
                      ${OpenCV_LIBS} ${ORT_LIBS})

target_include_directories(${LIBS_NAME} PUBLIC 
                      ${OpenCV_INCLUDE_DIRS} 
                      ${ORT_INCLUDE_DIRS})

set_target_properties(${LIBS_NAME} PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
                      LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
                      RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

# Add demo subdirectory
if(BUILD_EXAMPLE)
  add_subdirectory(demo)
endif()