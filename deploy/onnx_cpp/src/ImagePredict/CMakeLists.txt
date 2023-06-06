set(SUBSYS_NAME ImagePredict)

# opencv is necessory
if (WIN32)
  find_package(OpenCV REQUIRED HINTS ${THIRD_PARTY} NO_DEFAULT_PATH)
else ()
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64") # x86_64 aarch64
    find_package(OpenCV REQUIRED)
    # set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
    # file(GLOB OpenCV_LIBS /usr/lib/aarch64-linux-gnu/libopencv_*${CMAKE_SHARED_LIBRARY_SUFFIX})
  else()
    find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR} NO_DEFAULT_PATH)
  endif()
endif ()
# message(STATUS "OpenCV dir: ${OpenCV_DIR}")
# message(STATUS "OpenCV include dir: ${OpenCV_INCLUDE_DIRS}")
# message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# add ort dependence
if(WIN32)
  file(GLOB ORT_LIBS ${ORT_DIRS}/lib/*.lib)
else()
file(GLOB ORT_LIBS ${ORT_DIRS}/lib/*.so)
endif()
set(ORT_INCLUDE_DIRS "${ORT_DIRS}/include")

# config ImagePredict lib build
set(SRC_FILES ImagePredict.cpp)
set(HEADER_FILES ImagePredict.h)

source_group("Source Files" FILES ${SRC_FILES})
source_group("Header Files" FILES ${HEADER_FILES})

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_library(${SUBSYS_NAME} STATIC ${SRC_FILES} ${HEADER_FILES})

target_link_libraries(${SUBSYS_NAME} PUBLIC 
                      ${OpenCV_LIBS} ${ORT_LIBS})

target_include_directories(${SUBSYS_NAME} PUBLIC 
                      ${OpenCV_INCLUDE_DIRS} 
                      ${ORT_INCLUDE_DIRS})

set_target_properties(${SUBSYS_NAME} PROPERTIES
                      ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
                      LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
                      RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

# Add demo subdirectory
set(BUILD_DEMO true)
if(BUILD_DEMO)
    add_subdirectory(demo)
endif()