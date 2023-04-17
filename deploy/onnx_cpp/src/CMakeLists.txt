set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Each subdirectory is a library. Process them.
add_subdirectory(ImagePredict)

# Add demo subdirectory
set(BUILD_DEMO true)
if(BUILD_DEMO)
    add_subdirectory(demo)
endif()