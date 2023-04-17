# generate project target
add_executable(demo_ip demo_main.cpp ../ImagePredict/ImagePredict.cpp)
target_link_libraries(demo_ip ${DEPS})
message(STATUS "SRC directory is: ${SRC_DIR}")
include_directories(${SRC_DIR})

# The onnxruntime DLLs need to be copied to avoid memory errors.
if (MSVC)
file(GLOB ORT_DLLS "${ORT_DIRS}/lib/*.dll")
add_custom_command(TARGET "demo_ip"
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ORT_DLLS}
                    $<TARGET_FILE_DIR:demo_ip>)
endif (MSVC)