
set(SRC_FILES
    demo_main.cpp)

# generate project target
add_executable(demo_ip ${SRC_FILES})

target_link_libraries(demo_ip PUBLIC ${ORT_LIBS} ${OpenCV_LIBS})

target_link_libraries(demo_ip PRIVATE ${SUBSYS_NAME})

target_include_directories(demo_ip PUBLIC ${ORT_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

target_include_directories(demo_ip PRIVATE 
            "${CMAKE_SOURCE_DIR}/src/ImagePredict")

set_target_properties(demo_ip PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
            LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${OUTPUT_DIR}")

# The onnxruntime DLLs need to be copied to avoid memory errors.
if (MSVC)
file(GLOB ORT_DLLS "${ORT_DIRS}/lib/*.dll")
add_custom_command(TARGET "demo_ip"
                    POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    ${ORT_DLLS}
                    $<TARGET_FILE_DIR:demo_ip>)
endif (MSVC)