set(SUBSYS_NAME ImagePredict)

set(
    SRC_FILES
    ImagePredict.cpp
)
set(
    SRC_FILES_H
    ImagePredict.h
)

add_library(${SUBSYS_NAME} SHARED ${SRC_FILES} ${SRC_FILES_H})

target_link_libraries(${SUBSYS_NAME} ${DEPS})

target_include_directories(${SUBSYS_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})