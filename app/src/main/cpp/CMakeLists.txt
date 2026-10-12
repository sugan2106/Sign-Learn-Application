cmake_minimum_required(VERSION 3.22.1)
project(yolov5ncnn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# NCNN official CMake config (ABI-aware)
set(ncnn_DIR
        ${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20260113-android-vulkan/${ANDROID_ABI}/lib/cmake/ncnn
)

find_package(ncnn REQUIRED)

add_library(
        yolov5ncnn
        SHARED
        native-lib.cpp
)

find_library(log-lib log)
find_library(android-lib android)
find_library(jnigraphics-lib jnigraphics)

target_link_libraries(
        yolov5ncnn
        ncnn
        ${jnigraphics-lib}
        ${android-lib}
        ${log-lib}
)