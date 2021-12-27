cmake_minimum_required(VERSION 3.12)
project(proj4)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

find_package(OpenCV CONFIG REQUIRED)

add_executable(proj4 main.cpp face_binary_cls.cpp)
target_link_libraries(proj4 PRIVATE pthread)
target_link_libraries(proj4 PRIVATE opencv_core opencv_imgcodecs)