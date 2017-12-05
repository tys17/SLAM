cmake_minimum_required(VERSION 3.9)
project(SLAM)

find_package(OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 14)

add_executable(SLAM main.cpp)
target_link_libraries(SLAM ${OpenCV_LIBS})