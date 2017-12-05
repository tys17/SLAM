cmake_minimum_required(VERSION 3.9)
project(feature_extraction)

find_package(OpenCV REQUIRED )
set(CMAKE_CXX_STANDARD 14)

add_executable(feature_extraction main.cpp)
target_link_libraries(feature_extraction ${OpenCV_LIBS})