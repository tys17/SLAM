cmake_minimum_required(VERSION 3.9)
project(loop_closure)

find_package(OpenCV REQUIRED )
set(DBoW3_DIR /home/jianwei/contrib/DBow3/build)
find_package(DBoW3 REQUIRED )
include_directories(/usr/include/eigen3)
find_package(Ceres REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_executable(loop_closure main.cpp fileIO.cpp fileIO.h Tracking.cpp Tracking.h Database.cpp Database.h)
target_link_libraries(loop_closure ${OpenCV_LIBS})
target_link_libraries(loop_closure ${DBoW3_LIBS})
target_link_libraries(loop_closure ${Ceres_LIBS})