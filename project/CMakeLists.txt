cmake_minimum_required(VERSION 3.21)
project(project)

# 设置OpenCV DIR
set(OpenCV_DIR /opt/homebrew/Cellar/opencv)
# 寻找OpenCV.CMakeLists,以此找到包,并赋值各库相关变量
find_package(OpenCV REQUIRED)

# OpenCV_INCLUDE_DIRS是关于find_package的变量,
# 包含了一个路径，这样可以在代码中的#include做根目录
include_directories( /opt/homebrew/Cellar/opencv/4.5.3_2/include )


set(CMAKE_CXX_STANDARD 14)

add_executable(project main.cpp)

target_link_libraries( project ${OpenCV_LIBS} )