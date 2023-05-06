cmake_minimum_required(VERSION 3.22)

project(polynom_regression)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt6 COMPONENTS Widgets Core Gui REQUIRED)
find_package(Eigen3 REQUIRED)

set(TENSORFLOW_INCLUDE_DIR "/home/yzvihunov/Documents/src/repo/tf/tensorflow/bazel-bin/tensorflow/include")
set(TENSORFLOW_LIB_DIR "/home/yzvihunov/Documents/src/repo/tf/tensorflow/bazel-bin/tensorflow")

include_directories(
    ${TENSORFLOW_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

link_directories(${TENSORFLOW_LIB_DIR})

qt_add_executable(
    ${PROJECT_NAME} 
    
    src/MainWindow.cpp
    src/main.cpp
    src/logic_polynominal.cpp
    src/logic_linear.cpp
)

target_link_libraries(
    ${PROJECT_NAME}  
    
    PRIVATE 
    
    Qt6::Widgets
    libtensorflow_cc.so
    libtensorflow_framework.so
)

