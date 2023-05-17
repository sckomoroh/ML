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

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

include_directories(
    src
    ${TENSORFLOW_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

link_directories(${TENSORFLOW_LIB_DIR})

qt_add_executable(
    ${PROJECT_NAME}

    src/main.cpp

    src/ui/MainWindow.cpp

    src/regressions/LinearRegression.cpp
    src/regressions/PolynomialRegression.cpp
    src/regressions/LogisticRegression.cpp
    src/regressions/Logistic2ParamRegression.cpp

    # src/classification/Classification.cpp
)

target_link_libraries(
    ${PROJECT_NAME}

    PRIVATE

    Qt6::Widgets
    libtensorflow_cc.so
    libtensorflow_framework.so
)

add_executable(
    tests

    tests/Matrix_test.cpp
)

target_link_libraries(tests

    ${GTEST_BOTH_LIBRARIES}

    libtensorflow_cc.so
    libtensorflow_framework.so
)
