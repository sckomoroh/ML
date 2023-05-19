
cmake_minimum_required(VERSION 3.22)

project(ml)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(VENV_PATH /home/yzvihunov/Documents/src/ML/venv)
set(TENSORFLOW_INCLUDE_DIR ${VENV_PATH}/lib/python3.10/site-packages/tensorflow/include)
set(TENSORFLOW_LIB_DIR ${VENV_PATH}/lib/python3.10/site-packages/tensorflow/)

find_package(GTest REQUIRED)

include_directories(${GTEST_INCLUDE_DIRS})

include_directories(
    src
    ${TENSORFLOW_INCLUDE_DIR}
)

link_directories(${TENSORFLOW_LIB_DIR})

add_executable(
    ${PROJECT_NAME}

    src/main.cpp

    src/regressions/LinearRegression.h
    src/regressions/LinearRegression.cpp

    src/regressions/PolynomialRegression.h
    src/regressions/PolynomialRegression.cpp

    src/regressions/LogisticRegression.h
    src/regressions/LogisticRegression.cpp

    src/regressions/Logistic2ParamRegression.h
    src/regressions/Logistic2ParamRegression.cpp
)

target_link_libraries(
    ${PROJECT_NAME}

    tensorflow_cc
    tensorflow_framework
)

