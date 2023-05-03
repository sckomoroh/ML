cmake_minimum_required(VERSION 3.10)
project(tf_sample)

set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)

# Replace these with the paths to the 'tensorflow' and 'bazel-bin/tensorflow' directories
set(TENSORFLOW_INCLUDE_DIR "/home/yzvihunov/Documents/src/repo/tf/tensorflow/bazel-bin/tensorflow/include")
set(TENSORFLOW_LIB_DIR "/home/yzvihunov/Documents/src/repo/tf/tensorflow/bazel-bin/tensorflow")

include_directories(
    ${TENSORFLOW_INCLUDE_DIR}
    ${EIGEN3_INCLUDE_DIRS}
)

link_directories(${TENSORFLOW_LIB_DIR})

add_executable(tf_sample 
    src/main.cpp
)

target_link_libraries(tf_sample
    libtensorflow_cc.so
    libtensorflow_framework.so
)
