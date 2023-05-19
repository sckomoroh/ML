#include <gtest/gtest.h>

#include "tensorflow/core/framework/tensor.h"

TEST(Matrix, fromToVector)
{
    constexpr int COUNT = 10;

    Eigen::Tensor<float, 3> tensor;
    tensor.resize(10, 3, 2);
    int value = 0;
    auto dims = tensor.dimensions();
    for (int k = 0; k < dims[0]; k++) {
        for (int i = 0; i < dims[0]; i++) {
            for (int j = 0; j < dims[0]; j++) {
                tensor(i, j, k) = value++;
            }
        }
    }

    Eigen::Vector2f vector;
    for (int i = 0; i < 2; i++) {
        vector(i) = tensor(5, 2, i);
    }

    int a = 0;
}
