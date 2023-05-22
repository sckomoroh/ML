/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include "tensorflow/core/framework/tensor.h"

namespace utils ::tensor {

template <typename TEigenType>
tensorflow::Tensor tensorFromMatrix(const TEigenType& data)
{
    using Scalar = typename TEigenType::Scalar;

    tensorflow::Tensor tensor{tensorflow::DataTypeToEnum<Scalar>::value,
                              {data.rows(), data.cols()}};
    auto matrix = tensor.matrix<float>();
    for (int row = 0; row < data.rows(); row++) {
        for (int col = 0; col < data.cols(); col++) {
            matrix(row, col) = data(row, col);
        }
    }

    return tensor;
}

template <typename TEigenType>
tensorflow::Tensor tensorFromVector(const TEigenType& data)
{
    using Scalar = typename TEigenType::Scalar;

    tensorflow::Tensor tensor{tensorflow::DataTypeToEnum<Scalar>::value, {data.rows()}};
    auto matrix = tensor.matrix<float>();
    for (int row = 0; row < data.rows(); row++) {
        matrix(row) = data(row);
    }

    return tensor;
}

}  // namespace utils::tensor