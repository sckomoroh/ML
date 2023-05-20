/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include "tensorflow/core/framework/tensor.h"

namespace utils ::tensor {

template <typename TEigenType>
tensorflow::Tensor eigenMatrixToTensor(const TEigenType& eigenObject)
{
    using Scalar = typename TEigenType::Scalar;

    constexpr int Rows = TEigenType::RowsAtCompileTime;
    constexpr int Cols = TEigenType::ColsAtCompileTime;
    constexpr bool IsDynamic = (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic);

    tensorflow::Tensor tensor(tensorflow::DataTypeToEnum<Scalar>::value,
                              tensorflow::TensorShape({IsDynamic ? eigenObject.rows() : Rows,
                                                       IsDynamic ? eigenObject.cols() : Cols}));

    // Copy Eigen object data to TensorFlow tensor
    auto eigenData = eigenObject.data();
    auto tensorData = tensor.flat<Scalar>().data();
    std::copy(eigenData, eigenData + eigenObject.size(), tensorData);

    return tensor;
}

template <class TVector>
tensorflow::Tensor eigenVectorToTensor(const TVector& eigenVector)
{
    tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({eigenVector.size()}));

    // Copy Eigen vector data to TensorFlow tensor
    auto eigenVectorData = eigenVector.data();
    auto tensorData = tensor.flat<float>().data();
    std::copy(eigenVectorData, eigenVectorData + eigenVector.size(), tensorData);

    return tensor;
}

}  // namespace utils::tensor