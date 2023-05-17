/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace regression::polynomial {

constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 2, POINTS_COUNT>;

InputMatrix generateData();

class PolynomialRegression {
public:
    Eigen::Matrix<float, 6, 1> train(const InputMatrix& matrix, bool log = false);
};

}  // namespace regression::polynomial
