/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace regression::logistic {

constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 2, POINTS_COUNT>;

InputMatrix generateData();

class LogisticRegression {
public:
    Eigen::Vector2f train(const InputMatrix& matrix, bool log = false);
};

}  // namespace regression::logistic