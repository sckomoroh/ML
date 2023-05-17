/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace regression::logistic2d {

constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 3, POINTS_COUNT>;

InputMatrix generateData();

/**
 * @brief Predicts belonging to one of the sets by 2 parameters
 *
 * Predicts belonging to one of the sets by 2 parameters.
 * The regression equation  for 2 parameter is w0 + w1*param1 + w2*param2
 *
 * If you have more than 2 parameter so the equation will be like
 * w0 + w1*param1 + w2*param2 + ... + wN*paramN
 */
class Logistic2ParamRegression {
public:
    Eigen::Vector3f train(const InputMatrix& matrix, bool log = false);
};

}  // namespace regression::logistic2d