/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

#include "LinearRegression.h"

namespace regression {

namespace logistic2d {

constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 3, POINTS_COUNT>;

}  // namespace logistic2d

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
private:
    tensorflow::Scope mRoot;
    tensorflow::ops::Variable mWeights;
    tensorflow::ClientSession mSession;

public:
    Logistic2ParamRegression();

public:
    void trainModel(const logistic2d::InputMatrix& matrix, bool log = false);

    float getPrediction(float param1, float param2, bool useLite = true);

    static void demonstrate(LinearRegression* lineRegression = nullptr);

private:
    tensorflow::Output model(const tensorflow::ops::Placeholder& placeholder1,
                             const tensorflow::ops::Placeholder& placeholde2);
};

}  // namespace regression