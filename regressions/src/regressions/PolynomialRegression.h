/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace regression {

namespace polynomial {

constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 2, POINTS_COUNT>;

}  // namespace polynomial

class PolynomialRegression {
private:
    tensorflow::Scope mRoot;
    tensorflow::ops::Variable mWeights;
    tensorflow::ClientSession mSession;

public:
    PolynomialRegression();

public:
    void trainModel(const polynomial::InputMatrix& matrix, bool log = false);

    float getPrediction(float value);

    static void demonstrate();

private:
    tensorflow::Output model(const tensorflow::ops::Placeholder& placeholder);
};

}  // namespace regression
