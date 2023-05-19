/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace regression {

namespace logistic {
constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 2, POINTS_COUNT>;
}  // namespace logistic

class LogisticRegression {
private:
    tensorflow::Scope mRoot;
    tensorflow::ops::Variable mWeights;
    tensorflow::ClientSession mSession;

public:
    LogisticRegression();

public:
    void trainModel(const logistic::InputMatrix& matrix, bool log = false);

    float getPrediction(float value);

    static void demonstrate();

private:
    tensorflow::Output model(const tensorflow::ops::Placeholder& placeholder);
};

}  // namespace regression