#pragma once

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/cc/client/client_session.h"

namespace regression {

namespace linear {
constexpr int POINTS_COUNT = 100;

using InputMatrix = Eigen::Matrix<float, 2, Eigen::Dynamic>;
}  // namespace linear

class LinearRegression {
private:
    tensorflow::Scope mRoot;
    tensorflow::ops::Variable mWeights;
    tensorflow::ClientSession mSession;

public:
    LinearRegression();

public:
    void trainModel(const linear::InputMatrix& matrix, bool log = false);

    float getPrediction(float value);

    static void demonstrate();

private:
    tensorflow::Output model(const tensorflow::ops::Placeholder& placeholder);
};

}  // namespace regression
