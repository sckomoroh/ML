/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <array>
#include <map>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace regression {

constexpr int POINTS_COUNT = 100;
constexpr int PARAMS_COUNT = 15;
constexpr int CLASS_COUNT = 10;
constexpr int DATA_COUNT = POINTS_COUNT * CLASS_COUNT;

namespace classification {

using InputData = Eigen::Matrix<float, DATA_COUNT, PARAMS_COUNT>;
using MaskData = Eigen::Matrix<float, DATA_COUNT, CLASS_COUNT>;

}  // namespace classification

class Classification {
private:
    tensorflow::Scope mRoot;
    tensorflow::ops::Variable mWeights;
    tensorflow::ops::Variable mOffsets;
    tensorflow::ClientSession mSession;

public:
    Classification();

public:
    void trainModel(const classification::InputData& data, const classification::MaskData& mask);

    void verify(const classification::InputData& data);

    static void demonstrate();

    int getPrediction(const Eigen::Matrix<float, 1, PARAMS_COUNT>& value);

private:
    tensorflow::Output model(const tensorflow::ops::Placeholder& placeholder);

    void zeroVector(tensorflow::ops::Variable& var, int l);
    void zeroMatrix(tensorflow::ops::Variable& var, int l, int h);
};

}  // namespace regression
