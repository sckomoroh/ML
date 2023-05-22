/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace clustering {

namespace k_means {

constexpr int POINTS_COUNT = 50;
constexpr int PARAM_COUNT = 2;
constexpr int CLASS_COUNT = 2;

using InputData = Eigen::Matrix<float, POINTS_COUNT * CLASS_COUNT, PARAM_COUNT>;
using PredictionData = Eigen::Vector<int64_t, POINTS_COUNT * CLASS_COUNT>;

}  // namespace k_means

class ClusteringKMeans {
public:
    tensorflow::Scope mRoot;
    tensorflow::ClientSession mSession;
    tensorflow::ops::Variable mCentroids;

public:
    ClusteringKMeans();

public:
    void trainModel(const k_means::InputData& data);

    k_means::PredictionData getPrediction(const k_means::InputData& data);

    static void demonstrate();

private:
    tensorflow::Output model(const tensorflow::ops::Placeholder& pointsPlaceholder);
};

}  // namespace clustering
