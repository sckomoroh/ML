/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "LinearRegression.h"

#include <iostream>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

namespace regression {

constexpr float LEARNING_RATE = 0.01;
constexpr int TRAINING_EPOCHS = 100;
constexpr int POINTS_COUNT = 101;
constexpr float LAMBDA = 0.01;

float LinearRegression::function(std::vector<float> k, float X) { return k[0] * X + k[1]; }

std::vector<std::vector<IRegression::PointF>> LinearRegression::generateData()
{
    std::vector<IRegression::PointF> points;
    points.resize(POINTS_COUNT);
    for (int i = 0; i < POINTS_COUNT; ++i) {
        points[i].x = -1.0f + (float)i * (2.0f / float(POINTS_COUNT - 1));
    }

    for (int i = 0; i < POINTS_COUNT; ++i) {
        points[i].y += 2 * points[i].x + 4;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.33);

    for (int i = 0; i < POINTS_COUNT; ++i) {
        points[i].y += dis(gen);
    }

    return {points};
}

std::vector<float> LinearRegression::train(std::vector<std::vector<IRegression::PointF>> points,
                                           bool log)
{
    auto trainPoints = points[0];
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto weight = ops::Variable(root, {2}, tf::DataType::DT_FLOAT);
    auto weight0 = ops::Slice(root, weight, {0}, {1});
    auto weight1 = ops::Slice(root, weight, {1}, {1});

    auto predictionOp = ops::Add(root, ops::Multiply(root, X, weight0), weight1);

    // To avoid overweight
    // I dodn't know why but with using the ops::Square crashes the program
    auto L2Regularization = ops::Multiply(
        root, {LAMBDA},
        ops::AddN(root, std::vector<tf::Output>{ops::Multiply(root, weight0, weight0),
                                                ops::Multiply(root, weight1, weight1)}));

    tf::Output costOp = ops::Square(root, ops::Subtract(root, Y, predictionOp));
    costOp = ops::Add(root, costOp, L2Regularization);

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(weight);

    TF_CHECK_OK(tf::AddSymbolicGradients(root, {costOp}, weightOutputs, &gradients));

    auto trainOp = ops::ApplyGradientDescent(root, weight, LEARNING_RATE, gradients[0]);

    tf::ClientSession session{root};
    TF_CHECK_OK(session.Run({ops::Assign(root, weight, {0.0f, 0.0f})}, nullptr));

    std::vector<tf::Tensor> outputs;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        for (int i = 0; i < trainPoints.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trainPoints[i].x}, {Y, trainPoints[i].y}};
            TF_CHECK_OK(session.Run(feedType, {trainOp, costOp}, &outputs));
        }

        if (log) {
            float costValue = outputs[1].scalar<float>()();
            TF_CHECK_OK(session.Run({weight0, weight1}, &outputs));
            float k1 = outputs[0].scalar<float>()();
            float k2 = outputs[1].scalar<float>()();
            std::cerr << "Coeffs: " << k1 << "," << k2 << " Cost: " << costValue << std::endl;
        }
    }

    TF_CHECK_OK(session.Run({weight0, weight1}, &outputs));

    float k1 = outputs[0].scalar<float>()();
    float k2 = outputs[1].scalar<float>()();
    if (log) {
        std::cerr << "Coefficient: " << k1 << "," << k2 << std::endl;
    }

    return {k1, k2};
}

}  // namespace regression
