/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "LogisticRegression.h"

#include <cmath>

#include <iostream>
#include <random>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace regression {

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr int SAMPLES_COUNT = 100;
constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 1000;
constexpr float SENSITIVE_GATE = 0.0001;  // 0.01%
constexpr float LAMBDA = 0.01;

float LogisticRegression::function(std::vector<float> k, float x)
{
    return 1.0f / (1.0f + exp(-x));
}

std::vector<std::vector<IRegression::PointF>> LogisticRegression::generateData()
{
    std::vector<PointF> rightPoints;
    std::vector<PointF> leftPoints;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist1(-4, 2);
    std::normal_distribution<> dist2(4, 2);

    leftPoints.resize(SAMPLES_COUNT);
    rightPoints.resize(SAMPLES_COUNT);

    for (int i = 0; i < SAMPLES_COUNT; ++i) {
        leftPoints[i].x = dist1(gen);
        leftPoints[i].y = 0.0;
    }

    for (int i = 0; i < SAMPLES_COUNT; ++i) {
        rightPoints[i].x = dist2(gen);
        rightPoints[i].y = 1.0;
    }

    return {leftPoints, rightPoints};
}

std::vector<float> LogisticRegression::train(std::vector<std::vector<IRegression::PointF>> points,
                                             bool log)
{
    auto trainPoints = points[0];

    tf::Scope root = tf::Scope::NewRootScope();

    auto Param = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto weight = ops::Variable(root.WithOpName("parameter"), {2}, tf::DataType::DT_FLOAT);

    auto weight0 = ops::Slice(root, weight, {0}, {1});
    auto weight1 = ops::Slice(root, weight, {1}, {1});

    auto predictionOp =
        ops::Sigmoid(root, ops::Add(root, ops::Multiply(root, weight1, Param), weight0));

    // To avoid owerweight
    auto L2Regularization = ops::Multiply(
        root, {LAMBDA}, ops::Add(root, ops::Square(root, weight0), ops::Square(root, weight1)));

    // -Y * tf.log(predictionOp) - (1 - У) * tf.log(l - predictionOp)
    auto costOp =
        ops::Add(root, L2Regularization,
                 ops::ReduceMean(
                     root,
                     ops::Subtract(
                         root, ops::Multiply(root, ops::Neg(root, Y), ops::Log(root, predictionOp)),
                         ops::Multiply(root, ops::Subtract(root, 1.0f, Y),
                                       ops::Log(root, ops::Subtract(root, 1.0f, predictionOp)))),
                     {0}));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(weight);
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {costOp}, weightOutputs, &gradients));

    auto trainOp = ops::ApplyGradientDescent(root, weight, LEARNING_RATE, gradients[0]);

    tf::ClientSession session{root};
    TF_CHECK_OK(session.Run({ops::Assign(root, weight, {0.0f, 0.0f})}, nullptr));

    std::vector<tf::Tensor> outputs;
    float prevCost = 0.0f;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        float totalCost = 0.0f;
        for (int i = 0; i < trainPoints.size(); i++) {
            tf::ClientSession::FeedType feedType{{Param, trainPoints[i].x}, {Y, trainPoints[i].y}};
            TF_CHECK_OK(session.Run(feedType, {trainOp, costOp}, &outputs));
            float costValue = outputs[1].scalar<float>()();
            totalCost += costValue;
        }

        totalCost /= trainPoints.size();

        if (log) {
            std::cerr << "Previos cost: " << prevCost << " Current cost: " << totalCost
                      << " Diff: " << abs(prevCost - totalCost) << std::endl;
        }

        if (abs(prevCost - totalCost) < SENSITIVE_GATE) {
            if (log) {
                std::cerr << "Completed" << std::endl;
            }

            break;
        }
        prevCost = totalCost;
    }

    TF_CHECK_OK(session.Run({weight0, weight1}, &outputs));

    if (log) {
        std::cerr << "W0:" << outputs[0].scalar<float>()()
                  << " W1: " << outputs[1].scalar<float>()() << std::endl;
    }

    return {outputs[0].scalar<float>()(), outputs[1].scalar<float>()()};
}

}  // namespace regression
