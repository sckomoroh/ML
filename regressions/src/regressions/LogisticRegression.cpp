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
constexpr float LEARNING_RATE = 0.0001;
constexpr int TRAINING_EPOCHS = 1000;
constexpr float SENSITIVE_GATE = 0.0001;  // 0.01%

float LogisticRegression::function(std::vector<float> k, float X)
{
    X = X * k[1] + k[0];
    return 1.0f / (1.0f + exp(-X));
}

void LogisticRegression::generateData(std::vector<float>& trX, std::vector<float>& trY)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist1(-4, 2);
    std::normal_distribution<> dist2(4, 2);

    trX.resize(SAMPLES_COUNT * 2);
    trY.resize(SAMPLES_COUNT * 2);

    for (int i = 0; i < SAMPLES_COUNT; ++i) {
        trX[i] = dist1(gen);
        trY[i] = 0.0;
    }

    for (int i = SAMPLES_COUNT; i < SAMPLES_COUNT * 2; ++i) {
        trX[i] = dist2(gen);
        trY[i] = 1.0;
    }
}

std::vector<float> LogisticRegression::train(const std::vector<float>& trX,
                                             const std::vector<float>& trY,
                                             bool log)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto weight = ops::Variable(root.WithOpName("parameter"), {2}, tf::DataType::DT_FLOAT);

    auto weight0 = ops::Slice(root, weight, {0}, {1});
    auto weight1 = ops::Slice(root, weight, {1}, {1});

    auto model = ops::Sigmoid(root, ops::Add(root, ops::Multiply(root, weight1, X), weight0));

    // -Y * tf.log(model) - (1 - У) * tf.log(l - model)
    auto costOp = ops::ReduceMean(
        root,
        ops::Subtract(root, ops::Multiply(root, ops::Neg(root, Y), ops::Log(root, model)),
                      ops::Multiply(root, ops::Subtract(root, 1.0f, Y),
                                    ops::Log(root, ops::Subtract(root, 1.0f, model)))),
        {0});

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
        for (int i = 0; i < trX.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trX[i]}, {Y, trY[i]}};
            TF_CHECK_OK(session.Run(feedType, {trainOp, costOp}, &outputs));
            float costValue = outputs[1].scalar<float>()();
            totalCost += costValue;
        }

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

    return {outputs[0].scalar<float>()(), outputs[1].scalar<float>()()};
}

}  // namespace regression
