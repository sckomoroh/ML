/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "Logistic2ParamRegression.h"

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

constexpr int POINTS_COUNT = 500;
constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 2000;
constexpr float SENSITIVE_GATE = 0.0001;  // 0.01%
constexpr float LAMBDA = 0.0001;

float Logistic2ParamRegression::function(std::vector<float> k, float X)
{
    // The value cumputes outside
    return 0.0f;
}

std::vector<std::vector<IRegression::PointF>> Logistic2ParamRegression::generateData()
{
    std::vector<PointF> leftPoints;
    std::vector<PointF> rightPoints;
    leftPoints.resize(POINTS_COUNT);
    rightPoints.resize(POINTS_COUNT);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::normal_distribution<> distLeftX(3, 1);
    std::normal_distribution<> distLeftY(2, 1);
    std::normal_distribution<> distRightX(7, 1);
    std::normal_distribution<> distRightY(6, 1);

    for (int i = 0; i < POINTS_COUNT; ++i) {
        leftPoints[i].x = distLeftX(gen);
        leftPoints[i].y = distLeftY(gen);
        rightPoints[i].x = distRightX(gen);
        rightPoints[i].y = distRightY(gen);
    }

    return {leftPoints, rightPoints};
}

std::vector<float> Logistic2ParamRegression::train(
    std::vector<std::vector<IRegression::PointF>> points,
    bool log)
{
    std::vector<float> x1s(POINTS_COUNT * 2);
    std::vector<float> x2s(POINTS_COUNT * 2);
    std::vector<float> ys(POINTS_COUNT * 2);

    // Preprocess the input data
    for (auto i = 0; i < points[0].size(); i++) {
        x1s[i] = points[0][i].x;  // Param 1
        x2s[i] = points[0][i].y;  // Param 2
        ys[i] = 0.0f;
    }

    for (auto i = 0; i < points[1].size(); i++) {
        x1s[i + POINTS_COUNT] = points[1][i].x;  // Param 1
        x2s[i + POINTS_COUNT] = points[1][i].y;  // Param 2
        ys[i + POINTS_COUNT] = 1.0f;
    }

    tf::Scope root = tf::Scope::NewRootScope();

    auto Param1 = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Param2 = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto YS = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto weight = ops::Variable(root.WithOpName("parameter"), {3}, tf::DataType::DT_FLOAT);

    auto weight0 = ops::Slice(root, weight, {0}, {1});
    auto weight1 = ops::Slice(root, weight, {1}, {1});
    auto weight2 = ops::Slice(root, weight, {2}, {1});

    // tf.sigmoid(w[2] * x2s + w[1] * x1s + w[0])
    auto model = ops::Sigmoid(
        root,
        ops::AddN(root, std::vector<tf::Output>{ops::Multiply(root, weight2, Param2),
                                                ops::Multiply(root, weight1, Param1), weight0}));

    // To avoid owerweight
    auto L2Regularization =
        ops::Multiply(root, {LAMBDA},
                      ops::AddN(root, std::vector<tf::Output>{ops::Square(root, weight0),
                                                              ops::Square(root, weight1),
                                                              ops::Square(root, weight2)}));

    // -tf.reduce_mean(tf.math.log(y_model * ys + (1 - y_model) * (1 - ys)))
    tf::Output costOp;
    costOp = ops::Neg(
        root, ops::ReduceMean(
                  root,
                  ops::Log(root, ops::Add(root, ops::Multiply(root, model, YS),
                                          ops::Multiply(root, ops::Subtract(root, {1.0f}, model),
                                                        ops::Subtract(root, {1.0f}, YS)))),

                  {0}));

    costOp = ops::Add(root, L2Regularization, costOp);

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(weight);
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {costOp}, weightOutputs, &gradients));

    auto trainOp = ops::ApplyGradientDescent(root, weight, LEARNING_RATE, gradients[0]);

    tf::ClientSession session{root};
    TF_CHECK_OK(session.Run({ops::Assign(root, weight, {0.0f, 0.0f, 0.0f})}, nullptr));

    std::vector<tf::Tensor> outputs;
    float prevCost = 0.0f;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        float totalCost = 0.0f;
        for (int i = 0; i < x1s.size(); i++) {
            tf::ClientSession::FeedType feedType{{Param1, x1s[i]}, {Param2, x2s[i]}, {YS, ys[i]}};
            TF_CHECK_OK(session.Run(feedType, {trainOp, costOp}, &outputs));
            float costValue = outputs[1].scalar<float>()();
            totalCost += costValue;
        }

        totalCost /= POINTS_COUNT;

        if (log) {
            std::cerr << "Epoch: " << epoch << " Cost:" << totalCost << std::endl;
        }

        if (abs(prevCost - totalCost) < SENSITIVE_GATE) {
            if (log) {
                std::cerr << "Completed" << std::endl;
            }

            break;
        }
        prevCost = totalCost;
    }

    TF_CHECK_OK(session.Run({weight0, weight1, weight2}, &outputs));

    return {outputs[0].scalar<float>()(), outputs[1].scalar<float>()(),
            outputs[2].scalar<float>()()};
}

}  // namespace regression
