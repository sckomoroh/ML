/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "Logistic2ParamRegression.h"

#include <cmath>

#include <iostream>
#include <random>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace regression::logistic2d {

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 2000;
constexpr float SENSITIVE_GATE = 0.0001;  // 0.01%
constexpr float LAMBDA = 0.0001;

InputMatrix generateData()
{
    InputMatrix matrix = InputMatrix::Zero();
    float distance = 3.5f;
    for (auto i = 0; i < matrix.cols() / 2; i++) {
        matrix(0, i) = Eigen::internal::random<float>(-distance, distance) - 3.0f;
        matrix(1, i) = Eigen::internal::random<float>(-distance, distance) + 5.0f;
        matrix(2, i) = 0.0f;
    }

    for (auto i = matrix.cols() / 2; i < matrix.cols(); i++) {
        matrix(0, i) = Eigen::internal::random<float>(-distance, distance) + 5.0f;
        matrix(1, i) = Eigen::internal::random<float>(-distance, distance) - 2.0f;
        matrix(2, i) = 1.0f;
    }

    return matrix;
}

Eigen::Vector3f Logistic2ParamRegression::train(const InputMatrix& matrix, bool log)
{
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
        root, ops::Log(root, ops::Add(root, ops::Multiply(root, model, YS),
                                      ops::Multiply(root, ops::Subtract(root, {1.0f}, model),
                                                    ops::Subtract(root, {1.0f}, YS)))));

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
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{
                {Param1, matrix(0, i)}, {Param2, matrix(1, i)}, {YS, matrix(2, i)}};
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

    if (log) {
        std::cerr << "Coefficients: "
                  << "K0: " << outputs[0].scalar<float>()()
                  << " K1: " << outputs[1].scalar<float>()()
                  << " K2: " << outputs[2].scalar<float>()() << std::endl;
    }

    return {outputs[0].scalar<float>()(), outputs[1].scalar<float>()(),
            outputs[2].scalar<float>()()};
}

}  // namespace regression::logistic2d
