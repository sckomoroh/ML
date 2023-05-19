/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "LinearRegression.h"

#include <iostream>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/util/events_writer.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

namespace regression::linear {

constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 1000;
constexpr float LAMBDA = 0.01;

InputMatrix generateData()
{
    InputMatrix matrix;
    matrix.resize(2, POINTS_COUNT);
    matrix.setZero();
    matrix.row(0) = Eigen::VectorXf::LinSpaced(POINTS_COUNT, 1.0f, 5.0f);
    for (int i = 0; i < matrix.cols(); ++i) {
        matrix(1, i) =
            Eigen::internal::random<float>(matrix(0, i) - 0.5f, matrix(0, i) + 0.5) + 5.0f;
    }

    return matrix;
}

Eigen::Vector2f LinearRegression::train(const InputMatrix& matrix, bool log)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto weight = ops::Variable(root, {2}, tf::DataType::DT_FLOAT);
    auto weight0 = ops::Slice(root, weight, {0}, {1});
    auto weight1 = ops::Slice(root, weight, {1}, {1});

    // Y = x * w0 + w1
    tf::Input predictionOp = ops::Add(root, ops::Multiply(root, X, weight0), weight1);

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
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{{X, matrix(0, i)}, {Y, matrix(1, i)}};
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

    float k0 = outputs[0].scalar<float>()();
    float k1 = outputs[1].scalar<float>()();
    if (log) {
        std::cerr << "Coefficient: " << k0 << "," << k1 << std::endl;
    }

    return {k0, k1};
}

}  // namespace regression::linear
