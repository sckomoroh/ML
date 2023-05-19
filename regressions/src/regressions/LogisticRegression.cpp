/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "LogisticRegression.h"

#include <cmath>

#include <iostream>
#include <random>

#include "tensorflow/cc/framework/gradients.h"

namespace regression {

using namespace logistic;

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.01;
constexpr int TRAINING_EPOCHS = 5000;
constexpr float SENSITIVE_GATE = 0.0001;  // 0.01%
constexpr float LAMBDA = 0.001;

LogisticRegression::LogisticRegression()
    : mRoot{tf::Scope::NewRootScope()}
    , mWeights{mRoot, {2}, tf::DataType::DT_FLOAT}
    , mSession{mRoot}
{
}

void LogisticRegression::trainModel(const InputMatrix& matrix, bool log)
{
    auto Param = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    auto w0 = ops::Slice(mRoot, mWeights, {0}, {1});
    auto w1 = ops::Slice(mRoot, mWeights, {1}, {1});

    auto predictionOp = model(Param);

    // To avoid owerweight
    auto L2Regularization = ops::Multiply(
        mRoot, {LAMBDA}, ops::Add(mRoot, ops::Square(mRoot, w0), ops::Square(mRoot, w1)));

    // -Y * tf.log(predictionOp) - (1 - У) * tf.log(l - predictionOp)
    auto costOp =
        ops::Add(mRoot, L2Regularization,
                 ops::Subtract(
                     mRoot, ops::Multiply(mRoot, ops::Neg(mRoot, Y), ops::Log(mRoot, predictionOp)),
                     ops::Multiply(mRoot, ops::Subtract(mRoot, 1.0f, Y),
                                   ops::Log(mRoot, ops::Subtract(mRoot, 1.0f, predictionOp)))));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(mWeights);
    TF_CHECK_OK(tf::AddSymbolicGradients(mRoot, {costOp}, weightOutputs, &gradients));

    auto trainOp = ops::ApplyGradientDescent(mRoot, mWeights, LEARNING_RATE, gradients[0]);

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, mWeights, {0.0f, 0.0f})}, nullptr));

    std::vector<tf::Tensor> outputs;
    float prevCost = 0.0f;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        float totalCost = 0.0f;
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{{Param, matrix(0, i)}, {Y, matrix(1, i)}};
            TF_CHECK_OK(mSession.Run(feedType, {trainOp, costOp}, &outputs));
            float costValue = outputs[1].scalar<float>()();
            totalCost += costValue;
        }

        totalCost /= matrix.cols();

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

    TF_CHECK_OK(mSession.Run({mWeights}, &outputs));

    if (log) {
        std::cerr << "W0:" << outputs[0].scalar<float>()()
                  << " W1: " << outputs[1].scalar<float>()() << std::endl;
    }
}

float LogisticRegression::getPrediction(float value)
{
    auto X = ops::Placeholder(mRoot.WithOpName("value"), tf::DataType::DT_FLOAT);
    auto m = model(X);

    std::vector<tf::Tensor> outputs;
    tf::ClientSession::FeedType feed{{X, {value}}};

    TF_CHECK_OK(mSession.Run(feed, {m}, &outputs));

    return outputs[0].scalar<float>()();
}

tf::Output LogisticRegression::model(const tensorflow::ops::Placeholder& placeholder)
{
    // y = sigmoid(w0 + x*w1)

    auto w0 = ops::Slice(mRoot, mWeights, {0}, {1});
    auto w1 = ops::Slice(mRoot, mWeights, {1}, {1});

    return ops::Sigmoid(mRoot, ops::Add(mRoot, ops::Multiply(mRoot, w1, placeholder), w0));
}

void LogisticRegression::demonstrate()
{
    InputMatrix data = InputMatrix::Zero();
    for (auto i = 0; i < data.cols() / 2; i++) {
        data(0, i) = Eigen::internal::random<float>(-0.5f, +0.5f) - 1.0f;
        data(1, i) = 0.0f;
    }

    for (auto i = data.cols() / 2; i < data.cols(); i++) {
        data(0, i) = Eigen::internal::random<float>(-0.5f, +0.5f) + 1.0f;
        data(1, i) = 1.0f;
    }

    FILE* pipe = popen("gnuplot -persist", "w");

    fprintf(pipe, "plot '-' with points pt 7 lc rgb 'red', '-' with lines lc rgb 'blue'\n");

    for (int i = 0; i < logistic::POINTS_COUNT; i++) {
        auto x = data(0, i);
        auto y = data(1, i);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");

    // Regression side
    LogisticRegression regression;

    regression.trainModel(data);
    for (float x = -1.5f; x < 1.5f; x += 0.1f) {
        auto y = regression.getPrediction(x);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");
    fflush(pipe);
    fclose(pipe);
}

}  // namespace regression
