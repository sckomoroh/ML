/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "PolynomialRegression.h"

#include <iostream>
#include <limits>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr uint8_t COEFFS_COUNT = 6;
constexpr float LEARNING_RATE = 0.01;
constexpr int TRAINING_EPOCHS = 100;
constexpr int POINTS_COUNT = 101;
constexpr float LAMBDA = 0.01;

namespace regression {

using namespace polynomial;

PolynomialRegression::PolynomialRegression()
    : mRoot{tf::Scope::NewRootScope()}
    , mWeights{mRoot, {6}, tf::DataType::DT_FLOAT}
    , mSession{mRoot}
{
}

void PolynomialRegression::trainModel(const InputMatrix& matrix, bool log)
{
    auto X = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    tf::Output predictionOp = model(X);

    std::vector<ops::Slice> w;
    for (auto i = 0; i < COEFFS_COUNT; i++) {
        w.emplace_back(ops::Slice(mRoot, mWeights, {i}, {1}));
    }

    // To avoid overweight
    // I dodn't know why but with using the ops::Square the program crashes
    std::vector<tf::Output> terms;
    for (auto weight : w) {
        terms.push_back(ops::Multiply(mRoot, weight, weight));
    }
    auto L2Regularization =
        ops::Multiply(mRoot, ops::Const(mRoot, LAMBDA), ops::AddN(mRoot, terms));

    tf::Output costOp = ops::Add(mRoot, ops::Square(mRoot, ops::Subtract(mRoot, Y, predictionOp)),
                                 L2Regularization);

    std::vector<tf::Output> gradients;
    TF_CHECK_OK(tf::AddSymbolicGradients(mRoot, {costOp}, {mWeights}, &gradients));
    auto updateOp = ops::ApplyGradientDescent(mRoot, mWeights, LEARNING_RATE, gradients[0]);

    std::vector<tf::Tensor> outputs;

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, mWeights, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f})},
                             nullptr));

    // Start training
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        float totalCost = 0.0f;
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{{X, matrix(0, i)}, {Y, matrix(1, i)}};
            TF_CHECK_OK(mSession.Run(feedType, {updateOp, costOp}, &outputs));
            totalCost += outputs[1].scalar<float>()();
        }

        if (log) {
            std::cout << "Cost: " << totalCost << " Epoch " << epoch << " weights: ";
            for (int i = 0; i < COEFFS_COUNT; i++) {
                TF_CHECK_OK(mSession.Run({w[i]}, &outputs));
                std::cerr << outputs[0].scalar<float>()() << " ";
            }

            std::cerr << std::endl;
        }
    }

    if (log) {
        std::cerr << "Coefficients: ";
    }

    TF_CHECK_OK(mSession.Run({mWeights}, &outputs));
    Eigen::Matrix<float, 6, 1> weightsResult(COEFFS_COUNT);
    for (int i = 0; i < COEFFS_COUNT; i++) {
        if (log) {
            std::cerr << outputs[i].scalar<float>()() << " ";
        }
    }

    std::cerr << std::endl;
}

float PolynomialRegression::getPrediction(float value)
{
    auto X = ops::Placeholder(mRoot.WithOpName("value"), tf::DataType::DT_FLOAT);
    auto m = model(X);

    std::vector<tf::Tensor> outputs;
    tf::ClientSession::FeedType feed{{X, {value}}};

    TF_CHECK_OK(mSession.Run(feed, {m}, &outputs));

    return outputs[0].scalar<float>()();
}

tf::Output PolynomialRegression::model(const tensorflow::ops::Placeholder& placeholder)
{
    // w0 + w1*x + w2 * x^2 + w3 * x^3 + w4 * x^4 + w5 * x^5
    std::vector<ops::Slice> w;
    for (auto i = 0; i < COEFFS_COUNT; i++) {
        w.emplace_back(ops::Slice(mRoot, mWeights, {i}, {1}));
    }

    std::vector<tf::Output> terms;
    for (int i = 0; i < COEFFS_COUNT; i++) {
        auto term = ops::Multiply(mRoot, w[i], ops::Pow(mRoot, placeholder, static_cast<float>(i)));
        terms.push_back(term);
    }

    return ops::AddN(mRoot, terms);
}

void PolynomialRegression::demonstrate()
{
    InputMatrix data = InputMatrix::Zero();
    data.row(0) = Eigen::VectorXf::LinSpaced(POINTS_COUNT, -1.0f, 1.0f);

    std::vector<float> coeffs = {1, 2, 3, 4, 5, 6};

    for (int i = 0; i < POINTS_COUNT; ++i) {
        for (int j = 0; j < COEFFS_COUNT; ++j) {
            data(1, i) += coeffs[j] * std::pow(data(0, i), j);
        }
    }

    for (int i = 0; i < data.cols(); ++i) {
        data(1, i) = Eigen::internal::random<float>(data(1, i) - 1.5f, data(1, i) + 1.5) + 5.0f;
    }

    FILE* pipe = popen("gnuplot -persist", "w");

    fprintf(pipe, "plot '-' with points pt 7 lc rgb 'red', '-' with lines lc rgb 'blue'\n");

    for (int i = 0; i < polynomial::POINTS_COUNT; i++) {
        auto x = data(0, i);
        auto y = data(1, i);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");

    // Regression side
    PolynomialRegression regression;

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
