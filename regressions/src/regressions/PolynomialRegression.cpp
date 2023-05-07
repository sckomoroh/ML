/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "PolynomialRegression.h"

#include <iostream>
#include <limits>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradient_checker.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr uint8_t COEFFS_COUNT = 6;
constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 100;
constexpr int POINTS_COUNT = 101;

namespace regression {

float PolynomialRegression::function(std::vector<float> k, float X)
{
    float value = 0.0f;
    for (int j = 0; j < k.size(); ++j) {
        value += k[j] * std::pow(X, j);
    }

    return value;
}

void PolynomialRegression::generateData(std::vector<float>& trX, std::vector<float>& trY)
{
    float leftLimit = -1.0f;
    float rightLimit = 1.0f;
    trX.resize(POINTS_COUNT);
    for (int i = 0; i < POINTS_COUNT; ++i) {
        trX[i] = leftLimit + i * abs(leftLimit - rightLimit) / POINTS_COUNT;
    }

    float step = abs(leftLimit - rightLimit) / (float)POINTS_COUNT;
    for (float i = leftLimit; i < rightLimit; i = i + step) {
        trX[i] = i;
    }

    std::vector<float> trYCoeffs = {1, 2, 3, 4, 5, 6};
    trY = std::vector<float>(POINTS_COUNT, 0.0f);

    for (int i = 0; i < POINTS_COUNT; ++i) {
        for (int j = 0; j < COEFFS_COUNT; ++j) {
            trY[i] += trYCoeffs[j] * std::pow(trX[i], j);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.5);

    for (int i = 0; i < POINTS_COUNT; ++i) {
        trY[i] += dis(gen);
    }
}

std::vector<float> PolynomialRegression::train(const std::vector<float>& trX,
                                               const std::vector<float>& trY,
                                               bool log)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    std::vector<ops::Variable> weights;
    for (int i = 0; i < COEFFS_COUNT; i++) {
        weights.emplace_back(ops::Variable{root, {}, tf::DataType::DT_FLOAT});
    }

    // Define model y = w6*x^6+w5*x^5+w4*x^4+w3*x^3+w2*x^2+w1*x
    auto modelFunction = [&root, &weights](const ops::Placeholder& X) -> tf::Output {
        std::vector<tf::Output> terms;
        for (float i = 0; i < COEFFS_COUNT; i++) {
            auto term = ops::Multiply(root, weights[i], ops::Pow(root, X, i));
            terms.push_back(term);
        }
        return ops::AddN(root, terms);
    };

    tf::Output modelOp = modelFunction(X);

    tf::Output costOp = ops::Square(root, ops::Subtract(root, Y, modelOp));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightsOutputs(weights.begin(), weights.end());
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {costOp}, weightsOutputs, &gradients));

    std::vector<tf::Output> updateOps;
    for (int i = 0; i < COEFFS_COUNT; i++) {
        updateOps.push_back(
            ops::ApplyGradientDescent(root, weights[i], LEARNING_RATE, gradients[i]));
    }

    updateOps.push_back(costOp);

    std::vector<tf::Tensor> outputs;
    tf::ClientSession session{root};

    // Init coefficients variables
    for (auto i = 0; i < COEFFS_COUNT; i++) {
        TF_CHECK_OK(session.Run({ops::Assign(root, weights[i], 0.0f)}, nullptr));
    }

    // Start training
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        float totalCost = 0.0f;
        for (int i = 0; i < trX.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trX[i]}, {Y, trY[i]}};
            TF_CHECK_OK(session.Run(feedType, updateOps, &outputs));
            totalCost += outputs[6].scalar<float>()();
        }

        if (log) {
            std::cout << "Cost: " << totalCost << " Epoch " << epoch << " weights: ";
            for (int i = 0; i < COEFFS_COUNT; i++) {
                TF_CHECK_OK(session.Run({weights[i]}, &outputs));
                std::cerr << outputs[0].scalar<float>()() << " ";
            }

            std::cerr << std::endl;
        }
    }

    if (log) {
        std::cerr << "Coefficients: ";
    }

    std::vector<float> weightsResult(COEFFS_COUNT);
    for (int i = 0; i < COEFFS_COUNT; i++) {
        TF_CHECK_OK(session.Run({weights[i]}, &outputs));
        weightsResult[i] = outputs[0].scalar<float>()();
        if (log) {
            std::cerr << outputs[0].scalar<float>()() << " ";
        }
    }

    std::cerr << std::endl;

    return weightsResult;
}

}  // namespace regression
