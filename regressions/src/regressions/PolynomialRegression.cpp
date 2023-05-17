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

namespace regression::polynomial {

InputMatrix generateData()
{
    InputMatrix matrix = InputMatrix::Zero();
    matrix.row(0) = Eigen::VectorXf::LinSpaced(POINTS_COUNT, -1.0f, 1.0f);

    std::vector<float> trYCoeffs = {1, 2, 3, 4, 5, 6};

    for (int i = 0; i < POINTS_COUNT; ++i) {
        for (int j = 0; j < COEFFS_COUNT; ++j) {
            matrix(1, i) += trYCoeffs[j] * std::pow(matrix(0, i), j);
        }
    }

    for (int i = 0; i < matrix.cols(); ++i) {
        matrix(1, i) =
            Eigen::internal::random<float>(matrix(1, i) - 0.5f, matrix(1, i) + 0.5) + 5.0f;
    }

    return matrix;
}

Eigen::Matrix<float, 6, 1> PolynomialRegression::train(const InputMatrix& matrix, bool log)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    std::vector<ops::Variable> weights;
    for (int i = 0; i < COEFFS_COUNT; i++) {
        weights.emplace_back(ops::Variable{root, {}, tf::DataType::DT_FLOAT});
    }

    // Define model y = w5*x^5+w4*x^4+w3*x^3+w2*x^2+w1*x^1+w0
    auto predictionFunction = [&root, &weights](const ops::Placeholder& X) -> tf::Output {
        std::vector<tf::Output> terms;
        for (float i = 0; i < COEFFS_COUNT; i++) {
            auto term = ops::Multiply(root, weights[i], ops::Pow(root, X, i));
            terms.push_back(term);
        }

        return ops::AddN(root, terms);
    };

    tf::Output predictionOp = predictionFunction(X);

    // To avoid overweight
    // I dodn't know why but with using the ops::Square the program crashes
    std::vector<tf::Output> terms;
    for (auto weight : weights) {
        terms.push_back(ops::Multiply(root, weight, weight));
    }

    auto L2Regularization = ops::Multiply(root, ops::Const(root, LAMBDA), ops::AddN(root, terms));

    tf::Output costOp =
        ops::Add(root, ops::Square(root, ops::Subtract(root, Y, predictionOp)), L2Regularization);

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
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{{X, matrix(0, i)}, {Y, matrix(1, i)}};
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

    Eigen::Matrix<float, 6, 1> weightsResult(COEFFS_COUNT);
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

}  // namespace regression::polynomial
