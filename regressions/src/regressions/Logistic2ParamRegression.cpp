/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "Logistic2ParamRegression.h"

#include <iostream>

#include "tensorflow/cc/framework/gradients.h"

namespace regression {

using namespace logistic2d;

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 2000;
constexpr float SENSITIVE_GATE = 0.0001;  // 0.01%
constexpr float LAMBDA = 0.0001;

Logistic2ParamRegression::Logistic2ParamRegression()
    : mRoot{tf::Scope::NewRootScope()}
    , mWeights{mRoot, {3}, tf::DataType::DT_FLOAT}
    , mSession{mRoot}
{
}

void Logistic2ParamRegression::trainModel(const InputMatrix& matrix, bool log)
{
    auto w0 = ops::Slice(mRoot, mWeights, {0}, {1});
    auto w1 = ops::Slice(mRoot, mWeights, {1}, {1});
    auto w2 = ops::Slice(mRoot, mWeights, {2}, {1});

    auto Param1 = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto Param2 = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto YS = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    auto predictionOp = model(Param1, Param2);

    // To avoid owerweight
    auto L2Regularization = ops::Multiply(
        mRoot, {LAMBDA},
        ops::AddN(mRoot, std::vector<tf::Output>{ops::Square(mRoot, w0), ops::Square(mRoot, w1),
                                                 ops::Square(mRoot, w2)}));

    // -tf.reduce_mean(tf.math.log(y_model * ys + (1 - y_model) * (1 - ys)))
    tf::Output costOp;
    costOp = ops::Neg(
        mRoot,
        ops::Log(mRoot, ops::Add(mRoot, ops::Multiply(mRoot, predictionOp, YS),
                                 ops::Multiply(mRoot, ops::Subtract(mRoot, {1.0f}, predictionOp),
                                               ops::Subtract(mRoot, {1.0f}, YS)))));

    costOp = ops::Add(mRoot, L2Regularization, costOp);

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(mWeights);
    TF_CHECK_OK(tf::AddSymbolicGradients(mRoot, {costOp}, weightOutputs, &gradients));

    auto trainOp = ops::ApplyGradientDescent(mRoot, mWeights, LEARNING_RATE, gradients[0]);

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, mWeights, {0.0f, 0.0f, 0.0f})}, nullptr));

    std::vector<tf::Tensor> outputs;
    float prevCost = 0.0f;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        float totalCost = 0.0f;
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{
                {Param1, matrix(0, i)}, {Param2, matrix(1, i)}, {YS, matrix(2, i)}};
            TF_CHECK_OK(mSession.Run(feedType, {trainOp, costOp}, &outputs));
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

    TF_CHECK_OK(mSession.Run({w0, w1, w2}, &outputs));

    if (log) {
        std::cerr << "Coefficients: "
                  << "K0: " << outputs[0].scalar<float>()()
                  << " K1: " << outputs[1].scalar<float>()()
                  << " K2: " << outputs[2].scalar<float>()() << std::endl;
    }
}

float sigmoid(float x) { return 1.0 / (1.0 + std::exp(-x)); }

float Logistic2ParamRegression::getPrediction(float val1, float val2, bool useLite)
{
    if (useLite) {
        std::vector<tf::Tensor> outputs;

        TF_CHECK_OK(mSession.Run({mWeights}, &outputs));

        auto w0 = outputs[0].tensor<float, 1>()(0);
        auto w1 = outputs[0].tensor<float, 1>()(1);
        auto w2 = outputs[0].tensor<float, 1>()(2);

        return sigmoid(w2 * val2 + w1 * val1 + w0);
    }

    auto param1 = ops::Placeholder(mRoot.WithOpName("value"), tf::DataType::DT_FLOAT);
    auto param2 = ops::Placeholder(mRoot.WithOpName("value"), tf::DataType::DT_FLOAT);

    auto m = model(param1, param2);

    std::vector<tf::Tensor> outputs;
    tf::ClientSession::FeedType feed{{param1, {val1}}, {param2, {val2}}};

    TF_CHECK_OK(mSession.Run(feed, {m}, &outputs));

    return outputs[0].scalar<float>()();
}

tf::Output Logistic2ParamRegression::model(const tensorflow::ops::Placeholder& placeholder1,
                                           const tensorflow::ops::Placeholder& placeholde2)
{
    // tf.sigmoid(w[2] * x2s + w[1] * x1s + w[0])

    auto w0 = ops::Slice(mRoot, mWeights, {0}, {1});
    auto w1 = ops::Slice(mRoot, mWeights, {1}, {1});
    auto w2 = ops::Slice(mRoot, mWeights, {2}, {1});

    return ops::Sigmoid(mRoot, ops::AddN(mRoot, std::vector<tf::Output>{
                                                    ops::Multiply(mRoot, w2, placeholde2),
                                                    ops::Multiply(mRoot, w1, placeholder1), w0}));
}

void Logistic2ParamRegression::demonstrate(LinearRegression* lineRegression)
{
    InputMatrix data = InputMatrix::Zero();
    float distance = 3.5f;
    for (auto i = 0; i < data.cols() / 2; i++) {
        data(0, i) = Eigen::internal::random<float>(-distance, distance) - 3.0f;
        data(1, i) = Eigen::internal::random<float>(-distance, distance) + 5.0f;
        data(2, i) = 0.0f;
    }

    for (auto i = data.cols() / 2; i < data.cols(); i++) {
        data(0, i) = Eigen::internal::random<float>(-distance, distance) + 5.0f;
        data(1, i) = Eigen::internal::random<float>(-distance, distance) - 2.0f;
        data(2, i) = 1.0f;
    }

    Logistic2ParamRegression regression;
    regression.trainModel(data);

    FILE* pipe = popen("gnuplot -persist", "w");

    if (lineRegression) {
        fprintf(pipe,
                "plot '-' with points pt 7 lc rgb 'red', '-' with points pt 7 lc rgb 'green', "
                "'-' with lines lc rgb 'blue'\n");
    }
    else {
        fprintf(pipe,
                "plot '-' with points pt 7 lc rgb 'red', '-' with points pt 7 lc rgb 'green', "
                "'-' with points lc rgb 'blue'\n");
    }

    for (auto i = 0; i < data.cols() / 2; i++) {
        auto x = data(0, i);
        auto y = data(1, i);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");

    for (auto i = data.cols() / 2; i < data.cols(); i++) {
        auto x = data(0, i);
        auto y = data(1, i);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");

    linear::InputMatrix lineMatrix;
    Eigen::Vector<float, POINTS_COUNT> x1 = Eigen::VectorXf::LinSpaced(POINTS_COUNT, -6.0f, 6.0f);
    Eigen::Vector<float, POINTS_COUNT> x2 = Eigen::VectorXf::LinSpaced(POINTS_COUNT, -6.0f, 6.0f);
    for (auto i = 0; i < x1.rows(); i++) {
        for (auto j = 0; j < x2.rows(); j++) {
            auto z = regression.getPrediction(x1(i), x2(j));
            if (abs(z - 0.5f) < 0.01f) {
                if (!lineRegression) {
                    fprintf(pipe, "%f %f\n", x1(i), x2(j));
                }
                else {
                    lineMatrix.conservativeResize(lineMatrix.rows(), lineMatrix.cols() + 1);
                    lineMatrix(0, lineMatrix.cols() - 1) = x1(i);
                    lineMatrix(1, lineMatrix.cols() - 1) = x2(j);
                }
            }
        }
    }

    if (lineRegression) {
        lineRegression->trainModel(lineMatrix);
        for (float x = -6.5f; x < 6.5f; x += 0.1f) {
            auto y = lineRegression->getPrediction(x);
            fprintf(pipe, "%f %f\n", x, y);
        }
    }

    fprintf(pipe, "e\n");

    fflush(pipe);
    fclose(pipe);
}

}  // namespace regression
