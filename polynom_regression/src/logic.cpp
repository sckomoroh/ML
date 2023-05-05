#include <iostream>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr uint8_t COEFFS_COUNT = 6;
constexpr float LEARNING_RATE = 0.01;
constexpr int TRAINING_EPOCHS = 40;

void generateInputData(std::vector<float>& trX, std::vector<float>& trY)
{
    int num_points = 101;
    trX.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        trX[i] = -1.0f + (float)i * (2.0f / float(num_points - 1));
    }

    int num_coeffs = 6;
    std::vector<float> trY_coeffs = {1, 2, 3, 4, 5, 6};
    trY = std::vector<float>(num_points, 0.0f);

    for (int i = 0; i < num_points; ++i) {
        for (int j = 0; j < num_coeffs; ++j) {
            trY[i] += trY_coeffs[j] * std::pow(trX[i], j);
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.5);

    for (int i = 0; i < num_points; ++i) {
        trY[i] += dis(gen);
    }
}

std::vector<float> calculate(const std::vector<float>& trX, const std::vector<float>& trY)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    // Create vector with coefficients for polinom
    std::vector<ops::Variable> w;
    for (int i = 0; i < COEFFS_COUNT; i++) {
        w.emplace_back(ops::Variable{root, {}, tf::DataType::DT_FLOAT});
    }

    // Define model y = w6*x^6+w5*x^5+w4*x^4+w3*x^3+w2*x^2+w1*x
    auto model = [&root, &w](const ops::Placeholder& X) -> tf::Output {
        std::vector<tf::Output> terms;
        for (float i = 0; i < COEFFS_COUNT; i++) {
            auto term = ops::Multiply(root, w[i], ops::Pow(root, X, i));
            terms.push_back(term);
        }
        return ops::AddN(root, terms);
    };

    tf::Output y_model = model(X);

    auto cost = ops::Square(root, ops::Subtract(root, Y, y_model));

    std::vector<tf::Output> update_ops;
    for (int i = 0; i < COEFFS_COUNT; i++) {
        update_ops.push_back(ops::ApplyGradientDescent(root, w[i], LEARNING_RATE, cost));
    }

    std::vector<tf::Tensor> outputs;
    tf::ClientSession session{root};

    // Init coefficients placeholders
    for (auto i = 0; i < COEFFS_COUNT; i++) {
        TF_CHECK_OK(session.Run({ops::Assign(root, w[i], 0.0f)}, nullptr));
    }

    // Start trainig
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        for (int i = 0; i < trX.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trX[i]}, {Y, trY[i]}};

            // Perform gradient descent update
            TF_CHECK_OK(session.Run(feedType, update_ops, nullptr));

            // Update y_model with the new weights
            y_model = model(X);

            // Update the cost function based on the new y_model
            cost = ops::Square(root, ops::Sub(root, Y, y_model));
        }
        std::cerr << std::endl;
    }

    // Print coefficients
    std::cerr << "Coefficients: ";
    std::vector<float> w_val(COEFFS_COUNT);
    for (int i = 0; i < COEFFS_COUNT; i++) {
        TF_CHECK_OK(session.Run({w[i]}, &outputs));
        w_val[i] = outputs[0].scalar<float>()();
        std::cerr << outputs[0].scalar<float>()() << " ";
    }

    std::cerr << std::endl;

    return w_val;
}