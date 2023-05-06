#include "logic_linear.h"

#include <iostream>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

namespace linear {

constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 100;

void generateData(std::vector<float>& trX, std::vector<float>& trY)
{
    int num_points = 101;
    trX.resize(num_points);
    for (int i = 0; i < num_points; ++i) {
        trX[i] = -1.0f + (float)i * (2.0f / float(num_points - 1));
    }

    int num_coeffs = 6;
    trY = std::vector<float>(num_points, 0.0f);

    for (int i = 0; i < num_points; ++i) {
        trY[i] += 2 * trX[i];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.33);

    for (int i = 0; i < num_points; ++i) {
        trY[i] += dis(gen);
    }
}

float calculate(const std::vector<float>& trX, const std::vector<float>& trY)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto w = ops::Variable(root, {}, tf::DataType::DT_FLOAT);

    auto model = [&root](ops::Placeholder& X, ops::Variable& w) {
        return ops::Multiply(root, X, w);
    };

    auto y_model = model(X, w);

    auto cost = ops::Square(root, ops::Subtract(root, Y, y_model));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> w_outputs;
    w_outputs.push_back(w);
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {cost}, w_outputs, &gradients));

    auto train_op = ops::ApplyGradientDescent(root, w, LEARNING_RATE, gradients[0]);
    
    tf::ClientSession session{root};
    TF_CHECK_OK(session.Run({ops::Assign(root, w, 0.0f)}, nullptr));

    std::vector<tf::Tensor> outputs;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        for (int i = 0; i < trX.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trX[i]}, {Y, trY[i]}};
            TF_CHECK_OK(session.Run(feedType, {train_op, cost}, &outputs));
        }

        TF_CHECK_OK(session.Run({w}, &outputs));
        float coeff = outputs[0].scalar<float>()();
        float cost_val = outputs[1].scalar<float>()();
        std::cerr << "K: " << coeff << " Cost: " << cost_val << std::endl;
    }

    TF_CHECK_OK(session.Run({w}, &outputs));

    float coeff = outputs[0].scalar<float>()();
    std::cerr << "Coefficient: " << coeff << std::endl;

    return coeff;
}

}  // namespace linear
