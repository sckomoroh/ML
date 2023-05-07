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
constexpr int POINTS_COUNT = 101;

void generateData(std::vector<float>& trX, std::vector<float>& trY)
{
    trX.resize(POINTS_COUNT);
    for (int i = 0; i < POINTS_COUNT; ++i) {
        trX[i] = -1.0f + (float)i * (2.0f / float(POINTS_COUNT - 1));
    }

    trY = std::vector<float>(POINTS_COUNT, 0.0f);

    for (int i = 0; i < POINTS_COUNT; ++i) {
        trY[i] += 2 * trX[i];
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dis(0.0, 0.33);

    for (int i = 0; i < POINTS_COUNT; ++i) {
        trY[i] += dis(gen);
    }
}

float calculate(const std::vector<float>& trX, const std::vector<float>& trY, bool log)
{
    tf::Scope root = tf::Scope::NewRootScope();

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto weight = ops::Variable(root, {}, tf::DataType::DT_FLOAT);

    auto modelFunction = [&root](ops::Placeholder& X, ops::Variable& weight) {
        return ops::Multiply(root, X, weight);
    };

    auto modelOp = modelFunction(X, weight);

    auto costOp = ops::Square(root, ops::Subtract(root, Y, modelOp));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(weight);
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {costOp}, weightOutputs, &gradients));

    auto trainOp = ops::ApplyGradientDescent(root, weight, LEARNING_RATE, gradients[0]);

    tf::ClientSession session{root};
    TF_CHECK_OK(session.Run({ops::Assign(root, weight, 0.0f)}, nullptr));

    std::vector<tf::Tensor> outputs;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        for (int i = 0; i < trX.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trX[i]}, {Y, trY[i]}};
            TF_CHECK_OK(session.Run(feedType, {trainOp, costOp}, &outputs));
        }

        if (log) {
            TF_CHECK_OK(session.Run({weight}, &outputs));
            float coeff = outputs[0].scalar<float>()();
            float cost_val = outputs[1].scalar<float>()();
            std::cerr << "K: " << coeff << " Cost: " << cost_val << std::endl;
        }
    }

    TF_CHECK_OK(session.Run({weight}, &outputs));

    float coeff = outputs[0].scalar<float>()();
    if (log) {
        std::cerr << "Coefficient: " << coeff << std::endl;
    }

    return coeff;
}

}  // namespace linear
