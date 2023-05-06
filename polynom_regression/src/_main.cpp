#include <iostream>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

int main()
{
    tf::Scope root = tf::Scope::NewRootScope();

    float learning_rate = 0.01;
    int training_epochs = 40;

    std::vector<float> trX(101);
    float start = -1.0;
    float end = 1.0;
    float step = (end - start) / 100;

    for (int i = 0; i < trX.size(); i++) {
        trX[i] = start + step * i;
    }

    int num_coeffs = 6;
    std::vector<float> trY_coeffs = {1, 2, 3, 4, 5, 6};
    std::vector<float> trY(101, 0);

    for (int i = 0; i < trX.size(); i++) {
        for (int j = 0; j < num_coeffs; j++) {
            trY[i] += trY_coeffs[j] * std::pow(trX[i], j);
        }
    }

    auto X = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    std::vector<ops::Variable> w;
    for (int i = 0; i < num_coeffs; i++) {
        ops::Variable var(root, {}, tf::DataType::DT_FLOAT);
        w.emplace_back(std::move(var));
    }

    std::vector<tf::Output> init_ops;
    for (int i = 0; i < num_coeffs; i++) {
        init_ops.push_back(ops::Assign(root, w[i], 0.0f));
    }

    auto model = [&root, num_coeffs, &w](const ops::Placeholder& X) -> tf::Output {
        std::vector<tf::Output> terms;
        for (int i = 0; i < num_coeffs; i++) {
            auto term = ops::Multiply(root, w[i], ops::Pow(root, X, i));
            terms.push_back(term);
        }

        return ops::AddN(root, terms);
    };

    auto y_model = model(X);

    auto cost = ops::Square(root, ops::Sub(root, Y, y_model));

    std::vector<tf::Output> update_ops;
    for (int i = 0; i < num_coeffs; i++) {
        update_ops.push_back(ops::ApplyGradientDescent(root, w[i], learning_rate, cost));
    }

    tf::ClientSession session(root);
    std::vector<tf::Tensor> outputs;

    TF_CHECK_OK(session.Run(init_ops, &outputs));

    for (int i = 0; i < num_coeffs; i++) {
        TF_CHECK_OK(session.Run({w[i]}, nullptr));
    }

    for (int epoch = 0; epoch < training_epochs; epoch++) {
        for (int i = 0; i < trX.size(); i++) {
            tf::ClientSession::FeedType feedType{{X, trX[i]}, {Y, trY[i]}};
            session.Run(feedType, update_ops, nullptr);
        }
    }

    std::vector<float> w_val(num_coeffs);
    for (int i = 0; i < num_coeffs; i++) {
        TF_CHECK_OK(session.Run({w[i]}, &outputs));
        outputs[0].scalar<float>()();
        std::cout << outputs[0].scalar<float>()() << " ";
    }

    std::endl;
}
