#include "Classification.h"

#include <algorithm>
#include <numeric>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"

namespace classification {

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.01;

constexpr auto P1_VAL = "param1_values";
constexpr auto P2_VAL = "param2_values";
constexpr auto P3_VAL = "param3_values";

Classification::Classification()
{
    mScope = std::make_shared<tf::Scope>(tf::Scope::NewRootScope());
    mSession = std::make_shared<tf::ClientSession>(*mScope);
}

std::map<std::string, std::vector<float>> Classification::generateData()
{
    std::map<std::string, std::vector<float>> data;

    data[P1_VAL] = common::Utils::generateRange(1, 1, 10);
    data[P2_VAL] = common::Utils::generateRange(5, 5, 10);
    data[P3_VAL] = common::Utils::generateRange(8, 1, 10);

    common::Utils::randomizeData(data[P1_VAL], 0.0f, 1.0f);
    common::Utils::randomizeData(data[P2_VAL], 0.0f, 1.0f);
    common::Utils::randomizeData(data[P3_VAL], 0.0f, 1.0f);

    return data;
}

std::pair<matrixF, matrixF> Classification::prepareData(
    const std::map<std::string, std::vector<float>>& input)
{
    std::map<std::string, std::vector<float>> prop_masks;
    prop_masks[P1_VAL] = {1.0f, 0.0f, 0.0f};
    prop_masks[P2_VAL] = {0.0f, 1.0f, 0.0f};
    prop_masks[P3_VAL] = {0.0f, 0.0f, 1.0f};

    matrixF prop_values;
    matrixF masks;
    for (auto& property : input) {
        for (auto& val : property.second) {
            prop_values.push_back({val});
            masks.push_back(prop_masks[property.first]);
        }
    }

    return {prop_values, masks};
}

void zeroVector(tf::ClientSession& session, tf::Scope& root, ops::Variable& var, int l)
{
    auto init_value = tf::Tensor(tf::DataType::DT_FLOAT, {l});
    auto matrix = init_value.vec<float>();
    matrix.setZero();

    TF_CHECK_OK(session.Run({ops::Assign(root, var, init_value)}, nullptr));
}

void zeroMatrix(tf::ClientSession& session, tf::Scope& root, ops::Variable& var, int l, int h)
{
    auto init_value = tf::Tensor(tf::DataType::DT_FLOAT, {l, h});
    auto matrix = init_value.matrix<float>();
    matrix.setZero();

    TF_CHECK_OK(session.Run({ops::Assign(root, var, init_value)}, nullptr));
}

void printVector(tf::Tensor& input)
{
    printf("[ ");
    std::vector<float> data;
    auto vec = input.vec<float>();
    for (auto i = 0; i < vec.size(); i++) {
        fprintf(stderr, "%f ", vec(i));
    }
    printf("]\n");
}

std::pair<std::vector<float>, std::vector<float>> Classification::trainig(
    const std::pair<matrixF, matrixF>& data)
{
    auto& root = *mScope;
    auto& session = *mSession;

    auto trainig_epoch = 1000;
    auto training_rate = 0.001f;

    auto prop_value = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto mask = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto prop_per_count = 1;
    auto class_count = 3;

    auto weight = ops::Variable(root, {prop_per_count, class_count}, tf::DataType::DT_FLOAT);
    auto offset = ops::Variable(root, {class_count}, tf::DataType::DT_FLOAT);

    auto prediction =
        ops::Softmax(root, ops::Add(root, offset, ops::MatMul(root, prop_value, weight)));

    auto loss = ops::Neg(
        root, ops::ReduceSum(root, ops::Multiply(root, mask, ops::Log(root, prediction)), {1}));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(weight);
    weightOutputs.push_back(offset);
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {loss}, weightOutputs, &gradients));

    auto trainOp0 = ops::ApplyGradientDescent(root, weight, LEARNING_RATE, gradients[0]);
    auto trainOp1 = ops::ApplyGradientDescent(root, offset, LEARNING_RATE, gradients[1]);

    zeroVector(session, root, offset, class_count);
    zeroMatrix(session, root, weight, prop_per_count, class_count);

    auto& values = data.first;
    auto& masks = data.second;

    std::vector<tf::Tensor> outputs;
    for (auto epoch = 0; epoch < trainig_epoch; epoch++) {
        for (auto i = 0; i < values.size(); i++) {
            tf::Tensor prop_value_tensor{tf::DataType::DT_FLOAT, {1, 1}};
            std::copy_n(values[i].begin(), values[i].size(),
                        prop_value_tensor.flat<float>().data());

            tf::Tensor mask_tensor{tf::DataType::DT_FLOAT, {3}};
            std::copy_n(masks[i].begin(), masks[i].size(), mask_tensor.flat<float>().data());

            tf::ClientSession::FeedType feedType{{prop_value, prop_value_tensor},
                                                 {mask, mask_tensor}};
            TF_CHECK_OK(session.Run({feedType}, {trainOp0, trainOp1, loss}, &outputs));
        }

        if (epoch % 100 == 0) {
            fprintf(stderr, "Epoch: %d %f\n", epoch, outputs[2].scalar<float>()());
        }
    }

    TF_CHECK_OK(session.Run({weight, offset}, &outputs));
    auto flat_weight = outputs[0].flat<float>();
    auto flat_offset = outputs[1].flat<float>();

    std::vector<float> offset_vec(flat_offset.data(), flat_offset.data() + flat_offset.size());
    std::vector<float> weight_vec{flat_weight.data(), flat_weight.data() + flat_weight.size()};

    return {weight_vec, offset_vec};
}

void Classification::verify(const std::pair<matrixF, matrixF>& data,
                            std::pair<std::vector<float>, std::vector<float>> vars)
{
    auto& root = *mScope;
    auto& session = *mSession;

    auto& weight_val = vars.first;
    auto& offset_val = vars.second;

    std::vector<bool> result;

    auto prop_value = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto val_max = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    tf::Tensor weight_tensor{tf::DataType::DT_FLOAT, {1, 3}};
    std::copy_n(weight_val.begin(), weight_val.size(), weight_tensor.flat<float>().data());

    tf::Tensor offset_tensor{tf::DataType::DT_FLOAT, {3}};
    std::copy_n(offset_val.begin(), offset_val.size(), offset_tensor.flat<float>().data());

    auto weight = ops::Variable(root, weight_tensor.shape(), tf::DataType::DT_FLOAT);
    auto offset = ops::Variable(root, offset_tensor.shape(), tf::DataType::DT_FLOAT);

    TF_CHECK_OK(session.Run({ops::Assign(root, offset, offset_tensor)}, nullptr));
    TF_CHECK_OK(session.Run({ops::Assign(root, weight, weight_tensor)}, nullptr));

    auto prediction =
        ops::Softmax(root, ops::Add(root, offset, ops::MatMul(root, prop_value, weight)));

    auto& values = data.first;
    auto& masks = data.second;

    std::vector<tf::Tensor> outputs;
    for (auto i = 0; i < data.first.size(); i++) {
        tf::Tensor prop_value_tensor{tf::DataType::DT_FLOAT, {1, 1}};
        std::copy_n(values[i].begin(), values[i].size(), prop_value_tensor.flat<float>().data());

        tf::Tensor mask_tensor{tf::DataType::DT_FLOAT, {3}};
        std::copy_n(masks[i].begin(), masks[i].size(), mask_tensor.flat<float>().data());

        TF_CHECK_OK(session.Run({{prop_value, prop_value_tensor}}, {prediction}, &outputs));

        std::vector<float> tensor_values(outputs[0].flat<float>().data(),
                                         outputs[0].flat<float>().data() +
                                             outputs[0].flat<float>().size());

        auto maxPredictedIter = std::max_element(tensor_values.begin(), tensor_values.end());
        auto maxPpredicted = std::distance(tensor_values.begin(), maxPredictedIter);

        auto maxRealIter = std::max_element(masks[i].begin(), masks[i].end());
        auto maxReal = std::distance(masks[i].begin(), maxRealIter);

        result.push_back(maxPpredicted == maxReal);
    }

    auto sum = (float)std::accumulate(result.begin(), result.end(), 0);
    auto accurency = sum / (float)result.size();

    printf("Accurancy: %f\n", accurency);
}

}  // namespace classification
