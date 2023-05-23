/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "Classification.h"

#include <algorithm>
#include <numeric>

#include "Common/PrintUtils.h"
#include "Common/TensorUtils.h"

namespace regression {

using namespace classification;

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.0001f;
constexpr int TRAINIG_EPOCHS = 1000;

Classification::Classification()
    : mRoot{tf::Scope::NewRootScope()}
    , mWeights{ops::Variable(mRoot, {PARAMS_COUNT, CLASS_COUNT}, tf::DataType::DT_FLOAT)}
    , mOffsets{ops::Variable(mRoot, {CLASS_COUNT}, tf::DataType::DT_FLOAT)}
    , mSession{mRoot}
{
}

tensorflow::Output Classification::model(const tensorflow::ops::Placeholder& placeholder)
{
    return ops::Softmax(mRoot,
                        ops::Add(mRoot, mOffsets, ops::MatMul(mRoot, placeholder, mWeights)));
}

int Classification::getPrediction(const Eigen::Matrix<float, 1, PARAMS_COUNT>& value)
{
    auto propertyValue = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    auto valueTensor = utils::tensor::tensorFromMatrix(value);
    auto m = model(propertyValue);

    tf::ClientSession::FeedType feed{{propertyValue, valueTensor}};

    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(mSession.Run(feed, {m}, &outputs));

    TF_CHECK_OK(mSession.Run({ops::ArgMax(mRoot, outputs[0], 1)}, &outputs));

    return outputs[0].vec<int64_t>()(0);
}

void Classification::trainModel(const InputData& data, const classification::MaskData& mask)
{
    auto propertyValue = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto maskValue = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    auto predictionOp = model(propertyValue);

    auto lossOp = ops::Neg(
        mRoot,
        ops::ReduceSum(mRoot, ops::Multiply(mRoot, maskValue, ops::Log(mRoot, predictionOp)), {1}));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(mWeights);
    weightOutputs.push_back(mOffsets);

    TF_CHECK_OK(tf::AddSymbolicGradients(mRoot, {lossOp}, weightOutputs, &gradients));

    // Define the maximum and minimum values for the clipped gradients
    const float clip_value_min = -1.0f;
    const float clip_value_max = 1.0f;

    // Apply gradient clipping (Just as example for the future)
    //    auto clipped_gradient_0 = ops::ClipByValue(mRoot, gradients[0], clip_value_min, clip_value_max);
    //    auto clipped_gradient_1 = ops::ClipByValue(mRoot, gradients[1], clip_value_min, clip_value_max);

    auto trainOp1 = ops::ApplyGradientDescent(mRoot, mOffsets, LEARNING_RATE, gradients[1]);
    auto trainOp0 = ops::ApplyGradientDescent(mRoot, mWeights, LEARNING_RATE, gradients[0]);

    zeroMatrix(mWeights, PARAMS_COUNT, CLASS_COUNT);
    zeroVector(mOffsets, CLASS_COUNT);

    bool stop = false;
    std::vector<tf::Tensor> outputs;
    for (auto epoch = 0; epoch < TRAINIG_EPOCHS; epoch++) {
        for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
            Eigen::Matrix<float, 1, PARAMS_COUNT> paramsMatrix;
            paramsMatrix.setZero();
            for (auto i = 0; i < PARAMS_COUNT; i++) {
                paramsMatrix(0, i) = data(pointIndex, i);
            }

            Eigen::Vector<float, CLASS_COUNT> maskMatrix;
            maskMatrix.setZero();
            for (auto i = 0; i < CLASS_COUNT; i++) {
                maskMatrix(i) = mask(pointIndex, i);
            }

            auto maskTesor = utils::tensor::tensorFromVector(maskMatrix);
            auto propertyTensor = utils::tensor::tensorFromMatrix(paramsMatrix);

            tf::ClientSession::FeedType feedType{{propertyValue, propertyTensor},
                                                 {maskValue, maskTesor}};

            TF_CHECK_OK(mSession.Run({feedType}, {trainOp0, trainOp1, lossOp}, &outputs));
        }

        if (epoch % 100 == 0) {
            fprintf(stderr, "Epoch: %d %f\n", epoch, outputs[2].scalar<float>()());
        }
    }
}

void Classification::verify(const classification::InputData& data)
{
    auto maskMatrix = Eigen::Matrix<float, CLASS_COUNT, CLASS_COUNT>();
    maskMatrix.setZero();
    for (auto j = 0; j < CLASS_COUNT; j++) {
        maskMatrix(j, j) = 1.0f;
    }

    tf::Tensor predictions{tf::DataType::DT_FLOAT, {DATA_COUNT}};
    std::vector<tf::Tensor> outputs;
    for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
        for (auto classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
            Eigen::Matrix<float, 1, PARAMS_COUNT> params;
            params.setZero();
            for (auto i = 0; i < PARAMS_COUNT; i++) {
                params(0, i) = data(pointIndex + (classIndex * POINTS_COUNT), i);
            }

            auto prediction = getPrediction(params);
            auto maskTesor = utils::tensor::tensorFromVector(maskMatrix.col(classIndex));
            TF_CHECK_OK(mSession.Run({ops::ArgMax(mRoot, maskTesor, 0)}, &outputs));

            predictions.vec<float>()(pointIndex + (classIndex * POINTS_COUNT)) =
                static_cast<float>(outputs[0].scalar<int64_t>()() == prediction);
        }
    }

    TF_CHECK_OK(mSession.Run({ops::ReduceMean(mRoot, predictions, 0)}, &outputs));

    auto accurance = outputs[0].scalar<float>()();

    std::cout << "Accurancy: " << accurance << std::endl;
}

template <class TData>
TData shuffleMatrix(const TData& inputData, const Eigen::Vector<int64_t, DATA_COUNT>& indexes)
{
    TData result;
    for (auto i = 0; i < inputData.rows(); i++) {
        result.row(i) = inputData.row(indexes[i]);
    }

    return result;
}

void Classification::demonstrate()
{
    InputData inputData;
    MaskData maskData;

    for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
        for (auto classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
            for (auto paramIndex = 0; paramIndex < PARAMS_COUNT; paramIndex++) {
                auto center = (1.0f + static_cast<float>(classIndex)) * 2.0f +
                              (1.0f + static_cast<float>(paramIndex)) * 2.0f;
                inputData(pointIndex + (classIndex * POINTS_COUNT), paramIndex) =
                    Eigen::internal::random<float>(center - 0.2f, center + 0.5f);
            }
        }
    }

    for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
        for (auto classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
            auto index = pointIndex + (classIndex * POINTS_COUNT);
            maskData(index, classIndex) = 1.0f;
        }
    }

    // Shuffle train data to avoid shift weights to one class
    Eigen::Vector<int64_t, DATA_COUNT> vector =
        Eigen::Vector<int64_t, DATA_COUNT>::LinSpaced(DATA_COUNT, 0, DATA_COUNT);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(vector.data(), vector.data() + DATA_COUNT, g);

    maskData = shuffleMatrix(maskData, vector);
    inputData = shuffleMatrix(inputData, vector);

    Classification classification;
    classification.trainModel(inputData, maskData);

    InputData testData;

    for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
        for (auto classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
            for (auto paramIndex = 0; paramIndex < PARAMS_COUNT; paramIndex++) {
                auto center = (1.0f + static_cast<float>(classIndex)) * 2.0f +
                              (1.0f + static_cast<float>(paramIndex)) * 2.0f;
                auto paramValue = Eigen::internal::random<float>(center - .5f, center + 0.5f);
                testData(pointIndex + (classIndex * POINTS_COUNT), paramIndex) = paramValue;
            }
        }
    }

    classification.verify(testData);
}

void Classification::zeroVector(ops::Variable& var, int l)
{
    auto init_value = tf::Tensor(tf::DataType::DT_FLOAT, {l});
    auto matrix = init_value.vec<float>();
    matrix.setZero();

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, var, init_value)}, nullptr));
}

void Classification::zeroMatrix(ops::Variable& var, int l, int h)
{
    auto init_value = tf::Tensor(tf::DataType::DT_FLOAT, {l, h});
    auto matrix = init_value.matrix<float>();
    matrix.setZero();

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, var, init_value)}, nullptr));
}

}  // namespace regression
