#include "Classification.h"

#include <algorithm>
#include <numeric>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"

namespace regression::classification {

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.001f;
constexpr int TRAINIG_EPOCHS = 1000;

InputData generateData()
{
    InputData inputData;

    inputData.resize({POINTS_COUNT, CLASS_COUNT, PARAMS_COUNT});
    auto dims = inputData.dimensions();

    for (auto pointIndex = 0; pointIndex < dims[0]; pointIndex++) {
        for (auto classIndex = 0; classIndex < dims[1]; classIndex++) {
            for (auto paramIndex = 0; paramIndex < dims[2]; paramIndex++) {
                auto center =
                    static_cast<float>(classIndex) * 2.0f + static_cast<float>(paramIndex) * 0.5f;
                inputData(pointIndex, classIndex, paramIndex) =
                    Eigen::internal::random<float>(center - .02f, center + 0.2f);
            }
        }
    }

    return inputData;
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

// void printVector(tf::Tensor& input)
//{
//     printf("[ ");
//     std::vector<float> data;
//     auto vec = input.vec<float>();
//     for (auto i = 0; i < vec.size(); i++) {
//         fprintf(stderr, "%f ", vec(i));
//     }
//     printf("]\n");
// }

template <typename TEigenType>
tf::Tensor eigenMatrixToTensor(const TEigenType& eigenObject)
{
    using Scalar = typename TEigenType::Scalar;

    constexpr int Rows = TEigenType::RowsAtCompileTime;
    constexpr int Cols = TEigenType::ColsAtCompileTime;
    constexpr bool IsDynamic = (Rows == Eigen::Dynamic || Cols == Eigen::Dynamic);

    tf::Tensor tensor(tf::DataTypeToEnum<Scalar>::value,
                      tf::TensorShape({IsDynamic ? eigenObject.rows() : Rows,
                                       IsDynamic ? eigenObject.cols() : Cols}));

    // Copy Eigen object data to TensorFlow tensor
    auto eigenData = eigenObject.data();
    auto tensorData = tensor.flat<Scalar>().data();
    std::copy(eigenData, eigenData + eigenObject.size(), tensorData);

    return tensor;
}

template <class TVector>
tf::Tensor eigenVectorToTensor(const TVector& eigenVector)
{
    tf::Tensor tensor(tf::DT_FLOAT, tf::TensorShape({eigenVector.size()}));

    // Copy Eigen vector data to TensorFlow tensor
    auto eigenVectorData = eigenVector.data();
    auto tensorData = tensor.flat<float>().data();
    std::copy(eigenVectorData, eigenVectorData + eigenVector.size(), tensorData);

    return tensor;
}

//Eigen::Vector<float, Eigen::Dynamic> tensorToEigenVector(const tensorflow::Tensor& tensor) {}

void Classification::trainig(const InputData& data)
{
    auto root = tf::Scope::NewRootScope();
    auto session = tf::ClientSession(root);

    auto maskMatrix = Eigen::Matrix<float, CLASS_COUNT, CLASS_COUNT>();
    maskMatrix.setZero();
    for (auto j = 0; j < CLASS_COUNT; j++) {
        maskMatrix(j, j) = 1.0f;
    }

    auto propertyValue = ops::Placeholder(root, tf::DataType::DT_FLOAT);
    auto maskValue = ops::Placeholder(root, tf::DataType::DT_FLOAT);

    auto weight = ops::Variable(root, {PARAMS_COUNT, CLASS_COUNT}, tf::DataType::DT_FLOAT);
    auto offset = ops::Variable(root, {CLASS_COUNT}, tf::DataType::DT_FLOAT);

    auto prediction =
        ops::Softmax(root, ops::Add(root, offset, ops::MatMul(root, propertyValue, weight)));

    auto loss = ops::Neg(
        root,
        ops::ReduceSum(root, ops::Multiply(root, maskValue, ops::Log(root, prediction)), {1}));

    std::vector<tf::Output> gradients;
    std::vector<tf::Output> weightOutputs;
    weightOutputs.push_back(weight);
    weightOutputs.push_back(offset);
    TF_CHECK_OK(tf::AddSymbolicGradients(root, {loss}, weightOutputs, &gradients));

    auto trainOp0 = ops::ApplyGradientDescent(root, weight, LEARNING_RATE, gradients[0]);
    auto trainOp1 = ops::ApplyGradientDescent(root, offset, LEARNING_RATE, gradients[1]);

    zeroVector(session, root, offset, CLASS_COUNT);
    zeroMatrix(session, root, weight, PARAMS_COUNT, CLASS_COUNT);

    std::vector<tf::Tensor> outputs;
    for (auto epoch = 0; epoch < TRAINIG_EPOCHS; epoch++) {
        for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
            for (auto classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
                Eigen::Matrix<float, 1, PARAMS_COUNT> paramsMatrix;
                paramsMatrix.setZero();
                for (auto i = 0; i < PARAMS_COUNT; i++) {
                    paramsMatrix(0, i) = data(pointIndex, classIndex, i);
                }

                auto maskTesor = eigenVectorToTensor(maskMatrix.row(classIndex));
                auto propertyTensor = eigenMatrixToTensor(paramsMatrix);

                tf::ClientSession::FeedType feedType{{propertyValue, propertyTensor},
                                                     {maskValue, maskTesor}};

                TF_CHECK_OK(session.Run({feedType}, {trainOp0, trainOp1, loss}, &outputs));
            }
        }

        if (epoch % 100 == 0) {
            fprintf(stderr, "Epoch: %d %f\n", epoch, outputs[2].scalar<float>()());
        }
    }

    TF_CHECK_OK(session.Run({weight, offset}, &outputs));

    //    auto flatWeight = outputs[0].flat<float>();
    //    auto flatOffset = outputs[1].flat<float>();

    //return {};
}
/*
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
*/
}  // namespace regression::classification
