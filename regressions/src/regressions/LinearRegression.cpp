#include "LinearRegression.h"

#include "tensorflow/cc/framework/gradients.h"
#include "tensorflow/core/framework/tensor.h"

namespace regression {

using namespace regression::linear;

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr float LEARNING_RATE = 0.001;
constexpr int TRAINING_EPOCHS = 1000;
constexpr float LAMBDA = 0.01;

LinearRegression::LinearRegression()
    : mRoot{tf::Scope::NewRootScope()}
    , mWeights{ops::Variable(mRoot.WithOpName("Weights"), {2}, tf::DataType::DT_FLOAT)}
    , mSession{mRoot}
{
}

void LinearRegression::trainModel(const linear::InputMatrix& matrix, bool log)
{
    auto X = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto Y = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    auto w0 = ops::Slice(mRoot, mWeights, {0}, {1});
    auto w1 = ops::Slice(mRoot, mWeights, {1}, {1});

    auto L2Regularization =
        ops::Multiply(mRoot, {LAMBDA},
                      ops::AddN(mRoot, std::vector<tf::Output>{ops::Multiply(mRoot, w0, w0),
                                                               ops::Multiply(mRoot, w1, w1)}));

    auto predictionOp = model(X);
    auto costOp = ops::Add(mRoot, L2Regularization,
                           ops::Square(mRoot, ops::Subtract(mRoot, Y, predictionOp)));

    std::vector<tf::Output> weightOutputs{mWeights};
    std::vector<tf::Output> gradients;
    TF_CHECK_OK(tf::AddSymbolicGradients(mRoot, {costOp}, weightOutputs, &gradients));
    auto trainOp = ops::ApplyGradientDescent(mRoot, mWeights, LEARNING_RATE, gradients[0]);

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, mWeights, {0.0f, 0.0f})}, nullptr));

    std::vector<tf::Tensor> outputs;
    for (int epoch = 0; epoch < TRAINING_EPOCHS; epoch++) {
        for (int i = 0; i < matrix.cols(); i++) {
            tf::ClientSession::FeedType feedType{{X, matrix(0, i)}, {Y, matrix(1, i)}};
            TF_CHECK_OK(mSession.Run(feedType, {trainOp, costOp}, &outputs));
        }

        if (log) {
            float costValue = outputs[1].scalar<float>()();
            TF_CHECK_OK(mSession.Run({w0, w1}, &outputs));
            float k1 = outputs[0].scalar<float>()();
            float k2 = outputs[1].scalar<float>()();
            std::cerr << "Coeffs: " << k1 << "," << k2 << " Cost: " << costValue << std::endl;
        }
    }
}

float LinearRegression::getPrediction(float value)
{
    auto X = ops::Placeholder(mRoot.WithOpName("value"), tf::DataType::DT_FLOAT);
    auto m = model(X);

    std::vector<tf::Tensor> outputs;
    tf::ClientSession::FeedType feed{{X, {value}}};

    TF_CHECK_OK(mSession.Run(feed, {m}, &outputs));

    return outputs[0].scalar<float>()();
}

tf::Output LinearRegression::model(const tensorflow::ops::Placeholder& placeholder)
{
    // w0 + w1*x
    auto w0 = ops::Slice(mRoot.WithOpName("w0"), mWeights, {0}, {1});
    auto w1 = ops::Slice(mRoot.WithOpName("w1"), mWeights, {1}, {1});

    return ops::Add(mRoot, w0, ops::Multiply(mRoot, placeholder, w1));
}

void LinearRegression::demonstrate()
{
    // Data generation
    InputMatrix data;
    data.resize(2, POINTS_COUNT);
    data.setZero();
    data.row(0) = Eigen::VectorXf::LinSpaced(POINTS_COUNT, 1.0f, 5.0f);
    for (int i = 0; i < data.cols(); ++i) {
        data(1, i) = Eigen::internal::random<float>(data(0, i) - 0.5f, data(0, i) + 0.5) + 5.0f;
    }

    FILE* pipe = popen("gnuplot -persist", "w");

    fprintf(pipe, "plot "
                    "'-' with points pt 7 lc rgb 'red' notitle, "
                    "'-' with lines lc rgb 'blue' notitle\n");

    for (int i = 0; i < linear::POINTS_COUNT; i++) {
        auto x = data(0, i);
        auto y = data(1, i);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");

    // Regression side
    LinearRegression regression;

    regression.trainModel(data);
    for (float x = 0.5f; x < 5.5f; x += 0.1f) {
        auto y = regression.getPrediction(x);
        fprintf(pipe, "%f %f\n", x, y);
    }

    fprintf(pipe, "e\n");
    fflush(pipe);
    fclose(pipe);
}

};  // namespace regression
