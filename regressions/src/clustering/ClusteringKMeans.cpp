/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#include "ClusteringKMeans.h"

#include "Common/TensorUtils.h"

namespace clustering {

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

constexpr int EPOCH_COUNT = 100;

using namespace k_means;

ClusteringKMeans::ClusteringKMeans()
    : mRoot{tf::Scope::NewRootScope()}
    , mSession{mRoot}
    , mCentroids{ops::Variable(mRoot, {CLASS_COUNT, PARAM_COUNT}, tf::DataType::DT_FLOAT)}
{
}

tensorflow::Output ClusteringKMeans::model(const ops::Placeholder& pointsPlaceholder)
{
    auto expandedVectors = ops::ExpandDims(mRoot, pointsPlaceholder, 0);
    auto expandedCentroids = ops::ExpandDims(mRoot, mCentroids, 1);
    auto distances = ops::ReduceSum(
        mRoot, ops::Square(mRoot, ops::Subtract(mRoot, expandedVectors, expandedCentroids)), 2);
    return ops::ArgMin(mRoot, distances, 0);
}

template <class T>
void printTensor2(const tf::Tensor& tensor)
{
    auto matrix = tensor.matrix<T>();
    auto rows = matrix.dimension(0);
    auto cols = matrix.dimension(1);
    std::cerr << "[";
    for (int row = 0; row < rows; row++) {
        std::cerr << "[";
        for (int col = 0; col < cols; col++) {
            std::cerr << " " << matrix(row, col);
        }
        std::cerr << "]" << std::endl;
    }
    std::cerr << "]" << std::endl;
}

void ClusteringKMeans::trainModel(const k_means::InputData& data)
{
    auto pointsPlaceholder = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);
    auto oldCentroids = ops::Variable(mRoot, {CLASS_COUNT, PARAM_COUNT}, tf::DataType::DT_FLOAT);

    tf::Tensor init{tf::DataType::DT_FLOAT, {CLASS_COUNT, PARAM_COUNT}};
    init.matrix<float>().setZero();
    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, oldCentroids, init)}, nullptr));

    tf::Tensor centroitsInitTensor{tf::DataType::DT_FLOAT, {CLASS_COUNT, PARAM_COUNT}};
    for (auto i = 0; i < CLASS_COUNT; i++) {
        for (auto j = 0; j < PARAM_COUNT; j++) {
            centroitsInitTensor.matrix<float>()(i, j) = data(i, j);
        }
    }

    TF_CHECK_OK(mSession.Run({ops::Assign(mRoot, mCentroids, centroitsInitTensor)}, nullptr));

    int currentEpoch = 0;
    bool isConverged = false;
    auto mins = model(pointsPlaceholder);

    // Get sums of all points of centroids
    auto sums = ops::UnsortedSegmentSum(mRoot, pointsPlaceholder, mins, 2);
    // Get counts of points that belonds to the each segment
    auto counts = ops::UnsortedSegmentSum(mRoot, ops::OnesLike(mRoot, pointsPlaceholder), mins, 2);
    // Get the means values
    auto recomputeCentroids = ops::Assign(mRoot, mCentroids, ops::Div(mRoot, sums, counts));

    auto converged = ops::ReduceAll(mRoot, ops::Equal(mRoot, oldCentroids, mCentroids), {0, 1});
    auto saveOldCentroids = ops::Assign(mRoot, oldCentroids, mCentroids);

    TF_CHECK_OK(mSession.Run({saveOldCentroids}, nullptr));

    auto assignNewCentroids = [this](const tf::Tensor& t) {
        return ops::Assign(mRoot, mCentroids, t);
    };

    tf::Tensor pointsTensor = utils::tensor::tensorFromMatrix(data);

    std::vector<tf::Tensor> out;

    while (currentEpoch++ < EPOCH_COUNT && !isConverged) {
        tf::ClientSession::FeedType feed{{pointsPlaceholder, pointsTensor}};

        TF_CHECK_OK(mSession.Run(feed, {recomputeCentroids}, nullptr));
        TF_CHECK_OK(mSession.Run({converged}, &out));
        TF_CHECK_OK(mSession.Run(feed, {saveOldCentroids}, nullptr));

        isConverged = out[0].scalar<bool>()();
        std::cerr << "Epoch: " << currentEpoch << std::endl;
    }
}

PredictionData ClusteringKMeans::getPrediction(const k_means::InputData& data)
{
    auto pointsPlaceholder = ops::Placeholder(mRoot, tf::DataType::DT_FLOAT);

    auto mins = model(pointsPlaceholder);

    auto pointsTensor = utils::tensor::tensorFromMatrix(data);

    std::vector<tf::Tensor> out;
    tf::ClientSession::FeedType feed{{pointsPlaceholder, pointsTensor}};
    TF_CHECK_OK(mSession.Run(feed, {mins}, &out));

    return Eigen::Map<Eigen::Vector<int64_t, Eigen::Dynamic>>(out[0].vec<int64_t>().data(),
                                                              out[0].NumElements());
}

void ClusteringKMeans::demonstrate()
{
    InputData inputData;

    auto classIndex = 0;
    for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
        inputData(pointIndex + (classIndex * POINTS_COUNT), 0) =
            Eigen::internal::random<float>(0.0f, 2.0f);
        inputData(pointIndex + (classIndex * POINTS_COUNT), 1) =
            Eigen::internal::random<float>(0.0f, 2.0f);
    }

    classIndex = 1;
    for (auto pointIndex = 0; pointIndex < POINTS_COUNT; pointIndex++) {
        inputData(pointIndex + (classIndex * POINTS_COUNT), 0) =
            Eigen::internal::random<float>(2.0f, 4.0f);
        inputData(pointIndex + (classIndex * POINTS_COUNT), 1) =
            Eigen::internal::random<float>(2.0f, 4.0f);
    }

    ClusteringKMeans clustering;
    clustering.trainModel(inputData);
    auto pointIndexes = clustering.getPrediction(inputData);
    std::vector<tf::Tensor> out;
    TF_CHECK_OK(clustering.mSession.Run({clustering.mCentroids}, &out));

    FILE* pipe = popen("gnuplot -persist", "w");

    // Plot centroids
    fprintf(pipe, "plot "
                  "'-' with points pt 7 lc rgb 'blue' notitle, "
                  "'-' with points pt 7 lc rgb 'red' notitle, "
                  "'-' with points pt 7 lc rgb 'green' notitle, "
                  "'-' with points pt 7 lc rgb 'yellow' notitle\n");

    for (auto classIndex = 0; classIndex < CLASS_COUNT; classIndex++) {
        for (auto pointIndex = 0; pointIndex < POINTS_COUNT * CLASS_COUNT; pointIndex++) {
            auto pntIndex = pointIndexes(pointIndex);
            if (pntIndex == classIndex) {
                float x = inputData(pointIndex, 0);
                float y = inputData(pointIndex, 1);
                fprintf(pipe, "%f %f\n", x, y);
            }
        }
        fprintf(pipe, "e\n");
    }

    for (auto i = 0; i < CLASS_COUNT; i++) {
        float x = out[0].matrix<float>()(i, 0);
        float y = out[0].matrix<float>()(i, 1);
        fprintf(pipe, "%f %f\n", x, y);
        fprintf(pipe, "e\n");
    }

    fflush(pipe);
    fclose(pipe);
}

}  // namespace clustering
