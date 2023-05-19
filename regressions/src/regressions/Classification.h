#pragma once

#include <map>
#include <vector>

#include "tensorflow/core/framework/tensor.h"

namespace regression::classification {

constexpr int POINTS_COUNT = 100;
constexpr int PARAMS_COUNT = 2;
constexpr int CLASS_COUNT = 3;

using InputData = Eigen::Tensor<float, 3>;
using OutputData =
    std::pair<Eigen::Vector<float, CLASS_COUNT>, Eigen::Matrix<float, PARAMS_COUNT, CLASS_COUNT>>;

InputData generateData();

class Classification {
public:
public:
    void trainig(const InputData& data);

    void verify(const InputData& data, const OutputData& outputData);
};

}  // namespace regression::classification
