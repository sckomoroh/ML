#include <iostream>
#include <random>
#include <vector>

#include "tensorflow/cc/client/client_session.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/public/version.h"
#include "tensorflow/core/util/events_writer.h"

namespace tf = tensorflow;
namespace ops = tensorflow::ops;

// AVR = ALPHA * CUR_VAL + (1 - ALPHA) * PREV_AGV

int main()
{
    tf::Scope root = tf::Scope::NewRootScope();

    std::vector<float> inputData(100);
    std::generate(inputData.begin(), inputData.end(), []() { return std::rand() % 100; });

    auto alpha = ops::Const(root, 0.05f, {});
    auto one = ops::Const(root, 1.00f, {});
    ops::Variable previosAverage = ops::Variable{root, {}, tf::DataType::DT_FLOAT};
    ops::Placeholder currentValue{root, tf::DataType::DT_FLOAT};
    auto average = ops::Add(root, ops::Mul(root, ops::Subtract(root, one, alpha), previosAverage),
                            ops::Mul(root, alpha, currentValue));

    auto averageSummary = ops::ScalarSummary(root, "average", average);
    auto inputDataSummary = ops::ScalarSummary(root, "inputData", currentValue);
    std::vector<tensorflow::Output> summaries = {averageSummary, inputDataSummary};
    auto mergedSummaries = ops::MergeSummary(root, summaries);

    tf::EventsWriter writer("/home/yzvihunov/Documents/src/ML/log/events");

    tf::ClientSession session{root};
    std::vector<tf::Tensor> outputs;
    TF_CHECK_OK(session.Run({ops::Assign(root, previosAverage, 0.0f)}, nullptr));

    for (auto i = 0; i < inputData.size(); i++) {
        tf::Tensor currentValueTensor(tf::DT_FLOAT, {});
        currentValueTensor.scalar<float>()() = inputData[i];

        TF_CHECK_OK(session.Run({{currentValue, currentValueTensor}}, {average, mergedSummaries},
                                &outputs));

        currentValueTensor.scalar<float>()() = outputs[0].scalar<float>()();
        TF_CHECK_OK(session.Run({ops::Assign(root, previosAverage, currentValueTensor)}, nullptr));

        std::cerr << inputData[i] << " " << outputs[0].scalar<float>() << std::endl;

        const auto& summary_tensor = outputs[1];
        if (summary_tensor.dtype() == tensorflow::DT_STRING && summary_tensor.NumElements() == 1) {
            const auto& summary_string = summary_tensor.scalar<tensorflow::tstring>()();
            tf::Summary summary;
            summary.ParseFromString(summary_string);
            for (const auto& value : summary.value()) {
                tf::Event event;
                event.set_wall_time(i);
                event.set_step(i);
                event.mutable_summary()->add_value()->CopyFrom(value);
                writer.WriteEvent(event);
                TF_CHECK_OK(writer.Flush());
            }
        }
        else {
            std::cerr << "Unexpected summary tensor format." << std::endl;
        }
    }

    return 0;
}
