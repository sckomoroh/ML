/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "IRegression.h"
#include "Common/Types.h"

namespace regression {

class LinearRegression : public IRegression {
public:  // IRegression
    float function(std::vector<float> k, float X) override;

    std::vector<std::vector<common::PointF>> generateData() override;

    std::vector<float> train(std::vector<std::vector<common::PointF>> points, bool log = false) override;
};

}  // namespace regression
