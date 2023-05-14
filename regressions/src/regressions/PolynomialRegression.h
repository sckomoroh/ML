/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "Common/Types.h"
#include "IRegression.h"

namespace regression {

class PolynomialRegression : public IRegression {
public:  // IRegression
    float function(std::vector<float> k, float X) override;

    std::vector<std::vector<common::PointF>> generateData() override;

    std::vector<float> train(std::vector<std::vector<common::PointF>> points,
                             bool log = false) override;
};

}  // namespace regression
