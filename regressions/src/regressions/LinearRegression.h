/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "IRegression.h"

namespace regression {

class LinearRegression : public IRegression {
public:  // IRegression
    float function(std::vector<float> k, float X) override;

    void generateData(std::vector<float>& trX, std::vector<float>& trY) override;

    std::vector<float> train(const std::vector<float>& trX,
                             const std::vector<float>& trY,
                             bool log = false) override;
};

}  // namespace regression
