/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

namespace regression {

class IRegression {
public:
    virtual ~IRegression() = default;

public:
    virtual float function(std::vector<float> k, float X) = 0;

    virtual void generateData(std::vector<float>& trX, std::vector<float>& trY) = 0;

    virtual std::vector<float> train(const std::vector<float>& trX,
                                     const std::vector<float>& trY,
                                     bool log = false) = 0;
};

}  // namespace regression