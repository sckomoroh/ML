/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "Common/Types.h"

namespace regression {

class IRegression {
public:
    virtual ~IRegression() = default;

public:
    virtual float function(std::vector<float> k, float X) = 0;

    virtual std::vector<std::vector<common::PointF>> generateData() = 0;

    virtual std::vector<float> train(std::vector<std::vector<common::PointF>> points,
                                     bool log = false) = 0;
};

}  // namespace regression