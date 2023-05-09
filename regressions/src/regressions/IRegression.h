/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

namespace regression {

class IRegression {
public:
    struct PointF {
        float x = 0.0f;
        float y = 0.0f;
    };

public:
    virtual ~IRegression() = default;

public:
    virtual float function(std::vector<float> k, float X) = 0;

    virtual std::vector<std::vector<PointF>> generateData() = 0;

    virtual std::vector<float> train(std::vector<std::vector<PointF>> points, bool log = false) = 0;
};

}  // namespace regression