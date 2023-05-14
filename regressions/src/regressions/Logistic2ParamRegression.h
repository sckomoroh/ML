/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <vector>

#include "IRegression.h"

namespace regression {

/**
 * @brief Predicts belonging to one of the sets by 2 parameters
 * 
 * Predicts belonging to one of the sets by 2 parameters.
 * The regression equation  for 2 parameter is w0 + w1*param1 + w2*param2
 * 
 * If you have more than 2 parameter so the equation will be like
 * w0 + w1*param1 + w2*param2 + ... + wN*paramN
 */
class Logistic2ParamRegression : public IRegression {
public:  // IRegression
    float function(std::vector<float> k, float X) override;

    std::vector<std::vector<PointF>> generateData() override;

    std::vector<float> train(std::vector<std::vector<PointF>> points, bool log = false) override;
};

}  // namespace regression