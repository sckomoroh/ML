#pragma once

#include <vector>

namespace polynominal {

void generateData(std::vector<float>& trX, std::vector<float>& trY);

std::vector<float> calculate(const std::vector<float>& trX, const std::vector<float>& trY);

}  // namespace polynominal
