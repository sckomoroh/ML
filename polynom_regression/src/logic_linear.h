#pragma once

#include <vector>

namespace linear {

void generateData(std::vector<float>& trX, std::vector<float>& trY);

float calculate(const std::vector<float>& trX, const std::vector<float>& trY);

}  // namespace linear
