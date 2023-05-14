#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>

#include "Common/Utils.h"

namespace tensorflow {

class Scope;
class ClientSession;

}  // namespace tensorflow

namespace classification {

using matrixF = std::vector<std::vector<float>>;

class Classification {
private:
    std::shared_ptr<tensorflow::Scope> mScope;
    std::shared_ptr<tensorflow::ClientSession> mSession;

public:    
    Classification();
public:
    std::map<std::string, std::vector<float>> generateData();

    std::pair<matrixF, matrixF> prepareData(const std::map<std::string, std::vector<float>>& input);

    std::pair<std::vector<float>, std::vector<float>> trainig(
        const std::pair<matrixF, matrixF>& data);

    void verify(const std::pair<matrixF, matrixF>& data,
                std::pair<std::vector<float>, std::vector<float>>);
};

}  // namespace classification
