/**
 * Copyright 2023
 * Author: Yehor Zvihunov
 **/

#pragma once

#include <iostream>

#include "tensorflow/core/framework/tensor.h"

namespace utils::print {

template <class T>
void printTensor(const tensorflow::Tensor& tensor)
{
    switch (tensor.dims()) {
    case 0: {
        auto value = tensor.scalar<T>()();
        std::cerr << "Scalara: " << value << std::endl;
    } break;
    case 1: {
        auto vec = tensor.vec<T>();
        std::cerr << "Vector: [ ";
        for (int i = 0; i < tensor.dim_size(0); i++) {
            std::cerr << vec(i) << " ";
        }
        std::cerr << "]" << std::endl;
    } break;
    case 2: {
        auto mat = tensor.matrix<T>();
        std::cerr << "Matrix:\n";
        for (int i = 0; i < tensor.dim_size(0); i++) {
            std::cerr << "[ ";
            for (int j = 0; j < tensor.dim_size(0); j++) {
                std::cerr << mat(i, j) << " ";
            }
            std::cerr << "]" << std::endl;
        }
        std::cerr << std::endl;
    } break;
    }
}

}  // namespace utils