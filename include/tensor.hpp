#pragma once 

#include <vector>
#define TENSOR_DTYPE long double

class Tensor {
    TENSOR_DTYPE* data;
    std::vector<int> dimensions;
public:
    Tensor(std::vector<int> dimensions)
    {
        
    }
};