#pragma once

#include "config.h"
class Model {
public:
    Matrix weights;
    virtual Matrix forward(const Matrix& X) const = 0;
};