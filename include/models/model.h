#pragma once

#include "config.h"

// Base class for the classic (Matrix based) models trained with gradient_descent()
class Model {
public:
    Matrix weights;
    virtual ~Model() {}
    virtual Matrix forward(const Matrix& X) const = 0;
};
