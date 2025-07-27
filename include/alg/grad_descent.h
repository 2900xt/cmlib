#pragma once
#include "math/matrix.h"

// Loss function
FP_DTYPE mse_loss(const Matrix& X, const Matrix& y_actual_mat, const Matrix& weights);

// Gradient calculation
Matrix mse_gradient(const Matrix& X, const Matrix& y_actual_mat, const Matrix& weights);

// Gradient descent algorithm
void gradient_descent(
    const Matrix &x, 
    const Matrix &y, 
    Matrix &weights, 
    Vector &lossArray, 
    FP_DTYPE LR, 
    int epochs, 
    int debug = 1e9
); 