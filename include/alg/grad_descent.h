#pragma once
#include "config.h"
#include "models/model.h"

// Loss function: mean squared error (halved) of the model's predictions
FP_DTYPE mse_loss(const Matrix& X, const Matrix& y_actual_mat, const Model& model);

// Gradient of mse_loss w.r.t. every model weight, estimated with central finite differences
Matrix mse_gradient(const Matrix& X, const Matrix& y_actual_mat, Model& model);

// Gradient descent algorithm
void gradient_descent(
    const Matrix &x, 
    const Matrix &y, 
    Model &model, 
    Vector &lossArray, 
    FP_DTYPE LR, 
    int epochs, 
    int debug = 1e9
);
