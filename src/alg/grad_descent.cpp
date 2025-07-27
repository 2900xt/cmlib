#include "alg/grad_descent.h"
#include <iostream>

FP_DTYPE mse_loss(const Matrix& X, const Matrix& y_actual_mat, const Matrix& weights)
{
    Matrix y_test_mat = mdot(X, weights);
    Vector y_test = mtranspose(y_test_mat)[0];
    Vector y_actual = mtranspose(y_actual_mat)[0];

    FP_DTYPE cost = vsum(vsquare(vsubtract(y_actual, y_test)));
    return cost / (2*X.size());
}

Matrix mse_gradient(const Matrix& X, const Matrix& y_actual_mat, const Matrix& weights)
{
    Matrix y_test_mat = mdot(X, weights);

    Matrix diff = msubtract(y_test_mat, y_actual_mat);
    Matrix product = mdot(mtranspose(X), diff);
    Matrix grad = mmultiply(mdivide(product, X.size()), -1);
    return grad;
}

void gradient_descent(
    const Matrix &x, 
    const Matrix &y, 
    Matrix &weights, 
    Vector &lossArray, 
    FP_DTYPE LR, 
    int epochs, 
    int debug
){
    for(int epoch = 0; epoch < epochs; epoch++)
    {
        lossArray.push_back(mse_loss(x, y, weights));
        Matrix grad = mse_gradient(x, y, weights);
        weights = madd(weights, mmultiply(grad, LR));

        if(epoch % debug == 0) {
            std::cout << "Epoch: " << epoch << "; Loss: " << lossArray.back() << '\n';
        }
    }
} 