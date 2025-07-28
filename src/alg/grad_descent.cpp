#include "config.h"
#include "math/matrix.h"
#include "math/vector.h"
#include "models/model.h"

FP_DTYPE mse_loss(const Matrix& X, const Matrix& y_actual_mat, const Model& model)
{
    Matrix y_test_mat = model.forward(X);
    Vector y_test = mtranspose(y_test_mat)[0];
    Vector y_actual = mtranspose(y_actual_mat)[0];

    FP_DTYPE cost = vsum(vsquare(vsubtract(y_actual, y_test)));
    return cost / (2*X.size());
}

Matrix mse_gradient(const Matrix& X, const Matrix& y_actual_mat, const Model& model)
{
    Matrix y_test_mat = model.forward(X);

    Matrix diff = msubtract(y_test_mat, y_actual_mat);
    Matrix product = mdot(mtranspose(X), diff);
    Matrix grad = mmultiply(mdivide(product, X.size()), -1);
    return grad;
}

void gradient_descent(
    const Matrix &x, 
    const Matrix &y, 
    Model &model, 
    Vector &lossArray, 
    FP_DTYPE LR, 
    int epochs, 
    int debug = 1e9
){
    for(int epoch = 0; epoch < epochs; epoch++)
    {
        lossArray.push_back(mse_loss(x, y, model));
        Matrix grad = mse_gradient(x, y, model);
        model.weights = madd(model.weights, mmultiply(grad, LR));

        if(epoch % debug == 0) {
            std::cout << "Epoch: " << epoch << "; Loss: " << lossArray.back() << '\n';
        }
    }
} 