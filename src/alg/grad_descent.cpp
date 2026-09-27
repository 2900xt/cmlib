#include "config.h"
#include "math/matrix.h"
#include "math/vector.h"
#include "models/model.h"
#include "alg/grad_descent.h"

FP_DTYPE mse_loss(const Matrix& X, const Matrix& y_actual_mat, const Model& model)
{
    Matrix y_test_mat = model.forward(X);
    Vector y_test = mtranspose(y_test_mat)[0];
    Vector y_actual = mtranspose(y_actual_mat)[0];

    FP_DTYPE cost = vsum(vsquare(vsubtract(y_actual, y_test)));
    return cost / (2*X.size());
}

Matrix mse_gradient(const Matrix& X, const Matrix& y_actual_mat, Model& model)
{
    const FP_DTYPE eps = 1e-6;
    Matrix grad = mmake(model.weights.size(), model.weights[0].size());
    for(size_t i = 0; i < grad.size(); i++)
    {
        for(size_t j = 0; j < grad[i].size(); j++)
        {
            //df/dw ~ (f(w + eps) - f(w - eps)) / 2eps
            FP_DTYPE original = model.weights[i][j];
            model.weights[i][j] = original + eps;
            FP_DTYPE f1 = mse_loss(X, y_actual_mat, model);
            model.weights[i][j] = original - eps;
            FP_DTYPE f2 = mse_loss(X, y_actual_mat, model);
            model.weights[i][j] = original;

            grad[i][j] = (f1 - f2) / (2 * eps);
        }
    }

    return grad;
}

void gradient_descent(
    const Matrix &x, 
    const Matrix &y, 
    Model &model, 
    Vector &lossArray, 
    FP_DTYPE LR, 
    int epochs, 
    int debug
){
    for(int epoch = 0; epoch < epochs; epoch++)
    {
        lossArray.push_back(mse_loss(x, y, model));
        Matrix grad = mse_gradient(x, y, model);
        model.weights = msubtract(model.weights, mmultiply(grad, LR));

        if(epoch % debug == 0) {
            std::cout << "Epoch: " << epoch << "; Loss: " << lossArray.back() << "; Grad: " << grad << '\n';
        }
    }
    std::cout << "Final loss: " << mse_loss(x, y, model) << '\n';
}
