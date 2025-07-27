#include "config.hpp"
#include "vector.hpp"
#include "matrix.hpp"
#include "plot.hpp"

FP_DTYPE loss_fn(Matrix X, Matrix y_actual_mat, Matrix weights)
{
    Matrix y_test_mat = mmultiply(X, weights);
    Vector y_test = mtranspose(y_test_mat)[0];
    Vector y_actual = mtranspose(y_actual_mat)[0];

    FP_DTYPE cost = vsum(vsquare(vsubtract(y_actual, y_test)));
    return cost / (2*X.size());
}

int main()
{
    FP_DTYPE slope = 2.5, bias = 1.5;
    int num = 50;
    int sigma = 2;

    Matrix x = mmake(2, num, 1);
    x[1] = vrand(num, 10, 0);

    Matrix y = mmake(1, num);
    y[0] = vadd(vmultiply(x[1], slope), bias);

    Vector offset = vmultiply(vrand(num, 1, -1), sigma);
    y[0] = vadd(y[0], offset);

    x = mtranspose(x);
    y = mtranspose(y);
    
    //mprint(x);
    //mprint(y);

    //plot({{mtranspose(x)[1], mtranspose(y)[0], "with points"}}, "Test Data", "X", "Y");
    Matrix weights({{0}, {1.5}});

    FP_DTYPE LR = 0.01;
    int epochs = 20;
    Vector costArray{};

    cout << loss_fn(x, y, weights) << endl;
}