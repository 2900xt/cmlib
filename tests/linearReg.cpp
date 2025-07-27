#include "config.h"

int main()
{
    FP_DTYPE slope = 2.5, bias = 1.5;
    int num = 50;
    int sigma = 2;

    Matrix x = mmake(2, num, 1);
    x[1] = vrand(num, 0, 10);

    Matrix y = mmake(1, num);
    y[0] = vadd(vmultiply(x[1], slope), bias);

    Vector offset = vmultiply(vrand(num, 1, -1), sigma);
    y[0] = vadd(y[0], offset);

    x = mtranspose(x);
    y = mtranspose(y);
    
    //mprint(x);
    //mprint(y);

    //plot({{mtranspose(x)[1], mtranspose(y)[0], "with points"}}, "Test Data", "X", "Y");
    Matrix weights = mrand(2, 1);
    FP_DTYPE LR = 0.01;
    int epochs = 20;
    Vector costArray{};

    gradient_descent(x, y, weights, costArray, 0.001, 100);

    plot(
        {
            {mtranspose(x)[1], mtranspose(y)[0], "with points title 'Data'"},
            {mtranspose(x)[1], mtranspose(mdot(x, weights))[0], "with lines title 'Best Fit Line'"}
        },
        "Line of Best Fit in Data",
        "Input",
        "Feature Value"
    );
}