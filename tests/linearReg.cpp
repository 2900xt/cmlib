#include "config.h"
#include "math/matrix.h"
#include "math/vector.h"
#include "models/model.h"
#include "alg/grad_descent.h"
#include "plot/plot.h"

class linreg_model : public Model {
public:
    linreg_model() {
        this->weights = mrand(2, 1);
    }

    Matrix forward (const Matrix &X) const override 
    {
        return mdot(X, weights);
    }
};

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
    
    FP_DTYPE LR = 0.01;
    int epochs = 20;
    Vector costArray{};
    linreg_model model0{};

    gradient_descent(x, y, model0, costArray, 0.001, 100, 2);

    plot(
        {
            {mtranspose(x)[1], mtranspose(y)[0], "with points title 'Data'"},
            {mtranspose(x)[1], mtranspose(mdot(x, model0.weights))[0], "with lines title 'Best Fit Line'"}
        },
        "Line of Best Fit in Data",
        "Input",
        "Feature Value"
    );
}