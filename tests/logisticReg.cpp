#include "config.h"
#include "math/matrix.h"
#include "math/vector.h"
#include "models/model.h"
#include "alg/grad_descent.h"
#include "plot/plot.h"

class logreg_model : public Model {
public:
    logreg_model() {
        this->weights = mrand(2, 1);
    }

    Matrix forward (const Matrix &X) const override 
    {
        return msigmoid(mdot(X, weights));
    }
};

int main()
{
    FP_DTYPE hi = 1.0, lo = 0.0;
    FP_DTYPE mid = 0;
    int num = 50;

    FP_DTYPE sigma = 0.1;

    Matrix x = mmake(2, num, 1);
    x[1] = vrange(-5, 5, 50);

    Matrix y = mmake(1, num);
    for(int i = 0; i < num; i++)
    {
        if(x[1][i] > mid) y[0][i] = hi;
        else y[0][i] = lo;
    }

    Vector offset = vmultiply(vrand(num, 1, -1), sigma);
    y[0] = vadd(y[0], offset);

    x = mtranspose(x);
    y = mtranspose(y);
    
    //mprint(x);
    //mprint(y);

    //plot({{mtranspose(x)[1], mtranspose(y)[0], "with points"}}, "Test Data", "X", "Y");
    Vector costArray{};
    logreg_model model0{};

    gradient_descent(x, y, model0, costArray, 0.01, 10000);

    plot(
        {
            {mtranspose(x)[1], mtranspose(y)[0], "with points title 'Data'"},
            {mtranspose(x)[1], mtranspose(model0.forward(x))[0], "with lines title 'Best Fit Line'"}
        },
        "Line of Best Fit in Data",
        "Input",
        "Feature Value"
    );
}