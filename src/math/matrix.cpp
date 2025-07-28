#include "math/matrix.h"
#include <cmath>

Matrix mmake(int rows, int cols, FP_DTYPE value)
{
    Matrix out(rows, Vector(cols, value));
    return out;
}

Matrix mrand(int rows, int cols, FP_DTYPE mn, FP_DTYPE mx)
{
    Matrix out(rows);
    for(int i = 0; i < rows; i++)
    {
        out[i] = vrand(cols, mn, mx);
    }
    return out;
}

Matrix madd(const Matrix &a, const Matrix &b)
{
    if(b.size() != a.size() || b[0].size() != a[0].size())
    {
        std::cerr << "Cannot add matrices with mismatched sizes: " << b.size() << "x" << b[0].size() << ", " << a.size() << "x" << a[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = vadd(a[i], b[i]);
    }
    return out;
}

Matrix msubtract(const Matrix &a, const Matrix &b)
{
    if(b.size() != a.size() || b[0].size() != a[0].size())
    {
        std::cerr << "Cannot subtract matrices with mismatched sizes: " << b.size() << "x" << b[0].size() << ", " << a.size() << "x" << a[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = vsubtract(a[i], b[i]);
    }
    return out;
}

Matrix mmultiply(const Matrix &a, const Matrix &b)
{
    if(b.size() != a.size() || b[0].size() != a[0].size())
    {
        std::cerr << "Cannot (elem-wise) mutiply matrices with mismatched sizes: " << b.size() << "x" << b[0].size() << ", " << a.size() << "x" << a[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = vmultiply(a[i], b[i]);
    }
    return out;
}

Matrix mmultiply(const Matrix &a, FP_DTYPE val)
{
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < a.size(); i++)
    {
        out[i] = vmultiply(a[i], val);
    }
    return out;
}

Matrix mdivide(const Matrix &a, const Matrix &b)
{
    if(b.size() != a.size() || b[0].size() != a[0].size())
    {
        std::cerr << "Cannot (elem-wise) divide matrices with mismatched sizes: " << b.size() << "x" << b[0].size() << ", " << a.size() << "x" << a[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = vdivide(a[i], b[i]);
    }
    return out;
}

Matrix mdivide(const Matrix &a, FP_DTYPE val)
{
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < a.size(); i++)
    {
        out[i] = vdivide(a[i], val);
    }
    return out;
}

Matrix mdot(const Matrix &a, const Matrix &b)
{
    if(a[0].size() != b.size())
    {
        std::cerr << "Cannot dot matrices with mismatched sizes: " << a.size() << "x" << a[0].size() << ", " << b.size() << "x" << b[0].size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Matrix out = mmake(a.size(), b[0].size());
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < b[0].size(); j++)
        {
            FP_DTYPE sum = 0;
            for(int k = 0; k < a[0].size(); k++)
            {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }
    return out;
}

Matrix mtranspose(const Matrix &a)
{
    Matrix out = mmake(a[0].size(), a.size());
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < a[0].size(); j++)
        {
            out[j][i] = a[i][j];
        }
    }

    return out;
}

Vector msum(const Matrix &a, int dim)
{
    if(dim == 0)
    {
        //vertically sum???
        Vector out(a[0].size());
        for(int i = 0; i < a[0].size(); i++)
        {
            for(int j = 0; j < a.size(); j++)
            {
                out[i] += a[j][i];
            }
        }
        return out;
    }

    if(dim == 1)
    {
        Vector out(a.size());
        for(int i = 0; i < a.size(); i++)
        {
            for(int j = 0; j < a[0].size(); j++)
            {
                out[i] += a[i][j];
            }
        }
        return out;
    }

    FP_DTYPE sum = 0;
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < a[0].size(); j++)
        {
            sum += a[i][j];
        }
    }
    return {sum};
}

Matrix msigmoid(const Matrix &a)
{
    Matrix out = mmake(a.size(), a[0].size());
    for(int i = 0; i < a.size(); i++)
    {
        for(int j = 0; j < a[0].size(); j++)
        {
            out[i][j] = 1/(1 + exp(-a[i][j]));
        }
    }

    return out;
}

std::ostream& operator<<(std::ostream& os, const Matrix& a) 
{
    os << std::fixed << std::setprecision(2);
    os << "Shape: (" << a.size() << ", " << a[0].size() << ")\n";

    os << "[";
    for(int i = 0; i < a.size(); i++)
    {
        if(i != 0) os << ' ';
        os << a[i];
        if(i != a.size() - 1)
        {
            os << ",\n";
        }
    }
    os << "]";
    return os;
} 