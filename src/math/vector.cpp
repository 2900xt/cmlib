#include "math/vector.h"
#include <cstdlib>

Vector vrand(int n, FP_DTYPE mn, FP_DTYPE mx)
{
    Vector out(n);
    for(int i = 0; i < n; i++)
    {
        out[i] = (float)rand() / (float)RAND_MAX;
        out[i] = out[i] * (mx - mn) + mn;
    }
    return out;
}

Vector vadd(const Vector &a, const Vector &b)
{
    if(b.size() != a.size())
    {
        std::cerr << "Cannot add vectors with mismatched sizes: " << b.size() << ", " << a.size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Vector out(b.size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = b[i] + a[i];
    }
    return out;
}

Vector vadd(const Vector &a, FP_DTYPE b)
{
    Vector out(a);
    for(int i = 0; i < a.size(); i++)
    {
        out[i] = a[i] + b;
    }
    return out;
}

Vector vsubtract(const Vector &a, const Vector &b)
{
    if(b.size() != a.size())
    {
        std::cerr << "Cannot subtract vectors with mismatched sizes: " << b.size() << ", " << a.size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Vector out(b.size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = a[i] - b[i];
    }
    return out;
}

Vector vmultiply(const Vector &a, const Vector &b)
{
    if(b.size() != a.size())
    {
        std::cerr << "Cannot multiply vectors with mismatched sizes: " << b.size() << ", " << a.size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Vector out(b.size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = a[i] * b[i];
    }
    return out;
}

Vector vmultiply(const Vector &a, FP_DTYPE b)
{
    Vector out(a.size());
    for(int i = 0; i < a.size(); i++)
    {
        out[i] = a[i] * b;
    }
    return out;
}

Vector vdivide(const Vector &a, const Vector &b)
{
    if(b.size() != a.size())
    {
        std::cerr << "Cannot divide vectors with mismatched sizes: " << b.size() << ", " << a.size() << std::endl;
        exit(EXIT_FAILURE);
    }
    Vector out(b.size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = a[i] / b[i];
    }
    return out;
}

Vector vdivide(const Vector &a, FP_DTYPE b)
{
    Vector out(a.size());
    for(int i = 0; i < a.size(); i++)
    {
        out[i] = a[i] / b;
    }
    return out;
}

FP_DTYPE vdot(const Vector &a, const Vector &b)
{
    if(b.size() != a.size())
    {
        std::cerr << "Cannot dot vectors with mismatched sizes: " << b.size() << ", " << a.size() << std::endl;
        exit(EXIT_FAILURE);
    }
    FP_DTYPE out = 0;
    for(int i = 0; i < b.size(); i++)
    {
        out += b[i]*a[i];
    }
    return out;
}

FP_DTYPE vsum(const Vector &a)
{
    FP_DTYPE out = 0;
    for(int i = 0; i < a.size(); i++)
    {
        out += a[i];
    }
    return out;
}

Vector vsquare(const Vector &a)
{
    return vmultiply(a, a);
}

std::ostream& operator<<(std::ostream& os, const Vector& a) {
    os << '[';
    for(int i = 0; i < a.size(); i++)
    {
        os << a[i] << ",]"[i == a.size() - 1];
    }
    return os;
} 