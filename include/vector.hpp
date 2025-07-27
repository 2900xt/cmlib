#pragma once
#include <vector>
#include <iostream>
#include <random>
#include "config.hpp"
using namespace std;

Vector vrand(int n, FP_DTYPE mx, FP_DTYPE mn)
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
        cerr << "Cannot add vectors with mismatched sizes: " << b.size() << ", " << a.size() << endl;
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
        cerr << "Cannot subtract vectors with mismatched sizes: " << b.size() << ", " << a.size() << endl;
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
        cerr << "Cannot multiply vectors with mismatched sizes: " << b.size() << ", " << a.size() << endl;
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
        cerr << "Cannot divide vectors with mismatched sizes: " << b.size() << ", " << a.size() << endl;
        exit(EXIT_FAILURE);
    }
    Vector out(b.size());
    for(int i = 0; i < b.size(); i++)
    {
        out[i] = a[i] / b[i];
    }
    return out;
}

FP_DTYPE vdot(const Vector &a, const Vector &b)
{
    if(b.size() != a.size())
    {
        cerr << "Cannot dot vectors with mismatched sizes: " << b.size() << ", " << a.size() << endl;
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

void vprint(const Vector &a)
{
    cout << '[';
    for(int i = 0; i < a.size(); i++)
    {
        cout << a[i] << ",]"[i == a.size() - 1];
    }
}