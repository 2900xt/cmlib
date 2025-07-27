#pragma once
#include <vector>
#include <iostream>
#include "config.hpp"
using namespace std;

Vector vadd(Vector &a, Vector &b)
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

Vector vsubtract(Vector &a, Vector &b)
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

Vector vmultiply(Vector &a, Vector &b)
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

Vector vdivide(Vector &a, Vector &b)
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

FP_DTYPE vdot(Vector &a, Vector &b)
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