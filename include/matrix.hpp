#pragma once 
#include "vector.hpp"
#include "config.hpp"

using namespace std;

inline Matrix mmake(int rows, int cols, FP_DTYPE value = 0)
{
    Matrix out(rows, Vector(cols, value));
    return out;
}

inline Matrix mmake(vector<vector<FP_DTYPE>>& data)
{
    Matrix out = mmake(data.size(), data[0].size());
    for(int i = 0; i < data.size(); i++)
    {
        out[i] = Vector(data[i]);
    }
    return out;
}

Matrix madd(const Matrix &a, const Matrix &b)
{
    if(b.size() != a.size() || b[0].size() != a[0].size())
    {
        cerr << "Cannot add matrices with mismatched sizes: " << b.size() << "x" << b[0].size() << ", " << a.size() << "x" << a[0].size() << endl;
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
        cerr << "Cannot subtract matrices with mismatched sizes: " << b.size() << "x" << b[0].size() << ", " << a.size() << "x" << a[0].size() << endl;
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
    if(a[0].size() != b.size())
    {
        cerr << "Cannot multiply matrices with mismatched sizes: " << a.size() << "x" << a[0].size() << ", " << b.size() << "x" << b[0].size() << endl;
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

void mprint(const Matrix &a)
{
    cout << "[";
    for(int i = 0; i < a.size(); i++)
    {
        vprint(a[i]);
        if(i != a.size() - 1)
        {
            cout << ",\n";
        }
    }
    cout << "]\n";
}