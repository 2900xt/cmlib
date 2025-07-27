#pragma once 
#include "math/vector.h"
#include <iomanip>

#define Matrix std::vector<Vector>

// Matrix creation functions
Matrix mmake(int rows, int cols, FP_DTYPE value = 0);
Matrix mrand(int rows, int cols, FP_DTYPE mn = -1, FP_DTYPE mx = 1);

// Matrix arithmetic operations
Matrix madd(const Matrix &a, const Matrix &b);
Matrix msubtract(const Matrix &a, const Matrix &b);
Matrix mmultiply(const Matrix &a, const Matrix &b);
Matrix mmultiply(const Matrix &a, FP_DTYPE val);
Matrix mdivide(const Matrix &a, const Matrix &b);
Matrix mdivide(const Matrix &a, FP_DTYPE val);

// Matrix operations
Matrix mdot(const Matrix &a, const Matrix &b);
Matrix mtranspose(const Matrix &a);
Vector msum(const Matrix &a, int dim = -1);

// Output operator
std::ostream& operator<<(std::ostream& os, const Matrix& a); 