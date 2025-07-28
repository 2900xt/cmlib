#pragma once
#include <vector>
#include <iostream>

#define FP_DTYPE float
#define Vector std::vector<FP_DTYPE>

// Vector creation functions
Vector vrand(int n, FP_DTYPE mn, FP_DTYPE mx);
Vector vrange(FP_DTYPE start, FP_DTYPE end, int num);

// Vector arithmetic operations
Vector vadd(const Vector &a, const Vector &b);
Vector vadd(const Vector &a, FP_DTYPE b);
Vector vsubtract(const Vector &a, const Vector &b);
Vector vmultiply(const Vector &a, const Vector &b);
Vector vmultiply(const Vector &a, FP_DTYPE b);
Vector vdivide(const Vector &a, const Vector &b);
Vector vdivide(const Vector &a, FP_DTYPE b);

// Vector operations
FP_DTYPE vdot(const Vector &a, const Vector &b);
FP_DTYPE vsum(const Vector &a);
Vector vsquare(const Vector &a);

// Output operator
std::ostream& operator<<(std::ostream& os, const Vector& a); 