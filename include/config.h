#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstring>

// Floating point type used everywhere in the library
typedef long double FP_DTYPE;
typedef std::vector<FP_DTYPE> Vector;
typedef std::vector<Vector> Matrix;

// Forward declarations for all library modules
class Model;
