#pragma once

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cstring>

#define FP_DTYPE float
#define Vector std::vector<FP_DTYPE>
#define Matrix std::vector<Vector>

// Forward declarations for all library modules
#include "math/vector.h"
#include "math/matrix.h"
#include "alg/grad_descent.h"
#include "plot/plot.h" 