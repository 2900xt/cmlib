#pragma once
#include "math/vector.h"
#include <vector>
#include <string>

struct plot_data_params
{
    Vector x, y;
    std::string config;
};

void plot(std::vector<plot_data_params> data, const char *title, const char *x_label, const char *y_label); 