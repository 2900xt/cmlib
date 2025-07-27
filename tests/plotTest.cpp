#include "config.hpp"
#include "vector.hpp"
#include "plot.hpp"

int main()
{
    Vector x = {1.0, 2.0, 3.0, 4.0, 5.0};
    Vector y = {2.0, 3.0, 5.0, 7.0, 11.0};
    Vector x2 = {1.0, 2.0, 3.0, 4.0, 5.0};
    Vector y2 = {1.0, 4.0, 9.0, 16.0, 25.0};
    plot({{x, y, "with lines"}, {x2, y2, "with points"}}, "Sample Line Plot", "X Axis", "Y Axis");
    return 0;
}