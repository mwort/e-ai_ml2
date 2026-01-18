#include <cmath>

extern "C" double f_sin_over_1px2(double x)
{
    return std::sin(x) / (1.0 + x * x);
}
