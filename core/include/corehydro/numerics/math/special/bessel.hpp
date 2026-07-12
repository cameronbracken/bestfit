// ported from: Numerics/Mathematics/Special Functions/Bessel.cs @ a2c4dbf
// Ports I0 and I1 only (modified Bessel functions of the first kind, orders 0 and 1).
#pragma once
#include <cmath>
#include <limits>

#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::special::bessel {

// Modified Bessel function of the first kind, order 0: I0(x).
// Polynomial approximation from Abramowitz and Stegun, 9.8.1-9.8.2.
inline double i0(double x) {
    double ax = std::fabs(x);
    if (ax < 3.75) {
        double t = x / 3.75;
        t *= t;
        return 1.0 + t * (3.5156229 + t * (3.0899424 + t * (1.2067492
            + t * (0.2659732 + t * (0.0360768 + t * 0.0045813)))));
    } else {
        if (ax > 709.0) return std::numeric_limits<double>::infinity();
        double t = 3.75 / ax;
        return (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + t * (0.01328592
            + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706
            + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377))))))));
    }
}

// Modified Bessel function of the first kind, order 1: I1(x).
// Polynomial approximation from Abramowitz and Stegun, 9.8.3-9.8.4.
// Odd function: I1(-x) = -I1(x).
inline double i1(double x) {
    double ax = std::fabs(x);
    double result;
    if (ax < 3.75) {
        double t = x / 3.75;
        t *= t;
        result = ax * (0.5 + t * (0.87890594 + t * (0.51498869 + t * (0.15084934
            + t * (0.02658733 + t * (0.00301532 + t * 0.00032411))))));
    } else {
        double t = 3.75 / ax;
        result = (std::exp(ax) / std::sqrt(ax)) * (0.39894228 + t * (-0.03988024
            + t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 + t * (0.02282967
            + t * (-0.02895312 + t * (0.01787654 + t * (-0.00420059)))))))));
    }
    return x < 0.0 ? -result : result;
}

}  // namespace corehydro::numerics::math::special::bessel
