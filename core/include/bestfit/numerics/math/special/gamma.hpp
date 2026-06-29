// ported from: Numerics/Mathematics/Special Functions/Gamma.cs @ <pending-sha>
//             + Numerics/Mathematics/Special Functions/Evaluate.cs (PolynomialRev)
//
// Gamma function (Cephes), Lanczos approximation, and digamma. Algorithms and
// coefficient tables are copied verbatim from the C# source so results match the
// oracle values bit-closely.
#pragma once
#include <cmath>
#include <stdexcept>

#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::math::special {

namespace detail {

// Horner evaluation: coeffs[0]*x^n + ... + coeffs[n]  (mirrors Evaluate.PolynomialRev).
inline double polynomial_rev(const double* coeffs, double x, int n) {
    double value = coeffs[0];
    for (int i = 1; i <= n; ++i) {
        value *= x;
        value += coeffs[i];
    }
    return value;
}

inline constexpr double kLanczosP[9] = {
    0.99999999999980993,    676.5203681218851,     -1259.1392167224028,
    771.32342877765313,     -176.61502916214059,   12.507343278686905,
    -0.13857109526572012,   0.0000099843695780195716, 0.00000015056327351493116};

inline constexpr double kGammaP[7] = {
    1.60119522476751861407E-4, 1.19135147006586384913E-3, 1.04213797561761569935E-2,
    4.76367800457137231464E-2, 2.07448227648435975150E-1, 4.94214826801497100753E-1,
    9.99999999999999996796E-1};

inline constexpr double kGammaQ[8] = {
    -2.31581873324120129819E-5, 5.39605580493303397842E-4, -4.45641913851797240494E-3,
    1.18139785222060435552E-2,  3.58236398605498653373E-2, -2.34591795718243348568E-1,
    7.14304917030273074085E-2,  1.00000000000000000320E0};

inline constexpr double kStir[5] = {7.87311395793093628397E-4, -2.29549961613378126380E-4,
                                    -2.68132617805781232825E-3, 3.47222221605458667310E-3,
                                    8.33333333333482257126E-2};

}  // namespace detail

// Forward declaration: Lanczos uses Function for the reflection branch.
double function(double x);

inline double stirling(double x) {
    constexpr double MAXSTIR = 143.01608;
    double w = 1.0 / x;
    double y = std::exp(x);
    w = 1.0 + w * detail::polynomial_rev(detail::kStir, w, 4);
    if (x > MAXSTIR) {
        double v = std::pow(x, 0.5 * x - 0.25);
        if (std::isinf(v) && std::isinf(y)) {
            y = std::numeric_limits<double>::infinity();
        } else {
            y = v * (v / y);
        }
    } else {
        y = std::pow(x, x - 0.5) / y;
    }
    y = kSqrt2PI * y * w;
    return y;
}

// The Gamma function (Cephes implementation).
inline double function(double x) {
    double p, z;
    double q = std::fabs(x);

    if (q > 33.0) {
        if (x < 0.0) {
            p = std::floor(q);
            if (p == q) throw std::overflow_error("gamma overflow");
            z = q - p;
            if (z > 0.5) {
                p += 1.0;
                z = q - p;
            }
            z = q * std::sin(M_PI * z);
            if (z == 0.0) throw std::overflow_error("gamma overflow");
            z = std::fabs(z);
            z = M_PI / (z * stirling(q));
            return -z;
        } else {
            return stirling(x);
        }
    }

    z = 1.0;
    while (x >= 3.0) {
        x -= 1.0;
        z *= x;
    }
    while (x < 0.0) {
        if (x > -1.0E-9) return z / ((1.0 + kEuler * x) * x);
        z /= x;
        x += 1.0;
    }
    while (x < 2.0) {
        if (x == 0.0) throw std::runtime_error("gamma singularity");
        if (x < 1.0E-9) return z / ((1.0 + kEuler * x) * x);
        z /= x;
        x += 1.0;
    }
    if (x == 2.0 || x == 3.0) return z;
    x -= 2.0;
    p = detail::polynomial_rev(detail::kGammaP, x, 6);
    q = detail::polynomial_rev(detail::kGammaQ, x, 7);
    return z * p / q;
}

// Lanczos approximation of the Gamma function.
inline double lanczos(double x) {
    constexpr int g = 7;
    if (x < 0.5) {
        return M_PI / (std::sin(M_PI * x) * function(1.0 - x));
    }
    x -= 1.0;
    double y = detail::kLanczosP[0];
    for (int i = 1; i < g + 2; ++i) y += detail::kLanczosP[i] / (x + i);
    double t = x + g + 0.5;
    return kSqrt2PI * std::pow(t, x + 0.5) * std::exp(-t) * y;
}

// Digamma (Euler's psi), algorithm AS103.
inline double digamma(double X) {
    constexpr double SMALL = 0.000000001;
    constexpr double CRIT = 13.0;
    constexpr double C1 = 0.083333333333333329, C2 = -0.0083333333333333332,
                     C3 = 0.003968253968253968, C4 = -0.0041666666666666666,
                     C5 = 0.007575757575757576, C6 = -0.021092796092796094,
                     C7 = 0.083333333333333329, D1 = -0.57721566490153287;
    if (X <= 0.0) throw std::out_of_range("digamma: X must be > 0");
    double dg = 0.0;
    if (X > SMALL) {
        double Y = X;
        while (Y < CRIT) {
            dg -= 1.0 / Y;
            Y += 1.0;
        }
        dg += std::log(Y) - 0.5 / Y;
        Y = 1.0 / (Y * Y);
        double sum = ((((((C7 * Y + C6) * Y + C5) * Y + C4) * Y + C3) * Y + C2) * Y + C1) * Y;
        dg -= sum;
    } else {
        dg = D1 - 1.0 / X;
    }
    return dg;
}

}  // namespace bestfit::numerics::math::special
