// ported from: Numerics/Mathematics/Special Functions/Gamma.cs @ a2c4dbf
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
            z = q * std::sin(kPi * z);
            if (z == 0.0) throw std::overflow_error("gamma overflow");
            z = std::fabs(z);
            z = kPi / (z * stirling(q));
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
        return kPi / (std::sin(kPi * x) * function(1.0 - x));
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

// log_gamma: Lanczos-based log(Gamma(z)), ported from Gamma.LogGamma(z).
// Coefficient table (GammaDk) and algorithm mirror the C# source exactly.
namespace detail {
inline constexpr double kGammaDk[11] = {
    2.48574089138753565546e-5,  1.05142378581721974210,    -3.45687097222016235469,
    4.51227709466894823700,    -2.98285225323576655721,     1.05639711577126713077,
   -1.95428773191645869583e-1,  1.70970543404441224307e-2, -5.71926117404305781283e-4,
    4.63399473359905636708e-6, -2.71994908488607703910e-9};
}  // namespace detail

inline double log_gamma(double z) {
    constexpr int kGammaN = 10;
    constexpr double kGammaR = 10.900511;
    constexpr double kLnPi = 1.1447298858494001741434273513530587116472948129153;
    constexpr double kLogTwoSqrtEOverPi = 0.6207822376352452223455184457816472122518527279025978;
    if (z < 0.5) {
        double s = detail::kGammaDk[0];
        for (int i = 1; i <= kGammaN; ++i) s += detail::kGammaDk[i] / (i - z);
        return kLnPi - std::log(std::sin(kPi * z)) - std::log(s) - kLogTwoSqrtEOverPi
               - (0.5 - z) * std::log((0.5 - z + kGammaR) / std::exp(1.0));
    } else {
        double s = detail::kGammaDk[0];
        for (int i = 1; i <= kGammaN; ++i) s += detail::kGammaDk[i] / (z + i - 1.0);
        return std::log(s) + kLogTwoSqrtEOverPi
               + (z - 0.5) * std::log((z - 0.5 + kGammaR) / std::exp(1.0));
    }
}

// upper_incomplete: regularized upper incomplete gamma Q(a, x).
// Ported from Gamma.UpperIncomplete — continued-fraction algorithm.
inline double upper_incomplete(double a, double x);

// lower_incomplete: regularized lower incomplete gamma P(a, x).
// Ported from Gamma.LowerIncomplete — series expansion algorithm.
inline double lower_incomplete(double a, double x) {
    constexpr double kLogMax = 709.782712893384;
    constexpr double kDoubleEpsilon = 0.000000000000000111022302462516;
    if (a <= 0.0) return 1.0;
    if (x <= 0.0) return 0.0;
    if (x > 1.0 && x > a) return 1.0 - upper_incomplete(a, x);
    double ax = a * std::log(x) - x - log_gamma(a);
    if (ax < -kLogMax) return 0.0;
    ax = std::exp(ax);
    double r = a;
    double c = 1.0;
    double ans = 1.0;
    do {
        r += 1.0;
        c *= x / r;
        ans += c;
    } while (c / ans > kDoubleEpsilon);
    return ans * ax / a;
}

inline double upper_incomplete(double a, double x) {
    constexpr double kLogMax = 709.782712893384;
    constexpr double kDoubleEpsilon = 0.000000000000000111022302462516;
    constexpr double big = 4.5035996273705e+15;
    constexpr double biginv = 0.000000000000000222044604925031;
    if (x <= 0.0 || a <= 0.0) return 1.0;
    if (x < 1.0 || x < a) return 1.0 - lower_incomplete(a, x);
    if (std::isinf(x) && x > 0.0) return 0.0;
    double ax = a * std::log(x) - x - log_gamma(a);
    if (ax < -kLogMax) return 0.0;
    ax = std::exp(ax);
    // continued fraction
    double y = 1.0 - a;
    double z = x + y + 1.0;
    double c = 0.0;
    double pkm2 = 1.0, qkm2 = x;
    double pkm1 = x + 1.0, qkm1 = z * x;
    double ans = pkm1 / qkm1;
    double t;
    do {
        c += 1.0; y += 1.0; z += 2.0;
        double yc = y * c;
        double pk = pkm1 * z - pkm2 * yc;
        double qk = qkm1 * z - qkm2 * yc;
        double r;
        if (qk != 0.0) {
            r = pk / qk;
            t = std::fabs((ans - r) / r);
            ans = r;
        } else {
            t = 1.0;
        }
        pkm2 = pkm1; pkm1 = pk;
        qkm2 = qkm1; qkm1 = qk;
        if (std::fabs(pk) > big) {
            pkm2 *= biginv; pkm1 *= biginv;
            qkm2 *= biginv; qkm1 *= biginv;
        }
    } while (t > kDoubleEpsilon);
    return ans * ax;
}

}  // namespace bestfit::numerics::math::special
