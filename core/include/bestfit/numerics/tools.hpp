// ported from: Numerics/Utilities/Tools.cs @ a2c4dbf
// Shared numerical constants (+ Tools.Log10, the one Tools.cs function this port needs).
// P3.3 adds is_finite (Tools.IsFinite) and is_power_of_two (Tools.IsPowerOfTwo), needed by
// Fourier::fft (power-of-two length guard) and NumericalDerivative::gradient/hessian
// (non-finite-evaluation backtracking guard). Tools.NextPowerOfTwo is not ported -- no
// caller in this port's scope needs it. B3 adds sqr (Tools.Sqr, Tools.cs:146), needed by the
// ParameterPenalty/QuantilePenalty penalty functions. B5 adds sum_product (Tools.SumProduct,
// Tools.cs:425), needed by BFGS's strong-Wolfe line search, and normalized_distance
// (Tools.NormalizedDistance, Tools.cs:267), whose caller is MLSL (arrives in B6). B8 adds
// distance (Tools.Distance IList overload, Tools.cs:246), needed by the GMM iterative
// convergence check.
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

namespace bestfit::numerics {

inline constexpr double kDoubleMachineEpsilon = 1.11022302462516E-16;
inline constexpr double kPi = 3.14159265358979323846;  // M_PI is non-standard (absent on MSVC)
inline constexpr double kEuler = 0.5772156649015328606065120;  // Euler–Mascheroni γ
inline constexpr double kSqrt2 = 1.4142135623730950488016887;
inline constexpr double kSqrt2PI = 2.50662827463100050242E0;
inline constexpr double kLogSqrt2PI = 0.91893853320467274178032973640562;
inline constexpr double kLog2 = 0.69314718055994530941723212145818;  // ln(2) — L-moment estimation
inline constexpr double kE    = 2.71828182845904523536028747135266;  // e = exp(1)

// Clamped base-10 logarithm: ported from Tools.Log10. Values below 1E-16 that are not
// negative are clamped to 1E-16 before taking the log, avoiding -inf for near-zero
// inputs. Used by Linear's transform path (bestfit/numerics/data/interpolation/linear.hpp);
// Bilinear instead calls plain std::log10, matching that split in the C# source itself
// (see docs/upstream-csharp-issues.md).
inline double clamped_log10(double x) {
    if (x < 1E-16 && x >= 0.0) x = 1E-16;
    return std::log10(x);
}

// Returns true iff x is neither NaN nor +-infinity (mirrors Tools.IsFinite).
inline bool is_finite(double x) { return !(std::isnan(x) || std::isinf(x)); }

// Returns true iff n is a positive power of two (mirrors Tools.IsPowerOfTwo).
inline bool is_power_of_two(int n) { return n > 0 && (n & (n - 1)) == 0; }

// Returns the squared value of a (mirrors Tools.Sqr; added with B3 for the penalty classes).
inline double sqr(double a) { return a * a; }

// Returns the value of a with the sign of b (mirrors Tools.Sign, Tools.cs:125; added with B9
// for the SingularValueDecomposition port).
inline double sign(double a, double b) {
    return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
}

// Returns the sum product of two lists of values (mirrors Tools.SumProduct, Tools.cs:425;
// added with B5 for BFGS's strong-Wolfe line search). Empty or length-mismatched inputs
// return NaN, exactly as the C#.
inline double sum_product(const std::vector<double>& values1, const std::vector<double>& values2) {
    if (values1.empty()) return std::numeric_limits<double>::quiet_NaN();
    if (values2.size() != values1.size()) return std::numeric_limits<double>::quiet_NaN();
    double sum = 0.0;
    for (std::size_t i = 0; i < values1.size(); i++) sum += values1[i] * values2[i];
    return sum;
}

// Returns the Euclidean distance between two points after applying min-max normalization
// to each dimension, based on the given lower and upper bounds, so every parameter
// dimension contributes equally to the distance metric (mirrors Tools.NormalizedDistance,
// Tools.cs:267; added with B5 -- its caller is the MLSL global optimizer, ported in B6).
// A degenerate dimension (range <= 0 or NaN) contributes nothing, exactly as the C#.
inline double normalized_distance(const std::vector<double>& x, const std::vector<double>& y,
                                  const std::vector<double>& lower,
                                  const std::vector<double>& upper) {
    double d = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        double range = upper[i] - lower[i];
        if (range <= 0.0 || std::isnan(range)) {
            // Degenerate dimension; contribute nothing (identical after normalization).
            continue;
        }
        double xi = (x[i] - lower[i]) / range;
        double yi = (y[i] - lower[i]) / range;
        double dx = xi - yi;
        d += dx * dx;
    }
    return std::sqrt(d);
}

// Returns the Euclidean distance between two points (mirrors Tools.Distance's
// IList<double> overload, Tools.cs:246; added with B8 -- GeneralizedMethodOfMoments'
// iterative-strategy parameter-convergence check is the caller).
inline double distance(const std::vector<double>& x, const std::vector<double>& y) {
    double d = 0;
    for (std::size_t i = 0; i < x.size(); i++) {
        double dx = x[i] - y[i];
        d += dx * dx;
    }
    return std::sqrt(d);
}

}  // namespace bestfit::numerics
