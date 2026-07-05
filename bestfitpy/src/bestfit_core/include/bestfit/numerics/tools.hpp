// ported from: Numerics/Utilities/Tools.cs @ a2c4dbf
// Shared numerical constants (+ Tools.Log10, the one Tools.cs function this port needs).
// P3.3 adds is_finite (Tools.IsFinite) and is_power_of_two (Tools.IsPowerOfTwo), needed by
// Fourier::fft (power-of-two length guard) and NumericalDerivative::gradient/hessian
// (non-finite-evaluation backtracking guard). Tools.NextPowerOfTwo is not ported -- no
// caller in this port's scope needs it. B3 adds sqr (Tools.Sqr, Tools.cs:146), needed by the
// ParameterPenalty/QuantilePenalty penalty functions.
#pragma once
#include <cmath>

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

}  // namespace bestfit::numerics
