// ported from: Numerics/Utilities/Tools.cs @ <pending-sha>
// Shared numerical constants.
#pragma once

namespace bestfit::numerics {

inline constexpr double kDoubleMachineEpsilon = 1.11022302462516E-16;
inline constexpr double kPi = 3.14159265358979323846;  // M_PI is non-standard (absent on MSVC)
inline constexpr double kEuler = 0.5772156649015328606065120;  // Euler–Mascheroni γ
inline constexpr double kSqrt2 = 1.4142135623730950488016887;
inline constexpr double kSqrt2PI = 2.50662827463100050242E0;
inline constexpr double kLogSqrt2PI = 0.91893853320467274178032973640562;
inline constexpr double kLog2 = 0.69314718055994530941723212145818;  // ln(2) — L-moment estimation

}  // namespace bestfit::numerics
