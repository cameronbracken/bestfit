// ported from: Numerics/Data/Statistics/PlottingPositions.cs @ a2c4dbf
//
// The general Function(N, alpha) formula, the Weibull(N) entry point SNIS.cs's
// output-resampling step calls (`PlottingPositions.Weibull(OutputLength)`), and the
// Median/Blom/Cunnane/Gringorten/Hazen named formulas (M5's DataFrame plotting-position
// tests validate against them). KaplanMeier and the `Function(N, PlottingPostionType)`
// enum-dispatch overload that selects among them are not ported -- no caller in this
// port's scope needs them. Add them if a later task does.
//
// Nested under its own `plotting_positions` namespace (mirroring erf.hpp's `special::erf`
// pattern) rather than living bare in `corehydro::numerics::data`, since "function" is too
// generic a name to place unqualified in a namespace shared with statistics.hpp,
// correlation.hpp, histogram.hpp, and search.hpp.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace corehydro::numerics::data::plotting_positions {

// The general affine-symmetric plotting position formula:
// PP[i] = (i - alpha) / (N + 1 - 2*alpha) for i = 1..N (1-based in the source formula; the
// returned vector is 0-based, PP[i-1]). Assumes uncensored, complete data. `alpha` must be
// in [0, 1]; `n` must be > 2.
inline std::vector<double> function(int n, double alpha) {
    if (n <= 2) throw std::out_of_range("The sample size N must be greater than 2.");
    if (alpha < 0.0 || alpha > 1.0)
        throw std::out_of_range("The alpha coefficient must be between 0 and 1.");
    std::vector<double> pp(static_cast<std::size_t>(n));
    for (int i = 1; i <= n; ++i) pp[static_cast<std::size_t>(i - 1)] = (i - alpha) / (n + 1 - 2.0 * alpha);
    return pp;
}

// The Weibull plotting position formula (alpha = 0). Recommended for the uniform
// distribution; this is the entry point SNIS's output resampling uses.
inline std::vector<double> weibull(int n) { return function(n, 0.0); }

// The median (APL) plotting position formula (alpha = 0.3175); approximately
// median-unbiased for a wide range of distributions (C# Median, line 103).
inline std::vector<double> median(int n) { return function(n, 0.3175); }

// The Blom plotting position formula (alpha = 0.375); approximately unbiased for the
// Normal distribution (C# Blom, line 114).
inline std::vector<double> blom(int n) { return function(n, 0.375); }

// The Cunnane plotting position formula (alpha = 0.4); approximately unbiased for the
// Gumbel and GEV distributions (C# Cunnane, line 124).
inline std::vector<double> cunnane(int n) { return function(n, 0.4); }

// The Gringorten plotting position formula (alpha = 0.44); approximately unbiased for
// the Weibull distribution (C# Gringorten, line 134).
inline std::vector<double> gringorten(int n) { return function(n, 0.44); }

// The Hazen plotting position formula (alpha = 0.5); the median, distribution-agnostic
// formula (C# Hazen, line 145).
inline std::vector<double> hazen(int n) { return function(n, 0.5); }

}  // namespace corehydro::numerics::data::plotting_positions
