// ported from: Numerics/Data/Statistics/PlottingPositions.cs @ a2c4dbf
//
// Only the general Function(N, alpha) formula and the Weibull(N) entry point SNIS.cs's
// output-resampling step calls (`PlottingPositions.Weibull(OutputLength)`); the
// Blom/Cunnane/Gringorten/Hazen/KaplanMeier/Median named formulas, and the
// `Function(N, PlottingPostionType)` enum-dispatch overload that selects among them, are
// not ported -- no caller in this port's scope needs them. Add them if a later task does.
//
// Nested under its own `plotting_positions` namespace (mirroring erf.hpp's `special::erf`
// pattern) rather than living bare in `bestfit::numerics::data`, since "function" is too
// generic a name to place unqualified in a namespace shared with statistics.hpp,
// correlation.hpp, histogram.hpp, and search.hpp.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace bestfit::numerics::data::plotting_positions {

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

}  // namespace bestfit::numerics::data::plotting_positions
