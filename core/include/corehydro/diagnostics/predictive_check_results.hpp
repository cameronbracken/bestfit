// ported from: RMC-BestFit/src/RMC.BestFit/Diagnostics/PredictiveCheckResults.cs @ fc28c0c
//
// Results from posterior predictive checking with common test statistics: the number of
// replicates plus the five posterior predictive p-values (mean/SD/skewness/min/max) and the
// `HasPotentialMisfit(threshold)` predicate.
//
// C++ mapping: the C# `public class PredictiveCheckResults` with auto-properties ports as a
// plain value struct. C# `double X { get; set; } = double.NaN` -> `double` initialized to
// quiet_NaN(); `int NumberOfReplicates { get; set; }` -> `int`, default 0.
//
// HasPotentialMisfit (C# 83): the predicate is transcribed verbatim -- flags misfit when any
// p-value is `< threshold` or `> (1 - threshold)`, default threshold 0.05. NaN comparisons are
// false in both C# and C++ IEEE semantics, so an all-NaN (default) instance reports no misfit,
// exactly matching the C# `HasPotentialMisfit_NaNPValues_DoesNotThrow` test.
#pragma once
#include <limits>

namespace corehydro::diagnostics {

struct PredictiveCheckResults {
    // The number of replicates used (C# `NumberOfReplicates`).
    int number_of_replicates = 0;

    // P-value for the mean statistic (C# `MeanPValue`).
    double mean_p_value = std::numeric_limits<double>::quiet_NaN();

    // P-value for the standard deviation statistic (C# `SDPValue`).
    double sd_p_value = std::numeric_limits<double>::quiet_NaN();

    // P-value for the skewness statistic (C# `SkewnessPValue`).
    double skewness_p_value = std::numeric_limits<double>::quiet_NaN();

    // P-value for the minimum statistic (C# `MinPValue`).
    double min_p_value = std::numeric_limits<double>::quiet_NaN();

    // P-value for the maximum statistic (C# `MaxPValue`).
    double max_p_value = std::numeric_limits<double>::quiet_NaN();

    // Whether any p-value indicates potential model misfit (C# `HasPotentialMisfit`, C# 83):
    // true if any p-value is `< threshold` or `> (1 - threshold)`.
    bool has_potential_misfit(double threshold = 0.05) const {
        return mean_p_value < threshold || mean_p_value > (1 - threshold) ||
               sd_p_value < threshold || sd_p_value > (1 - threshold) ||
               skewness_p_value < threshold || skewness_p_value > (1 - threshold) ||
               min_p_value < threshold || min_p_value > (1 - threshold) ||
               max_p_value < threshold || max_p_value > (1 - threshold);
    }
};

}  // namespace corehydro::diagnostics
