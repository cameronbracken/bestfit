// ported from: RMC-BestFit/src/RMC.BestFit/Diagnostics/PredictiveSummary.cs @ fc28c0c
//
// Summary statistics for a predictive distribution (prior or posterior). A small result DTO
// holding quantile arrays [2.5%, 25%, 50%, 75%, 97.5%] for the mean/SD/min/max statistics
// computed across multiple predictive datasets, plus the number of valid draws.
//
// C++ mapping: the C# `public class PredictiveSummary` with auto-properties ports as a plain
// value struct (the neighbours in this diagnostics layer that are pure data carriers follow the
// same convention). C# `double[] X { get; set; } = Array.Empty<double>()` -> an empty
// `std::vector<double>` (default-constructed). `int NumberOfValidDraws { get; set; }` -> `int`,
// default 0. The field names mirror the C# properties (snake_cased per the core's convention).
#pragma once
#include <vector>

namespace bestfit::diagnostics {

struct PredictiveSummary {
    // The number of valid draws used to compute the summary (C# `NumberOfValidDraws`).
    int number_of_valid_draws = 0;

    // Quantiles [2.5%, 25%, 50%, 75%, 97.5%] of the mean statistic (C# `MeanQuantiles`).
    std::vector<double> mean_quantiles;

    // Quantiles [2.5%, 25%, 50%, 75%, 97.5%] of the standard deviation statistic
    // (C# `SDQuantiles`).
    std::vector<double> sd_quantiles;

    // Quantiles [2.5%, 25%, 50%, 75%, 97.5%] of the minimum statistic (C# `MinQuantiles`).
    std::vector<double> min_quantiles;

    // Quantiles [2.5%, 25%, 50%, 75%, 97.5%] of the maximum statistic (C# `MaxQuantiles`).
    std::vector<double> max_quantiles;
};

}  // namespace bestfit::diagnostics
