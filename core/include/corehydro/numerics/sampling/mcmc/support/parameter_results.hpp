// ported from: Numerics/Sampling/MCMC/Support/ParameterResults.cs @ 2a0357a
//
// Posterior summary + diagnostics for one MCMC parameter: mean/sd (via KernelDensity's
// sample product moments), median/CI (via Statistics.Percentile), a Histogram, and (set
// externally by MCMCResults, which calls MCMCDiagnostics) Rhat/ESS/Autocorrelation.
//
// OMITTED: the KernelDensity "PDF-graph" member (C#'s `KernelDensity` field, populated by
// `kde.CreatePDFGraph()`) -- CreatePDFGraph needs Stratify, which this port has not yet
// ported (see kernel_density.hpp's own header note). The Mean/StandardDeviation used for
// SummaryStatistics come directly from KernelDensity's sample-product-moment accessors
// (mean()/standard_deviation()), which don't need CreatePDFGraph or Stratify -- only the
// PDF-graph array itself (not consumed by any MCMC fixture assertion) is skipped.
//
// `sorted`: mirrors the C# ctor parameter. When false (the default), C# sorts the CALLER'S
// `values` array in place (`Array.Sort(values)`); this port instead takes `values` by value
// and sorts its own local copy, since std::vector's value semantics make in-place mutation
// of the caller's data both unnecessary and un-idiomatic here -- the sorted copy is used
// identically either way (KernelDensity/Histogram/percentile only read it).
#pragma once
#include <algorithm>
#include <array>
#include <optional>
#include <vector>

#include "corehydro/numerics/data/histogram.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/kernel_density.hpp"
#include "corehydro/numerics/sampling/mcmc/support/parameter_statistics.hpp"

namespace corehydro::numerics::sampling::mcmc {

class ParameterResults {
   public:
    // Constructs an empty parameter results (mirrors the C# JSON-deserialization ctor;
    // here just a default-constructed placeholder for array/vector slots before a real
    // instance is assigned).
    ParameterResults() = default;

    // Constructs new parameter results. `alpha`: confidence level (default 0.1 -> 90% CI).
    explicit ParameterResults(std::vector<double> values, double alpha = 0.1, bool sorted = false) {
        if (!sorted) std::sort(values.begin(), values.end());

        // Create Kernel Density Estimate (sample mean/sd only -- see file header).
        distributions::KernelDensity kde(values);

        summary_statistics.n = static_cast<int>(values.size());
        summary_statistics.mean = kde.mean();
        summary_statistics.standard_deviation = kde.standard_deviation();
        summary_statistics.median = data::percentile(values, 0.5, true);
        summary_statistics.lower_ci = data::percentile(values, alpha / 2.0, true);
        summary_statistics.upper_ci = data::percentile(values, 1.0 - alpha / 2.0, true);

        histogram.emplace(values);
    }

    // Parameter summary statistics.
    ParameterStatistics summary_statistics;

    // The histogram results. `nullopt` for a default-constructed (empty) instance -- see
    // ctor comment above; Histogram has no default state to fall back to (its ctor
    // requires at least one data point).
    std::optional<data::Histogram> histogram;

    // The autocorrelation function for each parameter, averaged across each chain. Set
    // externally by MCMCResults (mirrors the C# `Autocorrelation { get; set; }` property,
    // which is likewise never assigned inside this class's own ctor).
    std::vector<std::array<double, 2>> autocorrelation;
};

}  // namespace corehydro::numerics::sampling::mcmc
