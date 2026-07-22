// ported from: Numerics/Sampling/Bootstrap/BootstrapResults.cs @ 2a0357a
//
// `BootstrapCIMethod` (the confidence-interval method enum), `BootstrapStatisticResult` (a
// single statistic/parameter's CI summary), and the aggregate `BootstrapResults` container.
// Ported in full -- this tiny support file has no pivotal-vs-regular split of its own; both
// `Bootstrap<TData>::get_confidence_intervals` (P3.10) and the pivotal-only
// `GetRawPivotalConfidenceIntervals` (P3.11) share the same result types.
//
// Namespace note: C# `Bootstrap<TData>`/`BootstrapResults`/`BootstrapFit` all live directly
// in the flat `Numerics.Sampling` namespace (NOT a nested `Numerics.Sampling.Bootstrap`
// namespace), even though the source files sit in a `Sampling/Bootstrap/` folder. This port
// mirrors the folder nesting for file organization (per the task brief's `sampling/
// bootstrap/` header layout) but keeps the FLAT `corehydro::numerics::sampling` namespace for
// namespace parity with C# -- unlike `sampling::mcmc`, where the C# namespace really is
// nested (`Numerics.Sampling.MCMC`).
#pragma once
#include <vector>

namespace corehydro::numerics::sampling {

// Enumeration of bootstrap confidence interval methods.
enum class BootstrapCIMethod {
    Percentile,
    BiasCorrected,
    BCa,
    Normal,
    BootstrapT,
};

// Stores bootstrap confidence interval results for a single statistic or parameter.
struct BootstrapStatisticResult {
    double population_estimate = 0.0;
    double lower_ci = 0.0;
    double upper_ci = 0.0;
    int valid_count = 0;
    int total_count = 0;
    double standard_error = 0.0;
    double mean = 0.0;
};

// Stores complete bootstrap analysis results including confidence intervals for statistics
// and parameters.
struct BootstrapResults {
    BootstrapCIMethod method = BootstrapCIMethod::Percentile;
    double alpha = 0.0;
    std::vector<BootstrapStatisticResult> statistic_results;
    std::vector<BootstrapStatisticResult> parameter_results;
    int failed_replicates = 0;
};

}  // namespace corehydro::numerics::sampling
