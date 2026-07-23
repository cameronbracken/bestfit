// ported from: Numerics/Sampling/MCMC/Support/ParameterStatistics.cs @ 2a0357a
//
// Plain summary-statistics holder for one MCMC parameter's posterior samples: Rhat/ESS
// (filled in by MCMCDiagnostics via ParameterResults/MCMCResults), sample size, and the
// mean/median/sd/CI summary computed by ParameterResults' ctor. A struct (not a class with
// accessors) mirrors the C# type's role as a pure data-transfer object -- every field is a
// plain get/set property with no invariants to enforce.
#pragma once
#include <limits>

namespace corehydro::numerics::sampling::mcmc {

struct ParameterStatistics {
    // The Gelman-Rubin diagnostic.
    double rhat = std::numeric_limits<double>::quiet_NaN();

    // The effective sample size.
    double ess = std::numeric_limits<double>::quiet_NaN();

    // The total sample size.
    int n = 0;

    // The parameter mean.
    double mean = 0.0;

    // The parameter median.
    double median = 0.0;

    // The parameter standard deviation.
    double standard_deviation = 0.0;

    // The lower confidence interval.
    double lower_ci = 0.0;

    // The upper confidence interval.
    double upper_ci = 0.0;
};

}  // namespace corehydro::numerics::sampling::mcmc
