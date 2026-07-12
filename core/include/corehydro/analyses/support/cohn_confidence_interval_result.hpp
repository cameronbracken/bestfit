// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/Bulletin17CAnalysis.cs @ fc28c0c
//
// Result DTO for Bulletin17CAnalysis::compute_cohn_style_confidence_intervals (A9): the Cohn-style
// delta-method confidence intervals (the C# nested class CohnConfidenceIntervalResult, C# 3788-3833).
// It is a pure output holder, so it mirrors the C# auto-property set as public snake_case members
// (the A8 DTO naming decision) rather than the getter/setter shape BootstrapDiagnostics needs for
// its incrementing counters.
//
// NAMING NOTE (C#-governed, documented): the C# field is `ExceedanceProbabilities`, yet the C# body
// assigns it `ProbabilityOrdinates.ToArray()` -- in this codebase the ordinates ARE the exceedance
// probabilities, so `exceedance_probabilities` carries the ordinates verbatim (mirrored exactly).
//
// SKIPPED C# surface: the DTO has no ToXElement/FromXElement/INotifyPropertyChanged in the source and
// none is added here (the XML / WPF-binding surface is dropped project-wide).
#pragma once
#include <vector>

namespace corehydro::analyses {

struct CohnConfidenceIntervalResult {
    // C# `ExceedanceProbabilities`: the exceedance probabilities (== ProbabilityOrdinates) at which
    // the CIs were computed.
    std::vector<double> exceedance_probabilities;
    // C# `PointEstimates`: the quantile point estimates Q_hat_p, in log10 space (EvaluateQuantileSafe).
    std::vector<double> point_estimates;
    // C# `LowerCI` / `UpperCI`: the CI bounds in discharge space (pow(10, .) of the log10 bounds).
    std::vector<double> lower_ci;
    std::vector<double> upper_ci;
    // C# `ConfidenceLevel`: the confidence level used (e.g. 0.90 for a 90% CI).
    double confidence_level = 0.0;
    // C# `Beta1`: the regression coefficient of SE(Q_hat_p) on Q_hat_p per probability level.
    std::vector<double> beta1;
    // C# `Nu`: the effective Student-t degrees of freedom per probability level.
    std::vector<double> nu;
    // C# `QuantileVariance`: the delta-method Var(Q_hat_p) from the outer quadrature per level.
    std::vector<double> quantile_variance;
};

}  // namespace corehydro::analyses
