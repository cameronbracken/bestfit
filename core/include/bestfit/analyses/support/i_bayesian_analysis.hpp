// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Support/IBayesianAnalysis.cs @ fc28c0c
//
// Capability interface for Bayesian analyses: exposes the configured BayesianAnalysis estimator and
// the (nullable) uncertainty-analysis results. Derives virtually from IAnalysis so the shared
// interface base collapses to a single subobject in the IUnivariateAnalysis diamond (see
// analysis_base.hpp).
//
// Return-type choices:
//  * bayesian_analysis() -> NON-const `BayesianAnalysis&`. The C# `BayesianAnalysis { get; }`
//    returns the live estimator object; concrete analyses (A5/A7) hold it by value and mutate its
//    knobs (sampler, seed, PointEstimator), so the reference must permit mutation.
//  * analysis_results() -> `const UncertaintyAnalysisResults*`. The C# `UncertaintyAnalysisResults?`
//    is nullable (null until the analysis is estimated); a pointer models the nullability, const
//    because the results DTO is read-only to consumers.
#pragma once

#include "bestfit/analyses/support/i_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace bestfit::analyses {

class IBayesianAnalysis : public virtual IAnalysis {
   public:
    ~IBayesianAnalysis() override = default;

    // The Bayesian analysis (estimator) object (C# `BayesianAnalysis BayesianAnalysis { get; }`).
    virtual bestfit::estimation::BayesianAnalysis& bayesian_analysis() = 0;

    // The uncertainty analysis results, or null when the analysis is not estimated
    // (C# `UncertaintyAnalysisResults? AnalysisResults { get; }`).
    virtual const bestfit::numerics::distributions::UncertaintyAnalysisResults* analysis_results()
        const = 0;
};

}  // namespace bestfit::analyses
