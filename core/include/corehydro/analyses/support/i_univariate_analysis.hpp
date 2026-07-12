// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Support/IUnivariateAnalysis.cs @ fc28c0c
//
// Capability interface for univariate distribution analyses. Sits at the base of the interface
// diamond -- IUnivariateAnalysis -> IBayesianAnalysis -> IAnalysis (virtual) and IUnivariateAnalysis
// -> IProbabilityOrdinates -- exactly as the C# interface hierarchy declares it. IAnalysis is a
// virtual base (via IBayesianAnalysis), so no shared-state duplication arises; IProbabilityOrdinates
// carries no IAnalysis and needs no virtual base.
//
// The three accessors each return a NULLABLE `UnivariateDistributionBase*` (C#
// `UnivariateDistributionBase?` -- null when the analysis has not been estimated), non-const to
// mirror the C# non-const return.
#pragma once

#include "corehydro/analyses/support/i_bayesian_analysis.hpp"
#include "corehydro/analyses/support/i_probability_ordinates.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"

namespace corehydro::analyses {

class IUnivariateAnalysis : public IBayesianAnalysis, public IProbabilityOrdinates {
   public:
    ~IUnivariateAnalysis() override = default;

    // The distribution for a given output index (C# `GetDistribution(int)`).
    virtual corehydro::numerics::distributions::UnivariateDistributionBase* get_distribution(
        int index) = 0;

    // The point-estimate distribution using the analysis's currently configured PointEstimator
    // (C# `GetPointEstimateDistribution()`).
    virtual corehydro::numerics::distributions::UnivariateDistributionBase*
    get_point_estimate_distribution() = 0;

    // The point-estimate distribution using a caller-supplied estimator, without mutating the
    // analysis's own PointEstimator (C#
    // `GetPointEstimateDistribution(BayesianAnalysis.PointEstimateType)`).
    virtual corehydro::numerics::distributions::UnivariateDistributionBase*
    get_point_estimate_distribution(corehydro::estimation::PointEstimateType point_estimator) = 0;
};

}  // namespace corehydro::analyses
