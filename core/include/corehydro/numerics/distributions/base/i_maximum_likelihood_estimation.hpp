// ported from: Numerics/Distributions/Univariate/Parameter Estimation/IMaximumLikelihoodEstimation.cs @ 2a0357a
//
// Capability interface: a distribution that can report the (initials, lowers, uppers)
// bounds used to seed constrained maximum-likelihood optimization. Phase 1 already
// implemented this exact method non-virtually (same signature) on every distribution that
// needs it -- this mixin adds the polymorphic capability on top, so Phase 2's copula full
// MLE fit (bivariate_copula_estimation.hpp's MLE path, mirroring
// BivariateCopulaEstimation.MLE) can dynamic_cast a marginal distribution to its bounds
// without knowing its concrete type, exactly like the C# `IMaximumLikelihoodEstimation`
// interface lets BivariateCopulaEstimation.MLE dynamic_cast MarginalDistributionX/Y.
//
// The C# interface also declares `double[] MLE(IList<double> sample)`; it is NOT ported
// here because no caller -- including BivariateCopulaEstimation.MLE itself, which instead
// calls the marginal's IEstimation.Estimate(...) and then reads GetParameters -- invokes
// it polymorphically. Phase 1's per-distribution non-virtual `mle(sample)` helper (a
// different signature: returns the fitted parameter vector directly, used internally by
// each distribution's own `estimate()`) already covers every actual call site.
#pragma once
#include <vector>

namespace corehydro::numerics::distributions {

class IMaximumLikelihoodEstimation {
   public:
    virtual ~IMaximumLikelihoodEstimation() = default;

    virtual void get_parameter_constraints(const std::vector<double>& sample,
                                            std::vector<double>& initials,
                                            std::vector<double>& lowers,
                                            std::vector<double>& uppers) const = 0;
};

}  // namespace corehydro::numerics::distributions
