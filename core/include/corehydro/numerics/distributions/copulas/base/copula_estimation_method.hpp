// ported from: Numerics/Distributions/Bivariate Copulas/Base/CopulaEstimationMethod.cs @ a2c4dbf
//
// The three copula parameter-estimation methods mirrored by bivariate_copula_estimation.hpp's
// free `estimate(copula, x, y, method)` (the C# BivariateCopulaEstimation.Estimate static).
// There is no method-of-moments entry in this enum: the tau-based method of moments
// (`SetThetaFromTau` on the concrete Archimedean classes that support it) is not dispatched
// through BivariateCopulaEstimation.Estimate in the C# source -- callers invoke it directly
// on the concrete copula. The fixture/glue "tau" fit mode (fixtures/README.md) therefore
// bypasses this enum too and calls the concrete class's set_theta_from_tau directly.
#pragma once

namespace corehydro::numerics::distributions::copulas {

enum class CopulaEstimationMethod {
    // Requires parametric marginal distributions; the marginals and the copula dependency
    // are estimated simultaneously.
    FullLikelihood,
    // Semi-parametric: uses the plotting positions of the data (not the raw data) to
    // estimate the copula dependency via maximum likelihood.
    PseudoLikelihood,
    // Inference from margins (IFM): marginals are estimated independently first, then the
    // copula dependency is estimated by maximizing the likelihood given those marginals.
    InferenceFromMargins
};

}  // namespace corehydro::numerics::distributions::copulas
