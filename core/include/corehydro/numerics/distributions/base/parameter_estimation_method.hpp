// ported from: Numerics/Distributions/Univariate/Parameter Estimation/EstimationMethod.cs @ 2a0357a
//
// The parameter-estimation method passed to a distribution's Estimate(...). Shared by
// every univariate distribution that supports fitting.
#pragma once

namespace corehydro::numerics::distributions {

enum class ParameterEstimationMethod {
    MethodOfMoments,
    MethodOfLinearMoments,
    MaximumLikelihood
};

}  // namespace corehydro::numerics::distributions
