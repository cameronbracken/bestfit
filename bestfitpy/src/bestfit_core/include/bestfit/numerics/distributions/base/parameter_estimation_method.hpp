// ported from: Numerics/Distributions/Univariate/Parameter Estimation/ParameterEstimationMethod.cs @ a2c4dbf
//
// The parameter-estimation method passed to a distribution's Estimate(...). Shared by
// every univariate distribution that supports fitting.
#pragma once

namespace bestfit::numerics::distributions {

enum class ParameterEstimationMethod {
    MethodOfMoments,
    MethodOfLinearMoments,
    MaximumLikelihood
};

}  // namespace bestfit::numerics::distributions
