// ported from: Numerics/Distributions/Univariate/Parameter Estimation/ILinearMomentEstimation.cs @ <pending-sha>
//
// Capability interface: a distribution whose parameters map to/from sample L-moments.
// Mirrors the C# ILinearMomentEstimation mixin; the generic runner dynamic_casts to it
// for the L-moment round-trip assertions.
#pragma once
#include <vector>

namespace bestfit::numerics::distributions {

class ILinearMomentEstimation {
   public:
    virtual ~ILinearMomentEstimation() = default;
    // {L1, L2, T3, T4} -> parameters.
    virtual std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const = 0;
    // parameters -> {L1, L2, T3, T4}.
    virtual std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const = 0;
};

}  // namespace bestfit::numerics::distributions
