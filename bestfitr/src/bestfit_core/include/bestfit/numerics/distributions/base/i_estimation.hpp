// ported from: Numerics/Distributions/Univariate/Parameter Estimation/IEstimation.cs @ a2c4dbf
//
// Capability interface: a distribution that can be fit to a sample. Mirrors the C#
// IEstimation mixin -- the generic fixture runner / bindings dynamic_cast to this to
// fit, rather than every distribution exposing a fit on the common base (Uniform has none).
#pragma once
#include <vector>

#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"

namespace bestfit::numerics::distributions {

class IEstimation {
   public:
    virtual ~IEstimation() = default;
    virtual void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) = 0;
};

}  // namespace bestfit::numerics::distributions
