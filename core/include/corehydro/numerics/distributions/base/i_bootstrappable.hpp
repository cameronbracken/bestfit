// ported from: Numerics/Distributions/Univariate/Uncertainty Analysis/IBootstrappable.cs @ a2c4dbf
//
// Capability interface: a distribution that can be parametric-bootstrapped. Mirrors the C#
// IBootstrappable mixin (there `: IUnivariateDistribution`). Follows the same abstract,
// state-free base-mixin style as base/i_estimation.hpp: distributions that support the
// bootstrap multiply-inherit this and override `bootstrap(...)`; BootstrapAnalysis dynamic_casts
// to it (the C# `distribution as IBootstrappable` guard). The C# `Bootstrap` returns an
// IUnivariateDistribution; here it returns an owning std::unique_ptr<UnivariateDistributionBase>.
#pragma once
#include <memory>

#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"

namespace corehydro::numerics::distributions {

class UnivariateDistributionBase;

class IBootstrappable {
   public:
    virtual ~IBootstrappable() = default;

    // Bootstrap the distribution: draw a sample of `sample_size` from the current parameters,
    // re-fit by `method`, and return the fitted distribution. A negative/zero seed uses the
    // clock (mirrors the C# `seed = -1` default).
    virtual std::unique_ptr<UnivariateDistributionBase> bootstrap(
        ParameterEstimationMethod method, int sample_size, int seed = -1) const = 0;
};

}  // namespace corehydro::numerics::distributions
