// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/IQuantilePriors.cs @ c2e6192
//
// Capability interface for models that accept quantile priors (expert judgment about flood
// quantiles folded into the Bayesian prior). Ported as a pure-virtual mixin (the codebase's
// convention for C# capability interfaces -- see numerics/distributions/base/i_estimation.hpp);
// M8/M9 make UnivariateDistributionModel implement it.
//
// The C# `List<QuantilePrior> QuantilePriors { get; set; }` property ports as mutable + const
// reference accessors (the ModelBase::parameters() precedent) plus an explicit setter that
// replaces the whole list (the C# `set`). QuantilePrior is held by value; it deep-copies its
// owned prior distribution on copy (see quantile_prior.hpp).
#pragma once
#include <vector>

#include "corehydro/models/support/quantile_prior.hpp"

namespace corehydro::models {

class IQuantilePriors {
   public:
    virtual ~IQuantilePriors() = default;

    // The list of quantile prior distributions (C# `QuantilePriors { get; set; }`).
    virtual std::vector<QuantilePrior>& quantile_priors() = 0;
    virtual const std::vector<QuantilePrior>& quantile_priors() const = 0;
    virtual void set_quantile_priors(std::vector<QuantilePrior> quantile_priors) = 0;

    // Determines whether to enable quantile priors (C# `EnableQuantilePriors { get; set; }`).
    virtual bool enable_quantile_priors() const = 0;
    virtual void set_enable_quantile_priors(bool enable_quantile_priors) = 0;

    // Determines whether to use exact data only in the log-likelihood function
    // (C# `UseSingleQuantile { get; set; }`).
    virtual bool use_single_quantile() const = 0;
    virtual void set_use_single_quantile(bool use_single_quantile) = 0;

    // Process the quantile priors to be Gamma distributed differences.
    virtual void process_quantile_priors() = 0;

    // Set the default quantile prior distributions.
    virtual void set_default_quantile_priors() = 0;
};

}  // namespace corehydro::models
