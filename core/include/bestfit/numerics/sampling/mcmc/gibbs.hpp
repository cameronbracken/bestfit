// ported from: Numerics/Sampling/MCMC/Gibbs.cs @ a2c4dbf
//
// Gibbs sampler: the proposal for each iteration is fully delegated to a caller-supplied
// `Proposal` closure (C# `public delegate double[] Proposal(double[] parameters, Random
// prng);`) -- typically a conjugate-posterior draw, as in the model registry's
// "normal_conjugate_gibbs" entry. ChainIteration itself is a thin two-line dispatch: call the
// proposal, evaluate its log-likelihood, done (no accept/reject step -- a Gibbs draw from the
// full conditional is always "accepted").
//
// The ctor forces `NumberOfChains = 1`, `WarmupIterations = 1`, `ThinningInterval = 1`,
// `InitialIterations = 1`, `Iterations = 100000`, `OutputLength = 10000` by writing the base
// class's PROTECTED FIELDS directly (not the public reset()-triggering setters, which would
// each re-run Reset() -- C# assigns the private backing fields `_numberOfChains` etc.
// directly for the same reason) and then calls `reset()` explicitly ONCE at the end, exactly
// mirroring the C# ctor body. These are ordinary defaults, not permanently enforced settings:
// nothing stops a caller from overriding them afterward via the normal public setters (e.g. a
// fixture's `settings.iterations` for a short/fast digest case) -- `ValidateSettings` (the
// unmodified base implementation; Gibbs does not override it) accepts any combination that
// satisfies the base class's usual constraints.
#pragma once
#include <functional>
#include <memory>
#include <vector>

#include "bestfit/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"

namespace bestfit::numerics::sampling::mcmc {

class Gibbs : public MCMCSampler {
   public:
    // The proposal function for creating a proposal vector of parameters to evaluate:
    // `(current parameters, this chain's PRNG) -> proposed parameters`. Matches the C#
    // delegate `double[] Proposal(double[] parameters, Random prng)`.
    using Proposal = std::function<std::vector<double>(const std::vector<double>&, sampling::MersenneTwister&)>;

    // Constructs a new Gibbs sampler.
    Gibbs(std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors,
          LogLikelihood log_likelihood_function, Proposal proposal_function)
        : MCMCSampler(std::move(priors), std::move(log_likelihood_function)),
          proposal_function_(std::move(proposal_function)) {
        // Create default settings -- see file header for why these bypass the reset()-
        // triggering setters.
        number_of_chains_ = 1;
        warmup_iterations_ = 1;
        thinning_interval_ = 1;
        initial_iterations_ = 1;
        iterations_ = 100000;
        output_length = 10000;
        reset();
    }

    // The proposal function for creating a proposal vector of parameters to evaluate.
    const Proposal& proposal_function() const { return proposal_function_; }

   protected:
    ParameterSet chain_iteration(int index, ParameterSet state) override {
        auto xp = proposal_function_(state.values, chain_prngs_[static_cast<std::size_t>(index)]);
        return ParameterSet(xp, log_likelihood_function_(xp));
    }

   private:
    Proposal proposal_function_;
};

}  // namespace bestfit::numerics::sampling::mcmc
