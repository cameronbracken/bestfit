// corehydro ADDITION -- no upstream C# counterpart.
//
// A small, CLOSED registry mapping a named "model family" (e.g. an uninformative-Uniform-
// prior Normal model) plus a dataset to the (priors, log-likelihood) pair MCMCSampler needs.
// Mirrors copula_factory.hpp's role for copulas: this is NOT a port of anything under
// `Numerics.Sampling.MCMC` -- the C# equivalent is inline model-construction code that lives
// only inside a unit test (`Test_RWMH.cs`'s `Test_RWMH_NormalDist_RStan`), not in the
// library itself. This header gives the fixture runners and the oracle emitter a single,
// shared place to build the exact same models the fixtures reference by name, so the
// construction logic isn't duplicated four times (C++/R/Python/C#).
//
// Registry (now two entries): "uniform_constraints" (P3.5) and "normal_conjugate_gibbs" (this
// task) -- a conjugate Normal-InverseGamma model transcribed from Test_Gibbs_NormalDist_RStan
// (v2.1.4 rework: ConditionalMeanParameters/ConditionalVarianceParameters), whose `McmcModel`
// also carries a Gibbs `Proposal` closure (the third `McmcModel` member the P3.5 header
// comment anticipated).
//
// v2.1.4 SPLIT (superseding the pre-v2.1.4 note this replaced): the two distributions held in
// `priors` below (`mu_initialization_prior`/`sigma_initialization_prior`, used ONLY to seed
// the sampler's feasibility bounds / initial draws) are now DISTINCT objects from the two the
// proposal closure mutates every Gibbs step (`conditional_mean`/`conditional_variance`,
// default-constructed and captured only by the closure) -- mirroring the C# rework's
// `muInitializationPrior`/`sigmaInitializationPrior` versus the freshly constructed
// `conditionalMean`/`conditionalVariance` locals. Pre-v2.1.4, a single `mu_prior`/
// `sigma_prior` pair served BOTH roles (aliased into `priors` AND mutated by the closure); the
// `shared_ptr` choice below is retained for `conditional_mean`/`conditional_variance` purely
// so the same object persists (and is mutated in place) across repeated closure
// invocations -- there is no aliasing with `priors` to preserve anymore.
#pragma once
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/inverse_gamma.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/sampling/mcmc/gibbs.hpp"

namespace corehydro::numerics::sampling::mcmc {

// A built MCMC model: priors + a log-likelihood closure (which already includes the prior
// log-density where relevant -- see mcmc_sampler.hpp's `LogLikelihood` documentation; for
// the "uniform_constraints" entry the priors are uninformative/uniform, so the model's own
// prior contribution is a constant and is omitted from the closure, matching Test_RWMH.cs).
// `proposal`: only populated for Gibbs-family registry entries (default-constructed/empty
// `std::function` -- falsy via `operator bool` -- for every other entry).
struct McmcModel {
    std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors;
    LogLikelihood log_likelihood;
    Gibbs::Proposal proposal;
};

// Builds a named MCMC model. `family` is a univariate distribution type name (e.g.
// "Normal") constructible via distributions::create_distribution(); `data` is the sample
// the log-likelihood closure evaluates against.
inline McmcModel build_model(const std::string& name, const std::string& family,
                              const std::vector<double>& data) {
    if (name == "uniform_constraints") {
        // Transcribed from Test_RWMH.cs's Test_RWMH_NormalDist_RStan: uninformative
        // Uniform priors bounded by the family's own IMaximumLikelihoodEstimation
        // constraints (`family.GetParameterConstraints(sample)`'s lower/upper arrays), and
        // a log-likelihood closure that refits a fresh `family` instance's parameters at
        // each proposal and returns its sample log-likelihood -- exactly `new Normal(x[0],
        // x[1]).LogLikelihood(sample)`.
        auto probe = distributions::create_distribution(family);
        auto* imle = dynamic_cast<distributions::IMaximumLikelihoodEstimation*>(probe.get());
        if (imle == nullptr) {
            throw std::invalid_argument("model family '" + family +
                                         "' does not implement IMaximumLikelihoodEstimation");
        }
        std::vector<double> initials, lowers, uppers;
        imle->get_parameter_constraints(data, initials, lowers, uppers);

        std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors;
        priors.reserve(lowers.size());
        for (std::size_t i = 0; i < lowers.size(); ++i)
            priors.push_back(std::make_shared<distributions::Uniform>(lowers[i], uppers[i]));

        LogLikelihood log_likelihood = [family, data](const std::vector<double>& params) {
            auto d = distributions::create_distribution(family);
            d->set_parameters(params);
            return d->log_likelihood(data);
        };

        return McmcModel{std::move(priors), std::move(log_likelihood), Gibbs::Proposal{}};
    }
    if (name == "normal_conjugate_gibbs") {
        // Transcribed from Test_Gibbs_NormalDist_RStan (v2.1.4 rework:
        // ConditionalMeanParameters/ConditionalVarianceParameters). `family`/the factory are
        // unused here (the conjugate math below is Normal-InverseGamma-specific, unlike
        // "uniform_constraints"'s family-generic closure) -- guarded defensively.
        if (family != "Normal")
            throw std::invalid_argument(
                "model registry entry 'normal_conjugate_gibbs' only supports family 'Normal'");

        int n = static_cast<int>(data.size());
        double mu = corehydro::numerics::data::mean(data);
        double mu0 = 0.0, sigma0 = 5e5;
        // The conditional-variance update's own prior is the limiting non-informative
        // inverse-gamma (shape = scale = 0) -- see the proposal closure below. DISTINCT from
        // `initialization_shape`/`initialization_scale`, which only seed
        // `sigma_initialization_prior`'s feasibility bounds.
        double variance_prior_shape = 0.0, variance_prior_scale = 0.0;
        // Proper distributions used SOLELY to initialize the sampler state (feasibility
        // bounds / initial draws) -- see the file header's v2.1.4 split note.
        double initialization_shape = 2.0, initialization_scale = 0.001;
        auto mu_initialization_prior = std::make_shared<distributions::Normal>(mu0, sigma0);
        // InverseGamma(scale, shape) -- C# `new InverseGamma(initializationScale,
        // initializationShape)`; see inverse_gamma.hpp's ctor doc for the (scale, shape)
        // argument order.
        auto sigma_initialization_prior =
            std::make_shared<distributions::InverseGamma>(initialization_scale, initialization_shape);

        std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors{
            mu_initialization_prior, sigma_initialization_prior};

        LogLikelihood log_likelihood = [data](const std::vector<double>& params) {
            distributions::Normal dist(params[0], params[1]);
            return dist.log_likelihood(data);
        };

        // The two full-conditional distributions the proposal closure below mutates every
        // Gibbs step (`set_parameters` then `inverse_cdf`) -- default-constructed and owned
        // solely by this closure, NOT aliased with `priors` above; see the file header's
        // v2.1.4 split note.
        auto conditional_mean = std::make_shared<distributions::Normal>();
        auto conditional_variance = std::make_shared<distributions::InverseGamma>();

        Gibbs::Proposal proposal = [data, n, mu, mu0, sigma0, variance_prior_shape, variance_prior_scale,
                                     conditional_mean, conditional_variance](
                                        const std::vector<double>& x,
                                        sampling::MersenneTwister& random) -> std::vector<double> {
            // Sample the conditional mean given the current standard deviation
            // (ConditionalMeanParameters): the corrected conjugate-Normal posterior
            // mean/variance, replacing the pre-v2.1.4 `mu0 / 2.0` bug (see
            // docs/upstream-csharp-issues.md for the retired CONSISTENCY finding).
            double likelihood_variance = x[1] * x[1];
            double prior_variance = sigma0 * sigma0;
            double posterior_variance = 1.0 / (n / likelihood_variance + 1.0 / prior_variance);
            double posterior_mean = posterior_variance * (n * mu / likelihood_variance + mu0 / prior_variance);
            conditional_mean->set_parameters({posterior_mean, std::sqrt(posterior_variance)});
            double mup = conditional_mean->inverse_cdf(random.next_double());

            // Sample the conditional variance (ConditionalVarianceParameters), then return
            // its square root as sigma. Algebraically unchanged from the pre-v2.1.4 formula
            // (only the mean update above was buggy) -- see the file header.
            double sum_of_squared_errors = 0.0;
            for (double v : data) {
                double residual = v - mup;
                sum_of_squared_errors += residual * residual;
            }
            double scale = variance_prior_scale + sum_of_squared_errors / 2.0;
            double shape = variance_prior_shape + static_cast<double>(n) / 2.0;
            conditional_variance->set_parameters({scale, shape});
            double sig2p = conditional_variance->inverse_cdf(random.next_double());

            // Return the proposal vector.
            return {mup, std::sqrt(sig2p)};
        };

        return McmcModel{std::move(priors), std::move(log_likelihood), std::move(proposal)};
    }
    throw std::invalid_argument("unknown MCMC model registry entry: " + name);
}

}  // namespace corehydro::numerics::sampling::mcmc
