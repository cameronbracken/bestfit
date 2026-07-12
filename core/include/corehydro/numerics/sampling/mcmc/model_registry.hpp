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
// task) -- a conjugate Normal-InverseGamma model transcribed VERBATIM from
// Test_Gibbs_NormalDist_RStan, whose `McmcModel` also carries a Gibbs `Proposal` closure (the
// third `McmcModel` member the P3.5 header comment anticipated). The closure captures the
// SAME `shared_ptr<Normal>`/`shared_ptr<InverseGamma>` objects also held (as
// `UnivariateDistributionBase` pointers) in `priors` -- mirroring the C# source's reference-
// type aliasing (`muPrior`/`sigmaPrior` are the literal same objects the proposal closure and
// the sampler's `PriorDistributions` list both point at; `SetParameters` inside the closure
// mutates the SAME instance the sampler consults for feasibility/initialization). A closure
// that instead captured a COPY of the prior state (e.g. by value-capturing a dereferenced
// `Normal`/`InverseGamma`, or by re-looking-up a distribution via the factory) would diverge
// silently at draw >= 1, once the mutation has actually happened once -- this is why
// `McmcModel::priors` and the registry's own local prior variables are `shared_ptr`, not
// `unique_ptr`: only `shared_ptr` lets both the sampler and the closure alias the same
// mutable object.
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
        // Transcribed from Test_Gibbs_NormalDist_RStan VERBATIM. `family`/the factory are
        // unused here (the conjugate math below is Normal-InverseGamma-specific, unlike
        // "uniform_constraints"'s family-generic closure) -- guarded defensively.
        if (family != "Normal")
            throw std::invalid_argument(
                "model registry entry 'normal_conjugate_gibbs' only supports family 'Normal'");

        int n = static_cast<int>(data.size());
        double mu = corehydro::numerics::data::mean(data);
        double mu0 = 0.0, sigma0 = 5e5;
        auto mu_prior = std::make_shared<distributions::Normal>(mu0, sigma0);
        double alpha0 = 2.0, beta0 = 0.001;
        // InverseGamma(scale, shape) -- C# `new InverseGamma(beta0, alpha0)`; see
        // inverse_gamma.hpp's ctor doc for the (scale, shape) = (beta, alpha) argument order.
        auto sigma_prior = std::make_shared<distributions::InverseGamma>(beta0, alpha0);

        std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors{mu_prior, sigma_prior};

        LogLikelihood log_likelihood = [data](const std::vector<double>& params) {
            distributions::Normal dist(params[0], params[1]);
            return dist.log_likelihood(data);
        };

        // The conjugate proposal closure. Captures `mu_prior`/`sigma_prior` BY VALUE as
        // `shared_ptr` copies -- copying a `shared_ptr` copies the pointer, not the
        // pointee, so the closure aliases the SAME underlying Normal/InverseGamma instances
        // also held in `priors` above (see the file header's prior-aliasing note). Every
        // `set_parameters` call below mutates that shared instance in place, exactly
        // mirroring the C# closure's `muPrior.SetParameters(...)`/
        // `sigmaPrior.SetParameters(...)` reference-type mutation.
        Gibbs::Proposal proposal = [data, n, mu, mu0, sigma0, mu_prior, sigma_prior](
                                        const std::vector<double>& x,
                                        sampling::MersenneTwister& random) -> std::vector<double> {
            // Sample mu. Transcribed verbatim from Test_Gibbs.cs, INCLUDING the `mu0 / 2.0`
            // term -- the textbook conjugate-Normal posterior mean has `mu0 / sigma0^2` there
            // instead, and this line's own numerator/denominator otherwise omit the `/ sigma^2`
            // scaling the companion `sigma2` line below correctly applies. Unobservable in this
            // registry entry's `mu0 = 0` test data (the term vanishes) -- see the CONSISTENCY
            // finding in docs/upstream-csharp-issues.md before reusing this formula with a
            // non-zero mu0.
            double mun = (n * mu + mu0 / 2.0) / (n + 1.0 / (sigma0 * sigma0));
            double sigma2 = (x[1] * x[1]) / (n + (x[1] * x[1]) / (sigma0 * sigma0));
            mu_prior->set_parameters({mun, std::sqrt(sigma2)});
            double mup = mu_prior->inverse_cdf(random.next_double());

            // Sample sigma.
            double alpha1 = n / 2.0;
            double sse = 0.0;
            for (double v : data) sse += (v - mup) * (v - mup);
            double beta1 = sse / 2.0;
            sigma_prior->set_parameters({beta1, alpha1});
            double sig2p = sigma_prior->inverse_cdf(random.next_double());

            // Return the proposal vector.
            return {mup, std::sqrt(sig2p)};
        };

        return McmcModel{std::move(priors), std::move(log_likelihood), std::move(proposal)};
    }
    throw std::invalid_argument("unknown MCMC model registry entry: " + name);
}

}  // namespace corehydro::numerics::sampling::mcmc
