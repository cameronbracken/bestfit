// bestfit ADDITION -- no upstream C# counterpart.
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
// Keep the registry CLOSED (one entry today, "uniform_constraints"); the Gibbs sampler's
// conjugate-prior task adds a "normal_conjugate_gibbs" entry here (its `McmcModel` gains a
// third, Gibbs-specific member for the conjugate proposal at that point -- not needed yet).
#pragma once
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/sampling/mcmc/base/mcmc_sampler.hpp"

namespace bestfit::numerics::sampling::mcmc {

// A built MCMC model: priors + a log-likelihood closure (which already includes the prior
// log-density where relevant -- see mcmc_sampler.hpp's `LogLikelihood` documentation; for
// the "uniform_constraints" entry the priors are uninformative/uniform, so the model's own
// prior contribution is a constant and is omitted from the closure, matching Test_RWMH.cs).
struct McmcModel {
    std::vector<std::shared_ptr<distributions::UnivariateDistributionBase>> priors;
    LogLikelihood log_likelihood;
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

        return McmcModel{std::move(priors), std::move(log_likelihood)};
    }
    throw std::invalid_argument("unknown MCMC model registry entry: " + name);
}

}  // namespace bestfit::numerics::sampling::mcmc
