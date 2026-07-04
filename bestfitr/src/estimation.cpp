// cpp11 glue exposing the Phase-4 estimation surface (MaximumLikelihood, MaximumAPosteriori,
// and -- as of Task T12 -- BayesianAnalysis) of the shared C++ core to R. Like
// `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are inherently STATEFUL --
// one model construct + a single estimate() run backs every assertion in a case (see
// fixtures/README.md's model_estimation schema) -- so this file exposes ONE function per
// construct shape: `bf_estimation_run_` for MaximumLikelihood/MaximumAPosteriori (shared
// {target, family, dataset, optimizer} signature) and `bf_estimation_bayes_run_` (T12) for
// BayesianAnalysis (a disjoint {family, dataset, sampler, settings...} signature -- a sampler
// type + numeric knobs, not an optimizer string). Each builds the model, runs estimate() once,
// and returns every value test-fixtures.R's dispatcher needs in one named list. Core headers
// are vendored under src/bestfit_core/include (see tools/sync_core.py).
//
// `bic` DESIGN NOTE: unlike every other wired ML/MAP method, C# `GetBIC(sampleSize)` takes an
// actual sample size, not a 0-based index. Every other method's value is precomputed once here
// (in `bf_estimation_run_`, matching `bf_mcmc_run_`/`bf_bootstrap_run_`'s "precompute the full
// surface up front" contract), since none of them take a fixture-supplied argument. `bic` is
// the one exception: it is NOT precomputed. `bf_estimation_bic_` below rebuilds the same
// model/estimator (deterministic -- NelderMead/Brent have no randomness and
// DifferentialEvolution's default `prng_seed` is fixed, so re-running `estimate()` reproduces
// the exact same fit) and calls `e.get_bic(n)` live with whatever `n` the fixture's
// `args[0]` supplies, matching C++'s `dispatch_estimation` and the C# `GetBIC(sampleSize)`
// signature exactly. See `test-fixtures.R`'s `bic` dispatch arm.
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"

namespace est = bestfit::estimation;
namespace models = bestfit::models;
namespace dist = bestfit::numerics::distributions;
using namespace cpp11;

static est::OptimizationMethod parse_optimization_method(const std::string& s) {
    if (s == "Brent") return est::OptimizationMethod::Brent;
    if (s == "NelderMead") return est::OptimizationMethod::NelderMead;
    if (s == "DifferentialEvolution") return est::OptimizationMethod::DifferentialEvolution;
    stop("unknown model_estimation optimizer '%s'", s.c_str());
}

static est::SamplerType parse_sampler_type(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    stop("unknown model_estimation sampler '%s'", s.c_str());
}

// Builds the named model (`family` via the distribution factory + `dataset`), constructs
// `target`'s estimator (`optimizer`, default "DifferentialEvolution"), runs estimate() once,
// and returns a named list with the full surface test-fixtures.R's model_estimation
// dispatcher reads assertions from:
//   parameters       -- [n_params] vector (BestParameterSet.Values)
//   max_log_likelihood -- scalar
//   aic              -- scalar
//   covariance       -- [n_params x n_params] matrix
//   standard_errors  -- [n_params] vector
//   correlation      -- [n_params x n_params] matrix (T12)
//
// `bic` is deliberately NOT part of this surface -- see `bf_estimation_bic_` below and the
// file header DESIGN NOTE.
//
// WIRED (Task T11 + T12): parameter/max_log_likelihood/aic/bic/covariance/standard_error/
// correlation. `target == "BayesianAnalysis"` is NOT handled here (see
// `bf_estimation_bayes_run_` below -- disjoint construct shape).
[[cpp11::register]]
list bf_estimation_run_(std::string target, std::string family, doubles dataset, std::string optimizer) {
    std::vector<double> data(dataset.begin(), dataset.end());
    models::UnivariateDistributionModel model(dist::create_distribution(family), data);
    auto method = parse_optimization_method(optimizer);
    int n_params = model.number_of_parameters();

    writable::doubles parameters(n_params);
    double max_log_likelihood = 0.0, aic = 0.0;
    writable::doubles_matrix<by_column> covariance(n_params, n_params);
    writable::doubles standard_errors(n_params);
    writable::doubles_matrix<by_column> correlation(n_params, n_params);

    auto fill_from = [&](const auto& e) {
        const auto& best = e.best_parameter_set().values;
        for (int i = 0; i < n_params; ++i) parameters[i] = best[static_cast<std::size_t>(i)];
        max_log_likelihood = e.maximum_log_likelihood();
        aic = e.get_aic();
        auto cov = e.get_covariance_matrix();
        for (int i = 0; i < n_params; ++i)
            for (int j = 0; j < n_params; ++j) covariance(i, j) = cov(i, j);
        auto se = e.get_standard_errors();
        for (int i = 0; i < n_params; ++i) standard_errors[i] = se[static_cast<std::size_t>(i)];
        auto corr = e.get_correlation_matrix();
        for (int i = 0; i < n_params; ++i)
            for (int j = 0; j < n_params; ++j) correlation(i, j) = corr(i, j);
    };

    if (target == "MaximumLikelihood") {
        est::MaximumLikelihood e(model, method);
        if (!e.estimate()) stop("MaximumLikelihood::estimate() failed for a fixture case");
        fill_from(e);
    } else if (target == "MaximumAPosteriori") {
        est::MaximumAPosteriori e(model, method);
        if (!e.estimate()) stop("MaximumAPosteriori::estimate() failed for a fixture case");
        fill_from(e);
    } else {
        stop("unknown model_estimation target '%s' (BayesianAnalysis uses bf_estimation_bayes_run_)",
             target.c_str());
    }

    return writable::list({
        "parameters"_nm = parameters,
        "max_log_likelihood"_nm = writable::doubles({max_log_likelihood}),
        "aic"_nm = writable::doubles({aic}),
        "covariance"_nm = covariance,
        "standard_errors"_nm = standard_errors,
        "correlation"_nm = correlation,
    });
}

// `bic [n]` accessor: rebuilds the same model + named estimator, runs `estimate()` once (see
// the file header DESIGN NOTE for why this reproduces the exact same fit as
// `bf_estimation_run_`'s call), and returns `GetBIC(n)` evaluated live at the caller-supplied
// sample size `n` -- matching C++'s `dispatch_estimation` (`est->get_bic(a[0].get<int>())`)
// and the C# `GetBIC(sampleSize)` signature. Deliberately separate from `bf_estimation_run_`
// rather than folded into its returned list, since `n` is only known at assertion-dispatch
// time (a fixture case's `bic` assertion supplies it via `args[0]`), not at construction time.
[[cpp11::register]]
double bf_estimation_bic_(std::string target, std::string family, doubles dataset, std::string optimizer, int n) {
    std::vector<double> data(dataset.begin(), dataset.end());
    models::UnivariateDistributionModel model(dist::create_distribution(family), data);
    auto method = parse_optimization_method(optimizer);

    if (target == "MaximumLikelihood") {
        est::MaximumLikelihood e(model, method);
        if (!e.estimate()) stop("MaximumLikelihood::estimate() failed for a fixture case");
        return e.get_bic(n);
    }
    if (target == "MaximumAPosteriori") {
        est::MaximumAPosteriori e(model, method);
        if (!e.estimate()) stop("MaximumAPosteriori::estimate() failed for a fixture case");
        return e.get_bic(n);
    }
    stop("unknown model_estimation target '%s' (BayesianAnalysis has no bic method)", target.c_str());
}

// --- BayesianAnalysis (Task T12) -----------------------------------------------------------
//
// Disjoint construct shape from ML/MAP: a sampler type + numeric knobs, not an optimizer
// string, so this is a separate registered function rather than a `bf_estimation_run_` branch.
// Builds the model, constructs BayesianAnalysis(model, sampler), turns off the two "use
// defaults" flags so the explicit settings below aren't clobbered, applies whichever settings
// the fixture supplies (mirrors `bf_mcmc_run_`'s settings-application convention and the
// emitter's/C++ test_fixtures.cpp's `BuildEstimation` BayesianAnalysis arm), runs `estimate()`
// once, and returns every value test-fixtures.R's model_estimation dispatcher needs:
//   dic / waic / looic     -- scalars
//   posterior_mean         -- [n_params] vector
//   chain_values           -- [n_chains x n_iterations x n_params] flattened; see below
//
// `chain_value [chain, iter, param]` DESIGN NOTE: unlike the scalar/vector surface above, the
// chain digest is a 3-D lookup. R has no native 3-D-ragged-array-from-cpp11 shortcut as clean
// as a nested list, so this returns `chain_values` as a flat numeric vector plus `chain_dims`
// (n_chains, n_iterations, n_params); `dispatch_estimation`'s R-side helper below does the
// row-major index arithmetic (matching the C++/Python/C# access order:
// chains[chain][iter].values[param]).
[[cpp11::register]]
list bf_estimation_bayes_run_(std::string family, doubles dataset, std::string sampler,
                               int seed, int iterations, int warmup_iterations,
                               int number_of_chains, int thinning_interval,
                               int initial_iterations, int output_length) {
    std::vector<double> data(dataset.begin(), dataset.end());
    models::UnivariateDistributionModel model(dist::create_distribution(family), data);
    auto sampler_type = parse_sampler_type(sampler);

    est::BayesianAnalysis ba(model, sampler_type);
    ba.set_use_simulation_defaults(false);
    ba.set_use_advanced_simulation_defaults(false);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (iterations > 0) ba.set_iterations(iterations);
    if (warmup_iterations > 0) ba.set_warmup_iterations(warmup_iterations);
    if (number_of_chains > 0) ba.set_number_of_chains(number_of_chains);
    if (thinning_interval > 0) ba.set_thinning_interval(thinning_interval);
    if (initial_iterations > 0) ba.set_initial_iterations(initial_iterations);
    if (output_length > 0) ba.set_output_length(output_length);

    if (!ba.estimate()) stop("BayesianAnalysis::estimate() failed for a fixture case");

    int n_params = model.number_of_parameters();
    writable::doubles posterior_mean(n_params);
    const auto& pm = ba.results()->posterior_mean.values;
    for (int i = 0; i < n_params; ++i) posterior_mean[i] = pm[static_cast<std::size_t>(i)];

    const auto& chains = ba.sampler()->markov_chains();
    int n_chains = static_cast<int>(chains.size());
    int n_iterations = n_chains > 0 ? static_cast<int>(chains[0].size()) : 0;
    writable::doubles chain_values(static_cast<R_xlen_t>(n_chains) * n_iterations * n_params);
    R_xlen_t k = 0;
    for (int c = 0; c < n_chains; ++c)
        for (int it = 0; it < n_iterations; ++it)
            for (int p = 0; p < n_params; ++p)
                chain_values[k++] = chains[static_cast<std::size_t>(c)][static_cast<std::size_t>(it)]
                                        .values[static_cast<std::size_t>(p)];

    return writable::list({
        "dic"_nm = writable::doubles({ba.dic()}),
        "waic"_nm = writable::doubles({ba.waic()}),
        "looic"_nm = writable::doubles({ba.looic()}),
        "posterior_mean"_nm = posterior_mean,
        "chain_values"_nm = chain_values,
        "chain_dims"_nm = writable::integers({n_chains, n_iterations, n_params}),
    });
}
