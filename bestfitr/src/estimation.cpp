// cpp11 glue exposing the Phase-4 estimation surface (MaximumLikelihood/MaximumAPosteriori
// today; BayesianAnalysis deferred -- see below) of the shared C++ core to R. Like
// `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are inherently STATEFUL --
// one model construct + a single estimate() run backs every assertion in a case (see
// fixtures/README.md's model_estimation schema) -- so this file exposes ONE function,
// `bf_estimation_run_`, that builds the model, runs estimate() once, and returns every value
// test-fixtures.R's dispatcher needs in one named list. Core headers are vendored under
// src/bestfit_core/include (see tools/sync_core.py).
//
// `bic` DESIGN NOTE: unlike every other wired method, C# `GetBIC(sampleSize)` takes an actual
// sample size, not a 0-based index -- but this glue (like `bf_mcmc_run_`/`bf_bootstrap_run_`)
// precomputes the full result surface up front, before any assertion is dispatched. `bic` is
// therefore precomputed ONCE here at `sample_size = length(dataset)` (the only value any
// `model_estimation` fixture case has ever needed -- BIC is always evaluated at the fitted
// data's own sample size); `test-fixtures.R`'s `bic` dispatch arm reads this precomputed
// scalar directly and does not re-derive `n` from the fixture's `args`.
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/univariate_distribution_model.hpp"
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

// Builds the named model (`family` via the distribution factory + `dataset`), constructs
// `target`'s estimator (`optimizer`, default "DifferentialEvolution"), runs estimate() once,
// and returns a named list with the full surface test-fixtures.R's model_estimation
// dispatcher reads assertions from:
//   parameters       -- [n_params] vector (BestParameterSet.Values)
//   max_log_likelihood -- scalar
//   aic              -- scalar
//   bic              -- scalar (sample_size = length(dataset); see file header DESIGN NOTE)
//   covariance       -- [n_params x n_params] matrix
//   standard_errors  -- [n_params] vector
//
// WIRED (Task T11): parameter/max_log_likelihood/aic/bic/covariance/standard_error. LEFT FOR
// TASK T12 (see fixtures/README.md's model_estimation section): correlation/dic/waic/looic/
// posterior_mean -- the last four are BayesianAnalysis-only surface, and `target =
// "BayesianAnalysis"` is not yet constructible here (throws below), matching the C++ runner.
[[cpp11::register]]
list bf_estimation_run_(std::string target, std::string family, doubles dataset, std::string optimizer) {
    std::vector<double> data(dataset.begin(), dataset.end());
    models::UnivariateDistributionModel model(dist::create_distribution(family), data);
    auto method = parse_optimization_method(optimizer);
    int n_params = model.number_of_parameters();
    int sample_size = static_cast<int>(data.size());

    writable::doubles parameters(n_params);
    double max_log_likelihood = 0.0, aic = 0.0, bic = 0.0;
    writable::doubles_matrix<by_column> covariance(n_params, n_params);
    writable::doubles standard_errors(n_params);

    auto fill_from = [&](const auto& e) {
        const auto& best = e.best_parameter_set().values;
        for (int i = 0; i < n_params; ++i) parameters[i] = best[static_cast<std::size_t>(i)];
        max_log_likelihood = e.maximum_log_likelihood();
        aic = e.get_aic();
        bic = e.get_bic(sample_size);
        auto cov = e.get_covariance_matrix();
        for (int i = 0; i < n_params; ++i)
            for (int j = 0; j < n_params; ++j) covariance(i, j) = cov(i, j);
        auto se = e.get_standard_errors();
        for (int i = 0; i < n_params; ++i) standard_errors[i] = se[static_cast<std::size_t>(i)];
    };

    if (target == "MaximumLikelihood") {
        est::MaximumLikelihood e(model, method);
        if (!e.estimate()) stop("MaximumLikelihood::estimate() failed for a fixture case");
        fill_from(e);
    } else if (target == "MaximumAPosteriori") {
        est::MaximumAPosteriori e(model, method);
        if (!e.estimate()) stop("MaximumAPosteriori::estimate() failed for a fixture case");
        fill_from(e);
    } else if (target == "BayesianAnalysis") {
        stop(
            "model_estimation target 'BayesianAnalysis' is not yet wired in the fixture runner "
            "(deferred to Task T12); see fixtures/README.md's model_estimation section");
    } else {
        stop("unknown model_estimation target '%s'", target.c_str());
    }

    return writable::list({
        "parameters"_nm = parameters,
        "max_log_likelihood"_nm = writable::doubles({max_log_likelihood}),
        "aic"_nm = writable::doubles({aic}),
        "bic"_nm = writable::doubles({bic}),
        "covariance"_nm = covariance,
        "standard_errors"_nm = standard_errors,
    });
}
