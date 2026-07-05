// cpp11 glue exposing the estimation surface (MaximumLikelihood, MaximumAPosteriori,
// BayesianAnalysis, and -- as of M13 -- the seeded ISimulatable draw) of the shared C++ core
// to R. Like `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are inherently
// STATEFUL -- one model construct + a single estimate() run backs every assertion in a case
// (see fixtures/README.md's model_estimation schema) -- so this file exposes ONE function per
// construct shape: `bf_estimation_run_` for MaximumLikelihood/MaximumAPosteriori (shared
// {target, model_json, dataset, optimizer} signature), `bf_estimation_bayes_run_` (T12) for
// BayesianAnalysis (a disjoint {model_json, dataset, sampler, settings...} signature -- a
// sampler type + numeric knobs, not an optimizer string), and `bf_model_simulate_` (M13) for
// the estimator-less Simulation target. Each builds the model, runs its one stateful call,
// and returns every value test-fixtures.R's dispatcher needs. Core headers are vendored under
// src/bestfit_core/include (see tools/sync_core.py).
//
// M13 MODEL CONSTRUCTION: the flat Phase 4 `family` string became the serialized
// `construct.model` JSON object (`model_json`), parsed and built by the SHARED spec builder
// (bestfit/models/model_spec.hpp) -- the same code path the C++ test runner and the pybind11
// glue call, so all three harnesses construct byte-identical models (UnivariateDistribution
// incl. censored DataFrames + nonstationary trends, Mixture, CompetingRisks, PointProcess).
// The runner re-serializes the parsed fixture spec with jsonlite::toJSON(digits = I(17)),
// which round-trips doubles exactly. `dataset` stays a separate flat argument: the file-level
// `datasets` map is resolved R-side, exactly like every other fixture kind.
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
#include "bestfit/estimation/generalized_method_of_moments.hpp"
#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/model_spec.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"

namespace est = bestfit::estimation;
namespace models = bestfit::models;
using namespace cpp11;

// The shared construction path (see the file header): serialized `construct.model` JSON +
// the R-resolved flat dataset -> a ModelBase through bestfit/models/model_spec.hpp.
static std::unique_ptr<models::ModelBase> build_spec_model(const std::string& model_json,
                                                           const doubles& dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    return models::spec::build_model_from_json(model_json, data);
}

// Shared optimizer-method parser for ML/MAP AND the GMM `optimizer` knob. B11 extends it with
// the B7-un-gated BFGS/Powell/MultilevelSingleLinkage methods (with the "MLSL" alias).
static est::OptimizationMethod parse_optimization_method(const std::string& s) {
    if (s == "Brent") return est::OptimizationMethod::Brent;
    if (s == "NelderMead") return est::OptimizationMethod::NelderMead;
    if (s == "DifferentialEvolution") return est::OptimizationMethod::DifferentialEvolution;
    if (s == "BFGS") return est::OptimizationMethod::BFGS;
    if (s == "Powell") return est::OptimizationMethod::Powell;
    if (s == "MultilevelSingleLinkage" || s == "MLSL")
        return est::OptimizationMethod::MultilevelSingleLinkage;
    stop("unknown model_estimation optimizer '%s'", s.c_str());
}

// The GMM estimation-strategy knob (default Iterative, matching the C# GMM default).
static est::GeneralizedMethodOfMoments::GMMEstimationStrategy parse_gmm_strategy(
    const std::string& s) {
    using Strat = est::GeneralizedMethodOfMoments::GMMEstimationStrategy;
    if (s == "OneStep") return Strat::OneStep;
    if (s == "TwoStep") return Strat::TwoStep;
    if (s == "Iterative") return Strat::Iterative;
    stop("unknown GMM estimation strategy '%s'", s.c_str());
}

// Builds a concrete B17C model (NOT a ModelBase; see model_spec.hpp's build_bulletin17c_model
// note -- the GMM ctor takes it as IGMMModel&), fits it by GMM, and post_processes for the
// covariance stack + J-statistic. Shared by bf_estimation_gmm_run_ and bf_estimation_gmm_qvar_
// so both take the same deterministic path (BFGS + numerical Jacobian have no RNG, so a rebuild
// reproduces the same fit -- the lazy-rebuild contract `bf_estimation_bic_` relies on).
static std::unique_ptr<est::GeneralizedMethodOfMoments> build_and_fit_gmm(
    std::unique_ptr<models::Bulletin17CDistribution>& model, const std::string& model_json,
    const doubles& dataset, const std::string& strategy, const std::string& optimizer,
    int max_gmm_iterations) {
    std::vector<double> data(dataset.begin(), dataset.end());
    model = models::spec::build_bulletin17c_from_json(model_json, data);
    auto gmm = std::make_unique<est::GeneralizedMethodOfMoments>(
        *model, parse_optimization_method(optimizer));
    gmm->set_estimation_strategy(parse_gmm_strategy(strategy));
    if (max_gmm_iterations > 0) gmm->set_max_gmm_iterations(max_gmm_iterations);
    if (!gmm->estimate()) stop("GeneralizedMethodOfMoments::estimate() failed for a fixture case");
    gmm->post_process(/*use_sandwich=*/true, /*compute_jstat=*/true);
    return gmm;
}

static est::SamplerType parse_sampler_type(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    stop("unknown model_estimation sampler '%s'", s.c_str());
}

// Builds the model named by the serialized `construct.model` spec (`model_json`, via the
// shared spec builder -- see the file header), constructs `target`'s estimator (`optimizer`,
// default "DifferentialEvolution"), runs estimate() once, and returns a named list with the
// full surface test-fixtures.R's model_estimation dispatcher reads assertions from:
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
list bf_estimation_run_(std::string target, std::string model_json, doubles dataset, std::string optimizer) {
    std::unique_ptr<models::ModelBase> model_ptr = build_spec_model(model_json, dataset);
    models::ModelBase& model = *model_ptr;
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
double bf_estimation_bic_(std::string target, std::string model_json, doubles dataset, std::string optimizer, int n) {
    std::unique_ptr<models::ModelBase> model_ptr = build_spec_model(model_json, dataset);
    models::ModelBase& model = *model_ptr;
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
list bf_estimation_bayes_run_(std::string model_json, doubles dataset, std::string sampler,
                               int seed, int iterations, int warmup_iterations,
                               int number_of_chains, int thinning_interval,
                               int initial_iterations, int output_length) {
    std::unique_ptr<models::ModelBase> model_ptr = build_spec_model(model_json, dataset);
    models::ModelBase& model = *model_ptr;
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

// --- DataFrame assertion surface (M14) -------------------------------------------------
//
// Methods reachable from the model's DataFrame under ANY model_estimation target,
// corroborating the M1/M5 ctest oracles through the PUBLIC path. Builds a FRESH model via
// the shared spec builder (the frame surface is a pure function of the construct -- low
// outliers / thresholds are set at construction and plotting positions of the collections,
// never of the fit -- so a rebuild returns byte-identical values; the `bic` lazy-rebuild
// precedent). Returns everything test-fixtures.R's data-frame dispatch arms read:
//   number_of_low_outliers, low_outlier_threshold -- scalars (frame state)
//   pp_exact / pp_interval / pp_uncertain -- plotting-position vectors, in spec order, after
//     ONE calculate_plotting_positions() pass (idempotent). The threshold series is NOT
//     exposed: the C# assigns its positions to a sorted CLONE, so the original items never
//     carry one -- mirroring the C++/emitter dispatchers.
[[cpp11::register]]
list bf_model_data_frame_(std::string model_json, doubles dataset) {
    std::unique_ptr<models::ModelBase> model = build_spec_model(model_json, dataset);
    auto* udm = dynamic_cast<models::UnivariateDistributionModelBase*>(model.get());
    if (udm == nullptr || !udm->has_data_frame())
        stop("model_estimation data-frame method on a model without a DataFrame");
    auto& df = udm->data_frame();
    df.calculate_plotting_positions();

    auto positions = [](const auto& series) {
        writable::doubles out(static_cast<R_xlen_t>(series.count()));
        for (std::size_t i = 0; i < series.count(); ++i)
            out[static_cast<R_xlen_t>(i)] = series[i].plotting_position();
        return out;
    };
    return writable::list({
        "number_of_low_outliers"_nm = writable::doubles({static_cast<double>(df.number_of_low_outliers())}),
        "low_outlier_threshold"_nm = writable::doubles({df.low_outlier_threshold()}),
        "pp_exact"_nm = positions(df.exact_series()),
        "pp_interval"_nm = positions(df.interval_series()),
        "pp_uncertain"_nm = positions(df.uncertain_series()),
    });
}

// --- Simulation (M13) -----------------------------------------------------------------
//
// The estimator-less `Simulation` target: builds the model through the shared spec builder,
// calls the ISimulatable surface (`generate_random_values(sample_size, seed)`) ONCE, and
// returns the seeded draw vector; test-fixtures.R's `simulated_value [i]` dispatch indexes
// it. All four Phase 5 model types implement ISimulatable<std::vector<double>>; the
// dynamic_cast guard mirrors the C++ test runner's.
[[cpp11::register]]
doubles bf_model_simulate_(std::string model_json, doubles dataset, int sample_size, int seed) {
    std::unique_ptr<models::ModelBase> model = build_spec_model(model_json, dataset);
    auto* simulatable = dynamic_cast<models::ISimulatable<std::vector<double>>*>(model.get());
    if (simulatable == nullptr)
        stop("model_estimation Simulation target: model is not ISimulatable");
    std::vector<double> draws = simulatable->generate_random_values(sample_size, seed);
    writable::doubles out(static_cast<R_xlen_t>(draws.size()));
    for (std::size_t i = 0; i < draws.size(); ++i) out[static_cast<R_xlen_t>(i)] = draws[i];
    return out;
}

// --- GeneralizedMethodOfMoments (B11) --------------------------------------------------
//
// Disjoint construct shape from ML/MAP (a strategy + max_gmm_iterations instead of just an
// optimizer string) AND a different model type -- the concrete Bulletin17CDistribution the GMM
// ctor takes as IGMMModel& (NOT a ModelBase; see model_spec.hpp), built through the shared spec
// builder's dedicated bulletin17c entry point. So this is a separate registered function,
// mirroring the bf_estimation_bayes_run_ split. Fits once, post_processes, and returns every
// value test-fixtures.R's GMM dispatcher reads:
//   parameters / standard_errors     -- [p] vectors
//   covariance / correlation         -- [p x p] matrices
//   j_stat / j_stat_pval             -- scalars (pval is NaN when q - p == 0)
//   simulated                        -- [sample_size] seeded ISimulatable draw off the FITTED
//                                       model (present only when construct supplies sample_size),
//                                       read by the shared `simulated_value` dispatch arm.
// quantile_variance rides bf_estimation_gmm_qvar_ (per-assertion AEP, exactly like `bic`'s
// per-assertion sample size).
[[cpp11::register]]
list bf_estimation_gmm_run_(std::string model_json, doubles dataset, std::string strategy,
                            std::string optimizer, int max_gmm_iterations, int sample_size,
                            int seed) {
    std::unique_ptr<models::Bulletin17CDistribution> model;
    auto gmm = build_and_fit_gmm(model, model_json, dataset, strategy, optimizer, max_gmm_iterations);
    int p = model->number_of_parameters();

    writable::doubles parameters(p);
    const auto& best = gmm->best_parameter_set().values;
    for (int i = 0; i < p; ++i) parameters[i] = best[static_cast<std::size_t>(i)];
    writable::doubles standard_errors(p);
    auto se = gmm->get_standard_errors();
    for (int i = 0; i < p; ++i) standard_errors[i] = se[static_cast<std::size_t>(i)];
    writable::doubles_matrix<by_column> covariance(p, p);
    writable::doubles_matrix<by_column> correlation(p, p);
    auto cov = gmm->get_covariance_matrix();
    auto corr = gmm->get_correlation_matrix();
    for (int i = 0; i < p; ++i)
        for (int j = 0; j < p; ++j) {
            covariance(i, j) = cov(i, j);
            correlation(i, j) = corr(i, j);
        }

    writable::doubles simulated(static_cast<R_xlen_t>(0));
    if (sample_size > 0) {
        model->set_parameter_values(gmm->best_parameter_set().values);
        std::vector<double> draws = model->generate_random_values(sample_size, seed);
        simulated = writable::doubles(static_cast<R_xlen_t>(draws.size()));
        for (std::size_t i = 0; i < draws.size(); ++i) simulated[static_cast<R_xlen_t>(i)] = draws[i];
    }

    return writable::list({
        "parameters"_nm = parameters,
        "standard_errors"_nm = standard_errors,
        "covariance"_nm = covariance,
        "correlation"_nm = correlation,
        "j_stat"_nm = writable::doubles({gmm->jstat()}),
        "j_stat_pval"_nm = writable::doubles({gmm->jstat_pval()}),
        "simulated"_nm = simulated,
    });
}

// `quantile_variance [aep]`: like `bic [n]`, the AEP is only known at assertion-dispatch time,
// so this rebuilds the same deterministic fit and evaluates the B17C delta-method Var(Q_p) live.
// `aep` is the annual EXCEEDANCE probability; the C# QuantileVariance takes a NON-exceedance
// probability, so pass 1 - AEP.
[[cpp11::register]]
double bf_estimation_gmm_qvar_(std::string model_json, doubles dataset, std::string strategy,
                               std::string optimizer, int max_gmm_iterations, double aep) {
    std::unique_ptr<models::Bulletin17CDistribution> model;
    auto gmm = build_and_fit_gmm(model, model_json, dataset, strategy, optimizer, max_gmm_iterations);
    return model->quantile_variance(1.0 - aep, gmm->best_parameter_set().values,
                                    gmm->get_covariance_matrix().to_array());
}
