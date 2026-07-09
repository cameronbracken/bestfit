// pybind11 glue exposing the estimation surface (MaximumLikelihood, MaximumAPosteriori,
// BayesianAnalysis, and -- as of M13 -- the seeded ISimulatable draw) of the shared C++ core
// to Python. Like `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are
// inherently STATEFUL -- one model construct + a single estimate() run backs every assertion
// in a case (see fixtures/README.md's model_estimation schema) -- so this file exposes ONE
// function per construct shape: `estimation_run` for MaximumLikelihood/MaximumAPosteriori
// (shared {target, model_json, dataset, optimizer} signature), `estimation_bayes_run` (T12)
// for BayesianAnalysis (a disjoint {model_json, dataset, sampler, settings...} signature -- a
// sampler type + numeric knobs, not an optimizer string), and `model_simulate` (M13) for the
// estimator-less Simulation target, mirroring bestfitr's `bf_estimation_run_`/
// `bf_estimation_bayes_run_`/`bf_model_simulate_` split. Each builds the model, runs its one
// stateful call, and returns every value test_fixtures.py's dispatcher needs in one dict.
// Core headers are vendored under ../bestfit_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
//
// M13 MODEL CONSTRUCTION: the flat Phase 4 `family` string became the serialized
// `construct.model` JSON object (`model_json`), parsed and built by the SHARED spec builder
// (bestfit/models/model_spec.hpp) -- the same code path the C++ test runner and the cpp11
// glue call, so all three harnesses construct byte-identical models (UnivariateDistribution
// incl. censored DataFrames + nonstationary trends, Mixture, CompetingRisks, PointProcess).
// The runner re-serializes the parsed fixture spec with `json.dumps()`, which round-trips
// doubles exactly. `dataset` stays a separate flat argument: the file-level `datasets` map is
// resolved Python-side, exactly like every other fixture kind.
//
// `bic` DESIGN NOTE: unlike every other wired ML/MAP method, C# `GetBIC(sampleSize)` takes an
// actual sample size, not a 0-based index. Every other method's value is precomputed once here
// (in `estimation_run`, matching `mcmc_run`/`bootstrap_run`'s "precompute the full surface up
// front" contract), since none of them take a fixture-supplied argument. `bic` is the one
// exception: it is NOT precomputed. `estimation_bic` below rebuilds the same model/estimator
// (deterministic -- NelderMead/Brent have no randomness and DifferentialEvolution's default
// `prng_seed` is fixed, so re-running `estimate()` reproduces the exact same fit) and calls
// `e.get_bic(n)` live with whatever `n` the fixture's `args[0]` supplies, matching C++'s
// `dispatch_estimation` and the C# `GetBIC(sampleSize)` signature exactly. See
// `test_fixtures.py`'s `bic` dispatch arm.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
#include "bindings.hpp"

namespace py = pybind11;
namespace est = bestfit::estimation;
namespace models = bestfit::models;

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
    throw py::value_error("unknown model_estimation optimizer: " + s);
}

// The GMM estimation-strategy knob (default Iterative, matching the C# GMM default).
static est::GeneralizedMethodOfMoments::GMMEstimationStrategy parse_gmm_strategy(
    const std::string& s) {
    using Strat = est::GeneralizedMethodOfMoments::GMMEstimationStrategy;
    if (s == "OneStep") return Strat::OneStep;
    if (s == "TwoStep") return Strat::TwoStep;
    if (s == "Iterative") return Strat::Iterative;
    throw py::value_error("unknown GMM estimation strategy: " + s);
}

// Builds a B17C model, fits it by GMM, and (optionally) post_processes for the covariance
// stack + J-statistic. Shared by estimation_gmm_run and estimation_gmm_qvar so both take the
// exact same deterministic path (BFGS + numerical Jacobian have no RNG, so a rebuild
// reproduces the same fit -- the same lazy-rebuild contract `estimation_bic` relies on).
static std::unique_ptr<est::GeneralizedMethodOfMoments> build_and_fit_gmm(
    std::unique_ptr<models::Bulletin17CDistribution>& model, const std::string& model_json,
    const std::vector<double>& dataset, const std::string& strategy, const std::string& optimizer,
    int max_gmm_iterations) {
    model = models::spec::build_bulletin17c_from_json(model_json, dataset);
    auto gmm = std::make_unique<est::GeneralizedMethodOfMoments>(*model,
                                                                 parse_optimization_method(optimizer));
    gmm->set_estimation_strategy(parse_gmm_strategy(strategy));
    if (max_gmm_iterations > 0) gmm->set_max_gmm_iterations(max_gmm_iterations);
    if (!gmm->estimate())
        throw py::value_error("GeneralizedMethodOfMoments::estimate() failed for a fixture case");
    gmm->post_process(/*use_sandwich=*/true, /*compute_jstat=*/true);
    return gmm;
}

static est::SamplerType parse_sampler_type(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    throw py::value_error("unknown model_estimation sampler: " + s);
}

// Seeded ISimulatable draw, flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly across model types (P3). Most Phase 4-7 models are ISimulatable<std::vector<double>>;
// BivariateDistribution is ISimulatable<Matrix2D> (n-row x 2-col), flattened ROW-MAJOR
// (i = row*2 + col) -- the same order the C++/R glue and the README schema use.
static std::vector<double> simulate_flat(models::ModelBase* model, int sample_size, int seed) {
    if (auto* s = dynamic_cast<models::ISimulatable<std::vector<double>>*>(model))
        return s->generate_random_values(sample_size, seed);
    if (auto* s = dynamic_cast<models::ISimulatable<std::vector<std::vector<double>>>*>(model)) {
        std::vector<std::vector<double>> mat = s->generate_random_values(sample_size, seed);
        std::vector<double> flat;
        for (const auto& row : mat)
            for (double v : row) flat.push_back(v);
        return flat;
    }
    throw py::value_error(
        "model_estimation Simulation target: model is not ISimulatable<vector> or "
        "ISimulatable<Matrix2D>");
}

void register_estimation(py::module_& m) {
    // Builds the named model (`family` via the distribution factory + `dataset`), constructs
    // `target`'s estimator (`optimizer`, default "DifferentialEvolution"), runs estimate()
    // once, and returns a dict with the full surface test_fixtures.py's model_estimation
    // dispatcher reads assertions from:
    //   parameters          -- [n_params] list (BestParameterSet.Values)
    //   max_log_likelihood  -- scalar
    //   aic                 -- scalar
    //   covariance          -- [n_params][n_params] nested list
    //   standard_errors     -- [n_params] list
    //   correlation         -- [n_params][n_params] nested list (T12)
    //
    // `bic` is deliberately NOT part of this surface -- see `estimation_bic` below and the file
    // header DESIGN NOTE.
    //
    // WIRED (Task T11 + T12): parameter/max_log_likelihood/aic/bic/covariance/standard_error/
    // correlation. `target == "BayesianAnalysis"` is NOT handled here (see
    // `estimation_bayes_run` below -- disjoint construct shape); dic/waic/looic/posterior_mean/
    // chain_value are BayesianAnalysis-only surface exposed there instead.
    m.def(
        "estimation_run",
        [](const std::string& target, const std::string& model_json, const std::vector<double>& dataset,
           const std::string& optimizer, int sample_size, int seed) {
            std::unique_ptr<models::ModelBase> model_ptr =
                models::spec::build_model_from_json(model_json, dataset);
            models::ModelBase& model = *model_ptr;
            auto method = parse_optimization_method(optimizer);
            int n_params = model.number_of_parameters();

            py::dict out;
            std::vector<double> best_values;
            auto fill_from = [&](const auto& e) {
                best_values = e.best_parameter_set().values;
                out["parameters"] = e.best_parameter_set().values;
                out["max_log_likelihood"] = e.maximum_log_likelihood();
                out["aic"] = e.get_aic();

                // The covariance stack needs >= 2 parameters (the C# GetCovarianceMatrix throws
                // below that); the single-parameter bivariate copula fit skips it -- no fixture
                // asserts covariance/SE/correlation for a 1-param model. The C++ runner sidesteps
                // this by computing the covariance lazily, only when asserted.
                if (n_params >= 2) {
                    auto cov = e.get_covariance_matrix();
                    std::vector<std::vector<double>> covariance(static_cast<std::size_t>(n_params));
                    for (int i = 0; i < n_params; ++i) {
                        covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_params));
                        for (int j = 0; j < n_params; ++j)
                            covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cov(i, j);
                    }
                    out["covariance"] = covariance;
                    out["standard_errors"] = e.get_standard_errors();

                    auto corr = e.get_correlation_matrix();
                    std::vector<std::vector<double>> correlation(static_cast<std::size_t>(n_params));
                    for (int i = 0; i < n_params; ++i) {
                        correlation[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_params));
                        for (int j = 0; j < n_params; ++j)
                            correlation[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = corr(i, j);
                    }
                    out["correlation"] = correlation;
                }
            };

            if (target == "MaximumLikelihood") {
                est::MaximumLikelihood e(model, method);
                if (!e.estimate()) throw py::value_error("MaximumLikelihood::estimate() failed for a fixture case");
                fill_from(e);
            } else if (target == "MaximumAPosteriori") {
                est::MaximumAPosteriori e(model, method);
                if (!e.estimate())
                    throw py::value_error("MaximumAPosteriori::estimate() failed for a fixture case");
                fill_from(e);
            } else if (target == "BayesianAnalysis") {
                throw py::value_error(
                    "model_estimation target 'BayesianAnalysis' uses estimation_bayes_run, not "
                    "estimation_run (disjoint construct shape)");
            } else {
                throw py::value_error("unknown model_estimation target: " + target);
            }

            // Optional seeded-draw digest off the FITTED model (P3): pin the best parameters and
            // cache one seeded draw so one MLE smoke file covers parameter + max_log_likelihood +
            // a seeded draw, mirroring the C++/R/GMM arms. `simulate_flat` handles the bivariate
            // Matrix2D flatten.
            if (sample_size > 0) {
                model.set_parameter_values(best_values);
                out["simulated"] = simulate_flat(&model, sample_size, seed);
            }
            return out;
        },
        py::arg("target"), py::arg("model_json"), py::arg("dataset"), py::arg("optimizer"),
        py::arg("sample_size"), py::arg("seed"));

    // `bic [n]` accessor: rebuilds the same model + named estimator, runs `estimate()` once
    // (see the file header DESIGN NOTE for why this reproduces the exact same fit as
    // `estimation_run`'s call), and returns `GetBIC(n)` evaluated live at the caller-supplied
    // sample size `n` -- matching C++'s `dispatch_estimation`
    // (`est->get_bic(a[0].get<int>())`) and the C# `GetBIC(sampleSize)` signature.
    // Deliberately separate from `estimation_run` rather than folded into its returned dict,
    // since `n` is only known at assertion-dispatch time (a fixture case's `bic` assertion
    // supplies it via `args[0]`), not at construction time.
    m.def(
        "estimation_bic",
        [](const std::string& target, const std::string& model_json, const std::vector<double>& dataset,
           const std::string& optimizer, int n) {
            std::unique_ptr<models::ModelBase> model_ptr =
                models::spec::build_model_from_json(model_json, dataset);
            models::ModelBase& model = *model_ptr;
            auto method = parse_optimization_method(optimizer);

            if (target == "MaximumLikelihood") {
                est::MaximumLikelihood e(model, method);
                if (!e.estimate()) throw py::value_error("MaximumLikelihood::estimate() failed for a fixture case");
                return e.get_bic(n);
            }
            if (target == "MaximumAPosteriori") {
                est::MaximumAPosteriori e(model, method);
                if (!e.estimate())
                    throw py::value_error("MaximumAPosteriori::estimate() failed for a fixture case");
                return e.get_bic(n);
            }
            if (target == "BayesianAnalysis") {
                throw py::value_error(
                    "model_estimation target 'BayesianAnalysis' has no bic method");
            }
            throw py::value_error("unknown model_estimation target: " + target);
        },
        py::arg("target"), py::arg("model_json"), py::arg("dataset"), py::arg("optimizer"), py::arg("n"));

    // --- BayesianAnalysis (Task T12) -------------------------------------------------------
    //
    // Disjoint construct shape from ML/MAP: a sampler type + numeric knobs, not an optimizer
    // string, so this is a separate registered function rather than an `estimation_run`
    // branch. Builds the model, constructs BayesianAnalysis(model, sampler), turns off the two
    // "use defaults" flags so the explicit settings below aren't clobbered, applies whichever
    // settings the fixture supplies (mirrors `mcmc_run`'s settings-application convention and
    // the emitter's/C++ test_fixtures.cpp's/bestfitr's `bf_estimation_bayes_run_` BayesianAnalysis
    // construction), runs `estimate()` once, and returns every value test_fixtures.py's
    // model_estimation dispatcher needs:
    //   dic / waic / looic  -- scalars
    //   posterior_mean      -- [n_params] list
    //   chains              -- [n_chains][n_iterations][n_params] nested list (MarkovChains),
    //                          matching `mcmc_run`'s "chains" convention (unlike R, which has
    //                          no clean 3-D-from-cpp11 shortcut, pybind11/stl handles nested
    //                          std::vector natively, so no flat-plus-dims encoding is needed).
    // `chain_value [chain, iter, param]` on the Python side simply triple-indexes this nested
    // list -- matching the C++/R/C# access order: chains[chain][iter].values[param].
    m.def(
        "estimation_bayes_run",
        [](const std::string& model_json, const std::vector<double>& dataset, const std::string& sampler,
           const py::dict& settings) {
            std::unique_ptr<models::ModelBase> model_ptr =
                models::spec::build_model_from_json(model_json, dataset);
            models::ModelBase& model = *model_ptr;
            auto sampler_type = parse_sampler_type(sampler);

            est::BayesianAnalysis ba(model, sampler_type);
            ba.set_use_simulation_defaults(false);
            ba.set_use_advanced_simulation_defaults(false);
            if (settings.contains("seed")) ba.set_prng_seed(settings["seed"].cast<int>());
            if (settings.contains("iterations")) ba.set_iterations(settings["iterations"].cast<int>());
            if (settings.contains("warmup_iterations"))
                ba.set_warmup_iterations(settings["warmup_iterations"].cast<int>());
            if (settings.contains("number_of_chains"))
                ba.set_number_of_chains(settings["number_of_chains"].cast<int>());
            if (settings.contains("thinning_interval"))
                ba.set_thinning_interval(settings["thinning_interval"].cast<int>());
            if (settings.contains("initial_iterations"))
                ba.set_initial_iterations(settings["initial_iterations"].cast<int>());
            if (settings.contains("output_length")) ba.set_output_length(settings["output_length"].cast<int>());

            if (!ba.estimate()) throw py::value_error("BayesianAnalysis::estimate() failed for a fixture case");

            int n_params = model.number_of_parameters();
            std::vector<double> posterior_mean(static_cast<std::size_t>(n_params));
            const auto& pm = ba.results()->posterior_mean.values;
            for (int i = 0; i < n_params; ++i) posterior_mean[static_cast<std::size_t>(i)] = pm[static_cast<std::size_t>(i)];

            const auto& raw_chains = ba.sampler()->markov_chains();
            int n_chains = static_cast<int>(raw_chains.size());
            std::vector<std::vector<std::vector<double>>> chains(static_cast<std::size_t>(n_chains));
            for (int c = 0; c < n_chains; ++c) {
                const auto& chain = raw_chains[static_cast<std::size_t>(c)];
                chains[static_cast<std::size_t>(c)].resize(chain.size());
                for (std::size_t it = 0; it < chain.size(); ++it) chains[static_cast<std::size_t>(c)][it] = chain[it].values;
            }

            py::dict out;
            out["dic"] = ba.dic();
            out["waic"] = ba.waic();
            out["looic"] = ba.looic();
            out["posterior_mean"] = posterior_mean;
            out["chains"] = chains;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sampler"), py::arg("settings"));

    // --- DataFrame assertion surface (M14) ---------------------------------------------
    //
    // Methods reachable from the model's DataFrame under ANY model_estimation target,
    // corroborating the M1/M5 ctest oracles through the PUBLIC path. Builds a FRESH model
    // via the shared spec builder (the frame surface is a pure function of the construct --
    // low outliers / thresholds are set at construction and plotting positions of the
    // collections, never of the fit -- so a rebuild returns byte-identical values; the
    // `bic` lazy-rebuild precedent). Returns everything test_fixtures.py's data-frame
    // dispatch arms read:
    //   number_of_low_outliers, low_outlier_threshold -- scalars (frame state)
    //   pp_exact / pp_interval / pp_uncertain -- plotting-position lists, in spec order,
    //     after ONE calculate_plotting_positions() pass (idempotent). The threshold series
    //     is NOT exposed: the C# assigns its positions to a sorted CLONE, so the original
    //     items never carry one -- mirroring the C++/emitter/R dispatchers.
    m.def(
        "model_data_frame",
        [](const std::string& model_json, const std::vector<double>& dataset) {
            std::unique_ptr<models::ModelBase> model =
                models::spec::build_model_from_json(model_json, dataset);
            auto* udm = dynamic_cast<models::UnivariateDistributionModelBase*>(model.get());
            if (udm == nullptr || !udm->has_data_frame())
                throw py::value_error(
                    "model_estimation data-frame method on a model without a DataFrame");
            auto& df = udm->data_frame();
            df.calculate_plotting_positions();

            auto positions = [](const auto& series) {
                std::vector<double> out(series.count());
                for (std::size_t i = 0; i < series.count(); ++i) out[i] = series[i].plotting_position();
                return out;
            };
            py::dict out;
            out["number_of_low_outliers"] = df.number_of_low_outliers();
            out["low_outlier_threshold"] = df.low_outlier_threshold();
            out["pp_exact"] = positions(df.exact_series());
            out["pp_interval"] = positions(df.interval_series());
            out["pp_uncertain"] = positions(df.uncertain_series());
            return out;
        },
        py::arg("model_json"), py::arg("dataset"));

    // --- Simulation (M13) -------------------------------------------------------------
    //
    // The estimator-less `Simulation` target: builds the model through the shared spec
    // builder, calls the ISimulatable surface (`generate_random_values(sample_size, seed)`)
    // ONCE, and returns the seeded draw vector; test_fixtures.py's `simulated_value [i]`
    // dispatch indexes it. All four Phase 5 model types implement
    // ISimulatable<std::vector<double>>; the dynamic_cast guard mirrors the C++ test
    // runner's and bestfitr's.
    m.def(
        "model_simulate",
        [](const std::string& model_json, const std::vector<double>& dataset, int sample_size,
           int seed) {
            std::unique_ptr<models::ModelBase> model =
                models::spec::build_model_from_json(model_json, dataset);
            // simulate_flat handles both ISimulatable<vector<double>> and the bivariate
            // ISimulatable<Matrix2D> (flattened row-major) -- see its header note.
            return simulate_flat(model.get(), sample_size, seed);
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sample_size"), py::arg("seed"));

    // --- GeneralizedMethodOfMoments (B11) --------------------------------------------------
    //
    // Disjoint construct shape from ML/MAP (a strategy + max_gmm_iterations instead of just an
    // optimizer string) AND a different model type -- the concrete Bulletin17CDistribution the
    // GMM ctor takes as IGMMModel& (NOT a ModelBase; see model_spec.hpp's build_bulletin17c_model
    // note), built here through the shared spec builder's dedicated bulletin17c entry point. So
    // this is a separate registered function, mirroring the estimation_bayes_run split. Builds
    // the model, fits once, post_processes, and returns every value the GMM dispatcher reads:
    //   parameters/standard_errors     -- [p] lists
    //   covariance/correlation         -- [p][p] nested lists
    //   j_stat/j_stat_pval             -- scalars (pval is NaN when q - p == 0)
    //   simulated                      -- [sample_size] seeded ISimulatable draw off the FITTED
    //                                     model (present only when construct supplies sample_size),
    //                                     read by the shared `simulated_value` dispatch arm.
    // quantile_variance rides estimation_gmm_qvar (it needs a per-assertion AEP, exactly like
    // `bic`'s per-assertion sample size -- see below).
    m.def(
        "estimation_gmm_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& strategy, const std::string& optimizer, int max_gmm_iterations,
           int sample_size, int seed) {
            std::unique_ptr<models::Bulletin17CDistribution> model;
            auto gmm = build_and_fit_gmm(model, model_json, dataset, strategy, optimizer,
                                         max_gmm_iterations);
            int p = model->number_of_parameters();

            py::dict out;
            out["parameters"] = gmm->best_parameter_set().values;
            out["standard_errors"] = gmm->get_standard_errors();

            auto cov = gmm->get_covariance_matrix();
            auto corr = gmm->get_correlation_matrix();
            std::vector<std::vector<double>> covariance(static_cast<std::size_t>(p));
            std::vector<std::vector<double>> correlation(static_cast<std::size_t>(p));
            for (int i = 0; i < p; ++i) {
                covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
                correlation[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
                for (int j = 0; j < p; ++j) {
                    covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cov(i, j);
                    correlation[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = corr(i, j);
                }
            }
            out["covariance"] = covariance;
            out["correlation"] = correlation;
            out["j_stat"] = gmm->jstat();
            out["j_stat_pval"] = gmm->jstat_pval();

            if (sample_size > 0) {
                model->set_parameter_values(gmm->best_parameter_set().values);
                out["simulated"] = model->generate_random_values(sample_size, seed);
            }
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("strategy"), py::arg("optimizer"),
        py::arg("max_gmm_iterations"), py::arg("sample_size"), py::arg("seed"));

    // `quantile_variance [aep]`: like `bic [n]`, the AEP is only known at assertion-dispatch
    // time, so this rebuilds the same deterministic fit (see build_and_fit_gmm) and evaluates
    // the B17C delta-method Var(Q_p) live. args[0] is the annual EXCEEDANCE probability; the C#
    // QuantileVariance takes a NON-exceedance probability, so pass 1 - AEP.
    m.def(
        "estimation_gmm_qvar",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& strategy, const std::string& optimizer, int max_gmm_iterations,
           double aep) {
            std::unique_ptr<models::Bulletin17CDistribution> model;
            auto gmm = build_and_fit_gmm(model, model_json, dataset, strategy, optimizer,
                                         max_gmm_iterations);
            return model->quantile_variance(1.0 - aep, gmm->best_parameter_set().values,
                                            gmm->get_covariance_matrix().to_array());
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("strategy"), py::arg("optimizer"),
        py::arg("max_gmm_iterations"), py::arg("aep"));
}
