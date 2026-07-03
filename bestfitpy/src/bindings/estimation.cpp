// pybind11 glue exposing the Phase-4 estimation surface (MaximumLikelihood, MaximumAPosteriori,
// and -- as of Task T12 -- BayesianAnalysis) of the shared C++ core to Python. Like
// `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are inherently STATEFUL --
// one model construct + a single estimate() run backs every assertion in a case (see
// fixtures/README.md's model_estimation schema) -- so this file exposes TWO functions:
// `estimation_run` for MaximumLikelihood/MaximumAPosteriori (shared {target, family, dataset,
// optimizer} signature) and `estimation_bayes_run` (T12) for BayesianAnalysis (a disjoint
// {family, dataset, sampler, settings...} signature -- a sampler type + numeric knobs, not an
// optimizer string), mirroring bestfitr's `bf_estimation_run_`/`bf_estimation_bayes_run_` split.
// Each builds the model, runs estimate() once, and returns every value test_fixtures.py's
// dispatcher needs in one dict. Core headers are vendored under ../bestfit_core/include (see
// tools/sync_core.py).
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
#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace est = bestfit::estimation;
namespace models = bestfit::models;
namespace dist = bestfit::numerics::distributions;

static est::OptimizationMethod parse_optimization_method(const std::string& s) {
    if (s == "Brent") return est::OptimizationMethod::Brent;
    if (s == "NelderMead") return est::OptimizationMethod::NelderMead;
    if (s == "DifferentialEvolution") return est::OptimizationMethod::DifferentialEvolution;
    throw py::value_error("unknown model_estimation optimizer: " + s);
}

static est::SamplerType parse_sampler_type(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    throw py::value_error("unknown model_estimation sampler: " + s);
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
        [](const std::string& target, const std::string& family, const std::vector<double>& dataset,
           const std::string& optimizer) {
            models::UnivariateDistributionModel model(dist::create_distribution(family), dataset);
            auto method = parse_optimization_method(optimizer);
            int n_params = model.number_of_parameters();

            py::dict out;
            auto fill_from = [&](const auto& e) {
                out["parameters"] = e.best_parameter_set().values;
                out["max_log_likelihood"] = e.maximum_log_likelihood();
                out["aic"] = e.get_aic();

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

            return out;
        },
        py::arg("target"), py::arg("family"), py::arg("dataset"), py::arg("optimizer"));

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
        [](const std::string& target, const std::string& family, const std::vector<double>& dataset,
           const std::string& optimizer, int n) {
            models::UnivariateDistributionModel model(dist::create_distribution(family), dataset);
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
        py::arg("target"), py::arg("family"), py::arg("dataset"), py::arg("optimizer"), py::arg("n"));

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
        [](const std::string& family, const std::vector<double>& dataset, const std::string& sampler,
           const py::dict& settings) {
            models::UnivariateDistributionModel model(dist::create_distribution(family), dataset);
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
        py::arg("family"), py::arg("dataset"), py::arg("sampler"), py::arg("settings"));
}
