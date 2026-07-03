// pybind11 glue exposing the Phase-4 estimation surface (MaximumLikelihood/MaximumAPosteriori
// today; BayesianAnalysis deferred -- see below) of the shared C++ core to Python. Like
// `mcmc_sampler`/`bootstrap` fixtures, `model_estimation` fixtures are inherently STATEFUL --
// one model construct + a single estimate() run backs every assertion in a case (see
// fixtures/README.md's model_estimation schema) -- so this file exposes ONE function,
// `estimation_run`, that builds the model, runs estimate() once, and returns every value
// test_fixtures.py's dispatcher needs in one dict. Core headers are vendored under
// ../bestfit_core/include (see tools/sync_core.py).
//
// `bic` DESIGN NOTE: unlike every other wired method, C# `GetBIC(sampleSize)` takes an actual
// sample size, not a 0-based index -- but this glue (like `mcmc_run`/`bootstrap_run`)
// precomputes the full result surface up front, before any assertion is dispatched. `bic` is
// therefore precomputed ONCE here at `sample_size = len(dataset)` (the only value any
// `model_estimation` fixture case has ever needed -- BIC is always evaluated at the fitted
// data's own sample size); `test_fixtures.py`'s `bic` dispatch arm reads this precomputed
// scalar directly and does not re-derive `n` from the fixture's `args`.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

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

void register_estimation(py::module_& m) {
    // Builds the named model (`family` via the distribution factory + `dataset`), constructs
    // `target`'s estimator (`optimizer`, default "DifferentialEvolution"), runs estimate()
    // once, and returns a dict with the full surface test_fixtures.py's model_estimation
    // dispatcher reads assertions from:
    //   parameters          -- [n_params] list (BestParameterSet.Values)
    //   max_log_likelihood  -- scalar
    //   aic                 -- scalar
    //   bic                 -- scalar (sample_size = len(dataset); see file header DESIGN NOTE)
    //   covariance          -- [n_params][n_params] nested list
    //   standard_errors     -- [n_params] list
    //
    // WIRED (Task T11): parameter/max_log_likelihood/aic/bic/covariance/standard_error. LEFT
    // FOR TASK T12 (see fixtures/README.md's model_estimation section): correlation/dic/waic/
    // looic/posterior_mean -- the last four are BayesianAnalysis-only surface, and `target =
    // "BayesianAnalysis"` is not yet constructible here (raises below), matching the C++
    // runner.
    m.def(
        "estimation_run",
        [](const std::string& target, const std::string& family, const std::vector<double>& dataset,
           const std::string& optimizer) {
            models::UnivariateDistributionModel model(dist::create_distribution(family), dataset);
            auto method = parse_optimization_method(optimizer);
            int n_params = model.number_of_parameters();
            int sample_size = static_cast<int>(dataset.size());

            py::dict out;
            auto fill_from = [&](const auto& e) {
                out["parameters"] = e.best_parameter_set().values;
                out["max_log_likelihood"] = e.maximum_log_likelihood();
                out["aic"] = e.get_aic();
                out["bic"] = e.get_bic(sample_size);

                auto cov = e.get_covariance_matrix();
                std::vector<std::vector<double>> covariance(static_cast<std::size_t>(n_params));
                for (int i = 0; i < n_params; ++i) {
                    covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(n_params));
                    for (int j = 0; j < n_params; ++j)
                        covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cov(i, j);
                }
                out["covariance"] = covariance;
                out["standard_errors"] = e.get_standard_errors();
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
                    "model_estimation target 'BayesianAnalysis' is not yet wired in the fixture "
                    "runner (deferred to Task T12); see fixtures/README.md's model_estimation "
                    "section");
            } else {
                throw py::value_error("unknown model_estimation target: " + target);
            }

            return out;
        },
        py::arg("target"), py::arg("family"), py::arg("dataset"), py::arg("optimizer"));
}
