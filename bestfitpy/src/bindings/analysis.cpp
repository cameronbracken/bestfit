// pybind11 glue exposing the user-facing Analyses layer (A10) of the shared C++ core to Python:
// UnivariateAnalysis (Bayesian frequency curve), FittingAnalysis (multi-distribution GoF ranking),
// and Bulletin17CAnalysis (B17C flood-frequency + Cohn-style CIs). These are a bestfit-additions
// binding surface (no direct C# counterpart -- they wrap the ported analysis classes); each
// analysis is inherently STATEFUL, so this file exposes ONE run function per analysis that builds
// the model via the shared spec builder (bestfit/models/model_spec.hpp), runs the analysis once,
// and returns the full result surface as a dict. The signatures / spec assembly / seed plumbing
// mirror bestfitr's analysis.cpp exactly, so a seeded call returns identical numbers in either
// language. Core headers are vendored under ../bestfit_core/include (see tools/sync_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/analyses/distribution_fitting/fitting_analysis.hpp"
#include "bestfit/analyses/univariate/bulletin17c_analysis.hpp"
#include "bestfit/analyses/univariate/univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/data_frame/data_collections/exact_series.hpp"
#include "bestfit/models/model_spec.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace analyses = bestfit::analyses;
namespace models = bestfit::models;
namespace est = bestfit::estimation;
using UDT = bestfit::numerics::distributions::UnivariateDistributionType;

static est::SamplerType parse_analysis_sampler(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    throw py::value_error("unknown analysis sampler '" + s +
                          "' (use DEMCz, DEMCzs, ARWMH, or NUTS)");
}

static analyses::UncertaintyMethod parse_uncertainty_method(const std::string& s) {
    if (s == "MultivariateNormal") return analyses::UncertaintyMethod::MultivariateNormal;
    if (s == "Bootstrap") return analyses::UncertaintyMethod::Bootstrap;
    if (s == "LinkedMultivariateNormal" || s == "BiasCorrectedBootstrap")
        throw py::value_error("uncertainty method '" + s +
                              "' is deferred to Phase 9; use 'MultivariateNormal' or 'Bootstrap'");
    throw py::value_error("unknown uncertainty method '" + s + "'");
}

// The C# type name for a fitted candidate (mirrors the factory's forward name->type table for the
// 14 FittingAnalysis candidates). No reverse map exists in the core; this small local helper keeps
// the binding self-contained and matches bestfitr's analysis.cpp type_name.
static std::string type_name(UDT type) {
    switch (type) {
        case UDT::Exponential: return "Exponential";
        case UDT::GammaDistribution: return "GammaDistribution";
        case UDT::GeneralizedExtremeValue: return "GeneralizedExtremeValue";
        case UDT::GeneralizedLogistic: return "GeneralizedLogistic";
        case UDT::GeneralizedPareto: return "GeneralizedPareto";
        case UDT::Gumbel: return "Gumbel";
        case UDT::KappaFour: return "KappaFour";
        case UDT::LnNormal: return "LnNormal";
        case UDT::Logistic: return "Logistic";
        case UDT::LogNormal: return "LogNormal";
        case UDT::LogPearsonTypeIII: return "LogPearsonTypeIII";
        case UDT::Normal: return "Normal";
        case UDT::PearsonTypeIII: return "PearsonTypeIII";
        case UDT::Weibull: return "Weibull";
        default: return "Unknown";
    }
}

static void set_ordinates(bestfit::numerics::data::ProbabilityOrdinates& po,
                          const std::vector<double>& ep) {
    if (ep.empty()) return;
    po.clear();
    for (double p : ep) po.push_back(p);
}

void register_analysis(py::module_& m) {
    // UnivariateAnalysis (A5): Bayesian frequency curve. Returns {parameters, mode_curve,
    // mean_curve, lower_ci, upper_ci, aic, bic, dic, rmse}. The curve/CI lists are empty and the
    // scalars NaN when the fit yields no results.
    m.def(
        "analysis_univariate_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& sampler, int iterations, int output_length, double credible_level,
           int seed, const std::vector<double>& exceedance_probabilities) {
            std::unique_ptr<models::ModelBase> base =
                models::spec::build_model_from_json(model_json, dataset);
            auto* raw = dynamic_cast<models::UnivariateDistributionModel*>(base.get());
            if (raw == nullptr)
                throw py::value_error("univariate_analysis requires a univariate_distribution model spec");
            base.release();
            std::unique_ptr<models::UnivariateDistributionModel> model(raw);

            analyses::UnivariateAnalysis analysis(std::move(model));
            set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

            est::BayesianAnalysis& ba = analysis.bayesian_analysis();
            ba.set_type(parse_analysis_sampler(sampler));
            if (credible_level > 0.0 && credible_level < 1.0)
                ba.set_credible_interval_width(credible_level);
            if (seed >= 0) ba.set_prng_seed(seed);
            if (output_length > 0) ba.set_output_length(output_length);
            if (iterations > 0) {
                ba.set_iterations(iterations);
                ba.set_warmup_iterations(std::max(50, iterations / 2));
            }

            analysis.run();

            const double kNaN = std::numeric_limits<double>::quiet_NaN();
            py::dict out;
            std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
            double aic = kNaN, bic = kNaN, dic = kNaN, rmse = kNaN;

            const auto* results = analysis.analysis_results();
            if (results != nullptr) {
                auto* pe = analysis.get_point_estimate_distribution();
                if (pe != nullptr) parameters = pe->get_parameters();
                mode_curve = results->mode_curve;
                mean_curve = results->mean_curve;
                for (const auto& ci : results->confidence_intervals) {
                    lower_ci.push_back(ci[0]);
                    upper_ci.push_back(ci[1]);
                }
                aic = results->aic;
                bic = results->bic;
                dic = results->dic;
                rmse = results->rmse;
            }

            out["parameters"] = parameters;
            out["mode_curve"] = mode_curve;
            out["mean_curve"] = mean_curve;
            out["lower_ci"] = lower_ci;
            out["upper_ci"] = upper_ci;
            out["aic"] = aic;
            out["bic"] = bic;
            out["dic"] = dic;
            out["rmse"] = rmse;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("sampler"), py::arg("iterations"),
        py::arg("output_length"), py::arg("credible_level"), py::arg("seed"),
        py::arg("exceedance_probabilities"));

    // FittingAnalysis (A6): fit the 14 ported candidates by MLE, return the per-candidate GoF table.
    m.def(
        "analysis_fit_distributions",
        [](const std::vector<double>& dataset) {
            auto df = std::make_unique<models::DataFrame>();
            df->set_exact_series(models::ExactSeries(dataset));
            df->calculate_plotting_positions();

            analyses::FittingAnalysis analysis(std::move(df));
            analysis.run();

            const auto& fitted = analysis.fitted_distributions();
            std::vector<std::string> distribution;
            std::vector<double> aic, bic, rmse;
            std::vector<bool> converged;
            for (const auto& fd : fitted) {
                distribution.push_back(fd.distribution() != nullptr
                                           ? type_name(fd.distribution()->type())
                                           : std::string("Unknown"));
                aic.push_back(fd.aic());
                bic.push_back(fd.bic());
                rmse.push_back(fd.rmse());
                converged.push_back(fd.fit_succeeded());
            }

            py::dict out;
            out["distribution"] = distribution;
            out["aic"] = aic;
            out["bic"] = bic;
            out["rmse"] = rmse;
            out["converged"] = converged;
            return out;
        },
        py::arg("dataset"));

    // Bulletin17CAnalysis (A7-A9): GMM fit + Cohn-style delta-method CIs. Returns the CI surface
    // plus the fitted parameters and sandwich covariance (a nested p x p list; empty when
    // unestimated).
    m.def(
        "analysis_b17c_run",
        [](const std::string& model_json, const std::vector<double>& dataset,
           const std::string& uncertainty_method, int output_length, int seed,
           double confidence_level, const std::vector<double>& exceedance_probabilities) {
            std::unique_ptr<models::Bulletin17CDistribution> model =
                models::spec::build_bulletin17c_from_json(model_json, dataset);

            analyses::Bulletin17CAnalysis analysis(std::move(model));
            analysis.set_uncertainty_method(parse_uncertainty_method(uncertainty_method));
            set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

            est::BayesianAnalysis& ba = analysis.bayesian_analysis();
            if (confidence_level > 0.0 && confidence_level < 1.0)
                ba.set_credible_interval_width(confidence_level);
            if (seed >= 0) ba.set_prng_seed(seed);
            if (output_length > 0) ba.set_output_length(output_length);

            analysis.run();

            const double kNaN = std::numeric_limits<double>::quiet_NaN();
            py::dict out;
            std::vector<double> exceedance, point_estimates, lower_ci, upper_ci, beta1, nu,
                quantile_variance;
            double conf = kNaN;
            auto ci = analysis.compute_cohn_style_confidence_intervals();
            if (ci.has_value()) {
                exceedance = ci->exceedance_probabilities;
                point_estimates = ci->point_estimates;
                lower_ci = ci->lower_ci;
                upper_ci = ci->upper_ci;
                beta1 = ci->beta1;
                nu = ci->nu;
                quantile_variance = ci->quantile_variance;
                conf = ci->confidence_level;
            }

            std::vector<double> parameters;
            std::vector<std::vector<double>> covariance;
            if (analysis.gmm() != nullptr && analysis.gmm()->is_estimated()) {
                parameters = analysis.gmm()->best_parameter_set().values;
                int p = static_cast<int>(parameters.size());
                auto cov = analysis.gmm()->get_covariance_matrix();
                covariance.resize(static_cast<std::size_t>(p));
                for (int i = 0; i < p; ++i) {
                    covariance[static_cast<std::size_t>(i)].resize(static_cast<std::size_t>(p));
                    for (int j = 0; j < p; ++j)
                        covariance[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = cov(i, j);
                }
            }

            out["exceedance_probabilities"] = exceedance;
            out["point_estimates"] = point_estimates;
            out["lower_ci"] = lower_ci;
            out["upper_ci"] = upper_ci;
            out["confidence_level"] = conf;
            out["beta1"] = beta1;
            out["nu"] = nu;
            out["quantile_variance"] = quantile_variance;
            out["parameters"] = parameters;
            out["covariance"] = covariance;
            return out;
        },
        py::arg("model_json"), py::arg("dataset"), py::arg("uncertainty_method"),
        py::arg("output_length"), py::arg("seed"), py::arg("confidence_level"),
        py::arg("exceedance_probabilities"));
}
