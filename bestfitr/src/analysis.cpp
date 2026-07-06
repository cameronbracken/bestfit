// cpp11 glue exposing the user-facing Analyses layer (A10) of the shared C++ core to R:
// UnivariateAnalysis (Bayesian frequency curve), FittingAnalysis (multi-distribution GoF
// ranking), and Bulletin17CAnalysis (B17C flood-frequency + Cohn-style CIs). These are a
// bestfit-additions binding surface (no direct C# counterpart -- they wrap the ported analysis
// classes); each analysis is inherently STATEFUL, so this file exposes ONE run function per
// analysis that builds the model via the shared spec builder (bestfit/models/model_spec.hpp),
// runs the analysis once, and packs the full result surface into a named list. The exported R
// wrappers live in R/analysis.R; the model-neutral fixture harness (test-fixtures.R) also drives
// these same functions for the `analysis` fixture kind. Core headers are vendored under
// src/bestfit_core/include (see tools/sync_core.py).
#include <cpp11.hpp>

#include <algorithm>
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

namespace analyses = bestfit::analyses;
namespace models = bestfit::models;
namespace est = bestfit::estimation;
using UDT = bestfit::numerics::distributions::UnivariateDistributionType;
using namespace cpp11;

// The BayesianAnalysis sampler knob (shared with estimation.cpp's parse_sampler_type).
static est::SamplerType parse_analysis_sampler(const std::string& s) {
    if (s == "DEMCz") return est::SamplerType::DEMCz;
    if (s == "DEMCzs") return est::SamplerType::DEMCzs;
    if (s == "ARWMH") return est::SamplerType::ARWMH;
    if (s == "NUTS") return est::SamplerType::NUTS;
    stop("unknown analysis sampler '%s' (use DEMCz, DEMCzs, ARWMH, or NUTS)", s.c_str());
}

// The B17C uncertainty method knob. The two deferred (Phase 9) methods are rejected with a clear
// message rather than dispatched (the C++ analysis would throw from run() otherwise).
static analyses::UncertaintyMethod parse_uncertainty_method(const std::string& s) {
    if (s == "MultivariateNormal") return analyses::UncertaintyMethod::MultivariateNormal;
    if (s == "Bootstrap") return analyses::UncertaintyMethod::Bootstrap;
    if (s == "LinkedMultivariateNormal" || s == "BiasCorrectedBootstrap")
        stop("uncertainty method '%s' is deferred to Phase 9; use 'MultivariateNormal' or 'Bootstrap'",
             s.c_str());
    stop("unknown uncertainty method '%s'", s.c_str());
}

// The C# type name (matching the distribution factory names) for a fitted candidate. Covers the
// 14 FittingAnalysis candidates; a fallback keeps the binding total (never throws on an unexpected
// type). No reverse map exists in the core, so this small local helper mirrors the factory's
// forward name->type table for exactly the candidates FittingAnalysis produces.
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

// Applies a caller-supplied exceedance grid to an analysis's ProbabilityOrdinates when non-empty,
// leaving the 25 standard defaults in place otherwise (mirrors the C# reset-then-add pattern).
static void set_ordinates(bestfit::numerics::data::ProbabilityOrdinates& po, const doubles& ep) {
    if (ep.size() == 0) return;
    po.clear();
    for (R_xlen_t i = 0; i < ep.size(); ++i) po.push_back(ep[i]);
}

// UnivariateAnalysis (A5): build the UnivariateDistributionModel from the spec, set the Bayesian
// knobs, run(), and return {parameters, mode_curve, mean_curve, lower_ci, upper_ci, aic, bic, dic,
// rmse}. `parameters` are the point-estimate distribution parameters (PosteriorMean by default).
// When the fit yields no results (unestimated), the curve/CI vectors come back empty and the
// scalars NaN.
[[cpp11::register]]
list bf_analysis_univariate_run_(std::string model_json, doubles dataset, std::string sampler,
                                 int iterations, int output_length, double credible_level, int seed,
                                 doubles exceedance_probabilities) {
    std::vector<double> data(dataset.begin(), dataset.end());
    std::unique_ptr<models::ModelBase> base = models::spec::build_model_from_json(model_json, data);
    auto* raw = dynamic_cast<models::UnivariateDistributionModel*>(base.get());
    if (raw == nullptr)
        stop("univariate_analysis requires a univariate_distribution model spec");
    base.release();
    std::unique_ptr<models::UnivariateDistributionModel> model(raw);

    analyses::UnivariateAnalysis analysis(std::move(model));
    set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

    est::BayesianAnalysis& ba = analysis.bayesian_analysis();
    ba.set_type(parse_analysis_sampler(sampler));
    if (credible_level > 0.0 && credible_level < 1.0) ba.set_credible_interval_width(credible_level);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (output_length > 0) ba.set_output_length(output_length);
    if (iterations > 0) {
        ba.set_iterations(iterations);
        ba.set_warmup_iterations(std::max(50, iterations / 2));
    }

    analysis.run();

    writable::doubles parameters;
    writable::doubles mode_curve, mean_curve, lower_ci, upper_ci;
    double aic = NA_REAL, bic = NA_REAL, dic = NA_REAL, rmse = NA_REAL;

    const auto* results = analysis.analysis_results();
    if (results != nullptr) {
        auto* pe = analysis.get_point_estimate_distribution();
        if (pe != nullptr) {
            std::vector<double> p = pe->get_parameters();
            parameters = writable::doubles(p.begin(), p.end());
        }
        mode_curve = writable::doubles(results->mode_curve.begin(), results->mode_curve.end());
        mean_curve = writable::doubles(results->mean_curve.begin(), results->mean_curve.end());
        std::size_t n = results->confidence_intervals.size();
        lower_ci = writable::doubles(static_cast<R_xlen_t>(n));
        upper_ci = writable::doubles(static_cast<R_xlen_t>(n));
        for (std::size_t i = 0; i < n; ++i) {
            lower_ci[static_cast<R_xlen_t>(i)] = results->confidence_intervals[i][0];
            upper_ci[static_cast<R_xlen_t>(i)] = results->confidence_intervals[i][1];
        }
        aic = results->aic;
        bic = results->bic;
        dic = results->dic;
        rmse = results->rmse;
    }

    return writable::list({
        "parameters"_nm = parameters,
        "mode_curve"_nm = mode_curve,
        "mean_curve"_nm = mean_curve,
        "lower_ci"_nm = lower_ci,
        "upper_ci"_nm = upper_ci,
        "aic"_nm = writable::doubles({aic}),
        "bic"_nm = writable::doubles({bic}),
        "dic"_nm = writable::doubles({dic}),
        "rmse"_nm = writable::doubles({rmse}),
    });
}

// FittingAnalysis (A6): fit the 14 ported candidate distributions by MLE over an exact-only frame
// built from `dataset`, and return the per-candidate GoF table {distribution, aic, bic, rmse,
// converged}. Ranking is left to the R caller.
[[cpp11::register]]
list bf_analysis_fit_distributions_(doubles dataset) {
    std::vector<double> data(dataset.begin(), dataset.end());
    auto df = std::make_unique<models::DataFrame>();
    df->set_exact_series(models::ExactSeries(data));
    df->calculate_plotting_positions();

    analyses::FittingAnalysis analysis(std::move(df));
    analysis.run();

    const auto& fitted = analysis.fitted_distributions();
    R_xlen_t n = static_cast<R_xlen_t>(fitted.size());
    writable::strings distribution(n);
    writable::doubles aic(n), bic(n), rmse(n);
    writable::logicals converged(n);
    for (R_xlen_t i = 0; i < n; ++i) {
        const auto& fd = fitted[static_cast<std::size_t>(i)];
        distribution[i] = fd.distribution() != nullptr ? type_name(fd.distribution()->type())
                                                       : std::string("Unknown");
        aic[i] = fd.aic();
        bic[i] = fd.bic();
        rmse[i] = fd.rmse();
        converged[i] = cpp11::r_bool(fd.fit_succeeded());
    }

    return writable::list({
        "distribution"_nm = distribution,
        "aic"_nm = aic,
        "bic"_nm = bic,
        "rmse"_nm = rmse,
        "converged"_nm = converged,
    });
}

// Bulletin17CAnalysis (A7-A9): build the B17C distribution from the spec, fit by GMM under the
// requested uncertainty method, and return the Cohn-style delta-method CI surface plus the fitted
// parameters + sandwich covariance. When the Cohn optional is empty (unestimated) the CI vectors
// come back empty.
[[cpp11::register]]
list bf_analysis_b17c_run_(std::string model_json, doubles dataset, std::string uncertainty_method,
                           int output_length, int seed, double confidence_level,
                           doubles exceedance_probabilities) {
    std::vector<double> data(dataset.begin(), dataset.end());
    std::unique_ptr<models::Bulletin17CDistribution> model =
        models::spec::build_bulletin17c_from_json(model_json, data);

    analyses::Bulletin17CAnalysis analysis(std::move(model));
    analysis.set_uncertainty_method(parse_uncertainty_method(uncertainty_method));
    set_ordinates(analysis.probability_ordinates(), exceedance_probabilities);

    est::BayesianAnalysis& ba = analysis.bayesian_analysis();
    if (confidence_level > 0.0 && confidence_level < 1.0)
        ba.set_credible_interval_width(confidence_level);
    if (seed >= 0) ba.set_prng_seed(seed);
    if (output_length > 0) ba.set_output_length(output_length);

    analysis.run();

    writable::doubles exceedance, point_estimates, lower_ci, upper_ci, beta1, nu, quantile_variance;
    double conf = NA_REAL;
    auto ci = analysis.compute_cohn_style_confidence_intervals();
    if (ci.has_value()) {
        exceedance = writable::doubles(ci->exceedance_probabilities.begin(),
                                       ci->exceedance_probabilities.end());
        point_estimates = writable::doubles(ci->point_estimates.begin(), ci->point_estimates.end());
        lower_ci = writable::doubles(ci->lower_ci.begin(), ci->lower_ci.end());
        upper_ci = writable::doubles(ci->upper_ci.begin(), ci->upper_ci.end());
        beta1 = writable::doubles(ci->beta1.begin(), ci->beta1.end());
        nu = writable::doubles(ci->nu.begin(), ci->nu.end());
        quantile_variance =
            writable::doubles(ci->quantile_variance.begin(), ci->quantile_variance.end());
        conf = ci->confidence_level;
    }

    // Fitted parameters + sandwich covariance from the GMM (populated after a successful run()).
    // The covariance is returned as a flat row-major vector plus its dimension so the R wrapper
    // can reshape it to a p x p matrix (avoids a 0x0 matrix in the unestimated case).
    writable::doubles parameters;
    writable::doubles covariance;
    int cov_dim = 0;
    if (analysis.gmm() != nullptr && analysis.gmm()->is_estimated()) {
        const std::vector<double>& best = analysis.gmm()->best_parameter_set().values;
        parameters = writable::doubles(best.begin(), best.end());
        int p = static_cast<int>(best.size());
        cov_dim = p;
        auto cov = analysis.gmm()->get_covariance_matrix();
        covariance = writable::doubles(static_cast<R_xlen_t>(p) * p);
        for (int i = 0; i < p; ++i)
            for (int j = 0; j < p; ++j) covariance[i * p + j] = cov(i, j);
    }

    return writable::list({
        "exceedance_probabilities"_nm = exceedance,
        "point_estimates"_nm = point_estimates,
        "lower_ci"_nm = lower_ci,
        "upper_ci"_nm = upper_ci,
        "confidence_level"_nm = writable::doubles({conf}),
        "beta1"_nm = beta1,
        "nu"_nm = nu,
        "quantile_variance"_nm = quantile_variance,
        "parameters"_nm = parameters,
        "covariance"_nm = covariance,
        "covariance_dim"_nm = writable::integers({cov_dim}),
    });
}
