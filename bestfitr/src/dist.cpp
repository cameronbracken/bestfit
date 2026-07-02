// cpp11 glue exposing the polymorphic univariate-distribution surface of the shared C++
// core to R (Normal, Uniform, Exponential, ... -- everything built on
// UnivariateDistributionBase + the factory). GEV keeps its own bespoke glue in gev.cpp.
// Core headers are vendored under src/bestfit_core/include (see tools/sync_core.py).
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/competing_risks.hpp"
#include "bestfit/numerics/distributions/empirical_distribution.hpp"
#include "bestfit/numerics/distributions/kernel_density.hpp"
#include "bestfit/numerics/distributions/mixture.hpp"
#include "bestfit/numerics/distributions/truncated_distribution.hpp"

namespace dist = bestfit::numerics::distributions;
using namespace cpp11;

static std::unique_ptr<dist::UnivariateDistributionBase> make_dist(const std::string& target,
                                                                   doubles params) {
    auto d = dist::create_distribution(target);
    d->set_parameters(std::vector<double>(params.begin(), params.end()));
    return d;
}

static dist::ParameterEstimationMethod parse_method(const std::string& m) {
    if (m == "mom" || m == "moments") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom" || m == "lmoments")
        return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    if (m == "mle") return dist::ParameterEstimationMethod::MaximumLikelihood;
    stop("unknown estimation method '%s' (use 'mom', 'lmom', or 'mle')", m.c_str());
}

[[cpp11::register]]
doubles bf_dist_moments_(std::string target, doubles params) {
    auto d = make_dist(target, params);
    writable::doubles out({d->mean(), d->median(), d->mode(), d->standard_deviation(),
                           d->skewness(), d->kurtosis(), d->minimum(), d->maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double bf_dist_pdf_(std::string target, doubles params, double x) {
    return make_dist(target, params)->pdf(x);
}

[[cpp11::register]]
double bf_dist_cdf_(std::string target, doubles params, double x) {
    return make_dist(target, params)->cdf(x);
}

[[cpp11::register]]
double bf_dist_quantile_(std::string target, doubles params, double p) {
    return make_dist(target, params)->inverse_cdf(p);
}

[[cpp11::register]]
bool bf_dist_valid_(std::string target, doubles params) {
    return make_dist(target, params)->parameters_valid();
}

[[cpp11::register]]
doubles bf_dist_fit_(std::string target, doubles data, std::string method) {
    auto d = dist::create_distribution(target);
    auto* est = dynamic_cast<dist::IEstimation*>(d.get());
    if (est == nullptr) stop("distribution '%s' does not support estimation", target.c_str());
    est->estimate(std::vector<double>(data.begin(), data.end()), parse_method(method));
    return writable::doubles(d->get_parameters());
}

[[cpp11::register]]
doubles bf_dist_linear_moments_(std::string target, doubles params) {
    auto d = make_dist(target, params);
    auto* lm = dynamic_cast<dist::ILinearMomentEstimation*>(d.get());
    if (lm == nullptr) stop("distribution '%s' has no L-moments", target.c_str());
    return writable::doubles(lm->linear_moments_from_parameters(d->get_parameters()));
}

// --- Composite glue: TruncatedDistribution ------------------------------------------
// Accepts (base_target, base_params, lo, hi) and exposes the full distribution surface.
// Mirrors the bf_gev_* pattern: bespoke entry points that the R fixture runner uses.
// Future composites (Empirical, KernelDensity, Mixture, CompetingRisks) follow the same
// pattern with their own bf_<name>_* functions; the R runner dispatches by target string.

static dist::TruncatedDistribution make_trunc(const std::string& base_target,
                                               doubles base_params,
                                               double lo, double hi) {
    auto base = dist::create_distribution(base_target);
    base->set_parameters(std::vector<double>(base_params.begin(), base_params.end()));
    return dist::TruncatedDistribution(std::move(base), lo, hi);
}

[[cpp11::register]]
doubles bf_trunc_moments_(std::string base_target, doubles base_params,
                           double lo, double hi) {
    auto td = make_trunc(base_target, base_params, lo, hi);
    writable::doubles out({td.mean(), td.median(), td.mode(), td.standard_deviation(),
                           td.skewness(), td.kurtosis(), td.minimum(), td.maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double bf_trunc_pdf_(std::string base_target, doubles base_params, double lo, double hi, double x) {
    return make_trunc(base_target, base_params, lo, hi).pdf(x);
}

[[cpp11::register]]
double bf_trunc_cdf_(std::string base_target, doubles base_params, double lo, double hi, double x) {
    return make_trunc(base_target, base_params, lo, hi).cdf(x);
}

[[cpp11::register]]
double bf_trunc_quantile_(std::string base_target, doubles base_params,
                           double lo, double hi, double p) {
    return make_trunc(base_target, base_params, lo, hi).inverse_cdf(p);
}

[[cpp11::register]]
bool bf_trunc_valid_(std::string base_target, doubles base_params, double lo, double hi) {
    return make_trunc(base_target, base_params, lo, hi).parameters_valid();
}

// --- Composite glue: EmpiricalDistribution ------------------------------------------
// Accepts (x, p, p_transform) and exposes the full distribution surface.
// p_transform: "NormalZ" (default) or "None".

static dist::EmpiricalDistribution make_empirical(doubles x_vals, doubles p_vals,
                                                   std::string p_transform) {
    std::vector<double> xv(x_vals.begin(), x_vals.end());
    std::vector<double> pv(p_vals.begin(), p_vals.end());
    dist::EmpiricalTransform pt = dist::EmpiricalTransform::NormalZ;
    if (p_transform == "None") pt = dist::EmpiricalTransform::None;
    return dist::EmpiricalDistribution(std::move(xv), std::move(pv), pt);
}

[[cpp11::register]]
doubles bf_emp_moments_(doubles x_vals, doubles p_vals, std::string p_transform) {
    auto d = make_empirical(x_vals, p_vals, p_transform);
    writable::doubles out({d.mean(), d.median(), d.mode(), d.standard_deviation(),
                           d.skewness(), d.kurtosis(), d.minimum(), d.maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double bf_emp_pdf_(doubles x_vals, doubles p_vals, std::string p_transform, double x) {
    return make_empirical(x_vals, p_vals, p_transform).pdf(x);
}

[[cpp11::register]]
double bf_emp_cdf_(doubles x_vals, doubles p_vals, std::string p_transform, double x) {
    return make_empirical(x_vals, p_vals, p_transform).cdf(x);
}

[[cpp11::register]]
double bf_emp_quantile_(doubles x_vals, doubles p_vals, std::string p_transform, double prob) {
    return make_empirical(x_vals, p_vals, p_transform).inverse_cdf(prob);
}

[[cpp11::register]]
bool bf_emp_valid_(doubles x_vals, doubles p_vals, std::string p_transform) {
    return make_empirical(x_vals, p_vals, p_transform).parameters_valid();
}

// --- Composite glue: KernelDensity --------------------------------------------------
// Accepts (data, kernel, bandwidth=-1 means auto, bounded_by_data).
// kernel: "Gaussian" | "Epanechnikov" | "Triangular" | "Uniform"
// bandwidth: negative value means use Silverman's rule (auto).

static dist::KernelType parse_kernel_type(const std::string& s) {
    if (s == "Epanechnikov") return dist::KernelType::Epanechnikov;
    if (s == "Gaussian")     return dist::KernelType::Gaussian;
    if (s == "Triangular")   return dist::KernelType::Triangular;
    if (s == "Uniform")      return dist::KernelType::Uniform;
    stop("unknown kernel type '%s'", s.c_str());
}

static dist::KernelDensity make_kde(doubles data, std::string kernel,
                                    double bandwidth, bool bounded_by_data) {
    std::vector<double> dv(data.begin(), data.end());
    dist::KernelType kt = parse_kernel_type(kernel);
    dist::KernelDensity kde = bandwidth < 0.0
        ? dist::KernelDensity(dv, kt)
        : dist::KernelDensity(dv, kt, bandwidth);
    kde.set_bounded_by_data(bounded_by_data);
    return kde;
}

[[cpp11::register]]
doubles bf_kde_moments_(doubles data, std::string kernel, double bandwidth, bool bounded_by_data) {
    auto d = make_kde(data, kernel, bandwidth, bounded_by_data);
    writable::doubles out({d.mean(), d.median(), d.mode(), d.standard_deviation(),
                           d.skewness(), d.kurtosis(), d.minimum(), d.maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double bf_kde_pdf_(doubles data, std::string kernel, double bandwidth, bool bounded_by_data,
                   double x) {
    return make_kde(data, kernel, bandwidth, bounded_by_data).pdf(x);
}

[[cpp11::register]]
double bf_kde_cdf_(doubles data, std::string kernel, double bandwidth, bool bounded_by_data,
                   double x) {
    return make_kde(data, kernel, bandwidth, bounded_by_data).cdf(x);
}

[[cpp11::register]]
double bf_kde_quantile_(doubles data, std::string kernel, double bandwidth, bool bounded_by_data,
                        double prob) {
    return make_kde(data, kernel, bandwidth, bounded_by_data).inverse_cdf(prob);
}

[[cpp11::register]]
bool bf_kde_valid_(doubles data, std::string kernel, double bandwidth, bool bounded_by_data) {
    return make_kde(data, kernel, bandwidth, bounded_by_data).parameters_valid();
}

// --- Composite glue: Mixture --------------------------------------------------------
// Accepts (component_targets, component_params_list, weights) and exposes the full
// distribution surface. component_params_list is a list-of-doubles R list.
// Mirrors the bf_trunc_* and bf_kde_* pattern; R fixture runner dispatches by target.

static dist::Mixture make_mixture(strings comp_targets,
                                  list comp_params_list,
                                  doubles weights) {
    int K = static_cast<int>(comp_targets.size());
    std::vector<double> wts(weights.begin(), weights.end());
    std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
    comps.reserve(K);
    for (int i = 0; i < K; ++i) {
        auto d = dist::create_distribution(std::string(comp_targets[i]));
        doubles p = comp_params_list[i];
        d->set_parameters(std::vector<double>(p.begin(), p.end()));
        comps.push_back(std::move(d));
    }
    return dist::Mixture(std::move(wts), std::move(comps));
}

[[cpp11::register]]
doubles bf_mix_moments_(strings comp_targets, list comp_params_list, doubles weights) {
    auto d = make_mixture(comp_targets, comp_params_list, weights);
    writable::doubles out({d.mean(), d.median(), d.mode(), d.standard_deviation(),
                           d.skewness(), d.kurtosis(), d.minimum(), d.maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double bf_mix_pdf_(strings comp_targets, list comp_params_list, doubles weights, double x) {
    return make_mixture(comp_targets, comp_params_list, weights).pdf(x);
}

[[cpp11::register]]
double bf_mix_cdf_(strings comp_targets, list comp_params_list, doubles weights, double x) {
    return make_mixture(comp_targets, comp_params_list, weights).cdf(x);
}

[[cpp11::register]]
double bf_mix_quantile_(strings comp_targets, list comp_params_list, doubles weights, double prob) {
    return make_mixture(comp_targets, comp_params_list, weights).inverse_cdf(prob);
}

[[cpp11::register]]
bool bf_mix_valid_(strings comp_targets, list comp_params_list, doubles weights) {
    return make_mixture(comp_targets, comp_params_list, weights).parameters_valid();
}

// --- Composite glue: CompetingRisks -------------------------------------------------
// Accepts (component_targets, component_params_list, minimum_of_rv) and exposes the full
// distribution surface. component_params_list is a list-of-doubles R list.
// minimum_of_rv = TRUE for min-of-components (series system, default);
//                 FALSE for max-of-components (parallel system).

static dist::CompetingRisks make_competing_risks(strings comp_targets,
                                                  list comp_params_list,
                                                  bool minimum_of_rv) {
    int K = static_cast<int>(comp_targets.size());
    std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
    comps.reserve(K);
    for (int i = 0; i < K; ++i) {
        auto d = dist::create_distribution(std::string(comp_targets[i]));
        doubles p = comp_params_list[i];
        d->set_parameters(std::vector<double>(p.begin(), p.end()));
        comps.push_back(std::move(d));
    }
    dist::CompetingRisks cr(std::move(comps));
    cr.minimum_of_random_variables = minimum_of_rv;
    return cr;
}

[[cpp11::register]]
doubles bf_cr_moments_(strings comp_targets, list comp_params_list, bool minimum_of_rv) {
    auto d = make_competing_risks(comp_targets, comp_params_list, minimum_of_rv);
    writable::doubles out({d.mean(), d.median(), d.mode(), d.standard_deviation(),
                           d.skewness(), d.kurtosis(), d.minimum(), d.maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
double bf_cr_pdf_(strings comp_targets, list comp_params_list, bool minimum_of_rv, double x) {
    return make_competing_risks(comp_targets, comp_params_list, minimum_of_rv).pdf(x);
}

[[cpp11::register]]
double bf_cr_cdf_(strings comp_targets, list comp_params_list, bool minimum_of_rv, double x) {
    return make_competing_risks(comp_targets, comp_params_list, minimum_of_rv).cdf(x);
}

[[cpp11::register]]
double bf_cr_quantile_(strings comp_targets, list comp_params_list, bool minimum_of_rv, double prob) {
    return make_competing_risks(comp_targets, comp_params_list, minimum_of_rv).inverse_cdf(prob);
}

[[cpp11::register]]
bool bf_cr_valid_(strings comp_targets, list comp_params_list, bool minimum_of_rv) {
    return make_competing_risks(comp_targets, comp_params_list, minimum_of_rv).parameters_valid();
}
