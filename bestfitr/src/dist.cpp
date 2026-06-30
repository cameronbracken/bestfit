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
