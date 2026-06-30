// cpp11 glue exposing the GEV slice of the shared C++ core to R.
// The core headers are vendored under src/bestfit_core/include (see tools/sync_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"

namespace dist = bestfit::numerics::distributions;
using namespace cpp11;

static dist::EstimationMethod parse_method(const std::string& m) {
    if (m == "mom" || m == "moments") return dist::EstimationMethod::MethodOfMoments;
    if (m == "lmom" || m == "lmoments") return dist::EstimationMethod::MethodOfLinearMoments;
    if (m == "mle") return dist::EstimationMethod::MaximumLikelihood;
    stop("unknown estimation method '%s' (use 'mom', 'lmom', or 'mle')", m.c_str());
}

[[cpp11::register]]
doubles bf_gev_pdf_(doubles x, double location, double scale, double shape) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    writable::doubles out(x.size());
    for (R_xlen_t i = 0; i < x.size(); ++i) out[i] = g.pdf(x[i]);
    return out;
}

[[cpp11::register]]
doubles bf_gev_cdf_(doubles x, double location, double scale, double shape) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    writable::doubles out(x.size());
    for (R_xlen_t i = 0; i < x.size(); ++i) out[i] = g.cdf(x[i]);
    return out;
}

[[cpp11::register]]
doubles bf_gev_quantile_(doubles p, double location, double scale, double shape) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    writable::doubles out(p.size());
    for (R_xlen_t i = 0; i < p.size(); ++i) out[i] = g.inverse_cdf(p[i]);
    return out;
}

[[cpp11::register]]
doubles bf_gev_moments_(double location, double scale, double shape) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    writable::doubles out({g.mean(), g.median(), g.mode(), g.standard_deviation(), g.skewness(),
                           g.kurtosis(), g.minimum(), g.maximum()});
    out.names() = {"mean", "median", "mode", "sd", "skewness", "kurtosis", "minimum", "maximum"};
    return out;
}

[[cpp11::register]]
bool bf_gev_valid_(double location, double scale, double shape) {
    return dist::GeneralizedExtremeValue(location, scale, shape).parameters_valid();
}

[[cpp11::register]]
doubles bf_gev_fit_(doubles data, std::string method) {
    std::vector<double> sample(data.begin(), data.end());
    dist::GeneralizedExtremeValue g;
    g.estimate(sample, parse_method(method));
    writable::doubles out({g.xi(), g.alpha(), g.kappa()});
    out.names() = {"location", "scale", "shape"};
    return out;
}

[[cpp11::register]]
double bf_gev_quantile_variance_(double p, double location, double scale, double shape,
                                 int sample_size) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    return g.quantile_variance(p, sample_size);
}

[[cpp11::register]]
doubles bf_gev_linear_moments_(double location, double scale, double shape) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    auto lmom = g.linear_moments_from_parameters({location, scale, shape});
    return writable::doubles(lmom);
}

[[cpp11::register]]
doubles bf_gev_quantile_gradient_(double p, double location, double scale, double shape) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    return writable::doubles(g.quantile_gradient(p));
}

// Returns the flattened 3x3 covariance (row-major).
[[cpp11::register]]
doubles bf_gev_parameter_covariance_(double location, double scale, double shape,
                                     int sample_size) {
    dist::GeneralizedExtremeValue g(location, scale, shape);
    auto cov = g.parameter_covariance(sample_size);
    writable::doubles out(9);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) out[i * 3 + j] = cov[i][j];
    return out;
}
