// pybind11 glue exposing the polymorphic univariate-distribution surface of the shared
// C++ core to Python (Normal, Uniform, Exponential, ... -- everything on
// UnivariateDistributionBase + the factory). GEV keeps its own bespoke class in gev.cpp.
// Core headers are vendored under ../bestfit_core/include (see tools/sync_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace dist = bestfit::numerics::distributions;

static std::unique_ptr<dist::UnivariateDistributionBase> make_dist(
    const std::string& target, const std::vector<double>& params) {
    auto d = dist::create_distribution(target);
    d->set_parameters(params);
    return d;
}

static dist::ParameterEstimationMethod parse_method(const std::string& m) {
    if (m == "mom" || m == "moments") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom" || m == "lmoments")
        return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    if (m == "mle") return dist::ParameterEstimationMethod::MaximumLikelihood;
    throw py::value_error("unknown estimation method '" + m + "' (use 'mom', 'lmom', or 'mle')");
}

void register_distributions(py::module_& m) {
    m.def("dist_moments", [](const std::string& target, const std::vector<double>& params) {
        auto d = make_dist(target, params);
        // std::map keeps a stable, language-neutral key set; the Python wrapper orders output.
        return std::map<std::string, double>{
            {"mean", d->mean()},        {"median", d->median()},
            {"mode", d->mode()},        {"sd", d->standard_deviation()},
            {"skewness", d->skewness()}, {"kurtosis", d->kurtosis()},
            {"minimum", d->minimum()},  {"maximum", d->maximum()}};
    });
    m.def("dist_pdf", [](const std::string& t, const std::vector<double>& p, double x) {
        return make_dist(t, p)->pdf(x);
    });
    m.def("dist_cdf", [](const std::string& t, const std::vector<double>& p, double x) {
        return make_dist(t, p)->cdf(x);
    });
    m.def("dist_quantile", [](const std::string& t, const std::vector<double>& p, double prob) {
        return make_dist(t, p)->inverse_cdf(prob);
    });
    m.def("dist_valid", [](const std::string& t, const std::vector<double>& p) {
        return make_dist(t, p)->parameters_valid();
    });
    m.def("dist_fit", [](const std::string& target, const std::vector<double>& data,
                         const std::string& method) {
        auto d = dist::create_distribution(target);
        auto* est = dynamic_cast<dist::IEstimation*>(d.get());
        if (est == nullptr)
            throw py::value_error("distribution '" + target + "' does not support estimation");
        est->estimate(data, parse_method(method));
        return d->get_parameters();
    });
    m.def("dist_linear_moments", [](const std::string& t, const std::vector<double>& p) {
        auto d = make_dist(t, p);
        auto* lm = dynamic_cast<dist::ILinearMomentEstimation*>(d.get());
        if (lm == nullptr) throw py::value_error("distribution '" + t + "' has no L-moments");
        return lm->linear_moments_from_parameters(d->get_parameters());
    });
}
