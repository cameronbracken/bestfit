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

#include "bestfit/numerics/data/probability.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/competing_risks.hpp"
#include "bestfit/numerics/distributions/empirical_distribution.hpp"
#include "bestfit/numerics/distributions/kernel_density.hpp"
#include "bestfit/numerics/distributions/mixture.hpp"
#include "bestfit/numerics/distributions/truncated_distribution.hpp"
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

    // --- Composite glue: TruncatedDistribution -----------------------------------------
    // Accepts (base_target, base_params, lo, hi) and exposes the full distribution surface.
    // Future composites (Empirical, KernelDensity, Mixture, CompetingRisks) follow the same
    // pattern with their own trunc_<name>_* / empirical_* / etc. module-level functions.

    m.def("trunc_moments", [](const std::string& bt, const std::vector<double>& bp,
                               double lo, double hi) {
        auto base = dist::create_distribution(bt);
        base->set_parameters(bp);
        dist::TruncatedDistribution td(std::move(base), lo, hi);
        return std::map<std::string, double>{
            {"mean", td.mean()},         {"median", td.median()},
            {"mode", td.mode()},         {"sd", td.standard_deviation()},
            {"skewness", td.skewness()}, {"kurtosis", td.kurtosis()},
            {"minimum", td.minimum()},   {"maximum", td.maximum()}};
    });
    m.def("trunc_pdf", [](const std::string& bt, const std::vector<double>& bp,
                           double lo, double hi, double x) {
        auto base = dist::create_distribution(bt);
        base->set_parameters(bp);
        return dist::TruncatedDistribution(std::move(base), lo, hi).pdf(x);
    });
    m.def("trunc_cdf", [](const std::string& bt, const std::vector<double>& bp,
                           double lo, double hi, double x) {
        auto base = dist::create_distribution(bt);
        base->set_parameters(bp);
        return dist::TruncatedDistribution(std::move(base), lo, hi).cdf(x);
    });
    m.def("trunc_quantile", [](const std::string& bt, const std::vector<double>& bp,
                                double lo, double hi, double prob) {
        auto base = dist::create_distribution(bt);
        base->set_parameters(bp);
        return dist::TruncatedDistribution(std::move(base), lo, hi).inverse_cdf(prob);
    });
    m.def("trunc_valid", [](const std::string& bt, const std::vector<double>& bp,
                             double lo, double hi) {
        auto base = dist::create_distribution(bt);
        base->set_parameters(bp);
        return dist::TruncatedDistribution(std::move(base), lo, hi).parameters_valid();
    });

    // --- Composite glue: EmpiricalDistribution -----------------------------------------
    // Accepts (x_vals, p_vals, p_transform) and exposes the full distribution surface.
    // p_transform: "NormalZ" (default) or "None".

    auto parse_emp_transform = [](const std::string& s) {
        if (s == "None") return dist::EmpiricalTransform::None;
        if (s == "NormalZ") return dist::EmpiricalTransform::NormalZ;
        throw py::value_error("unknown p_transform: " + s);
        return dist::EmpiricalTransform::NormalZ;  // unreachable
    };

    m.def("emp_moments", [parse_emp_transform](const std::vector<double>& xv,
                                                const std::vector<double>& pv,
                                                const std::string& pt_str) {
        dist::EmpiricalDistribution d(xv, pv, parse_emp_transform(pt_str));
        return std::map<std::string, double>{
            {"mean", d.mean()},         {"median", d.median()},
            {"mode", d.mode()},         {"sd", d.standard_deviation()},
            {"skewness", d.skewness()}, {"kurtosis", d.kurtosis()},
            {"minimum", d.minimum()},   {"maximum", d.maximum()}};
    });
    m.def("emp_pdf", [parse_emp_transform](const std::vector<double>& xv,
                                            const std::vector<double>& pv,
                                            const std::string& pt_str, double x) {
        return dist::EmpiricalDistribution(xv, pv, parse_emp_transform(pt_str)).pdf(x);
    });
    m.def("emp_cdf", [parse_emp_transform](const std::vector<double>& xv,
                                            const std::vector<double>& pv,
                                            const std::string& pt_str, double x) {
        return dist::EmpiricalDistribution(xv, pv, parse_emp_transform(pt_str)).cdf(x);
    });
    m.def("emp_quantile", [parse_emp_transform](const std::vector<double>& xv,
                                                 const std::vector<double>& pv,
                                                 const std::string& pt_str, double prob) {
        return dist::EmpiricalDistribution(xv, pv, parse_emp_transform(pt_str)).inverse_cdf(prob);
    });
    m.def("emp_valid", [parse_emp_transform](const std::vector<double>& xv,
                                              const std::vector<double>& pv,
                                              const std::string& pt_str) {
        return dist::EmpiricalDistribution(xv, pv, parse_emp_transform(pt_str)).parameters_valid();
    });

    // --- Composite glue: KernelDensity -------------------------------------------------
    // Accepts (data, kernel, bandwidth, bounded_by_data) and exposes the full surface.
    // kernel: "Gaussian" | "Epanechnikov" | "Triangular" | "Uniform"
    // bandwidth: negative value means use Silverman's rule (auto).

    auto parse_kernel_type = [](const std::string& s) {
        if (s == "Epanechnikov") return dist::KernelType::Epanechnikov;
        if (s == "Gaussian")     return dist::KernelType::Gaussian;
        if (s == "Triangular")   return dist::KernelType::Triangular;
        if (s == "Uniform")      return dist::KernelType::Uniform;
        throw py::value_error("unknown kernel type: " + s);
        return dist::KernelType::Gaussian;  // unreachable
    };

    auto make_kde = [parse_kernel_type](const std::vector<double>& data,
                                        const std::string& kernel,
                                        double bandwidth,
                                        bool bounded_by_data) {
        dist::KernelType kt = parse_kernel_type(kernel);
        dist::KernelDensity kde = bandwidth < 0.0
            ? dist::KernelDensity(data, kt)
            : dist::KernelDensity(data, kt, bandwidth);
        kde.set_bounded_by_data(bounded_by_data);
        return kde;
    };

    m.def("kde_moments", [make_kde](const std::vector<double>& data, const std::string& kernel,
                                     double bandwidth, bool bounded_by_data) {
        auto d = make_kde(data, kernel, bandwidth, bounded_by_data);
        return std::map<std::string, double>{
            {"mean", d.mean()},         {"median", d.median()},
            {"mode", d.mode()},         {"sd", d.standard_deviation()},
            {"skewness", d.skewness()}, {"kurtosis", d.kurtosis()},
            {"minimum", d.minimum()},   {"maximum", d.maximum()}};
    });
    m.def("kde_pdf", [make_kde](const std::vector<double>& data, const std::string& kernel,
                                 double bandwidth, bool bounded_by_data, double x) {
        return make_kde(data, kernel, bandwidth, bounded_by_data).pdf(x);
    });
    m.def("kde_cdf", [make_kde](const std::vector<double>& data, const std::string& kernel,
                                 double bandwidth, bool bounded_by_data, double x) {
        return make_kde(data, kernel, bandwidth, bounded_by_data).cdf(x);
    });
    m.def("kde_quantile", [make_kde](const std::vector<double>& data, const std::string& kernel,
                                      double bandwidth, bool bounded_by_data, double prob) {
        return make_kde(data, kernel, bandwidth, bounded_by_data).inverse_cdf(prob);
    });
    m.def("kde_valid", [make_kde](const std::vector<double>& data, const std::string& kernel,
                                   double bandwidth, bool bounded_by_data) {
        return make_kde(data, kernel, bandwidth, bounded_by_data).parameters_valid();
    });

    // --- Composite glue: Mixture -------------------------------------------------------
    // Accepts (comp_targets, comp_params, weights) where comp_params is a list of param
    // vectors (one per component). Exposes the full distribution surface.

    auto make_mixture = [](const std::vector<std::string>& comp_targets,
                            const std::vector<std::vector<double>>& comp_params,
                            const std::vector<double>& weights) {
        int K = static_cast<int>(comp_targets.size());
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        comps.reserve(K);
        for (int i = 0; i < K; ++i) {
            auto d = dist::create_distribution(comp_targets[i]);
            d->set_parameters(comp_params[i]);
            comps.push_back(std::move(d));
        }
        return dist::Mixture(weights, std::move(comps));
    };

    m.def("mix_moments", [make_mixture](const std::vector<std::string>& ct,
                                         const std::vector<std::vector<double>>& cp,
                                         const std::vector<double>& wts) {
        auto d = make_mixture(ct, cp, wts);
        return std::map<std::string, double>{
            {"mean", d.mean()},         {"median", d.median()},
            {"mode", d.mode()},         {"sd", d.standard_deviation()},
            {"skewness", d.skewness()}, {"kurtosis", d.kurtosis()},
            {"minimum", d.minimum()},   {"maximum", d.maximum()}};
    });
    m.def("mix_pdf", [make_mixture](const std::vector<std::string>& ct,
                                     const std::vector<std::vector<double>>& cp,
                                     const std::vector<double>& wts, double x) {
        return make_mixture(ct, cp, wts).pdf(x);
    });
    m.def("mix_cdf", [make_mixture](const std::vector<std::string>& ct,
                                     const std::vector<std::vector<double>>& cp,
                                     const std::vector<double>& wts, double x) {
        return make_mixture(ct, cp, wts).cdf(x);
    });
    m.def("mix_quantile", [make_mixture](const std::vector<std::string>& ct,
                                          const std::vector<std::vector<double>>& cp,
                                          const std::vector<double>& wts, double prob) {
        return make_mixture(ct, cp, wts).inverse_cdf(prob);
    });
    m.def("mix_valid", [make_mixture](const std::vector<std::string>& ct,
                                       const std::vector<std::vector<double>>& cp,
                                       const std::vector<double>& wts) {
        return make_mixture(ct, cp, wts).parameters_valid();
    });

    // --- Composite glue: CompetingRisks -----------------------------------------------
    // Accepts (comp_targets, comp_params, minimum_of_rv, dependency, correlation) where
    // comp_params is a list of param vectors (one per component). minimum_of_rv=true ->
    // min-of-components (default); false -> max-of-components. dependency is one of
    // "Independent"/"PerfectlyPositive"/"PerfectlyNegative"/"CorrelationMatrix";
    // correlation is a list of row vectors (only consulted when dependency ==
    // "CorrelationMatrix", may be empty otherwise).

    auto parse_dependency = [](const std::string& d) {
        namespace prob = bestfit::numerics::data::probability;
        if (d == "Independent") return prob::DependencyType::Independent;
        if (d == "PerfectlyPositive") return prob::DependencyType::PerfectlyPositive;
        if (d == "PerfectlyNegative") return prob::DependencyType::PerfectlyNegative;
        if (d == "CorrelationMatrix") return prob::DependencyType::CorrelationMatrix;
        throw py::value_error("unknown dependency type '" + d + "'");
    };

    auto make_competing_risks = [parse_dependency](
                                     const std::vector<std::string>& comp_targets,
                                     const std::vector<std::vector<double>>& comp_params,
                                     bool minimum_of_rv, const std::string& dependency,
                                     const std::vector<std::vector<double>>& correlation) {
        int K = static_cast<int>(comp_targets.size());
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        comps.reserve(K);
        for (int i = 0; i < K; ++i) {
            auto d = dist::create_distribution(comp_targets[i]);
            d->set_parameters(comp_params[i]);
            comps.push_back(std::move(d));
        }
        dist::CompetingRisks cr(std::move(comps));
        cr.minimum_of_random_variables = minimum_of_rv;
        cr.dependency = parse_dependency(dependency);
        if (dependency == "CorrelationMatrix") cr.set_correlation_matrix(correlation);
        return cr;
    };

    m.def("cr_moments", [make_competing_risks](const std::vector<std::string>& ct,
                                                const std::vector<std::vector<double>>& cp,
                                                bool minimum_of_rv, const std::string& dependency,
                                                const std::vector<std::vector<double>>& correlation) {
        auto d = make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation);
        return std::map<std::string, double>{
            {"mean", d.mean()},         {"median", d.median()},
            {"mode", d.mode()},         {"sd", d.standard_deviation()},
            {"skewness", d.skewness()}, {"kurtosis", d.kurtosis()},
            {"minimum", d.minimum()},   {"maximum", d.maximum()}};
    });
    m.def("cr_pdf", [make_competing_risks](const std::vector<std::string>& ct,
                                            const std::vector<std::vector<double>>& cp,
                                            bool minimum_of_rv, const std::string& dependency,
                                            const std::vector<std::vector<double>>& correlation,
                                            double x) {
        return make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation).pdf(x);
    });
    m.def("cr_log_pdf", [make_competing_risks](const std::vector<std::string>& ct,
                                                const std::vector<std::vector<double>>& cp,
                                                bool minimum_of_rv, const std::string& dependency,
                                                const std::vector<std::vector<double>>& correlation,
                                                double x) {
        return make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation).log_pdf(x);
    });
    m.def("cr_cdf", [make_competing_risks](const std::vector<std::string>& ct,
                                            const std::vector<std::vector<double>>& cp,
                                            bool minimum_of_rv, const std::string& dependency,
                                            const std::vector<std::vector<double>>& correlation,
                                            double x) {
        return make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation).cdf(x);
    });
    m.def("cr_quantile", [make_competing_risks](const std::vector<std::string>& ct,
                                                  const std::vector<std::vector<double>>& cp,
                                                  bool minimum_of_rv, const std::string& dependency,
                                                  const std::vector<std::vector<double>>& correlation,
                                                  double prob) {
        return make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation).inverse_cdf(prob);
    });
    m.def("cr_valid", [make_competing_risks](const std::vector<std::string>& ct,
                                              const std::vector<std::vector<double>>& cp,
                                              bool minimum_of_rv, const std::string& dependency,
                                              const std::vector<std::vector<double>>& correlation) {
        return make_competing_risks(ct, cp, minimum_of_rv, dependency, correlation).parameters_valid();
    });
}
