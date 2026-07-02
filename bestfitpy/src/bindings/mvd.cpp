// pybind11 glue exposing the multivariate-distribution surface (Dirichlet, Multinomial,
// BivariateEmpirical) of the shared C++ core to Python. Mirrors the trunc_*/emp_* style
// in dist.cpp: bespoke per-target module functions, generic-by-method-string rather than
// one function per method, so a new multivariate target (MultivariateNormal,
// MultivariateStudentT) only adds one more <name>_val function here.
// Core headers are vendored under ../bestfit_core/include (see tools/sync_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <string>
#include <vector>

#include "bestfit/numerics/data/interpolation/transform.hpp"
#include "bestfit/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "bestfit/numerics/distributions/multivariate/dirichlet.hpp"
#include "bestfit/numerics/distributions/multivariate/multinomial.hpp"
#include "bestfit/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace mvd = bestfit::numerics::distributions;
namespace data = bestfit::numerics::data;

static data::Transform parse_transform(const std::string& s) {
    if (s == "None") return data::Transform::None;
    if (s == "Logarithmic") return data::Transform::Logarithmic;
    if (s == "NormalZ") return data::Transform::NormalZ;
    throw py::value_error("unknown transform: " + s);
}

void register_multivariate(py::module_& m) {
    // --- Dirichlet ---------------------------------------------------------------------
    // method + flat args in, double out. Methods: dimension, parameters_valid, alpha,
    // alpha_sum, mean, variance, mode, covariance, pdf, log_pdf, log_multivariate_beta.
    m.def("dirichlet_val", [](const std::string& method, const std::vector<double>& alpha,
                               const std::vector<double>& args) {
        mvd::Dirichlet d(alpha);
        if (method == "dimension") return static_cast<double>(d.dimension());
        if (method == "parameters_valid") return d.parameters_valid() ? 1.0 : 0.0;
        if (method == "alpha") return d.alpha()[static_cast<std::size_t>(args[0])];
        if (method == "alpha_sum") return d.alpha_sum();
        if (method == "mean") return d.mean()[static_cast<std::size_t>(args[0])];
        if (method == "variance") return d.variance()[static_cast<std::size_t>(args[0])];
        if (method == "mode") return d.mode()[static_cast<std::size_t>(args[0])];
        if (method == "covariance")
            return d.covariance(static_cast<int>(args[0]), static_cast<int>(args[1]));
        if (method == "pdf") return d.pdf(args);
        if (method == "log_pdf") return d.log_pdf(args);
        if (method == "log_multivariate_beta") return mvd::Dirichlet::log_multivariate_beta(args);
        throw py::value_error("unknown Dirichlet fixture method: " + method);
    });

    // --- Multinomial ---------------------------------------------------------------------
    // Methods: dimension, parameters_valid, number_of_trials, mean, variance, covariance,
    // pdf, log_pdf.
    m.def("multinomial_val", [](const std::string& method, int n, const std::vector<double>& p,
                                 const std::vector<double>& args) {
        mvd::Multinomial d(n, p);
        if (method == "dimension") return static_cast<double>(d.dimension());
        if (method == "parameters_valid") return d.parameters_valid() ? 1.0 : 0.0;
        if (method == "number_of_trials") return static_cast<double>(d.number_of_trials());
        if (method == "mean") return d.mean()[static_cast<std::size_t>(args[0])];
        if (method == "variance") return d.variance()[static_cast<std::size_t>(args[0])];
        if (method == "covariance")
            return d.covariance(static_cast<int>(args[0]), static_cast<int>(args[1]));
        if (method == "pdf") return d.pdf(args);
        if (method == "log_pdf") return d.log_pdf(args);
        throw py::value_error("unknown Multinomial fixture method: " + method);
    });

    // --- BivariateEmpirical -------------------------------------------------------------
    // p is a nested list (row i = x1[i], column j = x2[j]). transforms is
    // (x1_transform, x2_transform, p_transform) strings. Methods: dimension,
    // parameters_valid, cdf (vector arg, args = [x1, x2]), cdf_xy (same, scalar pair).
    m.def("bve_cdf", [](const std::string& method, const std::vector<double>& x1,
                         const std::vector<double>& x2,
                         const std::vector<std::vector<double>>& p,
                         const std::vector<std::string>& transforms,
                         const std::vector<double>& args) {
        mvd::BivariateEmpirical bv(x1, x2, p, parse_transform(transforms[0]),
                                    parse_transform(transforms[1]), parse_transform(transforms[2]));
        if (method == "dimension") return static_cast<double>(bv.dimension());
        if (method == "parameters_valid") return bv.parameters_valid() ? 1.0 : 0.0;
        if (method == "cdf") return bv.cdf(args[0], args[1]);
        if (method == "cdf_xy") return bv.cdf(args[0], args[1]);
        throw py::value_error("unknown BivariateEmpirical fixture method: " + method);
    });

    // --- MultivariateNormal (deterministic methods) -------------------------------------
    // Stateless per-call style: a fresh instance is constructed from (mean, covariance) on
    // every call. Covers everything that does not touch the seeded MVNUNI stream (mean,
    // median, mode, sd, variance, covariance, pdf, log_pdf, cdf for dim 1-2, mahalanobis,
    // inverse_cdf, dimension, parameters_valid). Seeded CDF (dim>=3) / MVNDST batches need
    // a single persistent instance across several calls -- see mvn_cdf_seq/mvn_mvndst_seq
    // (added alongside the MVNDST port).
    m.def("mvn_val", [](const std::string& method, const std::vector<double>& mean,
                         const std::vector<std::vector<double>>& covariance,
                         const std::vector<double>& args) {
        mvd::MultivariateNormal n(mean, covariance);
        if (method == "dimension") return static_cast<double>(n.dimension());
        if (method == "parameters_valid") return n.parameters_valid() ? 1.0 : 0.0;
        if (method == "mean") return n.mean()[static_cast<std::size_t>(args[0])];
        if (method == "median") return n.median()[static_cast<std::size_t>(args[0])];
        if (method == "mode") return n.mode()[static_cast<std::size_t>(args[0])];
        if (method == "sd") return n.standard_deviation()[static_cast<std::size_t>(args[0])];
        if (method == "variance") return n.variance()[static_cast<std::size_t>(args[0])];
        if (method == "covariance")
            return n.covariance(static_cast<int>(args[0]), static_cast<int>(args[1]));
        if (method == "pdf") return n.pdf(args);
        if (method == "log_pdf") return n.log_pdf(args);
        if (method == "cdf") return n.cdf(args);
        if (method == "mahalanobis") return n.mahalanobis(args);
        if (method == "inverse_cdf") {
            // args = [p_1..p_dim, index]
            std::vector<double> p(args.begin(), args.end() - 1);
            int idx = static_cast<int>(args.back());
            return n.inverse_cdf(p)[static_cast<std::size_t>(idx)];
        }
        throw py::value_error("unknown MultivariateNormal fixture method: " + method);
    });
}
