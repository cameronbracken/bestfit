// cpp11 glue exposing the multivariate-distribution surface (Dirichlet, Multinomial,
// BivariateEmpirical) of the shared C++ core to R. Mirrors the bf_trunc_*/bf_emp_* style
// in dist.cpp: bespoke per-target entry points, generic-by-method-string rather than one
// function per method, so a new multivariate target (MultivariateNormal,
// MultivariateStudentT) only adds one more bf_<name>_val_ function here.
// Core headers are vendored under src/bestfit_core/include (see tools/sync_core.py).
#include <cpp11.hpp>

#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/data/interpolation/transform.hpp"
#include "bestfit/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "bestfit/numerics/distributions/multivariate/dirichlet.hpp"
#include "bestfit/numerics/distributions/multivariate/multinomial.hpp"

namespace mvd = bestfit::numerics::distributions;
namespace data = bestfit::numerics::data;
using namespace cpp11;

// --- Dirichlet -----------------------------------------------------------------------
// method + flat args in, double out. Methods: dimension, parameters_valid, alpha,
// alpha_sum, mean, variance, mode, covariance, pdf, log_pdf, log_multivariate_beta.
// `args` carries index/point arguments per method (see bestfitr/tests/testthat's dispatch
// for the exact per-method argument shape, mirroring fixtures/README.md).

[[cpp11::register]]
double bf_dirichlet_val_(std::string method, doubles alpha, doubles args) {
    std::vector<double> a(alpha.begin(), alpha.end());
    std::vector<double> ar(args.begin(), args.end());
    mvd::Dirichlet d(a);

    if (method == "dimension") return d.dimension();
    if (method == "parameters_valid") return d.parameters_valid() ? 1.0 : 0.0;
    if (method == "alpha") return d.alpha()[static_cast<std::size_t>(ar[0])];
    if (method == "alpha_sum") return d.alpha_sum();
    if (method == "mean") return d.mean()[static_cast<std::size_t>(ar[0])];
    if (method == "variance") return d.variance()[static_cast<std::size_t>(ar[0])];
    if (method == "mode") return d.mode()[static_cast<std::size_t>(ar[0])];
    if (method == "covariance")
        return d.covariance(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
    if (method == "pdf") return d.pdf(ar);
    if (method == "log_pdf") return d.log_pdf(ar);
    if (method == "log_multivariate_beta") return mvd::Dirichlet::log_multivariate_beta(ar);
    stop("unknown Dirichlet fixture method '%s'", method.c_str());
}

// --- Multinomial -----------------------------------------------------------------------
// Methods: dimension, parameters_valid, number_of_trials, mean, variance, covariance,
// pdf, log_pdf.

[[cpp11::register]]
double bf_multinomial_val_(std::string method, int n, doubles p, doubles args) {
    std::vector<double> pv(p.begin(), p.end());
    std::vector<double> ar(args.begin(), args.end());
    mvd::Multinomial m(n, pv);

    if (method == "dimension") return m.dimension();
    if (method == "parameters_valid") return m.parameters_valid() ? 1.0 : 0.0;
    if (method == "number_of_trials") return m.number_of_trials();
    if (method == "mean") return m.mean()[static_cast<std::size_t>(ar[0])];
    if (method == "variance") return m.variance()[static_cast<std::size_t>(ar[0])];
    if (method == "covariance")
        return m.covariance(static_cast<int>(ar[0]), static_cast<int>(ar[1]));
    if (method == "pdf") return m.pdf(ar);
    if (method == "log_pdf") return m.log_pdf(ar);
    stop("unknown Multinomial fixture method '%s'", method.c_str());
}

// --- BivariateEmpirical ------------------------------------------------------------
// Grid passed flattened row-major (nrow = x1.size(), ncol = x2.size()) + nrow. transforms
// is a 3-element character vector: (x1_transform, x2_transform, p_transform), each
// "None"/"Logarithmic"/"NormalZ". Methods: dimension, parameters_valid, cdf (vector arg,
// args = [x1, x2]), cdf_xy (same, scalar-pair entry point).

static data::Transform parse_transform(const std::string& s) {
    if (s == "None") return data::Transform::None;
    if (s == "Logarithmic") return data::Transform::Logarithmic;
    if (s == "NormalZ") return data::Transform::NormalZ;
    stop("unknown transform '%s'", s.c_str());
}

[[cpp11::register]]
double bf_bve_cdf_(std::string method, doubles x1, doubles x2, doubles p_flat, int nrow,
                    strings transforms, doubles args) {
    std::vector<double> x1v(x1.begin(), x1.end());
    std::vector<double> x2v(x2.begin(), x2.end());
    std::size_t ncol = x2v.size();
    std::vector<std::vector<double>> p(static_cast<std::size_t>(nrow), std::vector<double>(ncol));
    for (int i = 0; i < nrow; ++i)
        for (std::size_t j = 0; j < ncol; ++j)
            p[static_cast<std::size_t>(i)][j] = p_flat[static_cast<int>(static_cast<std::size_t>(i) * ncol + j)];

    mvd::BivariateEmpirical bv(x1v, x2v, p, parse_transform(std::string(transforms[0])),
                                parse_transform(std::string(transforms[1])),
                                parse_transform(std::string(transforms[2])));

    if (method == "dimension") return bv.dimension();
    if (method == "parameters_valid") return bv.parameters_valid() ? 1.0 : 0.0;
    if (method == "cdf") return bv.cdf(args[0], args[1]);
    if (method == "cdf_xy") return bv.cdf(args[0], args[1]);
    stop("unknown BivariateEmpirical fixture method '%s'", method.c_str());
}
