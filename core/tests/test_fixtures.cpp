// Generic, fixture-driven validation for the C++ core.
//
// Reads the language-neutral oracle fixtures (the single source of truth shared with
// the R and Python packages) and checks every assertion. No oracle values live here --
// only the dispatch from fixture method names to the core API. The fixtures directory is
// passed as argv[1] (CMake points it at the repo's canonical fixtures/).
//
// Two code paths: the GEV slice keeps its bespoke dispatch (location/scale/shape names,
// standard-error methods); every other distribution goes through the polymorphic
// UnivariateDistributionBase + factory path, which is what new distributions plug into.
// Special functions use a flat target->lambda map.
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include "bestfit/analyses/distribution_fitting/fitting_analysis.hpp"
#include "bestfit/analyses/univariate/bulletin17c_analysis.hpp"
#include "bestfit/analyses/univariate/univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/estimation/generalized_method_of_moments.hpp"
#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/maximum_likelihood.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/data_frame/data_collections/exact_series.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/model_spec.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/data/correlation.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/histogram.hpp"
#include "bestfit/numerics/data/interpolation/search.hpp"
#include "bestfit/numerics/data/plotting_positions.hpp"
#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/copulas/base/bivariate_copula_estimation.hpp"
#include "bestfit/numerics/distributions/copulas/base/copula_factory.hpp"
#include "bestfit/numerics/distributions/copulas/clayton_copula.hpp"
#include "bestfit/numerics/distributions/empirical_distribution.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/kernel_density.hpp"
#include "bestfit/numerics/distributions/competing_risks.hpp"
#include "bestfit/numerics/distributions/mixture.hpp"
#include "bestfit/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "bestfit/numerics/distributions/multivariate/bivariate_empirical.hpp"
#include "bestfit/numerics/distributions/multivariate/dirichlet.hpp"
#include "bestfit/numerics/distributions/multivariate/multinomial.hpp"
#include "bestfit/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "bestfit/numerics/distributions/multivariate/multivariate_student_t.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/truncated_distribution.hpp"
#include "bestfit/numerics/data/running_covariance_matrix.hpp"
#include "bestfit/numerics/data/running_statistics.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/math/differentiation/numerical_derivative.hpp"
#include "bestfit/numerics/math/fourier/fourier.hpp"
#include "bestfit/numerics/math/linalg/cholesky_decomposition.hpp"
#include "bestfit/numerics/math/linalg/lu_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"
#include "bestfit/numerics/math/optimization/differential_evolution.hpp"
#include "bestfit/numerics/math/special/beta.hpp"
#include "bestfit/numerics/math/special/bessel.hpp"
#include "bestfit/numerics/math/special/erf.hpp"
#include "bestfit/numerics/math/special/factorial.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"
#include "bestfit/numerics/sampling/bootstrap/bootstrap.hpp"
#include "bestfit/numerics/sampling/bootstrap/model_registry.hpp"
#include "bestfit/numerics/sampling/mcmc/arwmh.hpp"
#include "bestfit/numerics/sampling/mcmc/demcz.hpp"
#include "bestfit/numerics/sampling/mcmc/demczs.hpp"
#include "bestfit/numerics/sampling/mcmc/gibbs.hpp"
#include "bestfit/numerics/sampling/mcmc/hmc.hpp"
#include "bestfit/numerics/sampling/mcmc/model_registry.hpp"
#include "bestfit/numerics/sampling/mcmc/nuts.hpp"
#include "bestfit/numerics/sampling/mcmc/rwmh.hpp"
#include "bestfit/numerics/sampling/mcmc/snis.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_diagnostics.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/utilities/extension_methods.hpp"
#include "check.hpp"
#include "third_party/json.hpp"

using json = nlohmann::json;
namespace fs = std::filesystem;
namespace dist = bestfit::numerics::distributions;
namespace prob = bestfit::numerics::data::probability;
using dist::EstimationMethod;
using dist::GeneralizedExtremeValue;

static double parse_num(const json& v) {
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        if (s == "nan") return std::numeric_limits<double>::quiet_NaN();
        if (s == "inf") return std::numeric_limits<double>::infinity();
        if (s == "-inf") return -std::numeric_limits<double>::infinity();
        throw std::runtime_error("unexpected string value: " + s);
    }
    return v.get<double>();
}

// --- Shared assertion checking ---------------------------------------------------------

static void check_value(double actual, const json& as, const std::string& where) {
    std::string mode = as["mode"].get<std::string>();
    bool ok;
    if (mode == "equal") {
        double e = parse_num(as["expected"]);
        ok = std::isnan(e) ? std::isnan(actual) : (actual == e);
    } else if (mode == "abs") {
        ok = std::fabs(actual - as["expected"].get<double>()) <= as["tol"].get<double>();
    } else if (mode == "rel") {
        double e = as["expected"].get<double>();
        ok = std::fabs(actual - e) / std::fabs(e) <= as["tol"].get<double>();
    } else {
        throw std::runtime_error("unknown comparison mode: " + mode);
    }
    if (ok)
        bftest::report_pass();
    else
        bftest::report_fail(__FILE__, __LINE__, where + ": value mismatch");
}

static void check_bool(bool actual, const json& as, const std::string& where) {
    if (actual == as["expected"].get<bool>())
        bftest::report_pass();
    else
        bftest::report_fail(__FILE__, __LINE__, where + ": bool mismatch");
}

// --- GEV slice (bespoke) ---------------------------------------------------------------

static EstimationMethod parse_method(const std::string& m) {
    if (m == "mom") return EstimationMethod::MethodOfMoments;
    if (m == "lmom") return EstimationMethod::MethodOfLinearMoments;
    return EstimationMethod::MaximumLikelihood;
}

static GeneralizedExtremeValue build_gev(const json& construct, const json& datasets) {
    if (construct.contains("params")) {
        auto p = construct["params"];
        return GeneralizedExtremeValue(parse_num(p[0]), parse_num(p[1]), parse_num(p[2]));
    }
    const auto& fit = construct["fit"];
    std::vector<double> data;
    for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
    GeneralizedExtremeValue g;
    g.estimate(data, parse_method(fit["method"].get<std::string>()));
    return g;
}

static double dispatch_gev(const GeneralizedExtremeValue& g, const std::string& m, const json& a) {
    if (m == "mean") return g.mean();
    if (m == "median") return g.median();
    if (m == "mode") return g.mode();
    if (m == "sd") return g.standard_deviation();
    if (m == "skewness") return g.skewness();
    if (m == "kurtosis") return g.kurtosis();
    if (m == "minimum") return g.minimum();
    if (m == "maximum") return g.maximum();
    if (m == "pdf") return g.pdf(a[0].get<double>());
    if (m == "cdf") return g.cdf(a[0].get<double>());
    if (m == "quantile") return g.inverse_cdf(a[0].get<double>());
    if (m == "param") {
        std::string n = a[0].get<std::string>();
        return n == "location" ? g.xi() : n == "scale" ? g.alpha() : g.kappa();
    }
    if (m == "linear_moment")
        return g.linear_moments_from_parameters({g.xi(), g.alpha(), g.kappa()})[a[0].get<int>()];
    if (m == "quantile_gradient") return g.quantile_gradient(a[0].get<double>())[a[1].get<int>()];
    if (m == "parameter_covariance")
        return g.parameter_covariance(a[0].get<int>())[a[1].get<int>()][a[2].get<int>()];
    if (m == "quantile_variance") return g.quantile_variance(a[0].get<double>(), a[1].get<int>());
    if (m == "quantile_se")
        return std::sqrt(g.quantile_variance(a[0].get<double>(), a[1].get<int>()));
    throw std::runtime_error("unknown GEV fixture method: " + m);
}

static void run_gev(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        GeneralizedExtremeValue g = build_gev(c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(g.parameters_valid(), as, where);
            else
                check_value(dispatch_gev(g, method, args), as, where);
        }
    }
}

// --- Special-function path ------------------------------------------------------------

namespace sf = bestfit::numerics::math::special;
namespace la = bestfit::numerics::math::linalg;
// Alias must not be named `stat` (collides with the MSVC/POSIX CRT symbol, like the
// glibc `gamma` clash documented in .claude/CLAUDE.md).
namespace bfdata = bestfit::numerics::data;
namespace bfsamp = bestfit::numerics::sampling;
namespace bfutil = bestfit::numerics::utilities;
namespace bffourier = bestfit::numerics::math::fourier;
namespace bfdiff = bestfit::numerics::math::differentiation;
namespace bfopt = bestfit::numerics::math::optimization;

// Correlation fixture args are [x..., y...] concatenated and split at the midpoint
// (equal-length samples) -- see fixtures/special_functions/correlation.json / README.md.
static void correlation_split(const std::vector<double>& a, std::vector<double>& x,
                               std::vector<double>& y) {
    std::size_t mid = a.size() / 2;
    x.assign(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(mid));
    y.assign(a.begin() + static_cast<std::ptrdiff_t>(mid), a.end());
}

// Cholesky fixture args are a flattened row-major n*n matrix, with n inferred from the
// args length per the convention documented in fixtures/special_functions/cholesky.json.
static int cholesky_square_n(std::size_t len) {
    int n = static_cast<int>(std::lround(std::sqrt(static_cast<double>(len))));
    if (static_cast<std::size_t>(n) * static_cast<std::size_t>(n) != len)
        throw std::runtime_error("Cholesky fixture args: length is not a perfect square");
    return n;
}

// solve_element args are [flattened n*n matrix, n-length rhs vector, index i], i.e.
// n*n + n + 1 == len; solve the quadratic for n.
static int cholesky_solve_n(std::size_t len) {
    double n_double = (-1.0 + std::sqrt(1.0 + 4.0 * (static_cast<double>(len) - 1.0))) / 2.0;
    int n = static_cast<int>(std::lround(n_double));
    if (static_cast<std::size_t>(n) * static_cast<std::size_t>(n) + static_cast<std::size_t>(n) + 1 != len)
        throw std::runtime_error("Cholesky fixture args: length does not fit n*n+n+1");
    return n;
}

static la::Matrix cholesky_matrix_from_flat(const std::vector<double>& a, int n) {
    std::vector<double> flat(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n) * n);
    return la::Matrix(n, n, flat);
}

// RunningCovariance fixture args: [size, num_pushes, data_flat(num_pushes*size), trailing
// index/indices] -- see fixtures/special_functions/running_covariance.json for the
// convention. Builds a RunningCovarianceMatrix and replays `num_pushes` push()es of
// `size`-length rows sliced from the flattened data.
static bfdata::RunningCovarianceMatrix running_covariance_build(const std::vector<double>& a, int size,
                                                                  int num_pushes) {
    bfdata::RunningCovarianceMatrix rcm(size);
    for (int p = 0; p < num_pushes; ++p) {
        std::vector<double> row(a.begin() + 2 + static_cast<std::ptrdiff_t>(p) * size,
                                 a.begin() + 2 + static_cast<std::ptrdiff_t>(p + 1) * size);
        rcm.push(row);
    }
    return rcm;
}

// RunningStatistics combine fixture args: [n1, sample1(n1 values), sample2(remaining
// values)] -- a "split-index" convention, distinct from Correlation's equal-length
// two-halves split (Test_Combine/Test_Add split their 69-value sample into UNEQUAL 48/21
// sub-samples, so a fixed midpoint doesn't apply). See
// fixtures/special_functions/running_statistics.json for the full convention. Uses
// operator+ (rather than calling RunningStatistics::combine() directly), which exercises
// both -- operator+ is a one-line forwarder to combine().
static bfdata::RunningStatistics running_statistics_combined(const std::vector<double>& a) {
    std::size_t n1 = static_cast<std::size_t>(a[0]);
    std::vector<double> sample1(a.begin() + 1, a.begin() + 1 + static_cast<std::ptrdiff_t>(n1));
    std::vector<double> sample2(a.begin() + 1 + static_cast<std::ptrdiff_t>(n1), a.end());
    return bfdata::RunningStatistics(sample1) + bfdata::RunningStatistics(sample2);
}

// Fourier fixture args conventions (fixtures/special_functions/fourier.json):
//  - Fourier.fft_at / Fourier.real_fft_at: args = [data..., inverse (0/1), index] -- n =
//    len(args) - 2; runs fft()/real_fft() in place on a copy of `data`, returns data[index].
//  - Fourier.correlation_at: args = [data1..., data2..., index] -- equal-length data1/data2
//    concatenated (n = (len(args)-1)/2), returns correlation(data1, data2)[index].
//  - Fourier.autocorrelation_at: args = [series..., lag_max, lag] -- n = len(args) - 2;
//    autocorrelation(series, (int)lag_max) (lag_max == -1 triggers the default auto-lag),
//    returns the acf value (column 1) at row `lag`.
static double fourier_fft_at(const std::vector<double>& a) {
    std::size_t n = a.size() - 2;
    std::vector<double> data(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    bool inverse = a[n] != 0.0;
    int index = static_cast<int>(a[n + 1]);
    bffourier::fft(data, inverse);
    return data[static_cast<std::size_t>(index)];
}
static double fourier_real_fft_at(const std::vector<double>& a) {
    std::size_t n = a.size() - 2;
    std::vector<double> data(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    bool inverse = a[n] != 0.0;
    int index = static_cast<int>(a[n + 1]);
    bffourier::real_fft(data, inverse);
    return data[static_cast<std::size_t>(index)];
}
static double fourier_correlation_at(const std::vector<double>& a) {
    std::size_t n = (a.size() - 1) / 2;
    std::vector<double> data1(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    std::vector<double> data2(a.begin() + static_cast<std::ptrdiff_t>(n),
                               a.begin() + static_cast<std::ptrdiff_t>(2 * n));
    int index = static_cast<int>(a[2 * n]);
    auto corr = bffourier::correlation(data1, data2);
    return corr[static_cast<std::size_t>(index)];
}
static double fourier_autocorrelation_at(const std::vector<double>& a) {
    std::size_t n = a.size() - 2;
    std::vector<double> series(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
    int lag_max = static_cast<int>(a[n]);
    int lag = static_cast<int>(a[n + 1]);
    auto acf = bffourier::autocorrelation(series, lag_max);
    if (!acf) throw std::runtime_error("Fourier.autocorrelation_at: autocorrelation returned no value");
    return (*acf)[static_cast<std::size_t>(lag)][1];
}

// Closed registry of named functions for the numerical_derivative fixture -- MUST match
// tools/oracle_emitter/Program.cs's resolver exactly (the emitter runs the REAL C#
// NumericalDerivative against these same two functions):
//  - "quadratic": f(x) = sum_i (x_i - i)^2, i 0-based; analytic gradient 2*(x_i - i),
//    analytic Hessian 2*I (diagonal only) -- a smooth, unbounded-friendly sanity check.
//  - "normal_loglik": Normal(mu=x0, sigma=x1).LogLikelihood(sample) on a small embedded
//    5-point sample -- the exact shape (a 2-parameter log-likelihood) MCMC's default
//    HMC/NUTS gradient differentiates.
static const std::vector<double>& numerical_derivative_normal_sample() {
    static const std::vector<double> sample = {9.0, 10.0, 11.0, 12.0, 13.0};
    return sample;
}
static double numerical_derivative_quadratic(const std::vector<double>& x) {
    double s = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        double d = x[i] - static_cast<double>(i);
        s += d * d;
    }
    return s;
}
static double numerical_derivative_normal_loglik(const std::vector<double>& x) {
    dist::Normal n(x[0], x[1]);
    return n.log_likelihood(numerical_derivative_normal_sample());
}

// numerical_derivative fixture args convention
// (fixtures/special_functions/numerical_derivative.json):
//   gradient_element: args = [p, theta(p values), lower(p values), upper(p values), index]
//   hessian_element:  args = [p, theta(p values), lower(p values), upper(p values), i, j]
// `p` is an explicit leading arg (not inferred from length, matching the
// Extensions.next_doubles_grid convention) for clarity. lower/upper always carry an
// explicit p-length bound array using the JSON "-inf"/"inf" literals for "unbounded"
// dimensions, rather than a presence flag -- AvailableLeft/AvailableRight's null-vs-value
// behavior is bitwise identical to a value of +-infinity (theta[j] - (-inf) == +inf either
// way), so this is a behavior-preserving flattening of the C# nullable-array API onto a
// flat numeric args convention. rel_step/abs_step/max_backtrack always use the library
// defaults (not fixture-configurable), matching every real call site (HMC.cs's Gradient
// call and Optimizer.cs's Hessian call both omit them).
static void numerical_derivative_parse(const std::vector<double>& a, std::vector<double>& theta,
                                        std::vector<double>& lower, std::vector<double>& upper,
                                        std::size_t& next) {
    std::size_t p = static_cast<std::size_t>(a[0]);
    theta.assign(a.begin() + 1, a.begin() + 1 + static_cast<std::ptrdiff_t>(p));
    lower.assign(a.begin() + 1 + static_cast<std::ptrdiff_t>(p), a.begin() + 1 + 2 * static_cast<std::ptrdiff_t>(p));
    upper.assign(a.begin() + 1 + 2 * static_cast<std::ptrdiff_t>(p),
                  a.begin() + 1 + 3 * static_cast<std::ptrdiff_t>(p));
    next = 1 + 3 * p;
}
static double numerical_derivative_gradient_element(const bfdiff::ScalarFunction& f, const std::vector<double>& a) {
    std::vector<double> theta, lower, upper;
    std::size_t next;
    numerical_derivative_parse(a, theta, lower, upper, next);
    int index = static_cast<int>(a[next]);
    auto grad = bfdiff::gradient(f, theta, lower, upper);
    return grad[static_cast<std::size_t>(index)];
}
static double numerical_derivative_hessian_element(const bfdiff::ScalarFunction& f, const std::vector<double>& a) {
    std::vector<double> theta, lower, upper;
    std::size_t next;
    numerical_derivative_parse(a, theta, lower, upper, next);
    int i = static_cast<int>(a[next]);
    int j = static_cast<int>(a[next + 1]);
    auto hess = bfdiff::hessian(f, theta, lower, upper);
    return hess[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
}

// DifferentialEvolution fixture args convention (fixtures/special_functions/differential_evolution.json):
//   args = [fn_id, direction, D, lower(D values), upper(D values), index]
// fn_id: 0 = "quadratic" (numerical_derivative_quadratic), 1 = "normal_loglik"
// (numerical_derivative_normal_loglik) -- REUSES the P3.3 closed named-function registry
// above (see numerical_derivative_{quadratic,normal_loglik}) rather than porting a second
// registry, so the emitter runs the REAL C# DifferentialEvolution against the identical
// objective this runner does. direction: 0 = minimize(), 1 = maximize(). index: 0..D-1
// selects best_parameter_set().values[index]; index == D selects
// best_parameter_set().fitness. Every other DifferentialEvolution/Optimizer knob
// (PRNGSeed, PopulationSize, Mutation, DitherRate, CrossoverProbability, MaxIterations,
// tolerances, ReportFailure) is left at its library default -- matching the only real call
// site (MCMCSampler.cs's MAP init, a later task), which overrides none of them except
// ReportFailure (irrelevant here: whether hitting Max*Reached throws-then-is-swallowed or
// never throws at all, BestParameterSet ends up identical either way -- see optimizer.hpp's
// file header for the full analysis).
static double differential_evolution_best_value(const std::vector<double>& a) {
    int fn_id = static_cast<int>(a[0]);
    int direction = static_cast<int>(a[1]);
    int D = static_cast<int>(a[2]);
    std::vector<double> lower(a.begin() + 3, a.begin() + 3 + D);
    std::vector<double> upper(a.begin() + 3 + D, a.begin() + 3 + 2 * D);
    int index = static_cast<int>(a[static_cast<std::size_t>(3 + 2 * D)]);

    bfopt::DifferentialEvolution::Objective f =
        fn_id == 0 ? bfopt::DifferentialEvolution::Objective(numerical_derivative_quadratic)
                   : bfopt::DifferentialEvolution::Objective(numerical_derivative_normal_loglik);
    bfopt::DifferentialEvolution de(f, D, lower, upper);
    if (direction == 0)
        de.minimize();
    else
        de.maximize();
    return index == D ? de.best_parameter_set().fitness
                       : de.best_parameter_set().values[static_cast<std::size_t>(index)];
}

// Histogram fixture args convention (fixtures/special_functions/histogram.json): args =
// [explicit_bins, data...] for the whole-histogram scalar targets (explicit_bins == 0 uses
// the Rice-Rule ctor; explicit_bins > 0 uses the explicit-bin-count ctor). The
// bin_*_at/get_bin_index_of element-lookup targets append one trailing probe value (a bin
// index, or an x value to look up) after `data`; `trailing` tells histogram_build() how
// many args at the end of `a` are NOT part of `data`.
static bfdata::Histogram histogram_build(const std::vector<double>& a, std::size_t trailing) {
    int explicit_bins = static_cast<int>(a[0]);
    std::vector<double> data(a.begin() + 1, a.end() - static_cast<std::ptrdiff_t>(trailing));
    if (explicit_bins > 0) return bfdata::Histogram(data, explicit_bins);
    return bfdata::Histogram(data);
}

// Dispatch table: maps "Module.method" → a free function of (vector<double>) → double.
static const std::map<std::string, std::function<double(const std::vector<double>&)>>&
special_function_table() {
    static const std::map<std::string, std::function<double(const std::vector<double>&)>> t = {
        // Cholesky family (args: flattened row-major matrix, n inferred from length --
        // see fixtures/special_functions/cholesky.json for the full convention)
        {"Cholesky.determinant", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            return la::CholeskyDecomposition(cholesky_matrix_from_flat(a, n)).determinant();
        }},
        {"Cholesky.log_determinant", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            return la::CholeskyDecomposition(cholesky_matrix_from_flat(a, n)).log_determinant();
        }},
        {"Cholesky.inverse_element", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size() - 2);
            la::CholeskyDecomposition chol(cholesky_matrix_from_flat(a, n));
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n]);
            int j = static_cast<int>(a[static_cast<std::size_t>(n) * n + 1]);
            return chol.inverse_a()(i, j);
        }},
        {"Cholesky.solve_element", [](const std::vector<double>& a) {
            int n = cholesky_solve_n(a.size());
            la::CholeskyDecomposition chol(cholesky_matrix_from_flat(a, n));
            std::vector<double> rhs(a.begin() + static_cast<std::ptrdiff_t>(n) * n,
                                     a.begin() + static_cast<std::ptrdiff_t>(n) * n + n);
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n + static_cast<std::size_t>(n)]);
            return chol.solve(la::Vector(std::move(rhs)))[i];
        }},
        // Returns 1.0 if the matrix is positive-definite (construction succeeds), 0.0 if
        // the ctor throws std::runtime_error (non-PD or NaN diagonal) -- pins the
        // exception condition against the real C# behavior (see Program.cs's resolver).
        {"Cholesky.is_positive_definite", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            try {
                return la::CholeskyDecomposition(cholesky_matrix_from_flat(a, n)).is_positive_definite()
                           ? 1.0
                           : 0.0;
            } catch (const std::runtime_error&) {
                return 0.0;
            }
        }},
        // Erf family
        {"Erf.function",      [](const std::vector<double>& a) { return sf::erf::function(a[0]); }},
        {"Erf.erfc",          [](const std::vector<double>& a) { return sf::erf::erfc(a[0]); }},
        {"Erf.inverse_erf",   [](const std::vector<double>& a) { return sf::erf::inverse_erf(a[0]); }},
        {"Erf.inverse_erfc",  [](const std::vector<double>& a) { return sf::erf::inverse_erfc(a[0]); }},
        // Gamma family
        {"Gamma.function",               [](const std::vector<double>& a) { return sf::function(a[0]); }},
        {"Gamma.log_gamma",              [](const std::vector<double>& a) { return sf::log_gamma(a[0]); }},
        {"Gamma.digamma",                [](const std::vector<double>& a) { return sf::digamma(a[0]); }},
        {"Gamma.trigamma",               [](const std::vector<double>& a) { return sf::trigamma(a[0]); }},
        {"Gamma.lower_incomplete",       [](const std::vector<double>& a) { return sf::lower_incomplete(a[0], a[1]); }},
        {"Gamma.upper_incomplete",       [](const std::vector<double>& a) { return sf::upper_incomplete(a[0], a[1]); }},
        {"Gamma.inverse_lower_incomplete", [](const std::vector<double>& a) { return sf::inverse_lower_incomplete(a[0], a[1]); }},
        {"Gamma.inverse_upper_incomplete", [](const std::vector<double>& a) { return sf::inverse_upper_incomplete(a[0], a[1]); }},
        // Beta family
        {"Beta.function",           [](const std::vector<double>& a) { return sf::beta::function(a[0], a[1]); }},
        {"Beta.incomplete",         [](const std::vector<double>& a) { return sf::beta::incomplete(a[0], a[1], a[2]); }},
        {"Beta.incbcf",             [](const std::vector<double>& a) { return sf::beta::detail::incbcf(a[0], a[1], a[2]); }},
        {"Beta.incbd",              [](const std::vector<double>& a) { return sf::beta::detail::incbd(a[0], a[1], a[2]); }},
        {"Beta.power_series",       [](const std::vector<double>& a) { return sf::beta::detail::power_series(a[0], a[1], a[2]); }},
        {"Beta.incomplete_inverse", [](const std::vector<double>& a) { return sf::beta::incomplete_inverse(a[0], a[1], a[2]); }},
        // Factorial family
        {"Factorial.function",             [](const std::vector<double>& a) { return sf::factorial::function(static_cast<int>(a[0])); }},
        {"Factorial.log_factorial",        [](const std::vector<double>& a) { return sf::factorial::log_factorial(static_cast<int>(a[0])); }},
        {"Factorial.binomial_coefficient", [](const std::vector<double>& a) { return sf::factorial::binomial_coefficient(static_cast<int>(a[0]), static_cast<int>(a[1])); }},
        // Bessel family
        {"Bessel.i0", [](const std::vector<double>& a) { return sf::bessel::i0(a[0]); }},
        {"Bessel.i1", [](const std::vector<double>& a) { return sf::bessel::i1(a[0]); }},
        // Correlation family (args: [x..., y...], split at the midpoint -- see
        // fixtures/special_functions/correlation.json for the full convention)
        {"Correlation.pearson", [](const std::vector<double>& a) {
            std::vector<double> x, y;
            correlation_split(a, x, y);
            return bfdata::pearson(x, y);
        }},
        {"Correlation.spearman", [](const std::vector<double>& a) {
            std::vector<double> x, y;
            correlation_split(a, x, y);
            return bfdata::spearman(x, y);
        }},
        {"Correlation.kendalls_tau", [](const std::vector<double>& a) {
            std::vector<double> x, y;
            correlation_split(a, x, y);
            return bfdata::kendalls_tau(x, y);
        }},
        // LU family (args: flattened row-major matrix, n inferred from length -- reuses
        // the Cholesky-fixture flatten helpers above, which are generic matrix-args
        // conventions, not Cholesky-specific; see fixtures/special_functions/lu_decomposition.json)
        {"LU.determinant", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size());
            return la::LUDecomposition(cholesky_matrix_from_flat(a, n)).determinant();
        }},
        {"LU.inverse_element", [](const std::vector<double>& a) {
            int n = cholesky_square_n(a.size() - 2);
            la::LUDecomposition lu(cholesky_matrix_from_flat(a, n));
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n]);
            int j = static_cast<int>(a[static_cast<std::size_t>(n) * n + 1]);
            return lu.inverse_a()(i, j);
        }},
        {"LU.solve_element", [](const std::vector<double>& a) {
            int n = cholesky_solve_n(a.size());
            la::LUDecomposition lu(cholesky_matrix_from_flat(a, n));
            std::vector<double> rhs(a.begin() + static_cast<std::ptrdiff_t>(n) * n,
                                     a.begin() + static_cast<std::ptrdiff_t>(n) * n + n);
            int i = static_cast<int>(a[static_cast<std::size_t>(n) * n + static_cast<std::size_t>(n)]);
            return lu.solve(la::Vector(std::move(rhs)))[i];
        }},
        // Percentile (args: [data_1..data_n, k, data_is_sorted (0.0/1.0)] -- see
        // fixtures/special_functions/percentile.json for the convention)
        {"Statistics.percentile", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> data(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            double k = a[n];
            bool sorted = a[n + 1] != 0.0;
            return bfdata::percentile(data, k, sorted);
        }},
        // Extensions/MersenneTwister ranged-draw family (see
        // fixtures/special_functions/extension_methods.json for the conventions)
        {"Extensions.next_doubles_grid", [](const std::vector<double>& a) {
            // args: [n, dim, seed, row, col]
            int n = static_cast<int>(a[0]);
            int dim = static_cast<int>(a[1]);
            bfsamp::MersenneTwister rng(static_cast<std::uint32_t>(a[2]));
            int row = static_cast<int>(a[3]);
            int col = static_cast<int>(a[4]);
            auto grid = bfutil::next_doubles(rng, n, dim);
            return grid[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
        }},
        {"Extensions.next_integers_at", [](const std::vector<double>& a) {
            // args: [n, seed, i]
            int n = static_cast<int>(a[0]);
            bfsamp::MersenneTwister rng(static_cast<std::uint32_t>(a[1]));
            int i = static_cast<int>(a[2]);
            auto values = bfutil::next_integers(rng, n);
            return static_cast<double>(values[static_cast<std::size_t>(i)]);
        }},
        {"Mt.next_range", [](const std::vector<double>& a) {
            // args: [seed, min, max, i] -- draws next(min, max) (i+1) times, 0-based,
            // returning the i-th draw.
            bfsamp::MersenneTwister rng(static_cast<std::uint32_t>(a[0]));
            int min_v = static_cast<int>(a[1]);
            int max_v = static_cast<int>(a[2]);
            int i = static_cast<int>(a[3]);
            int result = 0;
            for (int k = 0; k <= i; ++k) result = rng.next(min_v, max_v);
            return static_cast<double>(result);
        }},
        // RunningCovarianceMatrix family (args: [size, num_pushes, data_flat, trailing
        // index/indices] -- see fixtures/special_functions/running_covariance.json)
        {"RunningCovariance.mean_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto rcm = running_covariance_build(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            return rcm.mean()(i, 0);
        }},
        {"RunningCovariance.covariance_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto rcm = running_covariance_build(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return rcm.covariance()(i, j);
        }},
        {"RunningCovariance.sample_covariance_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto rcm = running_covariance_build(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return rcm.sample_covariance()(i, j);
        }},
        {"RunningCovariance.sample_correlation_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto rcm = running_covariance_build(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return rcm.sample_correlation()(i, j);
        }},
        {"RunningCovariance.population_covariance_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto rcm = running_covariance_build(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return rcm.population_covariance()(i, j);
        }},
        {"RunningCovariance.population_correlation_element", [](const std::vector<double>& a) {
            int size = static_cast<int>(a[0]);
            int num_pushes = static_cast<int>(a[1]);
            auto rcm = running_covariance_build(a, size, num_pushes);
            std::size_t base = 2 + static_cast<std::size_t>(num_pushes) * static_cast<std::size_t>(size);
            int i = static_cast<int>(a[base]);
            int j = static_cast<int>(a[base + 1]);
            return rcm.population_correlation()(i, j);
        }},
        // RunningStatistics family (args: the flat sample; see
        // fixtures/special_functions/running_statistics.json)
        {"RunningStatistics.mean", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).mean(); }},
        {"RunningStatistics.variance", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).variance(); }},
        {"RunningStatistics.standard_deviation", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).standard_deviation(); }},
        {"RunningStatistics.population_variance", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_variance(); }},
        {"RunningStatistics.population_standard_deviation", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_standard_deviation(); }},
        {"RunningStatistics.coefficient_of_variation", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).coefficient_of_variation(); }},
        {"RunningStatistics.skewness", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).skewness(); }},
        {"RunningStatistics.population_skewness", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_skewness(); }},
        {"RunningStatistics.kurtosis", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).kurtosis(); }},
        {"RunningStatistics.population_kurtosis", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).population_kurtosis(); }},
        {"RunningStatistics.minimum", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).minimum(); }},
        {"RunningStatistics.maximum", [](const std::vector<double>& a) { return bfdata::RunningStatistics(a).maximum(); }},
        {"RunningStatistics.count", [](const std::vector<double>& a) { return static_cast<double>(bfdata::RunningStatistics(a).count()); }},
        // RunningStatistics combine family (args: [n1, sample1(n1), sample2(m)] -- see
        // running_statistics_combined() above and fixtures/special_functions/running_statistics.json)
        {"RunningStatistics.combined_minimum", [](const std::vector<double>& a) { return running_statistics_combined(a).minimum(); }},
        {"RunningStatistics.combined_maximum", [](const std::vector<double>& a) { return running_statistics_combined(a).maximum(); }},
        {"RunningStatistics.combined_mean", [](const std::vector<double>& a) { return running_statistics_combined(a).mean(); }},
        {"RunningStatistics.combined_variance", [](const std::vector<double>& a) { return running_statistics_combined(a).variance(); }},
        {"RunningStatistics.combined_standard_deviation", [](const std::vector<double>& a) { return running_statistics_combined(a).standard_deviation(); }},
        {"RunningStatistics.combined_coefficient_of_variation", [](const std::vector<double>& a) { return running_statistics_combined(a).coefficient_of_variation(); }},
        {"RunningStatistics.combined_skewness", [](const std::vector<double>& a) { return running_statistics_combined(a).skewness(); }},
        {"RunningStatistics.combined_kurtosis", [](const std::vector<double>& a) { return running_statistics_combined(a).kurtosis(); }},
        // Fourier family (see fourier_*_at() above for the args conventions)
        {"Fourier.fft_at", fourier_fft_at},
        {"Fourier.real_fft_at", fourier_real_fft_at},
        {"Fourier.correlation_at", fourier_correlation_at},
        {"Fourier.autocorrelation_at", fourier_autocorrelation_at},
        // NumericalDerivative family (closed function registry; see
        // numerical_derivative_{quadratic,normal_loglik} and the args convention above)
        {"NumericalDerivative.gradient_element_quadratic", [](const std::vector<double>& a) {
            return numerical_derivative_gradient_element(numerical_derivative_quadratic, a);
        }},
        {"NumericalDerivative.gradient_element_normal_loglik", [](const std::vector<double>& a) {
            return numerical_derivative_gradient_element(numerical_derivative_normal_loglik, a);
        }},
        {"NumericalDerivative.hessian_element_quadratic", [](const std::vector<double>& a) {
            return numerical_derivative_hessian_element(numerical_derivative_quadratic, a);
        }},
        {"NumericalDerivative.hessian_element_normal_loglik", [](const std::vector<double>& a) {
            return numerical_derivative_hessian_element(numerical_derivative_normal_loglik, a);
        }},
        // DifferentialEvolution family (see differential_evolution_best_value() above and
        // fixtures/special_functions/differential_evolution.json for the args convention)
        {"DifferentialEvolution.best_value", differential_evolution_best_value},
        // Histogram family (args: [explicit_bins, data..., trailing probe?] -- see
        // histogram_build() above and fixtures/special_functions/histogram.json)
        {"Histogram.number_of_bins", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build(a, 0).number_of_bins());
        }},
        {"Histogram.bin_width", [](const std::vector<double>& a) { return histogram_build(a, 0).bin_width(); }},
        {"Histogram.lower_bound", [](const std::vector<double>& a) { return histogram_build(a, 0).lower_bound(); }},
        {"Histogram.upper_bound", [](const std::vector<double>& a) { return histogram_build(a, 0).upper_bound(); }},
        {"Histogram.data_count", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build(a, 0).data_count());
        }},
        {"Histogram.mean", [](const std::vector<double>& a) { return histogram_build(a, 0).mean(); }},
        {"Histogram.median", [](const std::vector<double>& a) { return histogram_build(a, 0).median(); }},
        {"Histogram.mode", [](const std::vector<double>& a) { return histogram_build(a, 0).mode(); }},
        {"Histogram.standard_deviation", [](const std::vector<double>& a) {
            return histogram_build(a, 0).standard_deviation();
        }},
        {"Histogram.bin_lower_bound_at", [](const std::vector<double>& a) {
            return histogram_build(a, 1).bin(static_cast<int>(a.back())).lower_bound;
        }},
        {"Histogram.bin_upper_bound_at", [](const std::vector<double>& a) {
            return histogram_build(a, 1).bin(static_cast<int>(a.back())).upper_bound;
        }},
        {"Histogram.bin_frequency_at", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build(a, 1).bin(static_cast<int>(a.back())).frequency);
        }},
        {"Histogram.get_bin_index_of", [](const std::vector<double>& a) {
            return static_cast<double>(histogram_build(a, 1).get_bin_index_of(a.back()));
        }},
        // PlottingPositions family (args: [N, alpha, i] for function_at; [N, i] for
        // weibull_at -- see fixtures/special_functions/plotting_positions.json)
        {"PlottingPositions.function_at", [](const std::vector<double>& a) {
            auto pp = bfdata::plotting_positions::function(static_cast<int>(a[0]), a[1]);
            return pp[static_cast<std::size_t>(static_cast<int>(a[2]))];
        }},
        {"PlottingPositions.weibull_at", [](const std::vector<double>& a) {
            auto pp = bfdata::plotting_positions::weibull(static_cast<int>(a[0]));
            return pp[static_cast<std::size_t>(static_cast<int>(a[1]))];
        }},
        // Search family (args: [values..., x, start] -- see
        // fixtures/special_functions/search.json)
        {"Search.sequential", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> values(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            return static_cast<double>(
                bfdata::search::sequential(a[n], values, static_cast<int>(a[n + 1])));
        }},
        {"Search.bisection", [](const std::vector<double>& a) {
            std::size_t n = a.size() - 2;
            std::vector<double> values(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n));
            return static_cast<double>(
                bfdata::search::bisection(a[n], values, static_cast<int>(a[n + 1])));
        }},
        // MCMCDiagnostics.MinimumSampleSize (args: [quantile, tolerance, probability] --
        // see fixtures/special_functions/mcmc_diagnostics.json)
        {"MCMCDiagnostics.minimum_sample_size", [](const std::vector<double>& a) {
            return static_cast<double>(
                bestfit::numerics::sampling::mcmc::minimum_sample_size(a[0], a[1], a[2]));
        }},
    };
    return t;
}

// Most special_function fixtures dispatch every case through one file-level `target`
// (e.g. "Erf.function"). The Cholesky fixture instead groups several related dispatch
// keys ("Cholesky.determinant", "Cholesky.inverse_element", ...) in one file, so each
// case may override the target; a case without its own `target` falls back to the
// file-level one, preserving the single-target files' existing behavior unchanged.
static void run_special_function(const json& spec) {
    std::string file_target = spec["target"].get<std::string>();
    const auto& table = special_function_table();
    for (const auto& c : spec["cases"]) {
        std::string target = c.value("target", file_target);
        auto it = table.find(target);
        if (it == table.end())
            throw std::runtime_error("unknown special-function target: " + target);
        const auto& fn = it->second;
        std::string name = c["name"].get<std::string>();
        std::vector<double> args;
        for (const auto& v : c["args"]) args.push_back(parse_num(v));
        double actual = fn(args);
        for (const auto& as : c["assertions"]) {
            std::string where = target + "/" + name + "/" + as["method"].get<std::string>();
            check_value(actual, as, where);
        }
    }
}

// --- Generic polymorphic path ----------------------------------------------------------

static dist::ParameterEstimationMethod parse_pe_method(const std::string& m) {
    if (m == "mom") return dist::ParameterEstimationMethod::MethodOfMoments;
    if (m == "lmom") return dist::ParameterEstimationMethod::MethodOfLinearMoments;
    return dist::ParameterEstimationMethod::MaximumLikelihood;
}

static std::unique_ptr<dist::UnivariateDistributionBase> build_generic(const std::string& target,
                                                                       const json& construct,
                                                                       const json& datasets) {
    auto d = dist::create_distribution(target);
    if (construct.contains("params")) {
        std::vector<double> p;
        for (const auto& v : construct["params"]) p.push_back(parse_num(v));
        d->set_parameters(p);
        return d;
    }
    const auto& fit = construct["fit"];
    std::vector<double> data;
    for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
    auto* est = dynamic_cast<dist::IEstimation*>(d.get());
    if (est == nullptr) throw std::runtime_error(target + " does not support estimation");
    est->estimate(data, parse_pe_method(fit["method"].get<std::string>()));
    return d;
}

// Parses a "dependency" fixture string into a Probability::DependencyType (CompetingRisks).
static prob::DependencyType parse_dependency(const std::string& d) {
    if (d == "Independent") return prob::DependencyType::Independent;
    if (d == "PerfectlyPositive") return prob::DependencyType::PerfectlyPositive;
    if (d == "PerfectlyNegative") return prob::DependencyType::PerfectlyNegative;
    if (d == "CorrelationMatrix") return prob::DependencyType::CorrelationMatrix;
    throw std::runtime_error("unknown dependency type: " + d);
}

// --- Composite distribution path (TruncatedDistribution, and future Empirical/Kernel/Mixture/CR) ---

// build_component: create a sub-distribution from {"target": "...", "params": [...]} (or "fit").
// Recursive -- components can nest (Mixture inside CompetingRisks, etc.).
static std::unique_ptr<dist::UnivariateDistributionBase> build_component(const json& desc,
                                                                          const json& datasets) {
    std::string target = desc["target"].get<std::string>();
    auto d = dist::create_distribution(target);
    if (desc.contains("params")) {
        std::vector<double> p;
        for (const auto& v : desc["params"]) p.push_back(parse_num(v));
        d->set_parameters(p);
    } else if (desc.contains("fit")) {
        const auto& fit = desc["fit"];
        std::vector<double> data;
        for (const auto& v : datasets[fit["dataset"].get<std::string>()]) data.push_back(v.get<double>());
        auto* est = dynamic_cast<dist::IEstimation*>(d.get());
        if (est == nullptr) throw std::runtime_error(target + " does not support estimation");
        est->estimate(data, parse_pe_method(fit["method"].get<std::string>()));
    }
    return d;
}

// build_composite: switch on composite targets; returns a UnivariateDistributionBase* so
// dispatch_generic can be reused without modification for pdf/cdf/moments/etc.
// Future composites (KernelDensity, Mixture, CompetingRisks) add a case here.
static std::unique_ptr<dist::UnivariateDistributionBase> build_composite(const std::string& target,
                                                                          const json& construct,
                                                                          const json& datasets) {
    if (target == "TruncatedDistribution") {
        auto base = build_component(construct["base"], datasets);
        const auto& bounds = construct["bounds"];
        double lo = parse_num(bounds[0]);
        double hi = parse_num(bounds[1]);
        return std::make_unique<dist::TruncatedDistribution>(std::move(base), lo, hi);
    }
    if (target == "Empirical") {
        std::vector<double> xv, pv;
        for (const auto& v : construct["x"]) xv.push_back(parse_num(v));
        for (const auto& v : construct["p"]) pv.push_back(parse_num(v));
        auto pt = dist::EmpiricalTransform::NormalZ;
        if (construct.contains("p_transform")) {
            std::string t = construct["p_transform"].get<std::string>();
            if (t == "None") pt = dist::EmpiricalTransform::None;
            else if (t == "NormalZ") pt = dist::EmpiricalTransform::NormalZ;
            else throw std::runtime_error("unknown p_transform: " + t);
        }
        return std::make_unique<dist::EmpiricalDistribution>(std::move(xv), std::move(pv), pt);
    }
    if (target == "KernelDensity") {
        const auto& ds_name = construct["data"].get<std::string>();
        std::vector<double> data;
        for (const auto& v : datasets[ds_name]) data.push_back(v.get<double>());
        std::string kernel_str = "Gaussian";
        if (construct.contains("kernel")) kernel_str = construct["kernel"].get<std::string>();
        dist::KernelType kt = dist::KernelType::Gaussian;
        if      (kernel_str == "Epanechnikov") kt = dist::KernelType::Epanechnikov;
        else if (kernel_str == "Gaussian")     kt = dist::KernelType::Gaussian;
        else if (kernel_str == "Triangular")   kt = dist::KernelType::Triangular;
        else if (kernel_str == "Uniform")      kt = dist::KernelType::Uniform;
        else throw std::runtime_error("unknown kernel type: " + kernel_str);
        std::unique_ptr<dist::KernelDensity> kde;
        if (construct.contains("bandwidth"))
            kde = std::make_unique<dist::KernelDensity>(std::move(data), kt,
                                                        construct["bandwidth"].get<double>());
        else
            kde = std::make_unique<dist::KernelDensity>(std::move(data), kt);
        if (construct.contains("bounded_by_data"))
            kde->set_bounded_by_data(construct["bounded_by_data"].get<bool>());
        return kde;
    }
    if (target == "Mixture") {
        const auto& wts_json = construct["weights"];
        const auto& comps_json = construct["components"];
        std::vector<double> wts;
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        for (const auto& w : wts_json) wts.push_back(w.get<double>());
        for (const auto& c : comps_json) comps.push_back(build_component(c, datasets));
        return std::make_unique<dist::Mixture>(std::move(wts), std::move(comps));
    }
    if (target == "CompetingRisks") {
        const auto& comps_json = construct["components"];
        std::vector<std::unique_ptr<dist::UnivariateDistributionBase>> comps;
        for (const auto& c : comps_json) comps.push_back(build_component(c, datasets));
        auto cr = std::make_unique<dist::CompetingRisks>(std::move(comps));
        if (construct.contains("minimum_of_random_variables"))
            cr->minimum_of_random_variables = construct["minimum_of_random_variables"].get<bool>();
        if (construct.contains("dependency"))
            cr->dependency = parse_dependency(construct["dependency"].get<std::string>());
        if (construct.contains("correlation")) {
            prob::Matrix2D corr;
            for (const auto& row : construct["correlation"]) {
                std::vector<double> r;
                for (const auto& v : row) r.push_back(parse_num(v));
                corr.push_back(std::move(r));
            }
            cr->set_correlation_matrix(std::move(corr));
        }
        return cr;
    }
    throw std::runtime_error("unknown composite target: " + target);
}

static bool is_composite_target(const std::string& target) {
    return target == "TruncatedDistribution" || target == "Empirical"
        || target == "KernelDensity" || target == "Mixture" || target == "CompetingRisks";
}

static double dispatch_generic(const dist::UnivariateDistributionBase& d, const std::string& m,
                               const json& a) {
    if (m == "mean") return d.mean();
    if (m == "median") return d.median();
    if (m == "mode") return d.mode();
    if (m == "sd") return d.standard_deviation();
    if (m == "skewness") return d.skewness();
    if (m == "kurtosis") return d.kurtosis();
    if (m == "minimum") return d.minimum();
    if (m == "maximum") return d.maximum();
    if (m == "pdf") return d.pdf(a[0].get<double>());
    if (m == "log_pdf") return d.log_pdf(a[0].get<double>());
    if (m == "cdf") return d.cdf(a[0].get<double>());
    if (m == "quantile") return d.inverse_cdf(a[0].get<double>());
    if (m == "param") return d.get_parameters()[a[0].get<int>()];
    if (m == "linear_moment") {
        const auto* lm = dynamic_cast<const dist::ILinearMomentEstimation*>(&d);
        if (lm == nullptr) throw std::runtime_error("distribution has no L-moments");
        return lm->linear_moments_from_parameters(d.get_parameters())[a[0].get<int>()];
    }
    throw std::runtime_error("unknown fixture method: " + m);
}

static void run_generic(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    bool composite = is_composite_target(target);
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        // Composite targets use build_composite; flat-param targets use build_generic.
        // dispatch_generic works for both since TruncatedDistribution is a UnivariateDistributionBase.
        std::unique_ptr<dist::UnivariateDistributionBase> d =
            composite ? build_composite(target, c["construct"], datasets)
                      : build_generic(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(d->parameters_valid(), as, where);
            else
                check_value(dispatch_generic(*d, method, args), as, where);
        }
    }
}

// --- goodness_of_fit path -------------------------------------------------------------

namespace gof = bestfit::numerics::data;

static double dispatch_gof(const std::string& fn, const std::vector<double>& args,
                            const std::vector<double>& obs, const std::vector<double>& mod) {
    if (fn == "AIC")  return gof::GoodnessOfFit::aic(static_cast<int>(args[0]), args[1]);
    if (fn == "AICc") return gof::GoodnessOfFit::aicc(static_cast<int>(args[0]),
                                                       static_cast<int>(args[1]), args[2]);
    if (fn == "BIC")  return gof::GoodnessOfFit::bic(static_cast<int>(args[0]),
                                                      static_cast<int>(args[1]), args[2]);
    if (fn == "MSE")  return gof::GoodnessOfFit::mse(obs, mod);
    if (fn == "MAE")  return gof::GoodnessOfFit::mae(obs, mod);
    if (fn == "NashSutcliffeEfficiency")  return gof::GoodnessOfFit::nash_sutcliffe_efficiency(obs, mod);
    if (fn == "KlingGuptaEfficiency")     return gof::GoodnessOfFit::kling_gupta_efficiency(obs, mod);
    if (fn == "KlingGuptaEfficiencyMod")  return gof::GoodnessOfFit::kling_gupta_efficiency_mod(obs, mod);
    if (fn == "PBIAS")                    return gof::GoodnessOfFit::pbias(obs, mod);
    if (fn == "RSR")                      return gof::GoodnessOfFit::rsr(obs, mod);
    if (fn == "IndexOfAgreement")         return gof::GoodnessOfFit::index_of_agreement(obs, mod);
    if (fn == "ModifiedIndexOfAgreement") return gof::GoodnessOfFit::modified_index_of_agreement(obs, mod);
    if (fn == "RefinedIndexOfAgreement")  return gof::GoodnessOfFit::refined_index_of_agreement(obs, mod);
    if (fn == "VolumetricEfficiency")     return gof::GoodnessOfFit::volumetric_efficiency(obs, mod);
    throw std::runtime_error("unknown goodness_of_fit function: " + fn);
}

static void run_goodness_of_fit(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        std::string name = c["name"].get<std::string>();
        std::string fn = c["function"].get<std::string>();
        std::vector<double> args;
        if (c.contains("args"))
            for (const auto& v : c["args"]) args.push_back(parse_num(v));
        std::vector<double> obs, mod;
        if (c.contains("observed_dataset"))
            for (const auto& v : datasets[c["observed_dataset"].get<std::string>()])
                obs.push_back(v.get<double>());
        if (c.contains("modeled_dataset"))
            for (const auto& v : datasets[c["modeled_dataset"].get<std::string>()])
                mod.push_back(v.get<double>());
        double actual = dispatch_gof(fn, args, obs, mod);
        for (const auto& as : c["assertions"]) {
            std::string where = "gof/" + name;
            check_value(actual, as, where);
        }
    }
}

// --- multivariate_distribution path -----------------------------------------------------
//
// Mirrors the univariate build_generic/dispatch_generic split, but multivariate targets
// have no shared arithmetic surface beyond dimension/pdf/log_pdf/cdf/parameters_valid (no
// factory, no common Mean/Variance/Covariance signature across Dirichlet/Multinomial/
// BivariateEmpirical), so dispatch_multivariate dynamic_casts to the concrete type for
// everything else. Extensible: additional multivariate targets add a case to each of
// build_multivariate/dispatch_multivariate.

static std::vector<double> parse_num_vec(const json& arr) {
    std::vector<double> v;
    for (const auto& e : arr) v.push_back(parse_num(e));
    return v;
}

static std::unique_ptr<dist::MultivariateDistribution> build_multivariate(const std::string& target,
                                                                           const json& construct) {
    if (target == "Dirichlet") {
        return std::make_unique<dist::Dirichlet>(parse_num_vec(construct["alpha"]));
    }
    if (target == "Multinomial") {
        int n = construct["n"].get<int>();
        return std::make_unique<dist::Multinomial>(n, parse_num_vec(construct["p"]));
    }
    if (target == "BivariateEmpirical") {
        std::vector<double> x1 = parse_num_vec(construct["x1"]);
        std::vector<double> x2 = parse_num_vec(construct["x2"]);
        std::vector<std::vector<double>> p;
        for (const auto& row : construct["p"]) p.push_back(parse_num_vec(row));
        auto parse_transform = [&](const char* key) {
            bfdata::Transform t = bfdata::Transform::None;
            if (!construct.contains(key)) return t;
            std::string s = construct[key].get<std::string>();
            if (s == "None") t = bfdata::Transform::None;
            else if (s == "Logarithmic") t = bfdata::Transform::Logarithmic;
            else if (s == "NormalZ") t = bfdata::Transform::NormalZ;
            else throw std::runtime_error("unknown transform: " + s);
            return t;
        };
        return std::make_unique<dist::BivariateEmpirical>(
            std::move(x1), std::move(x2), std::move(p), parse_transform("x1_transform"),
            parse_transform("x2_transform"), parse_transform("p_transform"));
    }
    if (target == "MultivariateNormal") {
        std::vector<double> mean = parse_num_vec(construct["mean"]);
        std::vector<std::vector<double>> cov;
        for (const auto& row : construct["covariance"]) cov.push_back(parse_num_vec(row));
        auto mvn = std::make_unique<dist::MultivariateNormal>(std::move(mean), std::move(cov));
        if (construct.contains("seed")) mvn->set_mvnuni_seed(construct["seed"].get<int>());
        if (construct.contains("max_evaluations")) mvn->set_max_evaluations(construct["max_evaluations"].get<int>());
        if (construct.contains("abs_error")) mvn->set_absolute_error(construct["abs_error"].get<double>());
        if (construct.contains("rel_error")) mvn->set_relative_error(construct["rel_error"].get<double>());
        return mvn;
    }
    if (target == "MultivariateStudentT") {
        double df = construct["df"].get<double>();
        std::vector<double> location = parse_num_vec(construct["location"]);
        std::vector<std::vector<double>> scale;
        for (const auto& row : construct["scale"]) scale.push_back(parse_num_vec(row));
        return std::make_unique<dist::MultivariateStudentT>(df, std::move(location), std::move(scale));
    }
    throw std::runtime_error("unknown multivariate target: " + target);
}

// Shared lookup for the "random_value"/"lhs_value" seeded-sampling oracle methods, common
// to every multivariate target that implements generate_random_values (all four) /
// latin_hypercube_random_values (MultivariateNormal, MultivariateStudentT only -- see
// fixtures/README.md). args = [sample_size, seed, row, col]: construct a FRESH draw (the
// method itself seeds its own MersenneTwister from `seed`, so this is stateless -- no
// persistent-instance batching needed, unlike MultivariateNormal's MVNUNI-seeded cdf/
// interval/mvndst path above).
template <typename Dist>
static double random_value_at(const Dist& d, const json& a) {
    auto sample = d.generate_random_values(a[0].get<int>(), a[1].get<int>());
    return sample[static_cast<std::size_t>(a[2].get<int>())][static_cast<std::size_t>(a[3].get<int>())];
}

template <typename Dist>
static double lhs_value_at(const Dist& d, const json& a) {
    auto sample = d.latin_hypercube_random_values(a[0].get<int>(), a[1].get<int>());
    return sample[static_cast<std::size_t>(a[2].get<int>())][static_cast<std::size_t>(a[3].get<int>())];
}

static double dispatch_multivariate(const dist::MultivariateDistribution& d, const std::string& target,
                                    const std::string& m, const json& a) {
    if (m == "dimension") return d.dimension();
    if (m == "pdf") return d.pdf(parse_num_vec(a[0]));
    if (m == "log_pdf") return d.log_pdf(parse_num_vec(a[0]));
    if (m == "cdf") return d.cdf(parse_num_vec(a[0]));

    if (target == "Dirichlet") {
        const auto& dd = dynamic_cast<const dist::Dirichlet&>(d);
        if (m == "alpha") return dd.alpha()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "alpha_sum") return dd.alpha_sum();
        if (m == "mean") return dd.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return dd.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "mode") return dd.mode()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return dd.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "log_multivariate_beta") return dist::Dirichlet::log_multivariate_beta(parse_num_vec(a));
        if (m == "random_value") return random_value_at(dd, a);
    } else if (target == "Multinomial") {
        const auto& mm = dynamic_cast<const dist::Multinomial&>(d);
        if (m == "number_of_trials") return mm.number_of_trials();
        if (m == "mean") return mm.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return mm.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return mm.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "random_value") return random_value_at(mm, a);
    } else if (target == "BivariateEmpirical") {
        const auto& bb = dynamic_cast<const dist::BivariateEmpirical&>(d);
        if (m == "cdf_xy") return bb.cdf(a[0].get<double>(), a[1].get<double>());
    } else if (target == "MultivariateNormal") {
        const auto& nn = dynamic_cast<const dist::MultivariateNormal&>(d);
        if (m == "mean") return nn.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "median") return nn.median()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "mode") return nn.mode()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "sd") return nn.standard_deviation()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return nn.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return nn.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "mahalanobis") return nn.mahalanobis(parse_num_vec(a[0]));
        if (m == "inverse_cdf") return nn.inverse_cdf(parse_num_vec(a[0]))[static_cast<std::size_t>(a[1].get<int>())];
        if (m == "interval") return nn.interval(parse_num_vec(a[0]), parse_num_vec(a[1]));
        if (m == "random_value") return random_value_at(nn, a);
        if (m == "lhs_value") return lhs_value_at(nn, a);
        if (m == "mvndst") {
            // args = [n, [lower...], [upper...], [infin...], [correl...], maxpts, abseps, releps]
            int n = a[0].get<int>();
            std::vector<double> lower = parse_num_vec(a[1]);
            std::vector<double> upper = parse_num_vec(a[2]);
            std::vector<int> infin;
            for (const auto& v : a[3]) infin.push_back(v.get<int>());
            std::vector<double> correl = parse_num_vec(a[4]);
            int maxpts = a[5].get<int>();
            double abseps = a[6].get<double>();
            double releps = a[7].get<double>();
            double error = 0, value = 0;
            int inform = 0;
            nn.mvndst(n, lower, upper, infin, correl, maxpts, abseps, releps, error, value, inform);
            return value;
        }
    } else if (target == "MultivariateStudentT") {
        const auto& tt = dynamic_cast<const dist::MultivariateStudentT&>(d);
        if (m == "degrees_of_freedom") return tt.degrees_of_freedom();
        if (m == "mean") return tt.mean()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "median") return tt.median()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "mode") return tt.mode()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "sd") return tt.standard_deviation()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "variance") return tt.variance()[static_cast<std::size_t>(a[0].get<int>())];
        if (m == "covariance") return tt.covariance(a[0].get<int>(), a[1].get<int>());
        if (m == "mahalanobis") return tt.mahalanobis(parse_num_vec(a[0]));
        if (m == "inverse_cdf") return tt.inverse_cdf(parse_num_vec(a[0]))[static_cast<std::size_t>(a[1].get<int>())];
        if (m == "random_value") return random_value_at(tt, a);
        if (m == "lhs_value") return lhs_value_at(tt, a);
    }
    throw std::runtime_error("unknown multivariate fixture method: " + target + "/" + m);
}

static void run_multivariate(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    for (const auto& c : spec["cases"]) {
        auto d = build_multivariate(target, c["construct"]);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(d->parameters_valid(), as, where);
            else
                check_value(dispatch_multivariate(*d, target, method, args), as, where);
        }
    }
}

// --- bivariate_copula path --------------------------------------------------------------
//
// Every copula shares BivariateCopula's uniform theta/get_copula_parameters/pdf/cdf/...
// API (unlike multivariate_distribution, which has no such common surface across
// Dirichlet/Multinomial/BivariateEmpirical/...), so build_copula/dispatch_copula are FULLY
// generic through the factory -- no per-target branching, matching copula_factory.hpp's
// header comment. The one exception is the "tau" method-of-moments fit: SetThetaFromTau is
// a member of each concrete Archimedean class in the C# source (not part of
// IBivariateCopula/IArchimedeanCopula), so it is not callable through the BivariateCopula
// base pointer; set_theta_from_tau_dispatch dynamic_casts by target name for that one
// method, mirroring dispatch_multivariate's per-target branches above. Each new
// tau-capable copula (Task 8: AMH, Gumbel, Joe) adds one branch there.

namespace cop = bestfit::numerics::distributions::copulas;

static void set_theta_from_tau_dispatch(cop::BivariateCopula& copula, const std::string& target,
                                        const std::vector<double>& x, const std::vector<double>& y) {
    if (target == "Clayton") {
        dynamic_cast<cop::ClaytonCopula&>(copula).set_theta_from_tau(x, y);
        return;
    }
    if (target == "AliMikhailHaq") {
        dynamic_cast<cop::AMHCopula&>(copula).set_theta_from_tau(x, y);
        return;
    }
    if (target == "Gumbel") {
        dynamic_cast<cop::GumbelCopula&>(copula).set_theta_from_tau(x, y);
        return;
    }
    // NOTE: JoeCopula has no SetThetaFromTau in the C# source (see joe_copula.hpp's file
    // header) despite the Phase 2 plan/README listing it as tau-capable -- intentionally
    // not branched here.
    throw std::runtime_error("copula '" + target + "' has no tau-based method-of-moments fit");
}

// Per fixtures/README.md: construct is {"theta": x} (optionally {"theta": x, "df": y} for
// 2-parameter copulas, and/or {"marginals": {"targets": [..], "params": [[..], [..]]}} to
// attach marginals directly via the C# `Copula(theta, marginX, marginY)` ctor -- used by the
// seeded "random_value" sampling oracles, which back-transform through the marginals when
// set) or {"fit": {"x": "<dataset>", "y": "<dataset>", "method": "tau"|"mpl"|"ifm"|"mle",
// "marginals": ["Normal", "Normal"]?}}. "x"/"y" are the sample data for tau/ifm/mle, or the
// precomputed plotting-position datasets for mpl (the runner stays thin: it never computes
// plotting positions itself). "ifm" pre-fits the marginals by MLE before estimating the
// copula (mirroring the C# Test_IFM_Fit flow); "mle" leaves the marginals unfitted and lets
// BivariateCopulaEstimation.MLE (bivariate_copula_estimation.hpp) fit everything jointly.
static std::unique_ptr<cop::BivariateCopula> build_copula(const std::string& target,
                                                            const json& construct,
                                                            const json& datasets) {
    auto c = cop::create_copula(target);
    if (construct.contains("theta")) {
        std::vector<double> params = {parse_num(construct["theta"])};
        if (construct.contains("df")) params.push_back(parse_num(construct["df"]));
        c->set_copula_parameters(params);
        if (construct.contains("marginals")) {
            const auto& marg = construct["marginals"];
            auto mx = dist::create_distribution(marg["targets"][0].get<std::string>());
            auto my = dist::create_distribution(marg["targets"][1].get<std::string>());
            mx->set_parameters(parse_num_vec(marg["params"][0]));
            my->set_parameters(parse_num_vec(marg["params"][1]));
            c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
            c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
        }
        return c;
    }

    const auto& fit = construct["fit"];
    std::vector<double> x, y;
    for (const auto& v : datasets[fit["x"].get<std::string>()]) x.push_back(parse_num(v));
    for (const auto& v : datasets[fit["y"].get<std::string>()]) y.push_back(parse_num(v));
    std::string method = fit["method"].get<std::string>();

    if (fit.contains("marginals")) {
        auto mx = dist::create_distribution(fit["marginals"][0].get<std::string>());
        auto my = dist::create_distribution(fit["marginals"][1].get<std::string>());
        if (method == "ifm") {
            auto* ex = dynamic_cast<dist::IEstimation*>(mx.get());
            auto* ey = dynamic_cast<dist::IEstimation*>(my.get());
            if (ex == nullptr || ey == nullptr)
                throw std::runtime_error("marginal does not support estimation");
            ex->estimate(x, dist::ParameterEstimationMethod::MaximumLikelihood);
            ey->estimate(y, dist::ParameterEstimationMethod::MaximumLikelihood);
        }
        c->marginal_distribution_x = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(mx));
        c->marginal_distribution_y = std::shared_ptr<dist::UnivariateDistributionBase>(std::move(my));
    }

    if (method == "tau") {
        set_theta_from_tau_dispatch(*c, target, x, y);
    } else if (method == "mpl") {
        cop::estimate(*c, x, y, cop::CopulaEstimationMethod::PseudoLikelihood);
    } else if (method == "ifm") {
        cop::estimate(*c, x, y, cop::CopulaEstimationMethod::InferenceFromMargins);
    } else if (method == "mle") {
        cop::estimate(*c, x, y, cop::CopulaEstimationMethod::FullLikelihood);
    } else {
        throw std::runtime_error("unknown copula fit method: " + method);
    }
    return c;
}

static double dispatch_copula(const cop::BivariateCopula& c, const std::string& m, const json& a) {
    if (m == "pdf") return c.pdf(a[0].get<double>(), a[1].get<double>());
    if (m == "log_pdf") return c.log_pdf(a[0].get<double>(), a[1].get<double>());
    if (m == "cdf") return c.cdf(a[0].get<double>(), a[1].get<double>());
    if (m == "inverse_cdf")
        return c.inverse_cdf(a[0].get<double>(), a[1].get<double>())[static_cast<std::size_t>(a[2].get<int>())];
    if (m == "upper_tail_dependence") return c.upper_tail_dependence();
    if (m == "lower_tail_dependence") return c.lower_tail_dependence();
    if (m == "theta") return c.theta();
    if (m == "df") return c.get_copula_parameters()[1];
    if (m == "or_exceedance") return c.or_joint_exceedance_probability(a[0].get<double>(), a[1].get<double>());
    if (m == "and_exceedance") return c.and_joint_exceedance_probability(a[0].get<double>(), a[1].get<double>());
    if (m == "marginal_param") {
        std::string which = a[0].get<std::string>();
        std::size_t idx = static_cast<std::size_t>(a[1].get<int>());
        const auto& marg = which == "x" ? c.marginal_distribution_x : c.marginal_distribution_y;
        return marg->get_parameters()[idx];
    }
    if (m == "random_value") {
        // args = [sample_size, seed, row, col]. Stateless: GenerateRandomValues seeds its
        // own internal LatinHypercube draw from `seed`, so no persistent-instance batching
        // is needed (mirrors random_value_at() for multivariate_distribution above).
        auto sample = c.generate_random_values(a[0].get<int>(), a[1].get<int>());
        return sample[static_cast<std::size_t>(a[2].get<int>())][static_cast<std::size_t>(a[3].get<int>())];
    }
    throw std::runtime_error("unknown copula fixture method: " + m);
}

static void run_bivariate_copula(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto cop = build_copula(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            if (as["mode"].get<std::string>() == "bool")
                check_bool(cop->parameters_valid(), as, where);
            else
                check_value(dispatch_copula(*cop, method, args), as, where);
        }
    }
}

// --- mcmc_sampler path -------------------------------------------------------------------
//
// One sampler run per case: build the model via the registry, construct the sampler (RWMH
// today; extensible via a target-name switch as later samplers land), apply non-default
// settings, sample() ONCE, and cache both the sampler and its post-processed MCMCResults for
// every assertion in the case (mirrors the "single stateful glue call; no seq machinery"
// contract fixtures/README.md documents for this kind).

namespace mcmc = bestfit::numerics::sampling::mcmc;

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema.
// "zeros": the literal `Matrix(D)` the C# Test_RWMH.cs test constructs (safe only when
// MAP initialization is expected to override it before first use -- see "identity" below).
// "identity": D x D identity matrix. NOT present in the upstream C# test; added because a
// Randomize-initialized RWMH with a literal all-zero proposal covariance throws
// (CholeskyDecomposition rejects a non-positive-definite matrix) on its very first
// ChainIteration -- confirmed against the real C# library. Any Randomize-init fixture case
// therefore needs a non-degenerate proposal_sigma; identity is the simplest one.
static la::Matrix parse_proposal_sigma(const json& settings, int dimension) {
    if (!settings.contains("proposal_sigma")) return la::Matrix(dimension);
    std::string s = settings["proposal_sigma"].get<std::string>();
    if (s == "zeros") return la::Matrix(dimension);
    if (s == "identity") return la::Matrix::identity(dimension);
    throw std::runtime_error("unknown proposal_sigma sentinel: " + s);
}

static mcmc::MCMCSampler::InitializationType parse_initialize(const std::string& s) {
    if (s == "MAP") return mcmc::MCMCSampler::InitializationType::MAP;
    if (s == "Randomize") return mcmc::MCMCSampler::InitializationType::Randomize;
    if (s == "UserDefined") return mcmc::MCMCSampler::InitializationType::UserDefined;
    throw std::runtime_error("unknown initialize value: " + s);
}

// Builds + configures + samples() one sampler from a {"model": {...}, "settings": {...}}
// construct. `sampler_target`: the fixture's file-level "target" (the sampler type, e.g.
// "RWMH"); a later task extends this with more cases as more samplers land.
static std::unique_ptr<mcmc::MCMCSampler> build_and_sample(const std::string& sampler_target,
                                                             const json& construct, const json& datasets) {
    const auto& model_spec = construct["model"];
    std::vector<double> data;
    for (const auto& v : datasets[model_spec["dataset"].get<std::string>()]) data.push_back(parse_num(v));
    auto model = mcmc::build_model(model_spec["name"].get<std::string>(),
                                    model_spec["family"].get<std::string>(), data);
    int d = static_cast<int>(model.priors.size());

    json settings = construct.value("settings", json::object());

    std::unique_ptr<mcmc::MCMCSampler> sampler;
    if (sampler_target == "RWMH") {
        sampler = std::make_unique<mcmc::RWMH>(model.priors, model.log_likelihood,
                                                parse_proposal_sigma(settings, d));
    } else if (sampler_target == "HMC") {
        std::optional<double> step_size =
            settings.contains("step_size") ? std::optional<double>(settings["step_size"].get<double>()) : std::nullopt;
        std::optional<int> steps =
            settings.contains("steps") ? std::optional<int>(settings["steps"].get<int>()) : std::nullopt;
        sampler = std::make_unique<mcmc::HMC>(model.priors, model.log_likelihood, std::nullopt,
                                               step_size.value_or(0.1), steps.value_or(50));
    } else if (sampler_target == "NUTS") {
        std::optional<double> step_size =
            settings.contains("step_size") ? std::optional<double>(settings["step_size"].get<double>()) : std::nullopt;
        std::optional<int> max_tree_depth = settings.contains("max_tree_depth")
                                                 ? std::optional<int>(settings["max_tree_depth"].get<int>())
                                                 : std::nullopt;
        auto nuts = std::make_unique<mcmc::NUTS>(model.priors, model.log_likelihood, std::nullopt,
                                                  step_size.value_or(0.1), max_tree_depth.value_or(10));
        if (settings.contains("adapt_mass_matrix")) nuts->adapt_mass_matrix = settings["adapt_mass_matrix"].get<bool>();
        sampler = std::move(nuts);
    } else if (sampler_target == "ARWMH") {
        auto arwmh = std::make_unique<mcmc::ARWMH>(model.priors, model.log_likelihood);
        if (settings.contains("scale")) arwmh->scale = settings["scale"].get<double>();
        if (settings.contains("beta")) arwmh->beta = settings["beta"].get<double>();
        sampler = std::move(arwmh);
    } else if (sampler_target == "Gibbs") {
        if (!model.proposal) throw std::runtime_error("Gibbs model has no proposal function");
        sampler = std::make_unique<mcmc::Gibbs>(model.priors, model.log_likelihood, model.proposal);
    } else if (sampler_target == "SNIS") {
        sampler = std::make_unique<mcmc::SNIS>(model.priors, model.log_likelihood);
    } else if (sampler_target == "DEMCz") {
        auto demcz = std::make_unique<mcmc::DEMCz>(model.priors, model.log_likelihood);
        if (settings.contains("jump")) demcz->jump = settings["jump"].get<double>();
        if (settings.contains("jump_threshold")) demcz->jump_threshold = settings["jump_threshold"].get<double>();
        if (settings.contains("noise")) demcz->set_noise(settings["noise"].get<double>());
        sampler = std::move(demcz);
    } else if (sampler_target == "DEMCzs") {
        auto demczs = std::make_unique<mcmc::DEMCzs>(model.priors, model.log_likelihood);
        if (settings.contains("jump")) demczs->jump = settings["jump"].get<double>();
        if (settings.contains("jump_threshold")) demczs->jump_threshold = settings["jump_threshold"].get<double>();
        if (settings.contains("snooker_threshold"))
            demczs->snooker_threshold = settings["snooker_threshold"].get<double>();
        if (settings.contains("noise")) demczs->set_noise(settings["noise"].get<double>());
        sampler = std::move(demczs);
    } else {
        throw std::runtime_error("unknown mcmc_sampler target: " + sampler_target);
    }

    if (settings.contains("initialize")) sampler->initialize = parse_initialize(settings["initialize"].get<std::string>());
    if (settings.contains("prng_seed")) sampler->set_prng_seed(settings["prng_seed"].get<int>());
    if (settings.contains("initial_iterations")) sampler->set_initial_iterations(settings["initial_iterations"].get<int>());
    if (settings.contains("warmup_iterations")) sampler->set_warmup_iterations(settings["warmup_iterations"].get<int>());
    if (settings.contains("iterations")) sampler->set_iterations(settings["iterations"].get<int>());
    if (settings.contains("number_of_chains")) sampler->set_number_of_chains(settings["number_of_chains"].get<int>());
    if (settings.contains("thinning_interval")) sampler->set_thinning_interval(settings["thinning_interval"].get<int>());
    if (settings.contains("output_length")) sampler->output_length = settings["output_length"].get<int>();

    sampler->sample();
    return sampler;
}

static double dispatch_mcmc(const mcmc::MCMCSampler& sampler, const mcmc::MCMCResults& results,
                             const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    if (m == "posterior_mean") return results.parameter_results[idx(0)].summary_statistics.mean;
    if (m == "posterior_sd") return results.parameter_results[idx(0)].summary_statistics.standard_deviation;
    if (m == "posterior_median") return results.parameter_results[idx(0)].summary_statistics.median;
    if (m == "posterior_lower_ci") return results.parameter_results[idx(0)].summary_statistics.lower_ci;
    if (m == "posterior_upper_ci") return results.parameter_results[idx(0)].summary_statistics.upper_ci;
    if (m == "chain_value") return sampler.markov_chains()[idx(0)][idx(1)].values[idx(2)];
    if (m == "chain_fitness") return sampler.markov_chains()[idx(0)][idx(1)].fitness;
    if (m == "map_value") return results.map.values[idx(0)];
    if (m == "map_fitness") return results.map.fitness;
    if (m == "acceptance_rate") return sampler.acceptance_rates()[idx(0)];
    if (m == "mean_log_likelihood") return sampler.mean_log_likelihood()[idx(0)];
    if (m == "rhat") return results.parameter_results[idx(0)].summary_statistics.rhat;
    if (m == "ess") return results.parameter_results[idx(0)].summary_statistics.ess;
    throw std::runtime_error("unknown mcmc_sampler fixture method: " + m);
}

static void run_mcmc_sampler(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto sampler = build_and_sample(target, c["construct"], datasets);
        mcmc::MCMCResults results(*sampler);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            check_value(dispatch_mcmc(*sampler, results, method, args), as, where);
        }
    }
}

// --- bootstrap path ----------------------------------------------------------------------
//
// One bootstrap run per case (mirrors mcmc_sampler's single-stateful-glue-call contract): build
// the model via the registry, configure (replicates/seed/max_retries), run() or
// run_with_studentized_bootstrap() ONCE, then get_confidence_intervals() ONCE with the case's
// ci_method/alpha; every assertion in the case reads that single cached (bootstrap, results)
// pair. See fixtures/README.md's bootstrap schema.

static bfsamp::BootstrapCIMethod parse_ci_method(const std::string& s) {
    if (s == "Percentile") return bfsamp::BootstrapCIMethod::Percentile;
    if (s == "BiasCorrected") return bfsamp::BootstrapCIMethod::BiasCorrected;
    if (s == "BCa") return bfsamp::BootstrapCIMethod::BCa;
    if (s == "Normal") return bfsamp::BootstrapCIMethod::Normal;
    if (s == "BootstrapT") return bfsamp::BootstrapCIMethod::BootstrapT;
    throw std::runtime_error("unknown bootstrap ci_method: " + s);
}

struct BootstrapCase {
    bfsamp::Bootstrap<std::vector<double>> boot;
    bfsamp::BootstrapResults results;
};

static BootstrapCase build_and_run_bootstrap(const json& construct, const json& datasets) {
    std::string model_name = construct["model"].get<std::string>();
    double mu = construct.value("mu", 0.0);
    double sigma = construct.value("sigma", 0.0);
    int sample_size = construct.value("sample_size", 0);
    std::vector<double> probabilities;
    for (const auto& v : construct["probabilities"]) probabilities.push_back(parse_num(v));
    std::vector<double> sample_data;
    if (construct.contains("dataset"))
        for (const auto& v : datasets[construct["dataset"].get<std::string>()]) sample_data.push_back(parse_num(v));

    auto boot = bfsamp::build_bootstrap_model(model_name, mu, sigma, sample_size, probabilities, sample_data);
    if (construct.contains("replicates")) boot.replicates = construct["replicates"].get<int>();
    if (construct.contains("seed")) boot.prng_seed = construct["seed"].get<int>();
    if (construct.contains("max_retries")) boot.max_retries = construct["max_retries"].get<int>();

    std::string run = construct.value("run", "regular");
    if (run == "regular")
        boot.run();
    else if (run == "studentized")
        boot.run_with_studentized_bootstrap();
    else
        throw std::runtime_error("unknown bootstrap run kind: " + run);

    auto method = parse_ci_method(construct["ci_method"].get<std::string>());
    double alpha = construct.value("alpha", 0.1);
    bfsamp::BootstrapResults results = boot.get_confidence_intervals(method, alpha);

    return BootstrapCase{std::move(boot), std::move(results)};
}

static double dispatch_bootstrap(const BootstrapCase& bc, const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    if (m == "statistic_lower_ci") return bc.results.statistic_results[idx(0)].lower_ci;
    if (m == "statistic_upper_ci") return bc.results.statistic_results[idx(0)].upper_ci;
    if (m == "parameter_lower_ci") return bc.results.parameter_results[idx(0)].lower_ci;
    if (m == "parameter_upper_ci") return bc.results.parameter_results[idx(0)].upper_ci;
    if (m == "population_estimate") return bc.results.parameter_results[idx(0)].population_estimate;
    if (m == "valid_count") return bc.results.statistic_results[idx(0)].valid_count;
    if (m == "replicate_value") return bc.boot.bootstrap_parameter_sets()[idx(0)].values[idx(1)];
    throw std::runtime_error("unknown bootstrap fixture method: " + m);
}

static void run_bootstrap(const json& spec) {
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto bc = build_and_run_bootstrap(c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = "Bootstrap/" + name + "/" + method;
            check_value(dispatch_bootstrap(bc, method, args), as, where);
        }
    }
}

// --- model_estimation path -----------------------------------------------------------------
//
// One estimate() run per case (mirrors mcmc_sampler's/bootstrap's single-stateful-glue-call
// contract): build the model named by `construct.model` through the SHARED spec builder
// (models/model_spec.hpp -- `type` selects UnivariateDistributionModel (default, incl.
// censored DataFrames and nonstationary trend specs), MixtureModel, CompetingRisksModel, or
// PointProcessModel; there is no separate closed-name registry like mcmc/model_registry.hpp
// needs, since the spec's factory calls plus each model's default-parameter machinery are
// already enough), construct the estimator named by the file-level `target`, call
// `estimate()` ONCE, then dispatch every assertion in the case against the cached
// (model, estimator) pair. The `Simulation` target (M13) builds the model, skips the
// estimator, and caches ONE seeded ISimulatable::generate_random_values draw instead; the
// `simulated_value [i]` method asserts individual draws (the chain_value digest precedent).
// See fixtures/README.md's model_estimation section for the schema.
//
// WIRED (T11 + T12): MaximumLikelihood / MaximumAPosteriori share `parameter [p]`,
// `max_log_likelihood []`, `aic []`, `bic [n]`, `covariance [i,j]`, `standard_error [p]`,
// `correlation [i,j]` (same method names/signatures on both classes; dispatched via one
// std::visit branch). BayesianAnalysis (T12) adds `dic []`, `waic []`, `looic []`,
// `posterior_mean [p]`, and the seeded `chain_value [chain,iter,param]` digest -- a disjoint
// surface handled by a separate std::visit branch (it shares no methods with ML/MAP).
namespace estimation = bestfit::estimation;

// Shared optimizer-method parser for ML/MAP (`optimizer` knob) AND the GMM `optimizer` knob.
// B11 extends it with the B7-un-gated BFGS/Powell/MultilevelSingleLinkage methods (with the
// "MLSL" alias), so fixtures can pin them for ML/MAP and select them for a GMM fit.
static estimation::OptimizationMethod parse_optimization_method(const std::string& s) {
    if (s == "Brent") return estimation::OptimizationMethod::Brent;
    if (s == "NelderMead") return estimation::OptimizationMethod::NelderMead;
    if (s == "DifferentialEvolution") return estimation::OptimizationMethod::DifferentialEvolution;
    if (s == "BFGS") return estimation::OptimizationMethod::BFGS;
    if (s == "Powell") return estimation::OptimizationMethod::Powell;
    if (s == "MultilevelSingleLinkage" || s == "MLSL")
        return estimation::OptimizationMethod::MultilevelSingleLinkage;
    throw std::runtime_error("unknown model_estimation optimizer: " + s);
}

// The GMM estimation-strategy knob (default Iterative, matching the C# GMM default).
static estimation::GeneralizedMethodOfMoments::GMMEstimationStrategy parse_gmm_strategy(
    const std::string& s) {
    using Strat = estimation::GeneralizedMethodOfMoments::GMMEstimationStrategy;
    if (s == "OneStep") return Strat::OneStep;
    if (s == "TwoStep") return Strat::TwoStep;
    if (s == "Iterative") return Strat::Iterative;
    throw std::runtime_error("unknown GMM estimation strategy: " + s);
}

static estimation::SamplerType parse_sampler_type(const std::string& s) {
    if (s == "DEMCz") return estimation::SamplerType::DEMCz;
    if (s == "DEMCzs") return estimation::SamplerType::DEMCzs;
    if (s == "ARWMH") return estimation::SamplerType::ARWMH;
    if (s == "NUTS") return estimation::SamplerType::NUTS;
    throw std::runtime_error("unknown model_estimation sampler: " + s);
}

// Holds the model (kept alive -- every estimator stores a reference, not a copy) plus whichever
// estimator `target` selected, already `estimate()`d once. BayesianAnalysis joins the variant
// (its method surface is disjoint from ML/MAP; `dispatch_estimation` branches on the held type).
// M13: the model is now the ModelBase every estimator accepts (all four Phase 5 model types --
// UnivariateDistributionModel, MixtureModel, CompetingRisksModel (NOT an IUnivariateModel; the
// C# omits it), PointProcessModel -- derive from it), built through the SHARED spec builder
// (models/model_spec.hpp) from the serialized `construct.model` object. The `Simulation`
// target holds no estimator (std::monostate) -- just the cached seeded ISimulatable draw.
struct EstimationCase {
    std::unique_ptr<bestfit::models::ModelBase> model;
    std::variant<std::monostate, std::unique_ptr<estimation::MaximumLikelihood>,
                 std::unique_ptr<estimation::MaximumAPosteriori>,
                 std::unique_ptr<estimation::BayesianAnalysis>,
                 std::unique_ptr<estimation::GeneralizedMethodOfMoments>>
        estimator;
    std::vector<double> simulated;  // Simulation target, and the GMM seeded-draw digest
    // GMM target only: the concrete B17C model the estimator references (NOT a ModelBase, so it
    // cannot live in `model`). Kept alive here so the estimator's IGMMModel& stays valid and the
    // quantile_variance arm can reach the model.
    std::unique_ptr<bestfit::models::Bulletin17CDistribution> b17c;
};

// Seeded ISimulatable draw, flattened to a 1-D vector so the `simulated_value [i]` digest works
// uniformly across model types. Most Phase 4-7 models are ISimulatable<std::vector<double>> and
// pass through unchanged; BivariateDistribution is ISimulatable<Matrix2D> (n-row x 2-col), so its
// draw is flattened ROW-MAJOR (i = row*2 + col) -- the same order the R/Python glue and the README
// schema use. Throws if the model is neither.
static std::vector<double> simulate_flat(bestfit::models::ModelBase* model, int sample_size,
                                         int seed) {
    if (auto* s = dynamic_cast<bestfit::models::ISimulatable<std::vector<double>>*>(model))
        return s->generate_random_values(sample_size, seed);
    if (auto* s = dynamic_cast<
            bestfit::models::ISimulatable<std::vector<std::vector<double>>>*>(model)) {
        std::vector<std::vector<double>> mat = s->generate_random_values(sample_size, seed);
        std::vector<double> flat;
        for (const auto& row : mat)
            for (double v : row) flat.push_back(v);
        return flat;
    }
    throw std::runtime_error(
        "model_estimation Simulation target: model is not ISimulatable<vector> or "
        "ISimulatable<Matrix2D>");
}

static EstimationCase build_and_run_estimation(const std::string& target, const json& construct,
                                                 const json& datasets) {
    const auto& model_spec = construct["model"];
    std::vector<double> data;
    if (model_spec.contains("dataset"))
        for (const auto& v : datasets[model_spec["dataset"].get<std::string>()]) data.push_back(parse_num(v));

    // GMM builds the CONCRETE Bulletin17CDistribution (not a ModelBase -- see model_spec.hpp's
    // build_bulletin17c_model wiring note) and fits it, optionally caching a seeded draw from the
    // fitted model for the `simulated_value` digest (the DRY choice: `simulated_value` is already
    // dispatched from ec.simulated for every target, so riding the GMM case needs no new arm).
    if (target == "GeneralizedMethodOfMoments") {
        auto b17c = bestfit::models::spec::build_bulletin17c_from_json(model_spec.dump(), data);
        auto method = construct.contains("optimizer")
                          ? parse_optimization_method(construct["optimizer"].get<std::string>())
                          : estimation::OptimizationMethod::BFGS;
        auto gmm = std::make_unique<estimation::GeneralizedMethodOfMoments>(*b17c, method);
        if (construct.contains("strategy"))
            gmm->set_estimation_strategy(parse_gmm_strategy(construct["strategy"].get<std::string>()));
        if (construct.contains("max_gmm_iterations"))
            gmm->set_max_gmm_iterations(construct["max_gmm_iterations"].get<int>());
        if (!gmm->estimate())
            throw std::runtime_error("GeneralizedMethodOfMoments::estimate() failed for a fixture case");
        // post_process(sandwich, jstat) caches Sigma + the J-statistic so the accessors return
        // deterministic cached values.
        gmm->post_process(/*use_sandwich=*/true, /*compute_jstat=*/true);
        std::vector<double> draws;
        if (construct.contains("sample_size")) {
            // Draw from the FITTED model: pin the estimator's best parameters into the B17C
            // parent, then take a seeded ISimulatable stream (deterministic across harnesses).
            b17c->set_parameter_values(gmm->best_parameter_set().values);
            draws = b17c->generate_random_values(construct["sample_size"].get<int>(),
                                                 construct.value("seed", -1));
        }
        return EstimationCase{nullptr, std::move(gmm), std::move(draws), std::move(b17c)};
    }

    // One shared construction path for all three harnesses: serialize the spec back to JSON
    // and hand it to models/model_spec.hpp (see that header for the schema).
    auto model = bestfit::models::spec::build_model_from_json(model_spec.dump(), data);

    if (target == "Simulation") {
        std::vector<double> draws = simulate_flat(model.get(), construct["sample_size"].get<int>(),
                                                  construct.value("seed", -1));
        return EstimationCase{std::move(model), std::monostate{}, std::move(draws)};
    }
    if (target == "MaximumLikelihood" || target == "MaximumAPosteriori") {
        auto method = construct.contains("optimizer")
                          ? parse_optimization_method(construct["optimizer"].get<std::string>())
                          : estimation::OptimizationMethod::DifferentialEvolution;
        // Optional seeded-draw digest off the FITTED model (P3): when `sample_size` is present,
        // pin the estimator's best parameters back into the model and cache one seeded draw --
        // the same shared `simulated_value` arm the Simulation/GMM targets use, letting one MLE
        // smoke file cover parameter + max_log_likelihood + a seeded draw for the new families.
        auto cache_draw = [&](bestfit::models::ModelBase& fitted, const std::vector<double>& best) {
            std::vector<double> draws;
            if (construct.contains("sample_size")) {
                fitted.set_parameter_values(best);
                draws = simulate_flat(&fitted, construct["sample_size"].get<int>(),
                                      construct.value("seed", -1));
            }
            return draws;
        };
        if (target == "MaximumLikelihood") {
            auto est = std::make_unique<estimation::MaximumLikelihood>(*model, method);
            if (!est->estimate())
                throw std::runtime_error("MaximumLikelihood::estimate() failed for a fixture case");
            auto draws = cache_draw(*model, est->best_parameter_set().values);
            return EstimationCase{std::move(model), std::move(est), std::move(draws)};
        }
        auto est = std::make_unique<estimation::MaximumAPosteriori>(*model, method);
        if (!est->estimate())
            throw std::runtime_error("MaximumAPosteriori::estimate() failed for a fixture case");
        auto draws = cache_draw(*model, est->best_parameter_set().values);
        return EstimationCase{std::move(model), std::move(est), std::move(draws)};
    }
    if (target == "BayesianAnalysis") {
        // Mirrors the oracle emitter's BuildEstimation: construct with the fixture's sampler type
        // (defaulting to DEMCzs), turn OFF the two "use defaults" flags so the explicit knobs
        // below aren't clobbered, apply the settings, then estimate() once. The seeded chain
        // digest reproduces bit-identically against the real C# (see bayes_normal.json).
        auto sampler_type = construct.contains("sampler")
                                ? parse_sampler_type(construct["sampler"].get<std::string>())
                                : estimation::SamplerType::DEMCzs;
        auto ba = std::make_unique<estimation::BayesianAnalysis>(*model, sampler_type);
        ba->set_use_simulation_defaults(false);
        ba->set_use_advanced_simulation_defaults(false);
        if (construct.contains("settings")) {
            const auto& s = construct["settings"];
            if (s.contains("seed")) ba->set_prng_seed(s["seed"].get<int>());
            if (s.contains("iterations")) ba->set_iterations(s["iterations"].get<int>());
            if (s.contains("warmup_iterations")) ba->set_warmup_iterations(s["warmup_iterations"].get<int>());
            if (s.contains("number_of_chains")) ba->set_number_of_chains(s["number_of_chains"].get<int>());
            if (s.contains("thinning_interval")) ba->set_thinning_interval(s["thinning_interval"].get<int>());
            if (s.contains("initial_iterations")) ba->set_initial_iterations(s["initial_iterations"].get<int>());
            if (s.contains("output_length")) ba->set_output_length(s["output_length"].get<int>());
        }
        if (!ba->estimate())
            throw std::runtime_error("BayesianAnalysis::estimate() failed for a fixture case");
        return EstimationCase{std::move(model), std::move(ba)};
    }
    throw std::runtime_error("unknown model_estimation target: " + target);
}

// The DataFrame assertion surface (M14): methods reachable from the model's DataFrame under
// ANY model_estimation target, corroborating the M1/M5 ctest oracles through the PUBLIC path.
// `plotting_position [kind, i]` reads item i's plotting position from the named series
// ("exact" | "interval" | "uncertain", in spec order) after ONE calculate_plotting_positions()
// pass (idempotent -- a pure function of the collections + the plotting parameter; the
// threshold series is NOT exposed because the C# assigns its positions to a sorted CLONE, so
// the original items never carry one). `number_of_low_outliers`/`low_outlier_threshold` read
// the frame's current state (set by the spec's `mgbt_low_outliers` MGBT trigger, or the
// explicit `low_outlier_threshold`).
static double dispatch_model_data_frame(bestfit::models::ModelBase& model, const std::string& m,
                                        const json& a) {
    auto* udm = dynamic_cast<bestfit::models::UnivariateDistributionModelBase*>(&model);
    if (udm == nullptr || !udm->has_data_frame())
        throw std::runtime_error("model_estimation data-frame method on a model without a DataFrame");
    auto& df = udm->data_frame();
    if (m == "number_of_low_outliers") return df.number_of_low_outliers();
    if (m == "low_outlier_threshold") return df.low_outlier_threshold();
    // plotting_position [kind, i]
    df.calculate_plotting_positions();
    std::string kind = a[0].get<std::string>();
    std::size_t i = static_cast<std::size_t>(a[1].get<int>());
    if (kind == "exact") return df.exact_series()[i].plotting_position();
    if (kind == "interval") return df.interval_series()[i].plotting_position();
    if (kind == "uncertain") return df.uncertain_series()[i].plotting_position();
    throw std::runtime_error("unknown plotting_position series kind: " + kind);
}

// GMM shares an accessor family with ML/MAP but its accessors are non-const (they cache Sigma
// on demand), so dispatch takes a non-const EstimationCase& (the caller's `ec` is a non-const
// local). quantile_variance lives on the B17C MODEL, not the estimator, so its arm recovers the
// concrete model from ec and feeds it the fitted parameters + the estimator's covariance.
static double dispatch_estimation(EstimationCase& ec, const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    // The seeded-simulation digest (M13): reads the vector cached at build time, so it works
    // for the Simulation target (no estimator) without touching the variant.
    if (m == "simulated_value") return ec.simulated.at(idx(0));
    // The M14 DataFrame surface works under any target (it reads the model, not the estimator).
    if (m == "plotting_position" || m == "number_of_low_outliers" || m == "low_outlier_threshold")
        return dispatch_model_data_frame(*ec.model, m, a);
    return std::visit(
        [&](auto& est) -> double {
            using Held = std::decay_t<decltype(est)>;
            if constexpr (std::is_same_v<Held, std::monostate>) {
                throw std::runtime_error("unknown Simulation fixture method: " + m);
            } else {
                using Estimator = std::decay_t<decltype(*est)>;
                if constexpr (std::is_same_v<Estimator,
                                             estimation::GeneralizedMethodOfMoments>) {
                    // GMM (B11): shares parameter/standard_error/covariance/correlation names
                    // with ML/MAP; adds j_stat/j_stat_pval and the B17C quantile_variance.
                    if (m == "parameter") return est->best_parameter_set().values[idx(0)];
                    if (m == "standard_error") return est->get_standard_errors()[idx(0)];
                    if (m == "covariance")
                        return est->get_covariance_matrix()(static_cast<int>(idx(0)), static_cast<int>(idx(1)));
                    if (m == "correlation")
                        return est->get_correlation_matrix()(static_cast<int>(idx(0)), static_cast<int>(idx(1)));
                    if (m == "j_stat") return est->jstat();
                    if (m == "j_stat_pval") return est->jstat_pval();
                    if (m == "quantile_variance") {
                        // args[0] is the annual EXCEEDANCE probability (AEP); the C#
                        // QuantileVariance takes a NON-exceedance probability, so pass 1 - AEP.
                        if (ec.b17c == nullptr)
                            throw std::runtime_error(
                                "quantile_variance requires a Bulletin17CDistribution model");
                        double aep = a[0].get<double>();
                        return ec.b17c->quantile_variance(1.0 - aep, est->best_parameter_set().values,
                                                          est->get_covariance_matrix().to_array());
                    }
                    throw std::runtime_error("unknown GMM fixture method: " + m);
                } else if constexpr (std::is_same_v<Estimator, estimation::BayesianAnalysis>) {
                    if (m == "dic") return est->dic();
                    if (m == "waic") return est->waic();
                    if (m == "looic") return est->looic();
                    if (m == "posterior_mean") return est->results()->posterior_mean.values[idx(0)];
                    if (m == "chain_value")
                        return est->sampler()->markov_chains()[idx(0)][idx(1)].values[idx(2)];
                    throw std::runtime_error("unknown BayesianAnalysis fixture method: " + m);
                } else {
                    if (m == "parameter") return est->best_parameter_set().values[idx(0)];
                    if (m == "max_log_likelihood") return est->maximum_log_likelihood();
                    if (m == "aic") return est->get_aic();
                    if (m == "bic") return est->get_bic(a[static_cast<std::size_t>(0)].get<int>());
                    if (m == "standard_error") return est->get_standard_errors()[idx(0)];
                    if (m == "covariance")
                        return est->get_covariance_matrix()(static_cast<int>(idx(0)), static_cast<int>(idx(1)));
                    if (m == "correlation")
                        return est->get_correlation_matrix()(static_cast<int>(idx(0)), static_cast<int>(idx(1)));
                    throw std::runtime_error("unknown model_estimation fixture method: " + m);
                }
            }
        },
        ec.estimator);
}

static void run_model_estimation(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        auto ec = build_and_run_estimation(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            check_value(dispatch_estimation(ec, method, args), as, where);
        }
    }
}

// --- analysis (Phase 8: user-facing Analyses layer) -------------------------------------
//
// A10: the three finished analyses (UnivariateAnalysis / FittingAnalysis / Bulletin17CAnalysis)
// become fixture-checkable through this stateful kind, mirroring model_estimation: one
// build+run per case caches a flat result surface, then every assertion dispatches against it.
// The construct fields map 1:1 onto the R/Python glue arguments (bf_analysis_*_ /
// analysis_*), so all three harnesses build byte-identical analyses from the same spec.
// A10 authors LOOSE, self-computed smoke oracles here; A11's emitter may tighten them.
namespace an = bestfit::analyses;

static an::UncertaintyMethod parse_uncertainty_method(const std::string& s) {
    if (s == "MultivariateNormal") return an::UncertaintyMethod::MultivariateNormal;
    if (s == "Bootstrap") return an::UncertaintyMethod::Bootstrap;
    throw std::runtime_error("unsupported/ deferred uncertainty method: " + s);
}

// Flat result surface every analysis assertion reads (only the fields the target populates are
// filled). Curve/CI vectors are indexed by the exceedance grid; the FittingAnalysis fields carry
// one entry per candidate.
struct AnalysisResult {
    std::vector<double> parameters, mode_curve, mean_curve, lower_ci, upper_ci;
    std::vector<double> exceedance, point_estimates, beta1, nu, quantile_variance;
    std::vector<double> cand_aic, cand_bic, cand_rmse, cand_converged;
    double aic = std::numeric_limits<double>::quiet_NaN();
    double bic = std::numeric_limits<double>::quiet_NaN();
    double dic = std::numeric_limits<double>::quiet_NaN();
    double rmse = std::numeric_limits<double>::quiet_NaN();
    double confidence_level = std::numeric_limits<double>::quiet_NaN();
    int candidate_count = 0;
};

static AnalysisResult build_and_run_analysis(const std::string& target, const json& construct,
                                             const json& datasets) {
    auto resolve_dataset = [&](const std::string& key) {
        std::vector<double> data;
        for (const auto& v : datasets[key]) data.push_back(parse_num(v));
        return data;
    };
    auto ordinates = [&]() {
        std::vector<double> ep;
        if (construct.contains("exceedance_probabilities"))
            for (const auto& v : construct["exceedance_probabilities"]) ep.push_back(parse_num(v));
        return ep;
    };
    auto apply_ordinates = [](bestfit::numerics::data::ProbabilityOrdinates& po,
                              const std::vector<double>& ep) {
        if (ep.empty()) return;
        po.clear();
        for (double p : ep) po.push_back(p);
    };

    AnalysisResult r;

    if (target == "FittingAnalysis") {
        std::vector<double> data = resolve_dataset(construct["dataset"].get<std::string>());
        auto df = std::make_unique<bestfit::models::DataFrame>();
        df->set_exact_series(bestfit::models::ExactSeries(data));
        df->calculate_plotting_positions();
        an::FittingAnalysis analysis(std::move(df));
        analysis.run();
        const auto& fitted = analysis.fitted_distributions();
        r.candidate_count = static_cast<int>(fitted.size());
        for (const auto& fd : fitted) {
            r.cand_aic.push_back(fd.aic());
            r.cand_bic.push_back(fd.bic());
            r.cand_rmse.push_back(fd.rmse());
            r.cand_converged.push_back(fd.fit_succeeded() ? 1.0 : 0.0);
        }
        return r;
    }

    const json& model_spec = construct["model"];
    std::vector<double> data = resolve_dataset(model_spec["dataset"].get<std::string>());

    if (target == "UnivariateAnalysis") {
        auto base = bestfit::models::spec::build_model_from_json(model_spec.dump(), data);
        auto* raw = dynamic_cast<bestfit::models::UnivariateDistributionModel*>(base.get());
        if (raw == nullptr) throw std::runtime_error("UnivariateAnalysis requires a univariate_distribution model");
        base.release();
        std::unique_ptr<bestfit::models::UnivariateDistributionModel> model(raw);
        an::UnivariateAnalysis analysis(std::move(model));
        apply_ordinates(analysis.probability_ordinates(), ordinates());
        auto& ba = analysis.bayesian_analysis();
        ba.set_type(parse_sampler_type(construct.value("sampler", std::string("DEMCzs"))));
        if (construct.contains("credible_level"))
            ba.set_credible_interval_width(construct["credible_level"].get<double>());
        if (construct.contains("seed")) ba.set_prng_seed(construct["seed"].get<int>());
        if (construct.contains("output_length")) ba.set_output_length(construct["output_length"].get<int>());
        if (construct.contains("iterations")) {
            int it = construct["iterations"].get<int>();
            ba.set_iterations(it);
            ba.set_warmup_iterations(std::max(50, it / 2));
        }
        // Optional explicit MCMC knobs (A11): the default thinning_interval=20 exposes a
        // C#-vs-C++ divergence in the thinned population-sampler stream (documented in
        // docs/upstream-csharp-issues.md); the fixture pins thinning_interval=1 (bayes_normal's
        // proven bit-identical path). All four runners honor the same override.
        if (construct.contains("thinning_interval"))
            ba.set_thinning_interval(construct["thinning_interval"].get<int>());
        if (construct.contains("number_of_chains"))
            ba.set_number_of_chains(construct["number_of_chains"].get<int>());
        if (construct.contains("initial_iterations"))
            ba.set_initial_iterations(construct["initial_iterations"].get<int>());
        analysis.run();
        const auto* results = analysis.analysis_results();
        if (results != nullptr) {
            auto* pe = analysis.get_point_estimate_distribution();
            if (pe != nullptr) r.parameters = pe->get_parameters();
            r.mode_curve = results->mode_curve;
            r.mean_curve = results->mean_curve;
            for (const auto& ci : results->confidence_intervals) {
                r.lower_ci.push_back(ci[0]);
                r.upper_ci.push_back(ci[1]);
            }
            r.aic = results->aic;
            r.bic = results->bic;
            r.dic = results->dic;
            r.rmse = results->rmse;
        }
        return r;
    }

    if (target == "Bulletin17CAnalysis") {
        auto model = bestfit::models::spec::build_bulletin17c_from_json(model_spec.dump(), data);
        an::Bulletin17CAnalysis analysis(std::move(model));
        analysis.set_uncertainty_method(
            parse_uncertainty_method(construct.value("uncertainty_method", std::string("MultivariateNormal"))));
        apply_ordinates(analysis.probability_ordinates(), ordinates());
        auto& ba = analysis.bayesian_analysis();
        if (construct.contains("confidence_level"))
            ba.set_credible_interval_width(construct["confidence_level"].get<double>());
        if (construct.contains("seed")) ba.set_prng_seed(construct["seed"].get<int>());
        if (construct.contains("output_length")) ba.set_output_length(construct["output_length"].get<int>());
        analysis.run();
        auto ci = analysis.compute_cohn_style_confidence_intervals();
        if (ci.has_value()) {
            r.exceedance = ci->exceedance_probabilities;
            r.point_estimates = ci->point_estimates;
            r.lower_ci = ci->lower_ci;
            r.upper_ci = ci->upper_ci;
            r.beta1 = ci->beta1;
            r.nu = ci->nu;
            r.quantile_variance = ci->quantile_variance;
            r.confidence_level = ci->confidence_level;
        }
        if (analysis.gmm() != nullptr && analysis.gmm()->is_estimated())
            r.parameters = analysis.gmm()->best_parameter_set().values;
        return r;
    }

    throw std::runtime_error("unknown analysis target: " + target);
}

static double dispatch_analysis(const AnalysisResult& r, const std::string& m, const json& a) {
    auto idx = [&](int i) { return static_cast<std::size_t>(a[static_cast<std::size_t>(i)].get<int>()); };
    if (m == "candidate_count") return static_cast<double>(r.candidate_count);
    if (m == "candidate_aic") return r.cand_aic.at(idx(0));
    if (m == "candidate_bic") return r.cand_bic.at(idx(0));
    if (m == "candidate_rmse") return r.cand_rmse.at(idx(0));
    if (m == "candidate_converged") return r.cand_converged.at(idx(0));
    if (m == "parameter") return r.parameters.at(idx(0));
    if (m == "mode_curve") return r.mode_curve.at(idx(0));
    if (m == "mean_curve") return r.mean_curve.at(idx(0));
    if (m == "lower_ci") return r.lower_ci.at(idx(0));
    if (m == "upper_ci") return r.upper_ci.at(idx(0));
    if (m == "exceedance_probability") return r.exceedance.at(idx(0));
    if (m == "point_estimate") return r.point_estimates.at(idx(0));
    if (m == "beta1") return r.beta1.at(idx(0));
    if (m == "nu") return r.nu.at(idx(0));
    if (m == "quantile_variance") return r.quantile_variance.at(idx(0));
    if (m == "aic") return r.aic;
    if (m == "bic") return r.bic;
    if (m == "dic") return r.dic;
    if (m == "rmse") return r.rmse;
    if (m == "confidence_level") return r.confidence_level;
    throw std::runtime_error("unknown analysis fixture method: " + m);
}

static void run_analysis(const json& spec) {
    std::string target = spec["target"].get<std::string>();
    json datasets = spec.value("datasets", json::object());
    for (const auto& c : spec["cases"]) {
        AnalysisResult r = build_and_run_analysis(target, c["construct"], datasets);
        std::string name = c["name"].get<std::string>();
        for (const auto& as : c["assertions"]) {
            std::string method = as["method"].get<std::string>();
            json args = as.contains("args") ? as["args"] : json::array();
            std::string where = target + "/" + name + "/" + method;
            check_value(dispatch_analysis(r, method, args), as, where);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <fixtures-dir>\n", argv[0]);
        return 2;
    }
    int files = 0;
    for (const auto& entry : fs::recursive_directory_iterator(argv[1])) {
        if (entry.path().extension() != ".json") continue;
        ++files;
        std::ifstream in(entry.path());
        json spec = json::parse(in);
        std::string kind = spec.value("kind", "");
        if (kind == "special_function") {
            run_special_function(spec);
        } else if (kind == "goodness_of_fit") {
            run_goodness_of_fit(spec);
        } else if (kind == "univariate_distribution") {
            if (spec.value("target", "") == "GeneralizedExtremeValue")
                run_gev(spec);
            else
                run_generic(spec);
        } else if (kind == "multivariate_distribution") {
            run_multivariate(spec);
        } else if (kind == "bivariate_copula") {
            run_bivariate_copula(spec);
        } else if (kind == "mcmc_sampler") {
            run_mcmc_sampler(spec);
        } else if (kind == "bootstrap") {
            run_bootstrap(spec);
        } else if (kind == "model_estimation") {
            run_model_estimation(spec);
        } else if (kind == "analysis") {
            run_analysis(spec);
        }
    }
    if (files == 0) {
        std::fprintf(stderr, "no fixtures found under %s\n", argv[1]);
        return 2;
    }
    return bftest::summary("fixtures");
}
