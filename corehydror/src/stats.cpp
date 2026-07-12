// cpp11 glue exposing the small data-statistics utilities of the shared C++ core:
// the Multiple Grubbs-Beck low-outlier test, Box-Cox and Yeo-Johnson transforms,
// plotting positions, and Latin hypercube sampling. Consumed by R/stats.R.
// Core headers are vendored under src/corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "corehydro/numerics/data/box_cox.hpp"
#include "corehydro/numerics/data/multiple_grubbs_beck_test.hpp"
#include "corehydro/numerics/data/plotting_positions.hpp"
#include "corehydro/numerics/data/yeo_johnson.hpp"
#include "corehydro/numerics/sampling/latin_hypercube.hpp"

namespace nd = corehydro::numerics::data;
namespace pp = corehydro::numerics::data::plotting_positions;
using namespace cpp11;

static std::vector<double> to_vec(doubles x) {
    return std::vector<double>(x.begin(), x.end());
}

[[cpp11::register]]
int ch_mgbt_test_(doubles x) {
    return nd::MultipleGrubbsBeckTest::function(to_vec(x));
}

[[cpp11::register]]
double ch_box_cox_lambda_(doubles values) {
    return nd::BoxCox::fit_lambda(to_vec(values));
}

[[cpp11::register]]
doubles ch_box_cox_(doubles values, double lambda) {
    return writable::doubles(nd::BoxCox::transform(to_vec(values), lambda));
}

[[cpp11::register]]
doubles ch_box_cox_inverse_(doubles values, double lambda) {
    return writable::doubles(nd::BoxCox::inverse_transform(to_vec(values), lambda));
}

[[cpp11::register]]
double ch_yeo_johnson_lambda_(doubles values) {
    return nd::YeoJohnson::fit_lambda(to_vec(values));
}

[[cpp11::register]]
doubles ch_yeo_johnson_(doubles values, double lambda) {
    return writable::doubles(nd::YeoJohnson::transform(to_vec(values), lambda));
}

[[cpp11::register]]
doubles ch_yeo_johnson_inverse_(doubles values, double lambda) {
    return writable::doubles(nd::YeoJohnson::inverse_transform(to_vec(values), lambda));
}

[[cpp11::register]]
doubles ch_plotting_positions_(int n, std::string method) {
    std::vector<double> out;
    if (method == "weibull") out = pp::weibull(n);
    else if (method == "median") out = pp::median(n);
    else if (method == "blom") out = pp::blom(n);
    else if (method == "cunnane") out = pp::cunnane(n);
    else if (method == "gringorten") out = pp::gringorten(n);
    else if (method == "hazen") out = pp::hazen(n);
    else stop("unknown plotting-position method '%s'", method.c_str());
    return writable::doubles(out);
}

[[cpp11::register]]
doubles ch_plotting_positions_alpha_(int n, double alpha) {
    return writable::doubles(pp::function(n, alpha));
}

// Returns the LHS sample flattened row-major with a dim attribute set R-side
// (doubles_matrix would transpose; keep the glue simple and reshape in R).
[[cpp11::register]]
doubles ch_latin_hypercube_(int sample_size, int dimension, int seed, bool median) {
    auto m = median
        ? corehydro::numerics::sampling::LatinHypercube::median(sample_size, dimension, seed)
        : corehydro::numerics::sampling::LatinHypercube::random(sample_size, dimension, seed);
    writable::doubles out(static_cast<R_xlen_t>(sample_size) * dimension);
    R_xlen_t k = 0;
    for (int i = 0; i < sample_size; ++i)
        for (int j = 0; j < dimension; ++j) out[k++] = m[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
    return out;
}
