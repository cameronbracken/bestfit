// pybind11 glue exposing the small data-statistics utilities of the shared C++ core:
// the Multiple Grubbs-Beck low-outlier test, Box-Cox and Yeo-Johnson transforms,
// plotting positions, and Latin hypercube sampling. Consumed by bestfitpy.stats.
// Core headers are vendored under ../bestfit_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "bestfit/numerics/data/box_cox.hpp"
#include "bestfit/numerics/data/multiple_grubbs_beck_test.hpp"
#include "bestfit/numerics/data/plotting_positions.hpp"
#include "bestfit/numerics/data/yeo_johnson.hpp"
#include "bestfit/numerics/sampling/latin_hypercube.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace nd = bestfit::numerics::data;
namespace pp = bestfit::numerics::data::plotting_positions;

static std::vector<double> plotting_positions_dispatch(int n, const std::string& method) {
    if (method == "weibull") return pp::weibull(n);
    if (method == "median") return pp::median(n);
    if (method == "blom") return pp::blom(n);
    if (method == "cunnane") return pp::cunnane(n);
    if (method == "gringorten") return pp::gringorten(n);
    if (method == "hazen") return pp::hazen(n);
    throw py::value_error("unknown plotting-position method '" + method +
                          "' (use weibull, median, blom, cunnane, gringorten, or hazen)");
}

void register_stats(py::module_& m) {
    m.def("mgbt_test", [](const std::vector<double>& x) {
        return nd::MultipleGrubbsBeckTest::function(x);
    });

    m.def("box_cox_lambda", [](const std::vector<double>& values) {
        return nd::BoxCox::fit_lambda(values);
    });
    m.def("box_cox", [](const std::vector<double>& values, double lambda) {
        return nd::BoxCox::transform(values, lambda);
    });
    m.def("box_cox_inverse", [](const std::vector<double>& values, double lambda) {
        return nd::BoxCox::inverse_transform(values, lambda);
    });

    m.def("yeo_johnson_lambda", [](const std::vector<double>& values) {
        return nd::YeoJohnson::fit_lambda(values);
    });
    m.def("yeo_johnson", [](const std::vector<double>& values, double lambda) {
        return nd::YeoJohnson::transform(values, lambda);
    });
    m.def("yeo_johnson_inverse", [](const std::vector<double>& values, double lambda) {
        return nd::YeoJohnson::inverse_transform(values, lambda);
    });

    m.def("plotting_positions", &plotting_positions_dispatch);
    m.def("plotting_positions_alpha", [](int n, double alpha) {
        return pp::function(n, alpha);
    });

    m.def("latin_hypercube", [](int sample_size, int dimension, int seed) {
        return bestfit::numerics::sampling::LatinHypercube::random(sample_size, dimension, seed);
    });
    m.def("latin_hypercube_median", [](int sample_size, int dimension, int seed) {
        return bestfit::numerics::sampling::LatinHypercube::median(sample_size, dimension, seed);
    });
}
