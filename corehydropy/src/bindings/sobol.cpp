// pybind11 glue exposing the Sobol quasi-random sequence to Python.
// Core header is vendored under ../corehydro_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
// The direction-numbers file is packaged under corehydropy/data/ and located at runtime via
//   importlib.resources.files("corehydropy") / "data" / "new-joe-kuo-6.21201"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>
#include <vector>

#include "corehydro/numerics/sampling/sobol.hpp"

namespace py = pybind11;
using corehydro::numerics::sampling::SobolSequence;

void register_sobol(py::module_& m) {
    // Generate n_steps consecutive Sobol points from the start of the sequence.
    // Returns a list of n_steps vectors (each of length dimension).
    // path: full path to the new-joe-kuo-6.21201 direction-numbers file
    //       (pass empty string for dimension == 1).
    m.def(
        "sobol_generate",
        [](int dimension, int n_steps, const std::string& path) {
            SobolSequence sobol(dimension, path);
            std::vector<std::vector<double>> out;
            out.reserve(n_steps);
            for (int i = 0; i < n_steps; ++i) out.push_back(sobol.next_double());
            return out;
        },
        py::arg("dimension"), py::arg("n_steps"), py::arg("path"));

    // Skip to a specific 1-based index and return that single point.
    // index == 1 returns the first point; index == 0 returns the zero vector.
    m.def(
        "sobol_skip_to",
        [](int dimension, int index, const std::string& path) {
            SobolSequence sobol(dimension, path);
            return sobol.skip_to(index);
        },
        py::arg("dimension"), py::arg("index"), py::arg("path"));
}
