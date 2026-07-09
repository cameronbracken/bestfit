// cpp11 glue exposing the Sobol quasi-random sequence to R.
// Core header is vendored under src/bestfit_core/include (a symlink into core/; regenerate real files with tools/materialize_core.py).
// The direction-numbers file is installed under inst/extdata/ and located at runtime via
//   system.file("extdata", "new-joe-kuo-6.21201", package = "bestfitr")
#include <cpp11.hpp>

#include <string>
#include <vector>

#include "bestfit/numerics/sampling/sobol.hpp"

using namespace cpp11;
using bestfit::numerics::sampling::SobolSequence;

// Generate n_steps consecutive Sobol points starting from the beginning of the sequence.
// Returns a double matrix with n_steps rows and dimension columns (column-major, R default).
// path: full path to the new-joe-kuo-6.21201 direction-numbers file.
[[cpp11::register]]
doubles_matrix<by_column> bf_sobol_generate_(int dimension, int n_steps,
                                              std::string path) {
    SobolSequence sobol(dimension, path);
    writable::doubles_matrix<by_column> out(n_steps, dimension);
    for (int i = 0; i < n_steps; ++i) {
        auto pt = sobol.next_double();
        for (int j = 0; j < dimension; ++j) {
            out(i, j) = pt[j];
        }
    }
    return out;
}

// Skip to a specific 1-based index in the sequence and return that single point.
// index == 1 returns the first point; index == 0 returns the zero vector (reset state).
[[cpp11::register]]
doubles bf_sobol_skip_to_(int dimension, int index, std::string path) {
    SobolSequence sobol(dimension, path);
    auto pt = sobol.skip_to(index);
    return writable::doubles(pt.begin(), pt.end());
}
