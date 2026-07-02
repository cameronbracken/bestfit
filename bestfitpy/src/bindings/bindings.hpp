// Shared declarations so the pybind11 module (defined in gev.cpp) can pull in
// bindings defined in dist.cpp, mvd.cpp, copula.cpp, sobol.cpp, and mcmc.cpp.
#pragma once
#include <pybind11/pybind11.h>

void register_distributions(pybind11::module_& m);
void register_multivariate(pybind11::module_& m);
void register_copulas(pybind11::module_& m);
void register_sobol(pybind11::module_& m);
void register_mcmc(pybind11::module_& m);
