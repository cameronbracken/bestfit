// Shared declarations so the pybind11 module (defined in gev.cpp) can pull in
// bindings defined in dist.cpp and sobol.cpp.
#pragma once
#include <pybind11/pybind11.h>

void register_distributions(pybind11::module_& m);
void register_sobol(pybind11::module_& m);
