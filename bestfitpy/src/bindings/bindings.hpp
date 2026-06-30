// Shared declaration so the pybind11 module (defined in gev.cpp) can pull in the
// polymorphic-distribution bindings defined in dist.cpp.
#pragma once
#include <pybind11/pybind11.h>

void register_distributions(pybind11::module_& m);
