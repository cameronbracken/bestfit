// ported from: Numerics/Functions/Link Functions/IdentityLink.cs @ 2a0357a
//
// Identity link function: h(x) = x. No transformation is applied; the canonical link
// for the Normal (Gaussian) GLM family. Domain: all real numbers.
// h(x) = x, h^-1(eta) = eta, h'(x) = 1.
#pragma once

#include "corehydro/numerics/functions/i_link_function.hpp"

namespace corehydro::numerics::functions {

class IdentityLink : public ILinkFunction {
   public:
    double link(double x) const override { return x; }

    double inverse_link(double eta) const override { return eta; }

    double d_link(double /*x*/) const override { return 1.0; }
};

}  // namespace corehydro::numerics::functions
