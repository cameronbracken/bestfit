// ported from: Numerics/Functions/Link Functions/IdentityLink.cs @ a2c4dbf
//
// Identity link function: h(x) = x. No transformation is applied; the canonical link
// for the Normal (Gaussian) GLM family. Domain: all real numbers.
// h(x) = x, h^-1(eta) = eta, h'(x) = 1.
#pragma once

#include "bestfit/numerics/functions/i_link_function.hpp"

namespace bestfit::numerics::functions {

class IdentityLink : public ILinkFunction {
   public:
    double link(double x) const override { return x; }

    double inverse_link(double eta) const override { return eta; }

    double d_link(double /*x*/) const override { return 1.0; }
};

}  // namespace bestfit::numerics::functions
