// ported from: Numerics/Functions/Link Functions/ComplementaryLogLogLink.cs @ a2c4dbf
//
// Complementary log-log link function mapping the unit interval (0, 1) to the
// unconstrained real line. Domain: x in (0, 1). h(x) = log(-log(1 - x)),
// h^-1(eta) = 1 - exp(-exp(eta)), h'(x) = 1 / ((1 - x) * (-log(1 - x))). Used for
// asymmetric binary response models (rare events); arises from extreme value (Gumbel)
// latent variable models. C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "corehydro/numerics/functions/i_link_function.hpp"

namespace corehydro::numerics::functions {

class ComplementaryLogLogLink : public ILinkFunction {
   public:
    double link(double x) const override {
        if (x <= 0.0 || x >= 1.0)
            throw std::out_of_range("ComplementaryLogLogLink requires x in (0, 1).");
        x = std::max(kMinX, std::min(kMaxX, x));
        return std::log(-std::log(1.0 - x));
    }

    double inverse_link(double eta) const override { return 1.0 - std::exp(-std::exp(eta)); }

    double d_link(double x) const override {
        if (x <= 0.0 || x >= 1.0)
            throw std::out_of_range("ComplementaryLogLogLink derivative requires x in (0, 1).");
        x = std::max(kMinX, std::min(kMaxX, x));
        // h'(x) = 1 / ((1 - x) * (-log(1 - x)))
        double one_minus_x = 1.0 - x;
        double neg_log_one_minus_x = -std::log(one_minus_x);
        return 1.0 / std::max(one_minus_x * neg_log_one_minus_x, 1e-16);
    }

   private:
    // Smallest/largest admissible x to avoid log(0) blow-ups near the boundary.
    static constexpr double kMinX = 1e-12;
    static constexpr double kMaxX = 1.0 - 1e-12;
};

}  // namespace corehydro::numerics::functions
