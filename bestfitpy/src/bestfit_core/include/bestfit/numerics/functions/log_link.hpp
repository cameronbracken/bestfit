// ported from: Numerics/Functions/Link Functions/LogLink.cs @ a2c4dbf
//
// Log link function mapping positive reals to the unconstrained real line.
// Domain: x > 0. h(x) = log(x), h^-1(eta) = exp(eta), h'(x) = 1/x.
// Canonical link for the Poisson and Exponential GLM families; commonly used for scale
// parameters that must remain positive. C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once
#include <cmath>
#include <stdexcept>

#include "bestfit/numerics/functions/i_link_function.hpp"

namespace bestfit::numerics::functions {

class LogLink : public ILinkFunction {
   public:
    double link(double x) const override {
        if (x <= 0.0) throw std::out_of_range("LogLink requires x > 0.");
        if (x < kMinX) x = kMinX;
        return std::log(x);
    }

    double inverse_link(double eta) const override { return std::exp(eta); }

    double d_link(double x) const override {
        if (x <= 0.0) throw std::out_of_range("LogLink derivative requires x > 0.");
        if (x < kMinX) x = kMinX;
        return 1.0 / x;
    }

   private:
    // Smallest admissible x to avoid log(0) and 1/x blow-ups.
    static constexpr double kMinX = 1e-12;
};

}  // namespace bestfit::numerics::functions
