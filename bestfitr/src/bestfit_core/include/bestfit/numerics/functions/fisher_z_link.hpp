// ported from: Numerics/Functions/Link Functions/FisherZLink.cs @ a2c4dbf
//
// Fisher z link function mapping correlations from (-1, 1) to the unconstrained real
// line. Domain: -1 < x < 1. h(x) = atanh(x), h^-1(eta) = tanh(eta),
// h'(x) = 1 / (1 - x^2). Commonly used for correlation parameters and other signed
// bounded parameters. C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once
#include <cmath>
#include <stdexcept>

#include "bestfit/numerics/functions/i_link_function.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::functions {

class FisherZLink : public ILinkFunction {
   public:
    double link(double x) const override {
        if (!bestfit::numerics::is_finite(x) || x <= -1.0 || x >= 1.0)
            throw std::out_of_range("FisherZLink requires -1 < x < 1.");

        return 0.5 * std::log((1.0 + x) / (1.0 - x));
    }

    double inverse_link(double eta) const override {
        if (eta >= 0.0) {
            double e = std::exp(-2.0 * eta);
            return (1.0 - e) / (1.0 + e);
        }

        double exp_eta = std::exp(2.0 * eta);
        return (exp_eta - 1.0) / (exp_eta + 1.0);
    }

    double d_link(double x) const override {
        if (!bestfit::numerics::is_finite(x) || x <= -1.0 || x >= 1.0)
            throw std::out_of_range("FisherZLink derivative requires -1 < x < 1.");

        return 1.0 / (1.0 - x * x);
    }
};

}  // namespace bestfit::numerics::functions
