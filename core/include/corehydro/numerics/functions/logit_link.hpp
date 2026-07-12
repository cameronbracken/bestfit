// ported from: Numerics/Functions/Link Functions/LogitLink.cs @ a2c4dbf
//
// Logit link function mapping the unit interval (0, 1) to the unconstrained real line.
// Domain: x in (0, 1). h(x) = log(x / (1 - x)), h^-1(eta) = 1 / (1 + exp(-eta)),
// h'(x) = 1 / (x(1 - x)). Canonical link for the Binomial GLM family. The inverse link
// (sigmoid) uses the numerically stable formulation to avoid overflow for large |eta|.
// C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "corehydro/numerics/functions/i_link_function.hpp"

namespace corehydro::numerics::functions {

class LogitLink : public ILinkFunction {
   public:
    double link(double x) const override {
        if (x <= 0.0 || x >= 1.0) throw std::out_of_range("LogitLink requires x in (0, 1).");
        x = std::max(kMinX, std::min(kMaxX, x));
        return std::log(x / (1.0 - x));
    }

    double inverse_link(double eta) const override {
        // Numerically stable sigmoid to avoid overflow for large |eta|.
        if (eta >= 0) {
            double e = std::exp(-eta);
            return 1.0 / (1.0 + e);
        } else {
            double e = std::exp(eta);
            return e / (1.0 + e);
        }
    }

    double d_link(double x) const override {
        if (x <= 0.0 || x >= 1.0)
            throw std::out_of_range("LogitLink derivative requires x in (0, 1).");
        x = std::max(kMinX, std::min(kMaxX, x));
        return 1.0 / (x * (1.0 - x));
    }

   private:
    // Smallest/largest admissible x to avoid log(0) blow-ups near the boundary.
    static constexpr double kMinX = 1e-12;
    static constexpr double kMaxX = 1.0 - 1e-12;
};

}  // namespace corehydro::numerics::functions
