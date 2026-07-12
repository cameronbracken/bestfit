// ported from: Numerics/Functions/Link Functions/ProbitLink.cs @ a2c4dbf
//
// Probit link function mapping the unit interval (0, 1) to the unconstrained real line
// using the standard normal quantile function. Domain: x in (0, 1). h(x) = Phi^-1(x),
// h^-1(eta) = Phi(eta), h'(x) = 1 / phi(Phi^-1(x)). Alternative link for the Binomial
// GLM family. Uses the ported Normal::standard_z / standard_cdf / standard_pdf (the C#
// calls Normal.StandardZ / StandardCDF / StandardPDF).
// C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once
#include <algorithm>
#include <stdexcept>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"

namespace corehydro::numerics::functions {

class ProbitLink : public ILinkFunction {
   public:
    double link(double x) const override {
        if (x <= 0.0 || x >= 1.0) throw std::out_of_range("ProbitLink requires x in (0, 1).");
        x = std::max(kMinX, std::min(kMaxX, x));
        return distributions::Normal::standard_z(x);
    }

    double inverse_link(double eta) const override {
        return distributions::Normal::standard_cdf(eta);
    }

    double d_link(double x) const override {
        if (x <= 0.0 || x >= 1.0)
            throw std::out_of_range("ProbitLink derivative requires x in (0, 1).");
        x = std::max(kMinX, std::min(kMaxX, x));
        double z = distributions::Normal::standard_z(x);
        double pdf = distributions::Normal::standard_pdf(z);
        return 1.0 / std::max(pdf, 1e-16);
    }

   private:
    // Smallest/largest admissible x to avoid quantile blow-ups near the boundary.
    static constexpr double kMinX = 1e-12;
    static constexpr double kMaxX = 1.0 - 1e-12;
};

}  // namespace corehydro::numerics::functions
