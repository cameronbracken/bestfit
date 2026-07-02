// ported from: Numerics/Distributions/Bivariate Copulas/FrankCopula.cs @ a2c4dbf
//
// The Frank copula. theta unbounded (-inf, +inf); custom closed-form PDF/CDF/InverseCDF
// (all five generator functions also have closed forms, including generator_prime_inverse
// -- no Brent root-solve is needed anywhere in this file, unlike AMH/Gumbel/Joe). Like
// AMHCopula, FrankCopula OVERRIDES ValidateParameter with a CORRECT implementation (`return
// null;` in range) -- it does NOT inherit ArchimedeanCopula's "always-false ParametersValid"
// bug (see amh_copula.hpp / clayton_copula.hpp / docs/upstream-csharp-issues.md).
//
// The C# FrankCopula class has NO SetThetaFromTau method (unlike Clayton/AMH/Gumbel) --
// its ParameterConstraints computes Kendall's tau from the sample data purely to pick the
// sign of the MPL/IFM/MLE search bracket ([0.001, 100] or [-100, -0.001]), not to fit theta
// directly. There is accordingly no "tau" method-of-moments fixture case for Frank (see
// frank_copula.json's source note and .superpowers/sdd/task-8-report.md).
#pragma once
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/numerics/data/correlation.hpp"
#include "bestfit/numerics/distributions/copulas/base/archimedean_copula.hpp"
#include "bestfit/numerics/distributions/copulas/base/copula_type.hpp"

namespace bestfit::numerics::distributions::copulas {

class FrankCopula : public ArchimedeanCopula {
   public:
    // Constructs a Frank copula with a dependency theta = 2.
    FrankCopula() { set_theta(2.0); }

    // Constructs a Frank copula with a specified theta.
    explicit FrankCopula(double theta) { set_theta(theta); }

    // Constructs a Frank copula with a specified theta and marginal distributions.
    FrankCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Frank; }

    double theta_minimum() const override { return -std::numeric_limits<double>::infinity(); }
    double theta_maximum() const override { return std::numeric_limits<double>::infinity(); }

    // Correct override (does NOT reproduce ArchimedeanCopula's ParametersValid bug -- see
    // file header).
    std::optional<std::string> validate_parameter(double parameter,
                                                    bool throw_exception) const override {
        if (parameter < theta_minimum()) {
            std::string msg = "The dependency parameter theta (theta) must be greater than or "
                               "equal to " +
                               std::to_string(theta_minimum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter > theta_maximum()) {
            std::string msg =
                "The dependency parameter theta (theta) must be less than or equal to " +
                std::to_string(theta_maximum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        return std::nullopt;
    }

    double generator(double t) const override {
        if (std::fabs(theta()) < 1e-10) return t;
        return -std::log((std::exp(-theta() * t) - 1.0) / (std::exp(-theta()) - 1.0));
    }

    double generator_inverse(double t) const override {
        return -std::log(std::exp(-theta() - t) - std::exp(-t) + 1.0) / theta();
    }

    double generator_prime(double t) const override {
        return theta() / (1.0 - std::exp(theta() * t));
    }

    double generator_prime2(double t) const override {
        return theta() * theta() * std::exp(theta() * t) /
               std::pow(1.0 - std::exp(theta() * t), 2.0);
    }

    double generator_prime_inverse(double t) const override {
        return std::log((theta() - t) / -t) / theta();
    }

    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double num = theta() * (std::exp(theta()) - 1.0) * std::exp(theta() * (1.0 + u + v));
        double den = std::pow(std::exp(theta() * (u + v)) - std::exp(theta() * (1.0 + u)) -
                                   std::exp(theta() * (1.0 + v)) + std::exp(theta()),
                               2.0);
        return num / den;
    }

    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        return -(1.0 / theta()) *
               std::log(1.0 + (std::exp(-theta() * u) - 1.0) * (std::exp(-theta() * v) - 1.0) /
                                  (std::exp(-theta()) - 1.0));
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double a = -std::fabs(theta());
        double vv = -1.0 / a *
                    std::log((-v * (std::exp(-a) - 1.0) / (std::exp(-a * u) * (v - 1.0) - v)) + 1.0);
        vv = theta() > 0.0 ? 1.0 - vv : vv;
        return {u, vv};
    }

    // Gets the upper tail dependence coefficient lambda_U = 0. The Frank copula has no tail
    // dependence.
    double upper_tail_dependence() const override { return 0.0; }

    // Gets the lower tail dependence coefficient lambda_L = 0. The Frank copula has no tail
    // dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<FrankCopula>(theta(), marginal_distribution_x, marginal_distribution_y);
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>& sample_data_x,
                                                  const std::vector<double>& sample_data_y) const override {
        double tau = bestfit::numerics::data::kendalls_tau(sample_data_x, sample_data_y);
        double L = tau > 0.0 ? 0.001 : -100.0;
        double U = tau > 0.0 ? 100.0 : -0.001;
        return {{L, U}};
    }
};

}  // namespace bestfit::numerics::distributions::copulas
