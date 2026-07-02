// ported from: Numerics/Distributions/Bivariate Copulas/NormalCopula.cs @ a2c4dbf
//
// The Gaussian (Normal) elliptical copula. theta = rho in [-1, +1]; extends BivariateCopula
// DIRECTLY (not ArchimedeanCopula -- the Normal copula has no Archimedean generator).
// PDF is the closed-form Gaussian copula density built from Normal::standard_z (the C#
// Normal.StandardZ quantile function). CDF delegates to the already-ported deterministic
// Drezner/Genz bivariate normal CDF, MultivariateNormal::bivariate_cdf -- passing the
// negated quantiles per the C# comment (BivariateCDF computes Phi2(-h,-k;r), so
// C(u,v) = Phi2(Phi^-1(u), Phi^-1(v); r) requires arguments -Phi^-1(u), -Phi^-1(v)).
// InverseCDF is the standard conditional-normal simulation formula. Zero tail dependence
// (a defining property of the Gaussian copula, unlike the Student's t copula).
//
// Unlike ArchimedeanCopula's ValidateParameter (see archimedean_copula.hpp's header for the
// upstream ParametersValid bug it reproduces), NormalCopula.ValidateParameter in the C#
// source returns `null` (not a non-null "Parameter is valid" sentinel) in its in-range
// branch -- ported here as std::nullopt, so ParametersValid behaves correctly.
#pragma once
#include <array>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "bestfit/numerics/distributions/copulas/base/copula_type.hpp"
#include "bestfit/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions::copulas {

class NormalCopula : public BivariateCopula {
   public:
    // Constructs a bivariate Gaussian copula with a correlation rho = 0.0.
    NormalCopula() { set_theta(0.0); }

    // Constructs a bivariate Gaussian copula with a specified correlation rho.
    explicit NormalCopula(double rho) { set_theta(rho); }

    // Constructs a bivariate Gaussian copula with a specified theta and marginal
    // distributions.
    NormalCopula(double rho, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
                 std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(rho);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Normal; }

    double theta_minimum() const override { return -1.0; }
    double theta_maximum() const override { return 1.0; }

    // Correct override (returns nullopt when in range) -- does NOT reproduce
    // ArchimedeanCopula's ParametersValid bug (see that file's header and
    // docs/upstream-csharp-issues.md).
    std::optional<std::string> validate_parameter(double parameter,
                                                    bool throw_exception) const override {
        if (parameter < theta_minimum()) {
            std::string msg =
                "The correlation parameter rho (rho) must be greater than " +
                std::to_string(theta_minimum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        if (parameter > theta_maximum()) {
            std::string msg = "The correlation parameter rho (rho) must be less than " +
                               std::to_string(theta_maximum()) + ".";
            if (throw_exception) throw std::out_of_range(msg);
            return msg;
        }
        return std::nullopt;
    }

    int number_of_copula_parameters() const override { return 1; }
    std::vector<double> get_copula_parameters() const override { return {theta()}; }
    void set_copula_parameters(const std::vector<double>& parameters) override {
        set_theta(parameters[0]);
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{-1.0 + bestfit::numerics::kDoubleMachineEpsilon,
                 1.0 - bestfit::numerics::kDoubleMachineEpsilon}};
    }

    double pdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double r = theta();
        double s = Normal::standard_z(u);
        double t = Normal::standard_z(v);
        return 1.0 / std::sqrt(1.0 - r * r) *
               std::exp(-(r * r * s * s + r * r * t * t - 2.0 * r * s * t) / (2.0 * (1.0 - r * r)));
    }

    double cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        // BivariateCDF implements Genz's BVND which computes Phi2(-h,-k;r). To get the
        // copula C(u,v) = Phi2(Phi^-1(u), Phi^-1(v); r), pass -Phi^-1(u) = Phi^-1(1-u) as
        // arguments.
        return bestfit::numerics::distributions::MultivariateNormal::bivariate_cdf(
            -Normal::standard_z(u), -Normal::standard_z(v), theta());
    }

    std::array<double, 2> inverse_cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double z1 = Normal::standard_z(u);
        double z2 = Normal::standard_z(v);
        double r = theta();
        double w2 = r * z1 + std::sqrt(1.0 - r * r) * z2;
        double vv = Normal::standard_cdf(w2);
        return {u, vv};
    }

    // Gets the upper tail dependence coefficient lambda_U = 0. The Normal copula has no
    // upper tail dependence.
    double upper_tail_dependence() const override { return 0.0; }

    // Gets the lower tail dependence coefficient lambda_L = 0. The Normal copula has no
    // lower tail dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<NormalCopula>(theta(), marginal_distribution_x, marginal_distribution_y);
    }
};

}  // namespace bestfit::numerics::distributions::copulas
