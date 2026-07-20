// ported from: Numerics/Distributions/Bivariate Copulas/JoeCopula.cs @ 2a0357a
//
// The Joe copula. theta in [1, +inf). No PDF/CDF override -- both resolve through
// ArchimedeanCopula's generic Genest-1986 forms built from the generator functions below.
// generator_prime_inverse has no closed form (Brent.Solve on [0, 1], mirroring
// AMH/Gumbel). InverseCDF also has no closed form: it solves the conditional distribution
// C(v|u) = p for v via a second, distinct Brent root-find over [0, 1] (mirroring
// GumbelCopula's InverseCDF override, not the generic ArchimedeanCopula::inverse_cdf).
// Like ClaytonCopula/GumbelCopula (and unlike AMH/Frank), JoeCopula does NOT override
// ValidateParameter, so it inherits ArchimedeanCopula's validate_parameter directly --
// including the v2.1.4 fix that makes ParametersValid report true for an in-range theta
// (previously always false; see clayton_copula.hpp / archimedean_copula.hpp /
// docs/upstream-csharp-issues.md). Clone() deep-copies attached marginals via
// BivariateCopula::clone_marginal (v2.1.4, Task 8).
//
// DEVIATION FROM THE PHASE 2 PLAN TEXT (C# source governs -- see .claude/CLAUDE.md):
// JoeCopula.cs has NO SetThetaFromTau method (grep across the whole "Bivariate Copulas"
// directory confirms it exists only on ClaytonCopula/AMHCopula/GumbelCopula), and
// Test_JoeCopula.cs correspondingly has no Test_MOM_Fit. The task brief and
// fixtures/README.md's "AliMikhailHaq, Gumbel, Joe" tau-capable list are therefore both
// wrong for Joe; this port omits set_theta_from_tau on JoeCopula and the fixture has no
// "tau" case for it (see joe_copula.json's source note and
// .superpowers/sdd/task-8-report.md).
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "corehydro/numerics/distributions/copulas/base/archimedean_copula.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"

namespace corehydro::numerics::distributions::copulas {

class JoeCopula : public ArchimedeanCopula {
   public:
    // Constructs a Joe copula with a dependency theta = 2.
    JoeCopula() { set_theta(2.0); }

    // Constructs a Joe copula with a specified theta.
    explicit JoeCopula(double theta) { set_theta(theta); }

    // Constructs a Joe copula with a specified theta and marginal distributions.
    JoeCopula(double theta, std::shared_ptr<UnivariateDistributionBase> marginal_distribution_x_,
              std::shared_ptr<UnivariateDistributionBase> marginal_distribution_y_) {
        set_theta(theta);
        marginal_distribution_x = std::move(marginal_distribution_x_);
        marginal_distribution_y = std::move(marginal_distribution_y_);
    }

    CopulaType type() const override { return CopulaType::Joe; }

    double theta_minimum() const override { return 1.0; }
    double theta_maximum() const override { return std::numeric_limits<double>::infinity(); }

    double generator(double t) const override {
        double a = 1.0 - t;
        return -std::log(1.0 - sign(a) * std::pow(std::fabs(a), theta()));
    }

    double generator_inverse(double t) const override {
        double a = 1.0 - std::exp(-t);
        return 1.0 - sign(a) * std::pow(std::fabs(a), 1.0 / theta());
    }

    double generator_prime(double t) const override {
        double a = 1.0 - t;
        return -(theta() * sign(a) * std::pow(std::fabs(a), theta() - 1.0)) /
               (1.0 - sign(a) * std::pow(std::fabs(a), theta()));
    }

    double generator_prime2(double t) const override {
        double a = 1.0 - t;
        double num = theta() * (theta() + sign(a) * std::pow(std::fabs(a), theta()) - 1.0) *
                     sign(a) * std::pow(std::fabs(a), theta() - 2.0);
        double aa = 1.0 - sign(a) * std::pow(std::fabs(a), theta());
        double den = sign(aa) * std::pow(std::fabs(aa), 2.0);
        return num / den;
    }

    double generator_prime_inverse(double t) const override {
        return corehydro::numerics::math::rootfinding::solve(
            [this, t](double x) { return generator_prime(x) - t; }, 0.0, 1.0);
    }

    // Solves the conditional distribution C(v|u) = p for v via Brent root-find (no closed
    // form for Joe).
    std::array<double, 2> inverse_cdf(double u, double v) const override {
        if (!parameters_valid()) validate_parameter(theta(), true);
        double p = v;
        double th = theta();
        double vv = corehydro::numerics::math::rootfinding::solve(
            [u, p, th](double x) {
                double a = std::pow(1.0 - u, th);
                double b = std::pow(1.0 - x, th);
                double vu = -(b - 1.0) * std::pow(a - a * b + b, (-th + 1.0) / th) *
                            std::pow(1.0 - u, th - 1.0);
                return vu - p;
            },
            0.0, 1.0);
        return {u, vv};
    }

    // Gets the upper tail dependence coefficient lambda_U = 2 - 2^(1/theta).
    double upper_tail_dependence() const override { return 2.0 - std::pow(2.0, 1.0 / theta()); }

    // Gets the lower tail dependence coefficient lambda_L = 0. The Joe copula has no lower
    // tail dependence.
    double lower_tail_dependence() const override { return 0.0; }

    std::unique_ptr<BivariateCopula> clone() const override {
        return std::make_unique<JoeCopula>(theta(), clone_marginal(marginal_distribution_x),
                                            clone_marginal(marginal_distribution_y));
    }

    math::linalg::Matrix2D parameter_constraints(const std::vector<double>&,
                                                  const std::vector<double>&) const override {
        return {{1.0, 100.0}};
    }

   private:
    // Math.Sign(a): -1, 0, or 1 (distinct from Tools.Sign's Fortran-style 2-arg transfer
    // used by brent_search.hpp).
    static double sign(double a) { return a > 0.0 ? 1.0 : (a < 0.0 ? -1.0 : 0.0); }
};

}  // namespace corehydro::numerics::distributions::copulas
