// ported from: Numerics/Distributions/Univariate/VonMises.cs @ a2c4dbf
//
// The von Mises distribution for circular data.
// Parameters: mu (mean direction, in radians, in [-π, π]) and
//             kappa (concentration, >= 0).
// Mirrors VonMises.cs method-for-method. Implements IEstimation (MLE only;
// other methods throw like the C#). Does NOT implement ILinearMomentEstimation
// (C# class does not implement it). CDF via adaptive Gauss-Kronrod (G10K21),
// InverseCDF via Brent's method on CDF, both mirroring the C# source exactly.
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/integration/adaptive_gauss_kronrod.hpp"
#include "bestfit/numerics/math/rootfinding/brent.hpp"
#include "bestfit/numerics/math/special/bessel.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class VonMises : public UnivariateDistributionBase,
                 public IEstimation,
                 public IMaximumLikelihoodEstimation {
   public:
    // Constructs a von Mises distribution with μ = 0 and κ = 1.
    VonMises() { set_parameters(0.0, 1.0); }

    // Constructs a von Mises distribution with given μ and κ.
    VonMises(double mu, double kappa) { set_parameters(mu, kappa); }

    double mu()    const { return mu_; }
    double kappa() const { return kappa_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::VonMises;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {mu_, kappa_}; }

    void set_parameters(double mu, double kappa) {
        mu_    = mu;
        kappa_ = kappa;
        parameters_valid_ = validate(mu, kappa);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1]);
    }

    // --- Moments / support ---
    // Mean direction = μ
    double mean() const override { return mu_; }
    // Median direction = μ (circular symmetry)
    double median() const override { return mu_; }
    // Mode direction = μ
    double mode() const override { return mu_; }

    // Circular standard deviation = √(1 − I₁(κ)/I₀(κ)).
    // For κ = 0: I₁(0)/I₀(0) = 0/1 = 0, so SD = 1.
    double standard_deviation() const override {
        double i0 = math::special::bessel::i0(kappa_);
        if (i0 == 0.0) return 1.0;
        double i1 = math::special::bessel::i1(kappa_);
        double ratio = i1 / i0;
        double v = 1.0 - ratio;
        if (v < 0.0) v = 0.0;
        return std::sqrt(v);
    }

    double skewness() const override { return 0.0; }

    // Circular kurtosis is not directly comparable to linear kurtosis (C# returns NaN).
    double kurtosis() const override { return std::numeric_limits<double>::quiet_NaN(); }

    double minimum() const override { return -kPi; }
    double maximum() const override { return  kPi; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (x < -kPi || x > kPi) return 0.0;
        double i0 = math::special::bessel::i0(kappa_);
        return std::exp(kappa_ * std::cos(x - mu_)) / (2.0 * kPi * i0);
    }

    // CDF via numerical integration from -π to x, mirroring the C# AdaptiveGaussKronrod call.
    double cdf(double x) const override {
        if (x <= -kPi) return 0.0;
        if (x >=  kPi) return 1.0;
        double mu    = mu_;
        double kappa = kappa_;
        auto integrand = [mu, kappa](double t) {
            return std::exp(kappa * std::cos(t - mu));
        };
        double integral = math::integration::integrate(integrand, -kPi, x);
        double i0 = math::special::bessel::i0(kappa_);
        return integral / (2.0 * kPi * i0);
    }

    // InverseCDF via Brent's method on CDF (mirrors C# InverseCDF).
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        auto f = [this, probability](double x) { return cdf(x) - probability; };
        return math::rootfinding::solve(f, -kPi, kPi);
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<VonMises>(mu_, kappa_);
    }

    // --- Estimation (MLE only; C# throws NotImplementedException for other methods) ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method != ParameterEstimationMethod::MaximumLikelihood) {
            throw std::logic_error(
                "VonMises: only MaximumLikelihood estimation is implemented");
        }
        set_parameters(mle(sample));
    }

    // GetParameterConstraints mirrors C#: initialVals from MLE; bounds [-π, π] × [0, upper].
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        initials = mle(sample);
        lowers   = {-kPi, 0.0};
        double kappa_upper = std::max(initials[1] * 10.0, 100.0);
        uppers   = {kPi, kappa_upper};
    }

    // MLE: closed-form for μ (circular mean), Newton-Raphson for κ.
    // Mirrors the C# MLE method exactly, including the Mardia & Jupp approximation and
    // the Newton-Raphson refinement (A(κ) = I₁(κ)/I₀(κ), A′(κ) = 1 − A²(κ) − A(κ)/κ).
    std::vector<double> mle(const std::vector<double>& sample) const {
        double sum_sin = 0.0, sum_cos = 0.0;
        for (double v : sample) {
            sum_sin += std::sin(v);
            sum_cos += std::cos(v);
        }
        double n   = static_cast<double>(sample.size());
        double mu  = std::atan2(sum_sin, sum_cos);

        // Mean resultant length R̄
        double r_bar = std::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / n;

        double kappa;
        if (r_bar < 0.53) {
            kappa = 2.0 * r_bar
                  + r_bar * r_bar * r_bar
                  + (5.0 / 6.0) * std::pow(r_bar, 5.0);
        } else if (r_bar < 0.85) {
            kappa = -0.4 + 1.39 * r_bar + 0.43 / (1.0 - r_bar);
        } else {
            kappa = 1.0 / (r_bar * r_bar * r_bar
                         - 4.0 * r_bar * r_bar
                         + 3.0 * r_bar);
        }

        // Newton-Raphson refinement (up to 20 iterations, mirrors C#)
        if (r_bar > 0.0 && r_bar < 1.0) {
            for (int i = 0; i < 20; ++i) {
                double i0 = math::special::bessel::i0(kappa);
                double i1 = math::special::bessel::i1(kappa);
                double a      = (i0 > 0.0) ? i1 / i0 : 0.0;
                double a_prime = 1.0 - a * a - a / kappa;
                if (std::fabs(a_prime) < 1e-30) break;
                double delta = (a - r_bar) / a_prime;
                kappa -= delta;
                if (kappa < 0.0) kappa = kDoubleMachineEpsilon;
                if (std::fabs(delta) < 1e-12) break;
            }
        } else if (r_bar >= 1.0) {
            kappa = std::numeric_limits<double>::max();
        } else {
            kappa = 0.0;
        }

        return {mu, kappa};
    }

   private:
    static bool validate(double mu, double kappa) {
        if (std::isnan(mu) || std::isinf(mu))   return false;
        if (mu < -kPi || mu > kPi)              return false;
        if (std::isnan(kappa) || std::isinf(kappa) || kappa < 0.0) return false;
        return true;
    }

    double mu_    = 0.0;
    double kappa_ = 1.0;
};

}  // namespace bestfit::numerics::distributions
