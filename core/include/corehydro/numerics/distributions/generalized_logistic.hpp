// ported from: Numerics/Distributions/Univariate/GeneralizedLogistic.cs @ a2c4dbf
//
// Generalized Logistic distribution: parameters ξ (location), α (scale), κ (shape).
// Mirrors the C# source method-for-method. The IBootstrappable, IStandardError, and
// WPF helpers are not ported (desktop / uncertainty-analysis concerns).
// κ→0 limit branch: standard logistic distribution.
// κ > 0: bounded above at ξ + α/κ.
// κ < 0: bounded below at ξ + α/κ.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_linear_moment_estimation.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/optimization/nelder_mead.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class GeneralizedLogistic : public UnivariateDistributionBase,
                            public IEstimation,
                            public ILinearMomentEstimation,
                            public IMaximumLikelihoodEstimation {
   public:
    GeneralizedLogistic() { set_parameters(100.0, 10.0, 0.0); }
    GeneralizedLogistic(double location, double scale, double shape) {
        set_parameters(location, scale, shape);
    }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::GeneralizedLogistic;
    }
    int number_of_parameters() const override { return 3; }
    std::vector<double> get_parameters() const override { return {xi_, alpha_, kappa_}; }

    void set_parameters(double location, double scale, double shape) {
        xi_ = location;
        alpha_ = scale;
        kappa_ = shape;
        parameters_valid_ = validate(location, scale, shape);
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2]);
    }

    // --- Moments / support ---
    double mean() const override {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero) return xi_;
        if (std::fabs(kappa_) < 1.0) {
            double U1 = g::function(1.0 + kappa_) * g::function(1.0 - kappa_);
            return xi_ + alpha_ / kappa_ * (1.0 - U1);
        }
        return kNaN;
    }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (std::fabs(kappa_) <= kNearZero) return xi_;
        return xi_ + alpha_ * (std::pow(1.0 + kappa_, -kappa_) - 1.0) / kappa_;
    }

    double standard_deviation() const override {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero)
            return alpha_ * kPi / std::sqrt(3.0);
        if (std::fabs(kappa_) < 0.5) {
            double U1 = g::function(1.0 + kappa_) * g::function(1.0 - kappa_);
            double U2 = g::function(1.0 + 2.0 * kappa_) * g::function(1.0 - 2.0 * kappa_);
            return std::sqrt(alpha_ * alpha_ / (kappa_ * kappa_) * (U2 - U1 * U1));
        }
        return kNaN;
    }

    double skewness() const override {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero) return 0.0;
        if (std::fabs(kappa_) < 1.0 / 3.0) {
            double U1 = g::function(1.0 + kappa_) * g::function(1.0 - kappa_);
            double U2 = g::function(1.0 + 2.0 * kappa_) * g::function(1.0 - 2.0 * kappa_);
            double U3 = g::function(1.0 + 3.0 * kappa_) * g::function(1.0 - 3.0 * kappa_);
            double num = -U3 + 3.0 * U1 * U2 - 2.0 * std::pow(U1, 3.0);
            double den = std::pow(U2 - U1 * U1, 1.5);
            return sign(kappa_) * num / den;
        }
        return kNaN;
    }

    double kurtosis() const override {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero) return 3.0 + 6.0 / 5.0;
        if (std::fabs(kappa_) < 0.25) {
            double U1 = g::function(1.0 + kappa_) * g::function(1.0 - kappa_);
            double U2 = g::function(1.0 + 2.0 * kappa_) * g::function(1.0 - 2.0 * kappa_);
            double U3 = g::function(1.0 + 3.0 * kappa_) * g::function(1.0 - 3.0 * kappa_);
            double U4 = g::function(1.0 + 4.0 * kappa_) * g::function(1.0 - 4.0 * kappa_);
            double knum = U4 - 4.0 * U3 * U1 - 3.0 * U2 * U2 + 12.0 * U2 * U1 * U1 -
                          6.0 * std::pow(U1, 4.0);
            double kden = std::pow(U2 - U1 * U1, 2.0);
            return 3.0 + knum / kden;
        }
        return kNaN;
    }

    double minimum() const override {
        if (kappa_ >= -kNearZero) return -kInf;
        return xi_ + alpha_ / kappa_;
    }

    double maximum() const override {
        if (kappa_ <= kNearZero) return kInf;
        return xi_ + alpha_ / kappa_;
    }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (x < minimum() || x > maximum()) return 0.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero)
            y = -std::log(1.0 - kappa_ * y) / kappa_;
        return 1.0 / alpha_ * std::exp(-(1.0 - kappa_) * y) /
               std::pow(1.0 + std::exp(-y), 2.0);
    }

    double cdf(double x) const override {
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero)
            y = -std::log(1.0 - kappa_ * y) / kappa_;
        return 1.0 / (1.0 + std::exp(-y));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (std::fabs(kappa_) <= kNearZero)
            return xi_ - alpha_ * std::log((1.0 - probability) / probability);
        return xi_ + alpha_ / kappa_ *
                         (1.0 - std::pow((1.0 - probability) / probability, kappa_));
    }

    // --- Parameter display names (X1; C# GeneralizedLogistic.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (\xCE\xBE)", "Scale (\xCE\xB1)", "Shape (\xCE\xBA)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBE", "\xCE\xB1", "\xCE\xBA"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GeneralizedLogistic>(xi_, alpha_, kappa_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample,
                  ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            set_parameters(direct_method_of_moments(data::product_moments(sample)));
        } else if (method == ParameterEstimationMethod::MethodOfLinearMoments) {
            set_parameters(parameters_from_linear_moments(data::linear_moments(sample)));
        } else {
            set_parameters(mle(sample));
        }
    }

    // Solve for shape κ from skewness using Brent's method with Lanczos gamma.
    double solve_for_kappa(double skew) const {
        namespace g = math::special;
        if (std::fabs(skew) < 10.0) {
            return math::rootfinding::solve(
                [skew](double x) {
                    double U1 = g::lanczos(1.0 + x) * g::lanczos(1.0 - x);
                    double U2 = g::lanczos(1.0 + 2.0 * x) * g::lanczos(1.0 - 2.0 * x);
                    double U3 = g::lanczos(1.0 + 3.0 * x) * g::lanczos(1.0 - 3.0 * x);
                    double k = sign(x) * (-U3 + 3.0 * U1 * U2 - 2.0 * std::pow(U1, 3.0)) /
                               std::pow(U2 - U1 * U1, 1.5);
                    return k - skew;
                },
                -(1.0 / 3.0), 1.0 / 3.0);
        }
        return kNaN;
    }

    // Direct method of moments using product moments.
    std::vector<double> direct_method_of_moments(const std::vector<double>& moments) const {
        namespace g = math::special;
        double k = solve_for_kappa(moments[2]);
        double a, x;
        if (std::fabs(k) <= kNearZero) {
            x = moments[0];
            a = moments[1] * std::sqrt(3.0) / kPi;
        } else {
            double U1 = g::function(1.0 + k) * g::function(1.0 - k);
            double U2 = g::function(1.0 + 2.0 * k) * g::function(1.0 - 2.0 * k);
            a = std::sqrt(moments[1] * moments[1] * k * k / (U2 - U1 * U1));
            x = moments[0] - a / k * (1.0 - U1);
        }
        return {x, a, k};
    }

    // Parameters from L-moments.
    // κ→0 limit: alpha = L2, xi = L1 (the sin(kappa*pi)/(kappa*pi) → 1 limit).
    std::vector<double> parameters_from_linear_moments(
        const std::vector<double>& moments) const override {
        double L1 = moments[0];
        double L2 = moments[1];
        double T3 = moments[2];
        // C# reads moments[3] (T4) but does not use it.
        double kappa = -T3;
        double alpha, xi;
        // INTENTIONAL DIVERGENCE from C#: upstream computes sin(kappa*pi)/(kappa*pi)
        // (and 1/kappa - pi/sin(kappa*pi)) with no guard, yielding NaN/Inf at kappa=0.
        // We return the correct L'Hopital limit instead. Oracle gate cannot verify this
        // point (C# returns NaN).
        if (std::fabs(kappa) <= kNearZero) {
            alpha = L2;
            xi = L1;
        } else {
            alpha = L2 * std::sin(kappa * kPi) / (kappa * kPi);
            xi = L1 - alpha * (1.0 / kappa - kPi / std::sin(kappa * kPi));
        }
        return {xi, alpha, kappa};
    }

    // L-moments from parameters.
    // κ→0 limit: L1=xi, L2=alpha, T3=0, T4=1/6.
    std::vector<double> linear_moments_from_parameters(
        const std::vector<double>& parameters) const override {
        double xi = parameters[0];
        double alpha = parameters[1];
        double kappa = parameters[2];
        if (std::fabs(kappa) >= 1.0)
            throw std::out_of_range("L-moments can only be defined for -1 < kappa < 1");
        double L1, L2;
        // INTENTIONAL DIVERGENCE from C#: upstream computes 1/kappa - pi/sin(kappa*pi)
        // (and kappa*pi/sin(kappa*pi)) with no guard, yielding NaN/Inf at kappa=0. We
        // return the correct L'Hopital limit instead. Oracle gate cannot verify this
        // point (C# returns NaN).
        if (std::fabs(kappa) <= kNearZero) {
            L1 = xi;
            L2 = alpha;
        } else {
            L1 = xi + alpha * (1.0 / kappa - kPi / std::sin(kappa * kPi));
            L2 = alpha * kappa * kPi / std::sin(kappa * kPi);
        }
        double T3 = -kappa;
        double T4 = (1.0 + 5.0 * kappa * kappa) / 6.0;
        return {L1, L2, T3, T4};
    }

    // Initial values and bounds for MLE optimization.
    void get_parameter_constraints(const std::vector<double>& sample,
                                   std::vector<double>& initials,
                                   std::vector<double>& lowers,
                                   std::vector<double>& uppers) const override {
        initials = parameters_from_linear_moments(data::linear_moments(sample));
        if (initials[0] == 0.0) initials[0] = kDoubleMachineEpsilon;
        lowers.resize(3);
        uppers.resize(3);
        lowers[0] = -std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        uppers[0] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[0])) + 1.0));
        lowers[1] = kDoubleMachineEpsilon;
        uppers[1] = std::pow(10.0, std::ceil(std::log10(std::fabs(initials[1]))) + 1.0);
        lowers[2] = -10.0;
        uppers[2] = 10.0;
        if (initials[2] <= lowers[2] || initials[2] >= uppers[2]) initials[2] = 0.0;
    }

    std::vector<double> mle(const std::vector<double>& sample) const {
        std::vector<double> initials, lowers, uppers;
        get_parameter_constraints(sample, initials, lowers, uppers);
        auto log_lh = [&sample](const std::vector<double>& x) {
            GeneralizedLogistic g;
            g.set_parameters(x[0], x[1], x[2]);
            return g.log_likelihood(sample);
        };
        math::optimization::NelderMead solver(log_lh, 3, initials, lowers, uppers);
        solver.maximize();
        return solver.best_parameters();
    }

   private:
    static double sign(double x) { return (x > 0.0) - (x < 0.0); }

    static bool validate(double location, double scale, double shape) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape)) return false;
        return true;
    }

    double xi_ = 0.0;
    double alpha_ = 0.0;
    double kappa_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
