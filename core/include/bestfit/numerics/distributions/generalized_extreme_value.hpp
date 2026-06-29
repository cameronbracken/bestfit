// ported from: Numerics/Distributions/Univariate/GeneralizedExtremeValue.cs @ <pending-sha>
//
// Generalized Extreme Value distribution: parameters ξ (location), α (scale),
// κ (shape). Distribution-core surface (moments, PDF/CDF/InverseCDF, log-likelihood).
// L-moment / MLE estimation lands in a later increment (needs Brent, Statistics,
// NelderMead). Logic mirrors the C# source method-for-method.
#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include "bestfit/numerics/math/special/gamma.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class GeneralizedExtremeValue {
   public:
    GeneralizedExtremeValue() { set_parameters(100.0, 10.0, 0.0); }
    GeneralizedExtremeValue(double location, double scale, double shape) {
        set_parameters(location, scale, shape);
    }

    double xi() const { return xi_; }
    double alpha() const { return alpha_; }
    double kappa() const { return kappa_; }
    bool parameters_valid() const { return parameters_valid_; }

    void set_parameters(double location, double scale, double shape) {
        xi_ = location;
        alpha_ = scale;
        kappa_ = shape;
        parameters_valid_ = validate(location, scale, shape);
    }

    // --- Moments ---
    double mean() const {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero) return xi_ + alpha_ * kEuler;
        if (std::fabs(kappa_) < 1.0) return xi_ + (alpha_ / kappa_ * (1.0 - g::function(1.0 + kappa_)));
        return kNaN;
    }

    double median() const {
        if (std::fabs(kappa_) <= kNearZero) return xi_ - alpha_ * std::log(std::log(2.0));
        return xi_ + alpha_ * (std::pow(std::log(2.0), -kappa_) - 1.0) / kappa_;
    }

    double mode() const {
        if (std::fabs(kappa_) <= kNearZero) return xi_;
        return xi_ + alpha_ * (std::pow(1.0 + kappa_, -kappa_) - 1.0) / kappa_;
    }

    double standard_deviation() const {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero)
            return std::sqrt(alpha_ * alpha_ * M_PI * M_PI / 6.0);
        if (std::fabs(kappa_) < 0.5) {
            double g1 = g::function(1.0 + kappa_);
            double g2 = g::function(1.0 + 2.0 * kappa_);
            return std::sqrt(alpha_ * alpha_ * (g2 - g1 * g1) / (kappa_ * kappa_));
        }
        return kNaN;
    }

    double skewness() const {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero) return 1.1396;
        if (std::fabs(kappa_) < 1.0 / 3.0) {
            double U1 = g::function(1.0 + kappa_);
            double U2 = g::function(1.0 + 2.0 * kappa_);
            double U3 = g::function(1.0 + 3.0 * kappa_);
            return sign(kappa_) * (-U3 + 3.0 * U1 * U2 - 2.0 * std::pow(U1, 3.0)) /
                   std::pow(U2 - U1 * U1, 1.5);
        }
        return kNaN;
    }

    double kurtosis() const {
        namespace g = math::special;
        if (std::fabs(kappa_) <= kNearZero) return 3.0 + 12.0 / 5.0;
        if (std::fabs(kappa_) < 0.25) {
            double U1 = g::function(1.0 + kappa_);
            double U2 = g::function(1.0 + 2.0 * kappa_);
            double U3 = g::function(1.0 + 3.0 * kappa_);
            double U4 = g::function(1.0 + 4.0 * kappa_);
            double num = U4 - 4.0 * U3 * U1 - 3.0 * U2 * U2 + 12.0 * U2 * U1 * U1 -
                         6.0 * std::pow(U1, 4.0);
            double den = std::pow(U2 - U1 * U1, 2.0);
            return 3.0 + num / den;
        }
        return kNaN;
    }

    double minimum() const {
        if (kappa_ >= -kNearZero) return -kInf;
        return xi_ + alpha_ / kappa_;
    }
    double maximum() const {
        if (kappa_ <= kNearZero) return kInf;
        return xi_ + alpha_ / kappa_;
    }

    // --- Distribution functions ---
    double pdf(double x) const {
        if (x < minimum() || x > maximum()) return 0.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero) y = -std::log(1.0 - kappa_ * y) / kappa_;
        return std::exp(-(1.0 - kappa_) * y - std::exp(-y)) / alpha_;
    }

    double cdf(double x) const {
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        double y = (x - xi_) / alpha_;
        if (std::fabs(kappa_) > kNearZero) y = -std::log(1.0 - kappa_ * y) / kappa_;
        return std::exp(-std::exp(-y));
    }

    double inverse_cdf(double probability) const {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (std::fabs(kappa_) <= kNearZero)
            return xi_ - alpha_ * std::log(-std::log(probability));
        return xi_ + alpha_ / kappa_ * (1.0 - std::pow(-std::log(probability), kappa_));
    }

    double log_pdf(double x) const { return std::log(pdf(x)); }

    double log_likelihood(const std::vector<double>& sample) const {
        double ll = 0.0;
        for (double v : sample) ll += log_pdf(v);
        if (std::isnan(ll) || std::isinf(ll)) return -kInf;
        return ll;
    }

   private:
    static constexpr double kNearZero = 1E-4;  // UnivariateDistributionBase.NearZero
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    static constexpr double kInf = std::numeric_limits<double>::infinity();

    static double sign(double x) { return (x > 0) - (x < 0); }

    static bool validate(double location, double scale, double shape) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape)) return false;
        return true;
    }

    double xi_ = 0.0;
    double alpha_ = 0.0;
    double kappa_ = 0.0;
    bool parameters_valid_ = false;
};

}  // namespace bestfit::numerics::distributions
