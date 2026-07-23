// ported from: Numerics/Distributions/Univariate/Pareto.cs @ 2a0357a
//
// The Pareto distribution with scale Xm and shape α. Logic mirrors the C# source
// method-for-method. The C# class has no estimation interfaces; only
// UnivariateDistributionBase is implemented.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Pareto : public UnivariateDistributionBase {
   public:
    Pareto() { set_parameters(1.0, 10.0); }
    Pareto(double scale, double shape) { set_parameters(scale, shape); }

    double xm() const { return xm_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Pareto;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {xm_, alpha_}; }

    void set_parameters(double scale, double shape) {
        xm_ = scale;
        alpha_ = shape;
        parameters_valid_ = validate(scale, shape);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override {
        if (alpha_ <= 1.0) return kInf;
        return alpha_ * xm_ / (alpha_ - 1.0);
    }
    double median() const override { return xm_ * std::pow(2.0, 1.0 / alpha_); }
    double mode() const override { return xm_; }
    double standard_deviation() const override {
        if (alpha_ <= 2.0) return kInf;
        return xm_ * std::sqrt(alpha_) / (std::fabs(alpha_ - 1.0) * std::sqrt(alpha_ - 2.0));
    }
    double skewness() const override {
        if (alpha_ <= 3.0) return kNaN;
        return 2.0 * (alpha_ + 1.0) / (alpha_ - 3.0) * std::sqrt((alpha_ - 2.0) / alpha_);
    }
    double kurtosis() const override {
        if (alpha_ <= 4.0) return kNaN;
        double num = 6.0 * (alpha_ * alpha_ * alpha_ + alpha_ * alpha_ - 6.0 * alpha_ - 2.0);
        double den = alpha_ * (alpha_ - 3.0) * (alpha_ - 4.0);
        return 3.0 + num / den;
    }
    double minimum() const override { return xm_; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (x < minimum() || x > maximum()) return 0.0;
        return alpha_ * std::pow(xm_, alpha_) / std::pow(x, alpha_ + 1.0);
    }

    double cdf(double x) const override {
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return 1.0 - std::pow(xm_ / x, alpha_);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return xm_ * std::pow(1.0 - probability, -1.0 / alpha_);
    }

    // --- Parameter display names (X1; C# Pareto.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Scale (Xm)", "Shape (\xCE\xB1)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"Xm", "\xCE\xB1"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Pareto>(xm_, alpha_);
    }

   private:
    static bool validate(double scale, double shape) {
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape) || shape <= 0.0) return false;
        return true;
    }

    double xm_ = 0.0;
    double alpha_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
