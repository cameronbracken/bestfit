// ported from: Numerics/Distributions/Univariate/Cauchy.cs @ a2c4dbf
//
// The Cauchy distribution with location X0 and scale γ (gamma). Logic mirrors the C# source
// method-for-method. Mean, standard deviation, skewness, and kurtosis are all undefined
// (NaN), matching the C# return values exactly. No estimation interfaces are implemented
// because the C# class implements none. The WPF helpers and IBootstrappable are not ported.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Cauchy : public UnivariateDistributionBase {
   public:
    Cauchy() { set_parameters(0.0, 1.0); }
    Cauchy(double location, double scale) { set_parameters(location, scale); }

    double x0() const { return x0_; }
    double gamma_scale() const { return gamma_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Cauchy;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {x0_, gamma_}; }

    void set_parameters(double location, double scale) {
        x0_ = location;
        gamma_ = scale;
        parameters_valid_ = validate(location, scale);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    // Mean, SD, skewness, kurtosis are undefined for the Cauchy distribution.
    double mean() const override { return kNaN; }
    double median() const override { return x0_; }
    double mode() const override { return x0_; }
    double standard_deviation() const override { return kNaN; }
    double skewness() const override { return kNaN; }
    double kurtosis() const override { return kNaN; }
    double minimum() const override { return -kInf; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        double z = (x - x0_) / gamma_;
        return 1.0 / (kPi * gamma_ * (1.0 + z * z));
    }

    double cdf(double x) const override {
        return 1.0 / kPi * std::atan2(x - x0_, gamma_) + 0.5;
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return x0_ + gamma_ * std::tan(kPi * (probability - 0.5));
    }

    // --- Parameter display names (X1; C# Cauchy.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Location (X0)", "Scale (\xCE\xB3)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"X0", "\xCE\xB3"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Cauchy>(x0_, gamma_);
    }

   private:
    static bool validate(double location, double scale) {
        if (std::isnan(location) || std::isinf(location)) return false;
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        return true;
    }

    double x0_ = 0.0;
    double gamma_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
