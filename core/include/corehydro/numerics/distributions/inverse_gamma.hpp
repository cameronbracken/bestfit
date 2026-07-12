// ported from: Numerics/Distributions/Univariate/InverseGamma.cs @ a2c4dbf
//
// Inverse-Gamma distribution with scale β (beta) and shape α (alpha). Logic mirrors
// the C# source method-for-method. No estimation interfaces are implemented upstream.
// CDF = UpperIncomplete(α, β/x); InverseCDF = β / InverseUpperIncomplete(α, p).
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_ig = corehydro::numerics::math::special;

class InverseGamma : public UnivariateDistributionBase {
   public:
    // Constructs an Inverse-Gamma distribution with scale β = 0.5 and shape α = 2.
    InverseGamma() { set_parameters(0.5, 2.0); }

    // Constructs an Inverse-Gamma distribution with given scale and shape.
    InverseGamma(double scale, double shape) { set_parameters(scale, shape); }

    double beta()  const { return beta_; }
    double alpha() const { return alpha_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::InverseGamma;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {beta_, alpha_}; }

    void set_parameters(double scale, double shape) {
        parameters_valid_ = validate(scale, shape);
        beta_  = scale;
        alpha_ = shape;
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override {
        if (alpha_ <= 1.0) return std::numeric_limits<double>::quiet_NaN();
        return beta_ / (alpha_ - 1.0);
    }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override { return beta_ / (alpha_ + 1.0); }

    double standard_deviation() const override {
        if (alpha_ <= 2.0) return std::numeric_limits<double>::quiet_NaN();
        return beta_ / (std::fabs(alpha_ - 1.0) * std::sqrt(alpha_ - 2.0));
    }

    double skewness() const override {
        if (alpha_ <= 3.0) return std::numeric_limits<double>::quiet_NaN();
        return 4.0 * std::sqrt(alpha_ - 2.0) / (alpha_ - 3.0);
    }

    double kurtosis() const override {
        if (alpha_ <= 4.0) return std::numeric_limits<double>::quiet_NaN();
        return 3.0 + 6.0 * (5.0 * alpha_ - 11.0) / ((alpha_ - 3.0) * (alpha_ - 4.0));
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("InverseGamma: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        if (x == 0.0) return std::numeric_limits<double>::quiet_NaN();
        return std::pow(beta_, alpha_) / sf_ig::function(alpha_)
               * std::pow(x, -alpha_ - 1.0) * std::exp(-beta_ / x);
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("InverseGamma: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return sf_ig::upper_incomplete(alpha_, beta_ / x);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("InverseGamma: invalid parameters");
        return beta_ / sf_ig::inverse_upper_incomplete(alpha_, probability);
    }

    // --- Parameter display names (X1; C# InverseGamma.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Scale (\xCE\xB2)", "Shape (\xCE\xB1)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xB2", "\xCE\xB1"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<InverseGamma>(beta_, alpha_);
    }

   private:
    static bool validate(double scale, double shape) {
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        if (std::isnan(shape) || std::isinf(shape) || shape <= 0.0) return false;
        return true;
    }

    double beta_  = 0.5;
    double alpha_ = 2.0;
};

}  // namespace corehydro::numerics::distributions
