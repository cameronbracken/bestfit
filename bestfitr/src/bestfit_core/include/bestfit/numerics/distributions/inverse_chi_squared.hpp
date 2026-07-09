// ported from: Numerics/Distributions/Univariate/InverseChiSquared.cs @ a2c4dbf
//
// Inverse Chi-Squared (Inv-χ²) distribution with degrees of freedom ν and scale σ.
// Mirrors the C# source method-for-method. No estimation interfaces upstream.
// CDF = UpperIncomplete(ν/2, ν·σ/(2x)); InverseCDF = ν·σ / (2·InverseUpperIncomplete(ν/2, p)).
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/special/gamma.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

namespace sf_ics = bestfit::numerics::math::special;

class InverseChiSquared : public UnivariateDistributionBase {
   public:
    // Constructs an Inverse Chi-Squared distribution with 10 degrees of freedom and σ = 1.
    InverseChiSquared() { set_parameters(10.0, 1.0); }

    // Constructs an Inverse Chi-Squared distribution with given degrees of freedom and scale σ.
    InverseChiSquared(int degrees_of_freedom, double scale) {
        set_parameters(static_cast<double>(degrees_of_freedom), scale);
    }

    int    degrees_of_freedom() const { return dof_; }
    double sigma()              const { return sigma_; }
    // Precision = 1/σ^2
    double precision()          const { return 1.0 / (sigma_ * sigma_); }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::InverseChiSquared;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override {
        return {static_cast<double>(dof_), sigma_};
    }

    void set_parameters(double dof, double sigma) {
        dof_   = static_cast<int>(dof);
        sigma_ = sigma;
        parameters_valid_ = validate(dof_, sigma_);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override {
        if (dof_ <= 2) return std::numeric_limits<double>::quiet_NaN();
        double v = static_cast<double>(dof_);
        return v * sigma_ / (v - 2.0);
    }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        double v = static_cast<double>(dof_);
        return v * sigma_ / (v + 2.0);
    }

    double standard_deviation() const override {
        if (dof_ <= 4) return std::numeric_limits<double>::quiet_NaN();
        double v = static_cast<double>(dof_);
        double t2 = sigma_;
        return std::sqrt(2.0 * v * v * t2 * t2 / ((v - 2.0) * (v - 2.0) * (v - 4.0)));
    }

    double skewness() const override {
        if (dof_ <= 6) return std::numeric_limits<double>::quiet_NaN();
        double v = static_cast<double>(dof_);
        return 4.0 / (v - 6.0) * std::sqrt(2.0 * (v - 4.0));
    }

    double kurtosis() const override {
        if (dof_ <= 8) return std::numeric_limits<double>::quiet_NaN();
        double v = static_cast<double>(dof_);
        return 3.0 + 12.0 * (5.0 * v - 22.0) / ((v - 6.0) * (v - 8.0));
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("InverseChiSquared: invalid parameters");
        if (x < minimum()) return 0.0;
        double v  = static_cast<double>(dof_);
        double t2 = sigma_;
        double a  = std::pow(t2 * v / 2.0, v / 2.0);
        double b  = sf_ics::function(v / 2.0);
        double c  = std::exp(-v * t2 / (2.0 * x));
        double d  = std::pow(x, 1.0 + v / 2.0);
        return a / b * (c / d);
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("InverseChiSquared: invalid parameters");
        if (x <= minimum()) return 0.0;
        double v  = static_cast<double>(dof_);
        double t2 = sigma_;
        return sf_ics::upper_incomplete(v / 2.0, v * t2 / (2.0 * x));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("InverseChiSquared: invalid parameters");
        double v  = static_cast<double>(dof_);
        double t2 = sigma_;
        return v * t2 / (2.0 * sf_ics::inverse_upper_incomplete(v / 2.0, probability));
    }

    // --- Parameter display names (X1; C# InverseChiSquared.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Degrees of Freedom (\xCE\xBD)", "Scale (\xCF\x83)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBD", "\xCF\x83"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<InverseChiSquared>(dof_, sigma_);
    }

   private:
    static bool validate(int dof, double sigma) {
        if (dof < 1) return false;
        if (std::isnan(sigma) || std::isinf(sigma) || sigma <= 0.0) return false;
        return true;
    }

    int    dof_   = 10;
    double sigma_ = 1.0;
};

}  // namespace bestfit::numerics::distributions
