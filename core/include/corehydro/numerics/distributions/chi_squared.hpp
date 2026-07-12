// ported from: Numerics/Distributions/Univariate/ChiSquared.cs @ a2c4dbf
//
// Chi-Squared (χ²) distribution with degrees of freedom ν. Logic mirrors the C# source
// method-for-method. The WPF helpers are not ported (desktop concerns).
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_chi2 = corehydro::numerics::math::special;

class ChiSquared : public UnivariateDistributionBase {
   public:
    // Constructs a Chi-Squared distribution with 10 degrees of freedom.
    ChiSquared() { set_parameters(10.0); }

    // Constructs a Chi-Squared distribution with given degrees of freedom.
    explicit ChiSquared(int degrees_of_freedom) {
        set_parameters(static_cast<double>(degrees_of_freedom));
    }

    int degrees_of_freedom() const { return dof_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::ChiSquared;
    }
    int number_of_parameters() const override { return 1; }
    std::vector<double> get_parameters() const override {
        return {static_cast<double>(dof_)};
    }

    void set_parameters(double v) {
        dof_ = static_cast<int>(v);
        parameters_valid_ = validate(dof_);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0]); }

    // --- Moments / support ---
    double mean() const override { return static_cast<double>(dof_); }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override { return std::max(static_cast<double>(dof_) - 2.0, 0.0); }

    double standard_deviation() const override {
        return std::sqrt(2.0 * static_cast<double>(dof_));
    }

    double skewness() const override {
        return std::sqrt(8.0 / static_cast<double>(dof_));
    }

    double kurtosis() const override {
        return 3.0 + 12.0 / static_cast<double>(dof_);
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("ChiSquared: invalid parameters");
        if (x < minimum()) return 0.0;
        double v = static_cast<double>(dof_);
        // Edge case x == 0: PDF(0) = 0 for v>2, 0.5 for v==2, +Inf for v<2
        if (x == 0.0) {
            if (v > 2.0) return 0.0;
            if (v == 2.0) return 0.5;
            return kInf;
        }
        // Compute in log-space to avoid overflow for large degrees of freedom
        double log_pdf = ((v - 2.0) / 2.0) * std::log(x) - x / 2.0
                         - (v / 2.0) * std::log(2.0) - sf_chi2::log_gamma(v / 2.0);
        return std::exp(log_pdf);
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("ChiSquared: invalid parameters");
        if (x <= minimum()) return 0.0;
        double v = static_cast<double>(dof_);
        return sf_chi2::lower_incomplete(v / 2.0, x / 2.0);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("ChiSquared: invalid parameters");
        double v = static_cast<double>(dof_);
        return sf_chi2::inverse_lower_incomplete(v / 2.0, probability) * 2.0;
    }

    // --- Parameter display names (X1; C# ChiSquared.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Degrees of Freedom (\xCE\xBD)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBD"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<ChiSquared>(dof_);
    }

   private:
    static bool validate(int dof) { return dof >= 1; }

    int dof_ = 10;
};

}  // namespace corehydro::numerics::distributions
