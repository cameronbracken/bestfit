// ported from: Numerics/Distributions/Univariate/Uniform.cs @ a2c4dbf
//
// The continuous Uniform distribution on [min, max]. Logic mirrors the C# source
// method-for-method. Uniform has no parameter-estimation methods upstream.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::numerics::distributions {

class Uniform : public UnivariateDistributionBase {
   public:
    Uniform() { set_parameters(0.0, 1.0); }
    Uniform(double min, double max) { set_parameters(min, max); }

    double min() const { return min_; }
    double max() const { return max_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override { return UnivariateDistributionType::Uniform; }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {min_, max_}; }

    void set_parameters(double min, double max) {
        min_ = min;
        max_ = max;
        parameters_valid_ = validate(min, max);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return (min_ + max_) / 2.0; }
    double median() const override { return (min_ + max_) / 2.0; }
    double mode() const override { return kNaN; }
    double standard_deviation() const override { return (max_ - min_) / std::sqrt(12.0); }
    double skewness() const override { return 0.0; }
    double kurtosis() const override { return 9.0 / 5.0; }
    double minimum() const override { return min_; }
    double maximum() const override { return max_; }

    // --- Distribution functions ---
    // PDF/CDF/InverseCDF mirror the C# `if (_parametersValid == false)
    // ValidateParameters(Min, Max, true)` guard: evaluating with invalid parameters
    // (NaN/inf bounds or min > max) throws (C# ArgumentOutOfRangeException ->
    // std::out_of_range). Added in M10 -- the MixtureModel degenerate-data contract
    // (constant data => auto-fit Uniform(eps, 0) prior) relies on this throw.
    double pdf(double x) const override {
        if (!parameters_valid_)
            throw std::out_of_range("The min cannot be greater than the max.");
        if (min_ == max_) return 0.0;
        if (x < minimum() || x > maximum()) return 0.0;
        return 1.0 / (max_ - min_);
    }

    double cdf(double x) const override {
        if (!parameters_valid_)
            throw std::out_of_range("The min cannot be greater than the max.");
        if (min_ == max_) return 1.0;
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return (x - min_) / (max_ - min_);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (min_ == max_) return min_;
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_)
            throw std::out_of_range("The min cannot be greater than the max.");
        return min_ + probability * (max_ - min_);
    }

    // --- Parameter display names (X1; C# Uniform.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Min", "Max"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"Min", "Max"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Uniform>(min_, max_);
    }

   private:
    static bool validate(double min, double max) {
        if (std::isnan(min) || std::isinf(min) || std::isnan(max) || std::isinf(max) || min > max)
            return false;
        return true;
    }

    double min_ = 0.0;
    double max_ = 1.0;
};

}  // namespace bestfit::numerics::distributions
