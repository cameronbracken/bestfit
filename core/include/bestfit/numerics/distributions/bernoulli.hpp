// ported from: Numerics/Distributions/Univariate/Bernoulli.cs @ a2c4dbf
//
// The Bernoulli distribution on two-point support {0, 1}. Single parameter p (probability of 1).
// Logic mirrors the C# source method-for-method. No estimation interfaces are implemented upstream.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::numerics::distributions {

class Bernoulli : public UnivariateDistributionBase {
   public:
    Bernoulli() { set_parameters({0.5}); }
    explicit Bernoulli(double probability) { set_parameters({probability}); }

    double probability() const { return p_; }
    double complement() const { return 1.0 - p_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Bernoulli;
    }
    int number_of_parameters() const override { return 1; }
    std::vector<double> get_parameters() const override { return {p_}; }

    void set_parameters(const std::vector<double>& params) override {
        p_ = params[0];
        parameters_valid_ = validate(p_);
    }

    // --- Moments / support ---
    double mean() const override { return p_; }

    double median() const override {
        if (p_ < 0.5) return 0.0;
        if (p_ > 0.5) return 1.0;
        return 0.5;
    }

    double mode() const override { return p_ > 0.5 ? 1.0 : 0.0; }

    double standard_deviation() const override {
        return std::sqrt(p_ * complement());
    }

    double skewness() const override {
        if (p_ == 0.0 || p_ == 1.0) return kNaN;
        return (complement() - p_) / std::sqrt(p_ * complement());
    }

    // C# formula: 3 + (1 - 6 * q * p) / (p * q)
    double kurtosis() const override {
        if (p_ == 0.0 || p_ == 1.0) return kNaN;
        double q = complement();
        return 3.0 + (1.0 - 6.0 * q * p_) / (p_ * q);
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return 1.0; }

    // --- Distribution functions ---
    double pdf(double k) const override {
        if (k < minimum() || k > maximum()) return 0.0;
        if (k == 0.0) return complement();
        if (k == 1.0) return p_;
        return 0.0;
    }

    // CDF: k < 0 → 0; k >= 1 → 1; else → complement
    double cdf(double k) const override {
        if (k < minimum()) return 0.0;
        if (k >= maximum()) return 1.0;
        return complement();
    }

    // InverseCDF: if prob > complement → 1; else → 0
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (probability > complement()) return 1.0;
        return 0.0;
    }

    // --- Parameter display names (X1; C# Bernoulli.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Probability (p)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"p"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Bernoulli>(p_);
    }

   private:
    static bool validate(double p) {
        if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0) return false;
        return true;
    }

    double p_ = 0.5;
};

}  // namespace bestfit::numerics::distributions
