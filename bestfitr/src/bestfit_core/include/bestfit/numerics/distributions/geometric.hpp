// ported from: Numerics/Distributions/Univariate/Geometric.cs @ a2c4dbf
//
// The Geometric distribution (number of failures before first success), support {0,1,2,...}.
// Single parameter p = probability of success. Logic mirrors the C# source method-for-method.
// No estimation interfaces are implemented upstream.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::numerics::distributions {

class Geometric : public UnivariateDistributionBase {
   public:
    Geometric() { set_parameters({0.5}); }
    explicit Geometric(double probability) { set_parameters({probability}); }

    double probability_of_success() const { return p_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Geometric;
    }
    int number_of_parameters() const override { return 1; }
    std::vector<double> get_parameters() const override { return {p_}; }

    void set_parameters(const std::vector<double>& params) override {
        p_ = params[0];
        parameters_valid_ = validate(p_);
    }

    // --- Moments / support ---
    double mean() const override { return (1.0 - p_) / p_; }

    double median() const override {
        if (p_ == 0.0) return kInf;
        if (p_ == 1.0) return 1.0;
        // C#: Math.Ceiling(-1 / Math.Log(1-p, 2)) - 1
        return std::ceil(-1.0 / (std::log(1.0 - p_) / std::log(2.0))) - 1.0;
    }

    double mode() const override { return 0.0; }

    double standard_deviation() const override {
        return std::sqrt(1.0 - p_) / p_;
    }

    double skewness() const override {
        return (2.0 - p_) / std::sqrt(1.0 - p_);
    }

    // C# formula: 3 + 6 + p^2 / (1 - p)  (mirrors upstream verbatim)
    double kurtosis() const override {
        return 3.0 + 6.0 + p_ * p_ / (1.0 - p_);
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    // PDF(k) = (1-p)^k * p
    double pdf(double k) const override {
        if (k < minimum() || k > maximum()) return 0.0;
        return std::pow(1.0 - p_, k) * p_;
    }

    // CDF(k) = 1 - (1-p)^(k+1)
    double cdf(double k) const override {
        if (k < minimum()) return 0.0;
        if (k >= maximum()) return 1.0;
        return 1.0 - std::pow(1.0 - p_, k + 1.0);
    }

    // InverseCDF: ceiling(log_{1-p}(1-prob)) - 1
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        // C#: Math.Ceiling(Math.Log(1-prob, 1-p)) - 1
        // Math.Log(x, base) = ln(x)/ln(base)
        return std::ceil(std::log(1.0 - probability) / std::log(1.0 - p_)) - 1.0;
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Geometric>(p_);
    }

   private:
    static bool validate(double p) {
        if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0) return false;
        return true;
    }

    double p_ = 0.5;
};

}  // namespace bestfit::numerics::distributions
