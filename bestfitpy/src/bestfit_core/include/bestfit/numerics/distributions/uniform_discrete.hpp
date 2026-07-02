// ported from: Numerics/Distributions/Univariate/UniformDiscrete.cs @ a2c4dbf
//
// The discrete Uniform distribution on integer support [min, max]. Logic mirrors the
// C# source method-for-method. No estimation interfaces are implemented upstream.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::numerics::distributions {

class UniformDiscrete : public UnivariateDistributionBase {
   public:
    UniformDiscrete() { set_parameters(0.0, 1.0); }
    UniformDiscrete(double min, double max) { set_parameters(min, max); }

    double min() const { return min_; }
    double max() const { return max_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::UniformDiscrete;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {min_, max_}; }

    void set_parameters(double min, double max) {
        min_ = min;
        max_ = max;
        parameters_valid_ = validate(min, max);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Derived quantity: N = Max - Min + 1 ---
    double n() const { return max_ - min_ + 1.0; }

    // --- Moments / support ---
    double mean() const override { return (min_ + max_) / 2.0; }
    double median() const override { return (min_ + max_) / 2.0; }
    double mode() const override { return kNaN; }
    double standard_deviation() const override {
        double nn = max_ - min_ + 1.0;
        return std::sqrt((nn * nn - 1.0) / 12.0);
    }
    double skewness() const override { return 0.0; }
    double kurtosis() const override {
        double nn = n();
        return 3.0 - 6.0 * (nn * nn + 1.0) / (5.0 * (nn * nn - 1.0));
    }
    double minimum() const override { return min_; }
    double maximum() const override { return max_; }

    // --- Distribution functions ---
    double pdf(double k) const override {
        if (k < minimum() || k > maximum()) return 0.0;
        return 1.0 / n();
    }

    double cdf(double k) const override {
        if (k < minimum()) return 0.0;
        if (k > maximum()) return 1.0;
        return (k - min_ + 1.0) / n();
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return std::floor(min_ + probability * n());
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<UniformDiscrete>(min_, max_);
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
