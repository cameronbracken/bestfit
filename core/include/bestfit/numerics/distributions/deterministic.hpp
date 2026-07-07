// ported from: Numerics/Distributions/Univariate/Deterministic.cs @ a2c4dbf
//
// Deterministic (degenerate) point-mass distribution with a single value. Logic mirrors
// the C# source method-for-method. PDF returns 1 at the value and 0 elsewhere (mirroring
// the C# integer-spike convention). Minimum == Maximum == Value (degenerate support).
// SD, skewness, and kurtosis are undefined (NaN). Implements IEstimation (MoM only).
// The WPF PDF-graph builder is not ported.
#pragma once
#include <string>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/i_estimation.hpp"
#include "bestfit/numerics/distributions/base/parameter_estimation_method.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::numerics::distributions {

class Deterministic : public UnivariateDistributionBase, public IEstimation {
   public:
    Deterministic() { set_parameters(0.5); }
    explicit Deterministic(double value) { set_parameters(value); }

    double value() const { return value_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Deterministic;
    }
    int number_of_parameters() const override { return 1; }
    std::vector<double> get_parameters() const override { return {value_}; }

    void set_parameters(double val) {
        value_ = val;
        parameters_valid_ = validate(val);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0]); }

    // --- Moments / support ---
    double mean() const override { return value_; }
    double median() const override { return value_; }
    double mode() const override { return value_; }
    // SD, skewness, kurtosis are undefined for a degenerate point-mass distribution.
    double standard_deviation() const override { return kNaN; }
    double skewness() const override { return kNaN; }
    double kurtosis() const override { return kNaN; }
    // Degenerate support: the entire probability mass is at the single point.
    double minimum() const override { return value_; }
    double maximum() const override { return value_; }

    // --- Distribution functions ---
    // PDF returns 1 at the mass point and 0 elsewhere (matches C# spike convention).
    double pdf(double x) const override {
        if (x != value_) return 0.0;
        return 1.0;
    }

    // CDF is a unit step at Value.
    double cdf(double x) const override {
        if (x < value_) return 0.0;
        return 1.0;
    }

    // InverseCDF always returns Value (the only point with non-zero mass).
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        return value_;
    }

    // --- Parameter display names (X1; C# Deterministic.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Value"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"Value"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Deterministic>(value_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            double sum = std::accumulate(sample.begin(), sample.end(), 0.0);
            set_parameters(sum / static_cast<double>(sample.size()));
        } else {
            throw std::invalid_argument("Deterministic: only MethodOfMoments is supported");
        }
    }

   private:
    static bool validate(double val) {
        if (std::isnan(val) || std::isinf(val)) return false;
        return true;
    }

    double value_ = 0.0;
};

}  // namespace bestfit::numerics::distributions
