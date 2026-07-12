// ported from: Numerics/Distributions/Univariate/Rayleigh.cs @ a2c4dbf
//
// The Rayleigh distribution with scale σ. Logic mirrors the C# source method-for-method.
// The C# MLE uses Gamma.LogGamma — std::lgamma from <cmath> is used here (C++17 standard).
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/i_estimation.hpp"
#include "corehydro/numerics/distributions/base/parameter_estimation_method.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class Rayleigh : public UnivariateDistributionBase, public IEstimation {
   public:
    Rayleigh() { set_parameters(10.0); }
    explicit Rayleigh(double scale) { set_parameters(scale); }

    double sigma() const { return sigma_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Rayleigh;
    }
    int number_of_parameters() const override { return 1; }
    std::vector<double> get_parameters() const override { return {sigma_}; }

    void set_parameters(double scale) {
        sigma_ = scale;
        parameters_valid_ = validate(scale);
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0]); }

    // --- Moments / support ---
    double mean() const override { return sigma_ * std::sqrt(kPi / 2.0); }
    double median() const override { return sigma_ * std::sqrt(std::log(4.0)); }
    double mode() const override { return sigma_; }
    double standard_deviation() const override {
        return std::sqrt((4.0 - kPi) / 2.0 * sigma_ * sigma_);
    }
    double skewness() const override {
        return 2.0 * std::sqrt(kPi) * (kPi - 3.0) / std::pow(4.0 - kPi, 1.5);
    }
    double kurtosis() const override {
        double num = -(6.0 * kPi * kPi - 24.0 * kPi + 16.0);
        double den = (4.0 - kPi) * (4.0 - kPi);
        return 3.0 + num / den;
    }
    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (x < minimum()) return 0.0;
        return x / (sigma_ * sigma_) * std::exp(-x * x / (2.0 * sigma_ * sigma_));
    }

    double cdf(double x) const override {
        if (x <= minimum()) return 0.0;
        return 1.0 - std::exp(-x * x / (2.0 * sigma_ * sigma_));
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return sigma_ * std::sqrt(-2.0 * std::log(1.0 - probability));
    }

    // --- Parameter display names (X1; C# Rayleigh.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Scale (\xCF\x83)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCF\x83"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Rayleigh>(sigma_);
    }

    // --- Estimation ---
    void estimate(const std::vector<double>& sample, ParameterEstimationMethod method) override {
        if (method == ParameterEstimationMethod::MethodOfMoments) {
            // sigma = mean / sqrt(pi/2); product_moments()[0] is the sample mean
            set_parameters(data::product_moments(sample)[0] / std::sqrt(kPi / 2.0));
        } else if (method == ParameterEstimationMethod::MaximumLikelihood) {
            // biased MLE estimator, then bias-corrected (mirrors C# Gamma.LogGamma via std::lgamma)
            double sum = 0.0;
            for (double v : sample) sum += v * v;
            double n = static_cast<double>(sample.size());
            double biased = std::sqrt(sum / (2.0 * n));
            // bias correction: exp(lgamma(n) + 0.5*log(n) - lgamma(n+0.5))
            double log_num = std::lgamma(n) + 0.5 * std::log(n);
            double log_den = std::lgamma(n + 0.5);
            double correction;
            if (std::isinf(log_num) && std::isinf(log_den)) {
                correction = 1.0;
            } else {
                correction = std::exp(log_num - log_den);
            }
            set_parameters(biased * correction);
        } else {
            throw std::invalid_argument("estimation method not implemented for Rayleigh");
        }
    }

   private:
    static bool validate(double scale) {
        if (std::isnan(scale) || std::isinf(scale) || scale <= 0.0) return false;
        return true;
    }

    double sigma_ = 0.0;
};

}  // namespace corehydro::numerics::distributions
