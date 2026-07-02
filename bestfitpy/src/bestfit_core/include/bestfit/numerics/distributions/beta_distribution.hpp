// ported from: Numerics/Distributions/Univariate/BetaDistribution.cs @ a2c4dbf
//
// The Beta distribution with shape parameters α (alpha) and β (beta), defined on (0,1).
// Logic mirrors the C# source method-for-method. The C# class derives only from
// UnivariateDistributionBase and implements no estimation interfaces, so none are ported.
// The WPF helpers are desktop concerns and are not ported.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/special/beta.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

class BetaDistribution : public UnivariateDistributionBase {
   public:
    // Constructs a Beta distribution with α = β = 2, defined in the interval (0,1).
    BetaDistribution() { set_parameters(2.0, 2.0); }
    // Constructs a Beta distribution with the given parameters α and β.
    BetaDistribution(double alpha, double beta) { set_parameters(alpha, beta); }

    double alpha() const { return alpha_; }
    double beta() const { return beta_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Beta;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override { return {alpha_, beta_}; }

    void set_parameters(double alpha, double beta) {
        parameters_valid_ = validate(alpha, beta);
        alpha_ = alpha;
        beta_ = beta;
    }
    void set_parameters(const std::vector<double>& p) override { set_parameters(p[0], p[1]); }

    // --- Moments / support ---
    double mean() const override { return alpha_ / (alpha_ + beta_); }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (alpha_ == 1.0 && beta_ == 1.0) return 0.5;
        if (alpha_ <= 1.0 && beta_ > 1.0) return 0.0;
        if (alpha_ > 1.0 && beta_ <= 1.0) return 1.0;
        return (alpha_ - 1.0) / (alpha_ + beta_ - 2.0);
    }

    double standard_deviation() const override {
        return std::sqrt(alpha_ * beta_ /
                         ((alpha_ + beta_) * (alpha_ + beta_) * (alpha_ + beta_ + 1.0)));
    }

    double skewness() const override {
        return 2.0 * (beta_ - alpha_) * std::sqrt(alpha_ + beta_ + 1.0) /
               ((alpha_ + beta_ + 2.0) * std::sqrt(alpha_ * beta_));
    }

    double kurtosis() const override {
        double num = 6.0 * ((alpha_ - beta_) * (alpha_ - beta_) * (alpha_ + beta_ + 1.0) -
                            alpha_ * beta_ * (alpha_ + beta_ + 2.0));
        double den = alpha_ * beta_ * (alpha_ + beta_ + 2.0) * (alpha_ + beta_ + 3.0);
        return 3.0 + num / den;
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return 1.0; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("BetaDistribution: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        double constant = 1.0 / math::special::beta::function(alpha_, beta_);
        double a = std::pow(x, alpha_ - 1.0);
        double b = std::pow(1.0 - x, beta_ - 1.0);
        return constant * a * b;
    }

    double cdf(double x) const override {
        if (!parameters_valid_) throw std::invalid_argument("BetaDistribution: invalid parameters");
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        return math::special::beta::incomplete(alpha_, beta_, x);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("BetaDistribution: invalid parameters");
        return math::special::beta::incomplete_inverse(alpha_, beta_, probability);
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<BetaDistribution>(alpha_, beta_);
    }

   private:
    static bool validate(double alpha, double beta) {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) return false;
        if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0) return false;
        return true;
    }

    double alpha_ = 2.0;
    double beta_ = 2.0;
};

}  // namespace bestfit::numerics::distributions
