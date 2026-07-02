// ported from: Numerics/Distributions/Univariate/Binomial.cs @ a2c4dbf
//
// The Binomial distribution with parameters p (probability of success) and n (number of trials).
// Discrete; PMF via binomial_coefficient; CDF via regularized incomplete beta (Beta.Incomplete).
// InverseCDF via integer walk 0..n. No estimation interfaces upstream.
// Logic mirrors the C# source method-for-method.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/special/beta.hpp"
#include "bestfit/numerics/math/special/factorial.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::distributions {

namespace sf_bin = bestfit::numerics::math::special;

class Binomial : public UnivariateDistributionBase {
   public:
    // Constructs a Binomial distribution with p=0.5 and n=10.
    Binomial() { set_parameters({0.5, 10.0}); }

    // Constructs a Binomial distribution with given probability and number of trials.
    Binomial(double probability, int number_of_trials) {
        set_parameters({probability, static_cast<double>(number_of_trials)});
    }

    double probability_of_success() const { return p_; }
    double complement() const { return 1.0 - p_; }
    int number_of_trials() const { return n_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Binomial;
    }
    int number_of_parameters() const override { return 2; }
    std::vector<double> get_parameters() const override {
        return {p_, static_cast<double>(n_)};
    }

    void set_parameters(const std::vector<double>& params) override {
        p_ = params[0];
        n_ = static_cast<int>(params[1]);
        parameters_valid_ = validate(p_, n_);
    }

    // --- Moments / support ---
    double mean() const override { return static_cast<double>(n_) * p_; }

    double median() const override { return std::ceil(static_cast<double>(n_) * p_); }

    double mode() const override {
        if (p_ == 1.0) return static_cast<double>(n_);
        if (p_ == 0.0) return 0.0;
        return static_cast<double>(static_cast<int>(
            std::floor((static_cast<double>(n_) + 1.0) * p_)));
    }

    double standard_deviation() const override {
        return std::sqrt(static_cast<double>(n_) * p_ * complement());
    }

    double skewness() const override {
        return (1.0 - 2.0 * p_) / std::sqrt(static_cast<double>(n_) * p_ * complement());
    }

    double kurtosis() const override {
        double nf = static_cast<double>(n_);
        double q = complement();
        return 3.0 + (1.0 - 6.0 * q * p_) / (nf * p_ * q);
    }

    double minimum() const override { return 0.0; }
    double maximum() const override { return static_cast<double>(n_); }

    // --- Distribution functions ---

    // PMF: floor(k) applied; returns 0 outside [0, n].
    // Handles p=0 and p=1 edge cases matching C# exactly.
    double pdf(double k) const override {
        if (!parameters_valid_) throw std::invalid_argument("Binomial: invalid parameters");
        k = std::floor(k);
        if (k < minimum() || k > maximum()) return 0.0;
        if (p_ == 0.0) return k == 0.0 ? 1.0 : 0.0;
        if (p_ == 1.0) return k == static_cast<double>(n_) ? 1.0 : 0.0;
        int ki = static_cast<int>(k);
        return sf_bin::factorial::binomial_coefficient(n_, ki) *
               std::pow(p_, k) * std::pow(complement(), static_cast<double>(n_) - k);
    }

    // CDF: floor(k) applied; uses Beta.Incomplete(n-k, k+1, complement) matching C#.
    double cdf(double k) const override {
        if (!parameters_valid_) throw std::invalid_argument("Binomial: invalid parameters");
        k = std::floor(k);
        if (k < minimum()) return 0.0;
        if (k >= maximum()) return 1.0;
        return sf_bin::beta::incomplete(
            static_cast<double>(n_) - k, k + 1.0, complement());
    }

    // InverseCDF: integer walk 0..n, return first i where CDF(i) >= probability.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("Binomial: invalid parameters");
        double k = 0.0;
        for (int i = 0; i <= n_; ++i) {
            if (cdf(static_cast<double>(i)) >= probability) {
                k = static_cast<double>(i);
                break;
            }
        }
        return k;
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Binomial>(p_, n_);
    }

   private:
    static bool validate(double p, int n) {
        if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0) return false;
        if (n <= 0) return false;
        return true;
    }

    double p_ = 0.5;
    int n_ = 10;
};

}  // namespace bestfit::numerics::distributions
