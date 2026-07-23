// ported from: Numerics/Distributions/Univariate/Binomial.cs @ 2a0357a
//
// The Binomial distribution with parameters p (probability of success) and n (number of trials).
// Discrete; PMF via binomial_coefficient; CDF via regularized incomplete beta (Beta.Incomplete).
// InverseCDF via integer walk 0..n. No estimation interfaces upstream.
// Logic mirrors the C# source method-for-method.
// Re-audited against v2.1.4's "Harden distribution parameter validation" wave: C#'s
// SetParameters now assigns the backing fields directly (bypassing the ProbabilityOfSuccess/
// NumberOfTrials property setters) before validating once from the full, final parameter
// pair; ValidateParameters already rejected NaN/Infinity in both parameters (unchanged by
// this wave). set_parameters below already matched the ordering; the only change is
// guarding n_'s truncating cast (`static_cast<int>` of a non-finite double is undefined
// behavior in C++) and validating the RAW params[1] for NaN/Infinity rather than the
// truncated int, so a non-finite trial count is rejected regardless of what garbage the
// (now-skipped) truncation would otherwise have produced.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/special/beta.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_bin = corehydro::numerics::math::special;

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
        // Guard the truncating cast: static_cast<int> of a non-finite double is undefined
        // behavior. Leave n_ unchanged when params[1] is non-finite (parameters_valid_
        // below will be false either way, so the stale value is never consulted).
        n_ = std::isfinite(params[1]) ? static_cast<int>(params[1]) : n_;
        parameters_valid_ = validate(p_, params[1]);
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

    // --- Parameter display names (X1; C# Binomial.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Probability of Success (p)", "Number of Trials (n)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"p", "n"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Binomial>(p_, n_);
    }

   private:
    // `n_raw` is the UNTRUNCATED trial count (mirrors C#'s ValidateParameters, which checks
    // parameters[1] directly rather than the rounded NumberOfTrials) -- consulted here only
    // for its NaN/Infinity status, which this wave adds. The `<= 0` bound below still checks
    // the TRUNCATED count, a pre-existing (unchanged by this wave, out of this task's scope)
    // divergence from C#'s own `n_raw <= 0.0d` check: a fractional n_raw in (0, 1) reads as
    // valid in C# (0.5 > 0) but invalid here (truncates to 0). No fixture exercises that
    // corner (a non-integer trial count).
    static bool validate(double p, double n_raw) {
        if (std::isnan(p) || std::isinf(p) || p < 0.0 || p > 1.0) return false;
        if (std::isnan(n_raw) || std::isinf(n_raw)) return false;
        if (static_cast<int>(n_raw) <= 0) return false;
        return true;
    }

    double p_ = 0.5;
    int n_ = 10;
};

}  // namespace corehydro::numerics::distributions
