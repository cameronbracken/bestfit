// ported from: Numerics/Distributions/Univariate/Poisson.cs @ a2c4dbf
//
// Poisson distribution with rate λ (lambda). Discrete; support {0, 1, 2, ...}.
// PMF via log-factorial; CDF via regularized upper incomplete gamma (UpperIncomplete(k+1, λ));
// InverseCDF via integer search near floor(λ). Mirrors the C# source method-for-method.
// No estimation interfaces are implemented upstream.
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

namespace sf_po = corehydro::numerics::math::special;

class Poisson : public UnivariateDistributionBase {
   public:
    // Constructs a Poisson distribution with λ = 1.
    Poisson() { set_parameters({1.0}); }

    // Constructs a Poisson distribution with a given rate λ.
    explicit Poisson(double rate) { set_parameters({rate}); }

    double lambda() const { return lambda_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::Poisson;
    }
    int number_of_parameters() const override { return 1; }
    std::vector<double> get_parameters() const override { return {lambda_}; }

    void set_parameters(const std::vector<double>& p) override {
        lambda_           = p[0];
        parameters_valid_ = validate(lambda_);
    }

    // --- Moments / support ---
    double mean() const override { return lambda_; }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override { return std::floor(lambda_); }

    double standard_deviation() const override { return std::sqrt(lambda_); }

    double skewness() const override { return 1.0 / std::sqrt(lambda_); }

    double kurtosis() const override { return 3.0 + 1.0 / lambda_; }

    double minimum() const override { return 0.0; }
    double maximum() const override { return kInf; }

    // --- Distribution functions ---

    // PMF: floor(k) is applied first (mirrors C# PDF(double k) → k = Math.Floor(k)).
    double pdf(double k) const override {
        if (!parameters_valid_) throw std::invalid_argument("Poisson: invalid parameters");
        k = std::floor(k);
        if (k < minimum() || k > maximum()) return 0.0;
        return std::exp(-lambda_ + k * std::log(lambda_)
                        - sf_po::factorial::log_factorial(static_cast<int>(k)));
    }

    // CDF: floor(k) applied; CDF(k) = UpperIncomplete(k+1, λ).
    double cdf(double k) const override {
        if (!parameters_valid_) throw std::invalid_argument("Poisson: invalid parameters");
        k = std::floor(k);
        if (k < minimum()) return 0.0;
        if (k > maximum()) return 1.0;
        return sf_po::upper_incomplete(k + 1.0, lambda_);
    }

    // InverseCDF: integer walk near floor(λ), then search upward.
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_) throw std::invalid_argument("Poisson: invalid parameters");

        int k = static_cast<int>(std::max(0.0, std::floor(lambda_)));

        // Move downward if needed
        while (k > 0 && cdf(static_cast<double>(k - 1)) >= probability)
            --k;

        // Move upward to find first k such that CDF(k) >= probability
        while (cdf(static_cast<double>(k)) < probability)
            ++k;

        return static_cast<double>(k);
    }

    // --- Parameter display names (X1; C# Poisson.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Rate (\xCE\xBB)"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xBB"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<Poisson>(lambda_);
    }

   private:
    static bool validate(double lam) {
        if (std::isnan(lam) || std::isinf(lam) || lam <= 0.0) return false;
        return true;
    }

    double lambda_ = 1.0;
};

}  // namespace corehydro::numerics::distributions
