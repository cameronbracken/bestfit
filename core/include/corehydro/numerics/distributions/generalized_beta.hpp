// ported from: Numerics/Distributions/Univariate/GeneralizedBeta.cs @ 2a0357a
//
// The four-parameter (generalized) Beta distribution with shape parameters α, β and support
// [min, max]. Logic mirrors the C# source method-for-method. The C# class derives only from
// UnivariateDistributionBase and implements no estimation interfaces, so none are ported.
// The static PERT(...) factory methods are consumed by the Pert distribution and are kept here.
// SetParametersFromMoments and the WPF helpers are not ported (application concerns).
//
// v2.1.4 (33dc1af "Fix audited Numerics port issues"): Mode widened the same way as
// BetaDistribution -- an explicit U-shape branch (both shape parameters < 1 -> support
// midpoint) plus boundary comparisons widened to <=/>= (Min when alpha<=1 && beta>=1, Max when
// alpha>=1 && beta<=1). Previously only the strict `alpha<=1 && beta<=1` U-shape case was
// special-cased; every other combination fell through to the unclamped interior formula, which
// could return a value outside [min, max] (e.g. GeneralizedBeta(0.42, 1.57, 0, 1).Mode was
// ~58, now 0.0 -- see docs/upstream-csharp-issues.md's "Beta / GeneralizedBeta Mode" entry,
// now resolved by this fix). Note the C# `Pert` distribution's own `Mode` returns its stored
// "most likely" parameter directly (never calls GeneralizedBeta.Mode), so PERT's degenerate
// midpoint behavior at min==mode==max is untouched by this change.
#pragma once
#include <string>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/math/special/beta.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::distributions {

class GeneralizedBeta : public UnivariateDistributionBase {
   public:
    // Constructs a Beta distribution with α = β = 2, defined in the interval (0,1).
    GeneralizedBeta() { set_parameters(2.0, 2.0, 0.0, 1.0); }
    // Constructs a Beta distribution with the given α and β, defined in the interval (0,1).
    GeneralizedBeta(double alpha, double beta) { set_parameters(alpha, beta, 0.0, 1.0); }
    // Constructs a Beta distribution with α, β and support [min, max].
    GeneralizedBeta(double alpha, double beta, double min, double max) {
        set_parameters(alpha, beta, min, max);
    }

    // Constructs a Beta distribution using the PERT estimation method (default scale λ = 4).
    static GeneralizedBeta pert(double min, double mode, double max) {
        return pert(min, mode, max, 4.0);
    }

    // Constructs a Beta distribution using the PERT estimation method with scale λ.
    static GeneralizedBeta pert(double min, double mode, double max, double scale) {
        if (min > max)
            throw std::out_of_range("The maximum value must be greater than the minimum value.");
        if (mode < min || mode > max)
            throw std::out_of_range("The mode must be between the minimum and maximum values.");
        double mean = (min + scale * mode + max) / (scale + 2.0);
        double alpha = 1.0 + scale / 2.0;
        if (!almost_equals(mean, mode)) {
            alpha = (mean - min) * (2.0 * mode - min - max) / ((mode - mean) * (max - min));
        }
        double beta = alpha * (max - mean) / (mean - min);
        return GeneralizedBeta(alpha, beta, min, max);
    }

    double alpha() const { return alpha_; }
    double beta() const { return beta_; }
    double min_val() const { return min_; }
    double max_val() const { return max_; }

    // --- Identity / parameters ---
    UnivariateDistributionType type() const override {
        return UnivariateDistributionType::GeneralizedBeta;
    }
    int number_of_parameters() const override { return 4; }
    std::vector<double> get_parameters() const override { return {alpha_, beta_, min_, max_}; }

    void set_parameters(double alpha, double beta, double min, double max) {
        parameters_valid_ = validate(alpha, beta, min, max);
        alpha_ = alpha;
        beta_ = beta;
        min_ = min;
        max_ = max;
    }
    void set_parameters(const std::vector<double>& p) override {
        set_parameters(p[0], p[1], p[2], p[3]);
    }

    // --- Moments / support ---
    double mean() const override {
        double m = alpha_ / (alpha_ + beta_);
        return m * (max_ - min_) + min_;
    }

    double median() const override { return inverse_cdf(0.5); }

    double mode() const override {
        if (alpha_ == 1.0 && beta_ == 1.0) return (min_ + max_) / 2.0;
        if (alpha_ < 1.0 && beta_ < 1.0) return (min_ + max_) / 2.0;
        if (alpha_ <= 1.0 && beta_ >= 1.0) return min_;
        if (alpha_ >= 1.0 && beta_ <= 1.0) return max_;
        double m = (alpha_ - 1.0) / (alpha_ + beta_ - 2.0);
        return m * (max_ - min_) + min_;
    }

    double standard_deviation() const override {
        double var = alpha_ * beta_ /
                     ((alpha_ + beta_) * (alpha_ + beta_) * (alpha_ + beta_ + 1.0));
        return std::sqrt(var) * (max_ - min_);
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

    double minimum() const override { return min_; }
    double maximum() const override { return max_; }

    // --- Distribution functions ---
    double pdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedBeta: invalid parameters");
        if (x < minimum() || x > maximum()) return 0.0;
        if (min_ == max_) return 0.0;
        double constant = 1.0 / math::special::beta::function(alpha_, beta_);
        double z = (x - min_) / (max_ - min_);
        double a = std::pow(z, alpha_ - 1.0);
        double b = std::pow(1.0 - z, beta_ - 1.0);
        return constant * a * b / (max_ - min_);
    }

    double cdf(double x) const override {
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedBeta: invalid parameters");
        if (min_ == max_) return 1.0;
        if (x <= minimum()) return 0.0;
        if (x >= maximum()) return 1.0;
        double z = (x - min_) / (max_ - min_);
        return math::special::beta::incomplete(alpha_, beta_, z);
    }

    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::out_of_range("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        if (!parameters_valid_)
            throw std::invalid_argument("GeneralizedBeta: invalid parameters");
        double z = math::special::beta::incomplete_inverse(alpha_, beta_, probability);
        return z * (max_ - min_) + min_;
    }

    // --- Parameter display names (X1; C# GeneralizedBeta.cs ParametersToString col0 +
    // ParameterNamesShortForm) ---
    std::vector<std::string> parameter_names() const override {
        return {"Shape (\xCE\xB1)", "Shape (\xCE\xB2)", "Min", "Max"};
    }
    std::vector<std::string> parameter_names_short_form() const override {
        return {"\xCE\xB1", "\xCE\xB2", "Min", "Max"};
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<GeneralizedBeta>(alpha_, beta_, min_, max_);
    }

   private:
    static bool almost_equals(double a, double b) {
        return std::fabs(a - b) <= kDoubleMachineEpsilon * std::fmax(std::fabs(a), std::fabs(b));
    }

    static bool validate(double alpha, double beta, double min, double max) {
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0) return false;
        if (std::isnan(beta) || std::isinf(beta) || beta <= 0.0) return false;
        if (std::isnan(min) || std::isinf(min) || std::isnan(max) || std::isinf(max) ||
            min >= max)
            return false;
        return true;
    }

    double alpha_ = 2.0;
    double beta_ = 2.0;
    double min_ = 0.0;
    double max_ = 1.0;
};

}  // namespace corehydro::numerics::distributions
