// ported from: Numerics/Distributions/Univariate/TruncatedDistribution.cs @ a2c4dbf
//
// A general truncated probability distribution. Wraps any UnivariateDistributionBase
// and restricts it to [min, max]: PDF is renormalized by F(max)-F(min); CDF and InverseCDF
// are shifted accordingly. Moments use AGK numerical integration over [min, max]; C# uses
// CentralMoments(1000) (the int overload: trapezoidal, 1000 steps over InverseCDF(1e-8)
// endpoints). AGK is more accurate and reproduces the C# values to oracle tolerance.
// Mode clamps the base mode to [min, max]: exact for unimodal bases (the truncated-PDF
// maximum is the base mode if in-bounds, else the nearest boundary), matching C# BrentSearch
// for all standard cases. NOTE: this would be wrong for a MULTIMODAL base (e.g. a Mixture)
// where truncation shifts the global PDF maximum to a different peak — replace with a
// BrentSearch-over-truncated-PDF port if such a base is ever truncated.
// type() delegates to the base distribution (mirrors C# `_baseDist.Type`).
// No IEstimation / ILinearMomentEstimation (TruncatedDistribution does not implement them in C#).
#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/math/integration/adaptive_gauss_kronrod.hpp"

namespace bestfit::numerics::distributions {

class TruncatedDistribution : public UnivariateDistributionBase {
   public:
    /// Construct a truncated distribution from a cloned base + bounds.
    TruncatedDistribution(const UnivariateDistributionBase& base, double min_val, double max_val)
        : base_dist_(base.clone()), min_(min_val), max_(max_val) {
        update_cached_cdfs();
        validate_and_set();
    }

    /// Construct from an rvalue unique_ptr (takes ownership).
    TruncatedDistribution(std::unique_ptr<UnivariateDistributionBase> base,
                          double min_val, double max_val)
        : base_dist_(std::move(base)), min_(min_val), max_(max_val) {
        update_cached_cdfs();
        validate_and_set();
    }

    // Accessors matching C# properties
    const UnivariateDistributionBase& base_distribution() const { return *base_dist_; }
    double min_bound() const { return min_; }
    double max_bound() const { return max_; }

    // --- Identity / parameters ---
    // Mirrors C#: `public override UnivariateDistributionType Type => _baseDist.Type;`
    UnivariateDistributionType type() const override { return base_dist_->type(); }
    int number_of_parameters() const override { return base_dist_->number_of_parameters() + 2; }

    // {base_params..., min, max}  -- mirrors C# GetParameters
    std::vector<double> get_parameters() const override {
        auto p = base_dist_->get_parameters();
        p.push_back(min_);
        p.push_back(max_);
        return p;
    }

    // Mirrors C# SetParameters: first N-2 go to base; last 2 are min/max.
    void set_parameters(const std::vector<double>& params) override {
        int n_base = base_dist_->number_of_parameters();
        std::vector<double> base_params(params.begin(), params.begin() + n_base);
        base_dist_->set_parameters(base_params);
        min_ = params[n_base];
        max_ = params[n_base + 1];
        update_cached_cdfs();
        validate_and_set();
        moments_computed_ = false;
    }

    // --- Moments / support ---
    double mean() const override {
        if (!moments_computed_) compute_moments();
        return u_[0];
    }
    double median() const override { return inverse_cdf(0.5); }
    double mode() const override {
        // Clamp base mode to [min_, max_]: exact for unimodal bases, matches C# BrentSearch
        // over [InverseCDF(0.001), InverseCDF(0.999)] for all standard fixture cases.
        return std::clamp(base_dist_->mode(), min_, max_);
    }
    double standard_deviation() const override {
        if (!moments_computed_) compute_moments();
        return u_[1];
    }
    double skewness() const override {
        if (!moments_computed_) compute_moments();
        return u_[2];
    }
    double kurtosis() const override {
        if (!moments_computed_) compute_moments();
        return u_[3];
    }
    // Mirrors C#: Minimum = Math.Max(_baseDist.Minimum, Min)
    double minimum() const override {
        return std::max(base_dist_->minimum(), min_);
    }
    // Mirrors C#: Maximum = Math.Min(_baseDist.Maximum, Max)
    double maximum() const override {
        return std::min(base_dist_->maximum(), max_);
    }

    // --- Distribution functions ---
    // Mirrors C#: PDF = basePDF(x) / (Fmax - Fmin) on [min, max], 0 outside.
    double pdf(double x) const override {
        if (x < min_ || x > max_) return 0.0;
        return base_dist_->pdf(x) / (f_max_ - f_min_);
    }

    // Mirrors C#: CDF = (baseCDF(x) - Fmin) / (Fmax - Fmin)
    double cdf(double x) const override {
        if (x <= min_) return 0.0;
        if (x >= max_) return 1.0;
        return (base_dist_->cdf(x) - f_min_) / (f_max_ - f_min_);
    }

    // Mirrors C#: InverseCDF = base.InverseCDF(p * (Fmax - Fmin) + Fmin)
    double inverse_cdf(double probability) const override {
        if (probability < 0.0 || probability > 1.0)
            throw std::invalid_argument("probability must be between 0 and 1");
        if (probability == 0.0) return minimum();
        if (probability == 1.0) return maximum();
        return base_dist_->inverse_cdf(probability * (f_max_ - f_min_) + f_min_);
    }

    // --- Parameter display names (X1; C# TruncatedDistribution.cs ParametersToString col0 +
    // ParameterNamesShortForm): the base distribution's names followed by "Min"/"Max" (C#
    // 78-107). ---
    std::vector<std::string> parameter_names() const override {
        std::vector<std::string> names = base_dist_->parameter_names();
        names.push_back("Min");
        names.push_back("Max");
        return names;
    }
    std::vector<std::string> parameter_names_short_form() const override {
        std::vector<std::string> names = base_dist_->parameter_names_short_form();
        names.push_back("Min");
        names.push_back("Max");
        return names;
    }

    std::unique_ptr<UnivariateDistributionBase> clone() const override {
        return std::make_unique<TruncatedDistribution>(*base_dist_, min_, max_);
    }

   private:
    std::unique_ptr<UnivariateDistributionBase> base_dist_;
    double min_;
    double max_;
    double f_min_;  // baseCDF(min_)
    double f_max_;  // baseCDF(max_)

    // Lazy moment cache (mutable so const moment accessors can populate it).
    mutable bool moments_computed_ = false;
    mutable double u_[4] = {kNaN, kNaN, kNaN, kNaN};  // [mean, sd, skewness, kurtosis]

    void update_cached_cdfs() {
        f_min_ = base_dist_->cdf(min_);
        f_max_ = base_dist_->cdf(max_);
    }

    void validate_and_set() {
        parameters_valid_ = base_dist_->parameters_valid() &&
                            !std::isnan(min_) && !std::isnan(max_) &&
                            !std::isinf(min_) && !std::isinf(max_) &&
                            min_ < max_ &&
                            std::fabs(f_max_ - f_min_) >= 1e-15;
    }

    // Numerical moments via AGK integration, mirroring C# CentralMoments(double tolerance=1e-8).
    // Integration bounds are [min_, max_] (the truncated support).
    void compute_moments() const {
        namespace agk = bestfit::numerics::math::integration;
        const double tol = 1e-8;
        const double a = min_, b = max_;
        if (a >= b) {
            u_[0] = a;
            u_[1] = u_[2] = u_[3] = kNaN;
            moments_computed_ = true;
            return;
        }
        // Mean: E[X] = integral x * f(x) dx
        u_[0] = agk::integrate([this](double x) { return x * pdf(x); }, a, b, tol, tol);
        const double mu = u_[0];
        // Standard deviation: sqrt(E[(X-mu)^2])
        u_[1] = std::sqrt(agk::integrate(
            [this, mu](double x) { return (x - mu) * (x - mu) * pdf(x); }, a, b, tol, tol));
        const double s = u_[1];
        // Skewness: E[((X-mu)/s)^3]
        u_[2] = agk::integrate(
            [this, mu, s](double x) {
                double z = (x - mu) / s;
                return z * z * z * pdf(x);
            }, a, b, tol, tol);
        // Kurtosis: E[((X-mu)/s)^4]
        u_[3] = agk::integrate(
            [this, mu, s](double x) {
                double z = (x - mu) / s;
                return z * z * z * z * pdf(x);
            }, a, b, tol, tol);
        moments_computed_ = true;
    }
};

}  // namespace bestfit::numerics::distributions
