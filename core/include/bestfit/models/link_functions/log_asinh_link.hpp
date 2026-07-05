// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/LogASinHLink.cs @ fc28c0c
//
// A positive-support sinh-arcsinh link applied to log-relative scale. Maps a positive
// parameter (a scale / standard deviation) to an unconstrained link-space variable by
// applying the Jones-Pewsey sinh-arcsinh transformation to the log-relative deviation
// from a fitted center. Let sigma0 > 0 be the fitted center, s > 0 a dimensionless
// log-scale, and r = log(sigma / sigma0), z = r / s. Then:
//   Forward (link): eta = sinh(delta * asinh(z) - epsilon)
//   Inverse: sigma = sigma0 * exp(s * sinh((asinh(eta) + epsilon) / delta))
//   Derivative: deta/dsigma = (delta / (s * sigma)) * cosh(delta * asinh(z) - epsilon)
//               / sqrt(1 + z^2)
// When epsilon = 0 and delta = 1 the link reduces to eta = log(sigma / sigma0) / s.
// Reference: Jones, M.C. and Pewsey, A. (2009). Sinh-arcsinh distributions.
// Biometrika, 96(4), 761-780.
//
// The C# class is a mutable property-bag whose setters validate/floor values; those
// property setters are ported verbatim as set_* methods (unlike ASinHLink, whose plain
// auto-properties port as unvalidated setters). The XElement constructor / ToXElement /
// ParseDouble are dropped (serialization is a desktop concern). The Asinh / SafeSinh /
// SafeCosh / SafeExp / SafeSqrtOnePlusSquare helpers (clamped at +-700 = MaxSafeLog)
// are private per-class copies, mirroring the C# file structure; NOTE this class's
// Asinh differs deliberately from the ASinHLink/SESLink copies (1e-8 linear threshold
// and a log(2)+log(ax) branch above 1e154), exactly per the C#.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

#include "bestfit/numerics/functions/i_link_function.hpp"

namespace bestfit::models::link_functions {

class LogASinHLink final : public numerics::functions::ILinkFunction {
   public:
    // Constructs a new symmetric log-sinh-arcsinh link with default settings.
    LogASinHLink() = default;

    // Constructs a new log-sinh-arcsinh link with the specified center and log-scale.
    // Assignments run through the validating setters, exactly like the C# property
    // assignments in its constructor.
    LogASinHLink(double sigma0, double log_scale, double epsilon = 0.0, double delta = 1.0) {
        set_sigma0(sigma0);
        set_log_scale(log_scale);
        set_epsilon(epsilon);
        set_delta(delta);
    }

    // Center sigma0 > 0, typically the fitted scale parameter.
    double sigma0() const { return sigma0_; }
    void set_sigma0(double value) {
        sigma0_ = is_positive_finite(value) ? value : std::max(eps_, kDefaultEps);
    }

    // Dimensionless scale used to standardize log(sigma / sigma0). For uncertainty
    // propagation this is usually derived from a relative standard error, e.g.
    // sqrt(log(1 + CV^2)).
    double log_scale() const { return log_scale_; }
    void set_log_scale(double value) {
        log_scale_ = is_positive_finite(value) ? value : std::max(eps_, kDefaultEps);
    }

    // Fixed asymmetry parameter used when use_adaptive_epsilon is false. Positive
    // epsilon inflates the upper sigma tail on the log-relative scale.
    double epsilon() const { return epsilon_; }
    void set_epsilon(double value) { epsilon_ = value; }

    // Tail-shape parameter delta > 0. For normal link-space draws, delta < 1 produces
    // heavier sigma-space tails and delta > 1 compresses them.
    double delta() const { return delta_; }
    void set_delta(double value) {
        delta_ = is_positive_finite(value) ? value : std::max(eps_, kDefaultEps);
    }

    // Enables adaptive asymmetry driven by parent_indicator (epsilon is then ignored).
    bool use_adaptive_epsilon() const { return use_adaptive_epsilon_; }
    void set_use_adaptive_epsilon(bool value) { use_adaptive_epsilon_ = value; }

    // Parent diagnostic controlling adaptive asymmetry sign and strength.
    double parent_indicator() const { return parent_indicator_; }
    void set_parent_indicator(double value) { parent_indicator_ = value; }

    // Maximum magnitude of adaptive asymmetry epsilon.
    double epsilon_max() const { return epsilon_max_; }
    void set_epsilon_max(double value) {
        epsilon_max_ = std::isfinite(value) ? std::max(0.0, value) : 0.0;
    }

    // Sensitivity of adaptive epsilon to parent_indicator.
    double epsilon_slope() const { return epsilon_slope_; }
    void set_epsilon_slope(double value) {
        epsilon_slope_ = std::isfinite(value) ? std::max(0.0, value) : 0.0;
    }

    // Numerical floor for positive sigma, sigma0, log-scale, and delta values.
    double eps() const { return eps_; }
    void set_eps(double value) { eps_ = is_positive_finite(value) ? value : kDefaultEps; }

    // Evaluates the link function. Nonpositive sigma values are floored at eps(),
    // preventing transient numerical failures during finite differencing or covariance
    // transformations.
    double link(double sigma) const override {
        double s = std::max(log_scale_, eps_);
        double d = std::max(delta_, eps_);
        double eps = effective_epsilon();
        double z = log_ratio(sigma) / s;
        return safe_sinh((d * asinh_local(z)) - eps);
    }

    // Evaluates the inverse link function (always positive).
    double inverse_link(double eta) const override {
        double sigma0 = std::max(sigma0_, eps_);
        double s = std::max(log_scale_, eps_);
        double d = std::max(delta_, eps_);
        double eps = effective_epsilon();

        double log_relative = s * safe_sinh((asinh_local(eta) + eps) / d);
        double log_sigma = std::log(sigma0) + log_relative;
        return safe_exp(log_sigma);
    }

    // Evaluates the derivative of the link function with respect to sigma.
    double d_link(double sigma) const override {
        double x = std::max(sigma, eps_);
        double s = std::max(log_scale_, eps_);
        double d = std::max(delta_, eps_);
        double eps = effective_epsilon();
        double z = log_ratio(x) / s;
        double transformed = (d * asinh_local(z)) - eps;
        double denominator = s * x * safe_sqrt_one_plus_square(z);
        double derivative = d * safe_cosh(transformed) / std::max(denominator, eps_);

        if (!std::isfinite(derivative)) return std::numeric_limits<double>::max();

        return std::max(derivative, eps_);
    }

   private:
    // Default numerical floor for positive quantities.
    static constexpr double kDefaultEps = 1e-12;

    // Maximum log argument used in exponential and hyperbolic evaluations.
    static constexpr double kMaxSafeLog = 700.0;

    double sigma0_ = 1.0;
    double log_scale_ = 1.0;
    double epsilon_ = 0.0;
    double delta_ = 1.0;
    bool use_adaptive_epsilon_ = false;
    double parent_indicator_ = 0.0;
    double epsilon_max_ = 0.5;
    double epsilon_slope_ = 1.0;
    double eps_ = kDefaultEps;

    // Computes the effective epsilon value from fixed or adaptive settings.
    double effective_epsilon() const {
        if (!use_adaptive_epsilon_) return epsilon_;

        return epsilon_max_ * std::tanh(epsilon_slope_ * parent_indicator_);
    }

    // Computes log(sigma / sigma0) with positive floors.
    double log_ratio(double sigma) const {
        double x = std::max(sigma, eps_);
        double sigma0 = std::max(sigma0_, eps_);
        return std::log(x) - std::log(sigma0);
    }

    // Determines whether a value is finite and positive.
    static bool is_positive_finite(double value) { return std::isfinite(value) && value > 0.0; }

    // Computes a numerically stable inverse hyperbolic sine (per-class C# copy; NOTE
    // the 1e-8 threshold and 1e154 large-argument branch differ from the other links'
    // Asinh copies, per the C#).
    static double asinh_local(double x) {
        if (!std::isfinite(x)) return x;

        double ax = std::fabs(x);
        if (ax < 1e-8) return x;

        double value = ax > 1e154 ? std::log(ax) + std::log(2.0)
                                  : std::log(ax + std::sqrt((ax * ax) + 1.0));

        return std::copysign(value, x);
    }

    // Computes sinh with overflow protection (argument clamped to +-700).
    static double safe_sinh(double x) {
        double bounded = std::clamp(x, -kMaxSafeLog, kMaxSafeLog);
        return std::sinh(bounded);
    }

    // Computes cosh with overflow protection (argument clamped to +-700).
    static double safe_cosh(double x) {
        double bounded = std::clamp(x, -kMaxSafeLog, kMaxSafeLog);
        return std::cosh(bounded);
    }

    // Computes exp with overflow and underflow protection (argument clamped to +-700).
    static double safe_exp(double x) {
        double bounded = std::clamp(x, -kMaxSafeLog, kMaxSafeLog);
        return std::exp(bounded);
    }

    // Computes sqrt(1 + x^2) without overflowing for very large x.
    static double safe_sqrt_one_plus_square(double x) {
        double ax = std::fabs(x);
        if (ax > 1e154) return ax;

        return std::sqrt(1.0 + (x * x));
    }
};

}  // namespace bestfit::models::link_functions
