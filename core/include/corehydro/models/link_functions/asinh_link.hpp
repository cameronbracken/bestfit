// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/ASinHLink.cs @ c2e6192
//
// A centered sinh-arcsinh link function for unbounded parameters, with optional
// asymmetry (epsilon) and tail-shape (delta) control (Jones-Pewsey sinh-arcsinh form
// around a fitted center; fully analytical, no Newton solver). Let z = (x - x0) / s be
// the standardized parameter deviation. Then:
//   Forward (link):   eta = sinh(delta * asinh(z) - epsilon)
//   Inverse:          x = x0 + s * sinh((asinh(eta) + epsilon) / delta)
//   Derivative:       deta/dx = (delta / s) * cosh(delta * asinh(z) - epsilon)
//                               / sqrt(1 + z^2)
// Adaptive asymmetry: when use_adaptive_epsilon is true, epsilon is computed from
// parent_indicator via epsilon_eff = epsilon_max * tanh(epsilon_slope * parent_indicator).
// Reference: Jones, M.C. and Pewsey, A. (2009). Sinh-arcsinh distributions.
// Biometrika, 96(4), 761-780.
//
// The C# class is a mutable property-bag; this port keeps plain getters/setters (the
// repo "never mutate" rule is relaxed for these model/binding objects). The XElement
// constructor / ToXElement / ParseDouble are dropped (serialization is a desktop
// concern). The Asinh helper is a private per-class copy, mirroring the C# file
// structure (the C# .NET Framework 4.8.1 fallback for the missing Math.Asinh).
#pragma once
#include <algorithm>
#include <cmath>

#include "corehydro/numerics/functions/i_link_function.hpp"

namespace corehydro::models::link_functions {

class ASinHLink final : public numerics::functions::ILinkFunction {
   public:
    // Constructs a new ASinHLink with default parameters (symmetric, Gamma0 = 0,
    // Scale = 1, Delta = 1).
    ASinHLink() = default;

    // Constructs a new symmetric ASinHLink with specified center and scale
    // (epsilon = 0, delta = 1).
    ASinHLink(double gamma0, double scale) : gamma0_(gamma0), scale_(std::max(scale, kEps)) {}

    // Constructs an ASinHLink with specified center, scale, asymmetry, and tail weight.
    ASinHLink(double gamma0, double scale, double epsilon, double delta = 1.0)
        : gamma0_(gamma0),
          scale_(std::max(scale, kEps)),
          epsilon_(epsilon),
          delta_(std::max(delta, kEps)) {}

    // Center x0, typically the GMM point estimate for the linked parameter.
    double gamma0() const { return gamma0_; }
    void set_gamma0(double value) { gamma0_ = value; }

    // Scale parameter s used to standardize deviations from gamma0. Typically the
    // parameter standard error from the GMM sandwich covariance.
    double scale() const { return scale_; }
    void set_scale(double value) { scale_ = value; }

    // If use_adaptive_epsilon is false, this fixed epsilon is used for asymmetry.
    double epsilon() const { return epsilon_; }
    void set_epsilon(double value) { epsilon_ = value; }

    // Tail-shape parameter delta (> 0).
    double delta() const { return delta_; }
    void set_delta(double value) { delta_ = value; }

    // Enable adaptive asymmetry driven by parent_indicator.
    bool use_adaptive_epsilon() const { return use_adaptive_epsilon_; }
    void set_use_adaptive_epsilon(bool value) { use_adaptive_epsilon_ = value; }

    // Parent indicator controlling asymmetry sign and strength.
    double parent_indicator() const { return parent_indicator_; }
    void set_parent_indicator(double value) { parent_indicator_ = value; }

    // Maximum magnitude of the adaptive asymmetry epsilon_max (>= 0).
    double epsilon_max() const { return epsilon_max_; }
    void set_epsilon_max(double value) { epsilon_max_ = value; }

    // Sensitivity of epsilon_eff to the parent indicator.
    double epsilon_slope() const { return epsilon_slope_; }
    void set_epsilon_slope(double value) { epsilon_slope_ = value; }

    // Evaluates the link function: eta = sinh(delta * asinh(z) - epsilon) where
    // z = (x - x0) / s. When epsilon = 0 and delta = 1 this reduces to eta = z.
    double link(double gamma) const override {
        double s = std::max(scale_, kEps);
        double d = std::max(delta_, kEps);
        double eps = effective_epsilon();
        double z = (gamma - gamma0_) / s;
        return std::sinh(d * asinh_local(z) - eps);
    }

    // Evaluates the inverse link: x = x0 + s * sinh((asinh(eta) + epsilon) / delta).
    double inverse_link(double eta) const override {
        double s = std::max(scale_, kEps);
        double d = std::max(delta_, kEps);
        double eps = effective_epsilon();
        return gamma0_ + s * std::sinh((asinh_local(eta) + eps) / d);
    }

    // Evaluates the derivative of the link function:
    // deta/dx = (delta / s) * cosh(delta * asinh(z) - epsilon) / sqrt(1 + z^2).
    double d_link(double gamma) const override {
        double s = std::max(scale_, kEps);
        double d = std::max(delta_, kEps);
        double eps = effective_epsilon();
        double z = (gamma - gamma0_) / s;
        double asinh_z = asinh_local(z);
        double cosh_term = std::cosh(d * asinh_z - eps);
        double sqrt_term = std::sqrt(1.0 + z * z);
        return (d / s) * cosh_term / sqrt_term;
    }

   private:
    // Numerical floor for the scale and delta parameters.
    static constexpr double kEps = 1e-12;

    double gamma0_ = 0.0;
    double scale_ = 1.0;
    double epsilon_ = 0.0;
    double delta_ = 1.0;
    bool use_adaptive_epsilon_ = false;
    double parent_indicator_ = 0.0;
    double epsilon_max_ = 0.5;
    double epsilon_slope_ = 1.0;

    // Computes the effective epsilon given current settings. When adaptive, maps
    // parent_indicator to (-epsilon_max, +epsilon_max) via tanh saturation.
    double effective_epsilon() const {
        if (!use_adaptive_epsilon_) return epsilon_;

        return epsilon_max_ * std::tanh(epsilon_slope_ * parent_indicator_);
    }

    // Computes the inverse hyperbolic sine (asinh) of x. Per-class copy of the C#
    // implementation for .NET Framework 4.8.1, which lacks Math.Asinh.
    static double asinh_local(double x) {
        double ax = std::fabs(x);
        if (ax < 1e-6) return x;
        return std::log(x + std::sqrt(x * x + 1.0));
    }
};

}  // namespace corehydro::models::link_functions
