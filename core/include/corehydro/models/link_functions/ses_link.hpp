// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/SESLink.cs @ fc28c0c
//
// A skew-exponential-sinh (SES) link function with tunable asymmetry and fat tails.
// Unbounded, strictly monotone, near-linear around 0.
//   Forward (inverse link): gamma(eta) = (1/a) * exp(lambda*eta) * sinh(a*eta),
//     with a > 0 and |lambda| < 1.
//   Derivative: dgamma/deta = exp(lambda*eta) * [(lambda/a) sinh(a*eta) + cosh(a*eta)].
//   Link h(x) solves eta = h(gamma) as the unique root of gamma(eta) - gamma = 0
//     (Newton method, MaxIterations / Tolerance on |delta eta|).
// a controls overall curvature / fat tails; lambda controls tail asymmetry. Local slope
// at 0 is 1 so the optimizer behaves like the identity near gamma = 0.
//
// The C# class is a mutable property-bag; this port keeps plain getters/setters (the
// repo "never mutate" rule is relaxed for these model/binding objects). The XElement
// constructor / ToXElement are dropped (serialization is a desktop concern). The
// LastInverseConverged / LastInverseResidual diagnostics the C# updates inside Link are
// `mutable` members here because ILinkFunction::link is const; they mirror the C#
// get-private-set properties exactly. The C# Debug.WriteLine on non-convergence ports
// as a silent no-throw guard (the diagnostics carry the information). The Asinh and
// SafeExp helpers are private per-class copies, mirroring the C# file structure.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::link_functions {

class SESLink : public numerics::functions::ILinkFunction {
   public:
    // Constructs a new SkewExponentialSinhLink with default settings (a = 1.0, adaptive
    // lambda enabled).
    SESLink() = default;

    // Constructs a new SkewExponentialSinhLink with custom settings. NOTE (per C#): the
    // constructor's lambda_max default is 0.4 while the property initializer used by the
    // default constructor is 0.8.
    explicit SESLink(double a, bool use_adaptive_lambda = true, double parent_indicator = 0.0,
                     double lambda_max = 0.4, double lambda_slope = 1.0, int max_iterations = 20,
                     double tolerance = 1e-12)
        : a_(a),
          use_adaptive_lambda_(use_adaptive_lambda),
          parent_indicator_(parent_indicator),
          lambda_max_(lambda_max),
          lambda_slope_(lambda_slope),
          max_iterations_(std::max(1, max_iterations)) {
        set_tolerance(std::max(1e-16, tolerance));
    }

    // Curvature / symmetric tail-heaviness (a > 0). Larger a => fatter tails on both sides.
    double a() const { return a_; }
    void set_a(double value) { a_ = value; }

    // If use_adaptive_lambda is false, this fixed lambda is used (|lambda| < 1).
    double lambda() const { return lambda_; }
    void set_lambda(double value) { lambda_ = value; }

    // Enable adaptive asymmetry driven by parent_indicator.
    bool use_adaptive_lambda() const { return use_adaptive_lambda_; }
    void set_use_adaptive_lambda(bool value) { use_adaptive_lambda_ = value; }

    // Parent indicator controlling asymmetry sign and strength (e.g., parent gamma-hat).
    double parent_indicator() const { return parent_indicator_; }
    void set_parent_indicator(double value) { parent_indicator_ = value; }

    // Maximum magnitude of the adaptive asymmetry (0 <= lambda_max < 1).
    double lambda_max() const { return lambda_max_; }
    void set_lambda_max(double value) { lambda_max_ = value; }

    // Sensitivity of lambda_eff to the parent indicator.
    double lambda_slope() const { return lambda_slope_; }
    void set_lambda_slope(double value) { lambda_slope_ = value; }

    // Maximum Newton iterations for the inverse (link) solve.
    int max_iterations() const { return max_iterations_; }
    void set_max_iterations(int value) { max_iterations_ = value; }

    // Newton tolerance on parameter update (|delta eta|). Clamped to a minimum of 1e-16:
    // a non-positive value would make the convergence loop never converge.
    double tolerance() const { return tolerance_; }
    void set_tolerance(double value) { tolerance_ = std::max(value, 1e-16); }

    // Whether the most recent link() call's Newton iteration converged (|delta eta| <
    // tolerance within max_iterations). Defaults to true until the first
    // non-converging call. Diagnostic only.
    bool last_inverse_converged() const { return last_inverse_converged_; }

    // |delta eta| at the final iteration of the most recent link() call. Diagnostic
    // only. NaN if link() has not yet been called.
    double last_inverse_residual() const { return last_inverse_residual_; }

    // --- ILinkFunction ---

    double link(double x) const override {
        // Solve for eta given gamma = x using Newton with a good asymptotic/central
        // initial guess.
        double a = std::max(a_, 1e-12);
        double lambda = effective_lambda();

        double eta = initial_guess(x, a, lambda);

        bool converged = false;
        double last_step = std::numeric_limits<double>::quiet_NaN();
        for (int it = 0; it < max_iterations_; ++it) {
            double f = gamma_of_eta(eta, a, lambda) - x;  // f(eta) = gamma(eta) - gamma
            double fp = d_gamma_d_eta(eta, a, lambda);    // f'(eta) = dgamma/deta
            fp = std::max(fp, 1e-16);
            double step = f / fp;
            last_step = std::fabs(step);

            double new_eta = eta - step;
            if (!corehydro::numerics::is_finite(new_eta)) new_eta = 0.5 * eta;

            if (std::fabs(step) < tolerance_) {
                eta = new_eta;
                converged = true;
                break;
            }

            // Mild damping in very extreme tails. Intentionally evaluated AFTER the
            // convergence check so a step that satisfies tolerance exits via the
            // un-damped value (preserves Newton's quadratic convergence near the root).
            // Damping only kicks in for the next iteration when the current step is
            // large (> 4 in eta-space), preventing runaway updates.
            if (std::fabs(step) > 4.0) new_eta = eta - sign(step) * 4.0;

            eta = new_eta;
        }

        last_inverse_converged_ = converged;
        last_inverse_residual_ = last_step;
        // C# emits a Debug.WriteLine when !converged; ported as a silent no-throw guard
        // (the last_inverse_* diagnostics expose the outcome).

        return eta;
    }

    double inverse_link(double eta) const override {
        double a = std::max(a_, 1e-12);
        double lambda = effective_lambda();
        return gamma_of_eta(eta, a, lambda);
    }

    double d_link(double x) const override {
        // h'(x) = deta/dgamma = 1 / (dgamma/deta) evaluated at eta = h(x)
        double eta = link(x);
        double a = std::max(a_, 1e-12);
        double lambda = effective_lambda();
        double dgdeta = d_gamma_d_eta(eta, a, lambda);
        dgdeta = std::max(dgdeta, 1e-16);
        return 1.0 / dgdeta;
    }

   private:
    double a_ = 1.0;
    double lambda_ = 0.4;
    bool use_adaptive_lambda_ = true;
    double parent_indicator_ = 0.0;
    double lambda_max_ = 0.8;
    double lambda_slope_ = 1.0;
    int max_iterations_ = 20;
    double tolerance_ = 1e-12;
    mutable bool last_inverse_converged_ = true;
    mutable double last_inverse_residual_ = std::numeric_limits<double>::quiet_NaN();

    // --- Core formulas ---

    // Computes gamma(eta) = (1/a) e^{lambda eta} sinh(a eta).
    static double gamma_of_eta(double eta, double a, double lambda) {
        return safe_exp(lambda * eta) * std::sinh(a * eta) / a;
    }

    // Computes the derivative dgamma/deta = e^{lambda eta} [(lambda/a) sinh(a eta) +
    // cosh(a eta)].
    static double d_gamma_d_eta(double eta, double a, double lambda) {
        double e = safe_exp(lambda * eta);
        double s = std::sinh(a * eta);
        double c = std::cosh(a * eta);
        return e * ((lambda / a) * s + c);
    }

    // --- Adaptive lambda ---

    // Computes the effective lambda given current settings. Ensures |lambda_eff| < 1 for
    // strict monotonicity.
    double effective_lambda() const {
        if (!use_adaptive_lambda_) return clamp_lambda(lambda_);

        // Map parent_indicator to (-lambda_max, +lambda_max) with tanh so that:
        //  - sign follows parent_indicator (directional inflation),
        //  - magnitude grows with |parent_indicator| and saturates at lambda_max,
        //  - parent_indicator ~ 0 => lambda_eff ~ 0 => symmetric tails.
        double lam_eff = lambda_max_ * std::tanh(lambda_slope_ * parent_indicator_);
        return clamp_lambda(lam_eff);
    }

    // Clamps lambda to strictly within (-1, 1) to preserve global monotonicity.
    static double clamp_lambda(double lambda) {
        if (lambda >= 1.0) return 0.999;
        if (lambda <= -1.0) return -0.999;
        return lambda;
    }

    // --- Initialization & utilities ---

    // Generates an initial guess for the Newton solver using both central and
    // tail-based approximations.
    static double initial_guess(double gamma, double a, double lambda) {
        // Central fallback: symmetric guess
        double central = asinh_local(a * gamma) / a;

        // Tail-based guess to speed convergence when |gamma| is large
        if (corehydro::numerics::is_finite(gamma) && std::fabs(gamma) > 1.0) {
            double denom_pos = a + lambda;
            double denom_neg = a - lambda;

            if (gamma > 0.0 && denom_pos > 1e-10) {
                double val = std::log(2.0 * a * gamma);
                if (corehydro::numerics::is_finite(val)) return val / denom_pos;
            } else if (gamma < 0.0 && denom_neg > 1e-10) {
                double val = std::log(2.0 * a * (-gamma));
                if (corehydro::numerics::is_finite(val)) return -val / denom_neg;
            }
        }

        return central;
    }

    // Computes the inverse hyperbolic sine (asinh) of x. Per-class copy of the C#
    // implementation for .NET Framework 4.8.1, which lacks Math.Asinh.
    static double asinh_local(double x) {
        double ax = std::fabs(x);
        if (ax < 1e-6) return x;
        return std::log(x + std::sqrt(x * x + 1.0));
    }

    // Computes exp(z) with protection against overflow by clamping z to [-700, 700].
    static double safe_exp(double z) {
        if (z > 700.0) return std::exp(700.0);
        if (z < -700.0) return std::exp(-700.0);
        return std::exp(z);
    }

    // C# Math.Sign as a double (-1, 0, +1).
    static double sign(double x) { return x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0); }
};

}  // namespace corehydro::models::link_functions
