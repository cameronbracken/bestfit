// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/LogSESLink.cs @ fc28c0c
//
// Asymmetric heavy-tailed link on log(sigma) using the Skew-Exponential-Sinh (SES) map:
//   r(eta) = (1/a) * exp(lambda*eta) * sinh(a*eta), where r = log(sigma/sigma0),
//     a > 0, |lambda| < 1.
//   Inverse: sigma = sigma0 * exp(r(eta)). Monotone; tunable asymmetry and tail weight.
// Link(x) solves r(eta) = log(sigma/sigma0) for eta by Newton (same solver shape as
// SESLink but with the tail-guess threshold at |r| > 0.5 rather than |gamma| > 1.0).
//
// The C# class is a mutable property-bag; this port keeps plain getters/setters (the
// repo "never mutate" rule is relaxed for these model/binding objects). The XElement
// constructor / ToXElement are dropped (serialization is a desktop concern). The
// LastInverseConverged / LastInverseResidual diagnostics are `mutable` (see
// ses_link.hpp). The C# Debug.WriteLine on non-convergence ports as a silent no-throw
// guard. The Asinh and SafeExp helpers are private per-class copies, mirroring the C#
// file structure.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

#include "bestfit/numerics/functions/i_link_function.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::models::link_functions {

class LogSESLink final : public numerics::functions::ILinkFunction {
   public:
    // Constructs a new LogSESLink with default parameters.
    LogSESLink() = default;

    // Constructs a new LogSESLink with specified parameters.
    explicit LogSESLink(double sigma0, double a = 1.0, double lambda = 0.2)
        : sigma0_(std::max(sigma0, 1e-12)),
          a_(std::max(a, 1e-12)),
          lambda_(clamp_lambda(lambda)) {}

    // Center sigma0 > 0 (fix per fit; e.g., parent sigma-hat).
    double sigma0() const { return sigma0_; }
    void set_sigma0(double value) { sigma0_ = value; }

    // Curvature a > 0 (heavier tails as a increases).
    double a() const { return a_; }
    void set_a(double value) { a_ = value; }

    // If use_adaptive_lambda is false, this fixed lambda is used (|lambda| < 1).
    double lambda() const { return lambda_; }
    void set_lambda(double value) { lambda_ = value; }

    // Enable adaptive asymmetry driven by parent_indicator.
    bool use_adaptive_lambda() const { return use_adaptive_lambda_; }
    void set_use_adaptive_lambda(bool value) { use_adaptive_lambda_ = value; }

    // Parent indicator controlling asymmetry sign and strength (e.g., max(0, gamma-hat)).
    double parent_indicator() const { return parent_indicator_; }
    void set_parent_indicator(double value) { parent_indicator_ = value; }

    // Maximum magnitude of the adaptive asymmetry (0 <= lambda_max < 1).
    double lambda_max() const { return lambda_max_; }
    void set_lambda_max(double value) { lambda_max_ = value; }

    // Sensitivity of adaptive lambda to parent_indicator.
    double lambda_slope() const { return lambda_slope_; }
    void set_lambda_slope(double value) { lambda_slope_ = value; }

    // Maximum Newton iterations for the inverse (link) solve.
    int max_iterations() const { return max_iterations_; }
    void set_max_iterations(int value) { max_iterations_ = value; }

    // Newton tolerance on parameter update (|delta eta|). Clamped to a minimum of 1e-16
    // (mirrors SESLink::tolerance: a non-positive value would never converge).
    double tolerance() const { return tolerance_; }
    void set_tolerance(double value) { tolerance_ = std::max(value, 1e-16); }

    // Numerical floor for sigma and internal terms to prevent underflow.
    double eps() const { return eps_; }
    void set_eps(double value) { eps_ = value; }

    // Whether the most recent link() call's Newton iteration converged. Defaults to
    // true until the first non-converging call. Diagnostic only.
    bool last_inverse_converged() const { return last_inverse_converged_; }

    // |delta eta| at the final iteration of the most recent link() call. Diagnostic
    // only. NaN if link() has not yet been called.
    double last_inverse_residual() const { return last_inverse_residual_; }

    // --- ILinkFunction ---

    double link(double x) const override {
        // Solve r(eta) = ln(sigma/sigma0) for eta via Newton (monotone => unique).
        double sigma = std::max(x, eps_);
        double r = std::log(sigma / std::max(sigma0_, eps_));
        double a = std::max(a_, 1e-12);
        double lambda = effective_lambda();

        double eta = initial_guess(r, a, lambda);

        bool converged = false;
        double last_step = std::numeric_limits<double>::quiet_NaN();
        for (int it = 0; it < max_iterations_; ++it) {
            double f = r_of_eta(eta, a, lambda) - r;
            double fp = d_r_d_eta(eta, a, lambda);
            fp = std::max(fp, 1e-16);
            double step = f / fp;
            last_step = std::fabs(step);
            double new_eta = eta - step;

            if (!bestfit::numerics::is_finite(new_eta)) new_eta = 0.5 * eta;
            if (std::fabs(step) < tolerance_) {
                eta = new_eta;
                converged = true;
                break;
            }
            if (std::fabs(step) > 4.0) new_eta = eta - sign(step) * 4.0;

            eta = new_eta;
        }

        last_inverse_converged_ = converged;
        last_inverse_residual_ = last_step;
        // C# emits a Debug.WriteLine when !converged; ported as a silent no-throw guard.

        return eta;
    }

    double inverse_link(double eta) const override {
        double a = std::max(a_, 1e-12);
        double lambda = effective_lambda();
        double r = r_of_eta(eta, a, lambda);
        return std::max(sigma0_, eps_) * std::exp(r);
    }

    double d_link(double x /* deta/dsigma */) const override {
        double sigma = std::max(x, eps_);
        double r = std::log(sigma / std::max(sigma0_, eps_));
        (void)r;  // computed in the C# but unused there too (Link re-derives it)
        double a = std::max(a_, 1e-12);
        double lambda = effective_lambda();

        // deta/dsigma = (1/sigma) * 1 / (dr/deta)
        double eta = link(sigma);
        double drdeta = d_r_d_eta(eta, a, lambda);
        drdeta = std::max(drdeta, 1e-16);
        return (1.0 / sigma) * (1.0 / drdeta);
    }

   private:
    double sigma0_ = 1.0;
    double a_ = 1.0;
    double lambda_ = 0.2;
    bool use_adaptive_lambda_ = false;
    double parent_indicator_ = 0.0;
    double lambda_max_ = 0.4;
    double lambda_slope_ = 1.0;
    int max_iterations_ = 20;
    double tolerance_ = 1e-12;
    double eps_ = 1e-12;
    mutable bool last_inverse_converged_ = true;
    mutable double last_inverse_residual_ = std::numeric_limits<double>::quiet_NaN();

    // Computes r(eta) = (1/a) e^{lambda eta} sinh(a eta).
    static double r_of_eta(double eta, double a, double lambda) {
        return safe_exp(lambda * eta) * std::sinh(a * eta) / a;
    }

    // Computes the derivative dr/deta = e^{lambda eta} [(lambda/a) sinh(a eta) +
    // cosh(a eta)].
    static double d_r_d_eta(double eta, double a, double lambda) {
        double e = safe_exp(lambda * eta);
        double s = std::sinh(a * eta);
        double c = std::cosh(a * eta);
        return e * ((lambda / a) * s + c);
    }

    // Generates an initial guess for the Newton solver based on central and tail-based
    // approximations.
    static double initial_guess(double r, double a, double lambda) {
        // Central fallback
        double central = asinh_local(a * r) / a;

        // Tail-based guess
        if (bestfit::numerics::is_finite(r) && std::fabs(r) > 0.5) {
            double denom_pos = a + lambda;
            double denom_neg = a - lambda;

            if (r > 0.0 && denom_pos > 1e-10) {
                double val = std::log(2.0 * a * r);
                if (bestfit::numerics::is_finite(val)) return val / denom_pos;
            } else if (r < 0.0 && denom_neg > 1e-10) {
                double val = std::log(2.0 * a * (-r));
                if (bestfit::numerics::is_finite(val)) return -val / denom_neg;
            }
        }
        return central;
    }

    // Computes the effective lambda given current settings. Ensures |lambda_eff| < 1
    // for strict monotonicity.
    double effective_lambda() const {
        if (!use_adaptive_lambda_) return clamp_lambda(lambda_);

        // Map parent_indicator to (-lambda_max, +lambda_max) with tanh (see ses_link.hpp).
        double lam_eff = lambda_max_ * std::tanh(lambda_slope_ * parent_indicator_);
        return clamp_lambda(lam_eff);
    }

    // Clamps lambda to the valid range (-1, 1) to ensure monotonicity.
    static double clamp_lambda(double lambda) {
        if (lambda >= 1.0) return 0.999;
        if (lambda <= -1.0) return -0.999;
        return lambda;
    }

    // Computes the inverse hyperbolic sine (asinh) of x (per-class C# copy).
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

}  // namespace bestfit::models::link_functions
