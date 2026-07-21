// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/NumericalDiff.cs @ c2e6192
//
// Centralized numerical differentiation with adaptive step sizing and flat-spot
// detection, used by the BestFit MLE/MAP estimators (Phase 4).
//
// All methods use the (|theta| + 1) x RelStep formula for initial step sizes,
// matching the Numerics library convention. When a flat spot is detected (derivative
// effectively zero, e.g. due to a regime-switching discontinuity), the step is
// automatically escalated by StepGrowthFactor up to MaxStep. This addresses
// distributions like LP3, PT3, GEV, GP, GN, and GL that switch to limit-form
// approximations when shape/skew parameters cross the NearZero = 1e-4 threshold
// (defined in UnivariateDistributionBase, see
// corehydro/numerics/distributions/base/univariate_distribution_base.hpp).
//
// Port scope (Task T3): ComputeHessian (called by MLE's DataLogLikelihood path and
// MAP's LogLikelihood path) and ComputePointwiseGradients (called by MLE/MAP for the
// sandwich/influence/Cook's-D diagnostics), plus every private/internal helper those
// two transitively use: InitialStep, ComputeStepSizes, AdaptiveHessianDiagStep,
// EvalPerturbed, Clamp, IsPointwiseFlat, IsFinite. C# `double[]? lowerBounds` /
// `upperBounds` (null meaning "unbounded") are represented here as possibly-empty
// `std::vector<double>` -- an empty vector means "no bound", matching C#'s null.
//
// Phase 6 (Task B8, the GeneralizedMethodOfMoments follow-up the Phase 4 sever note
// pointed at) completes the file: ComputeGradient (C# 282-362, GMM's GetPenaltyGradient
// and its BFGS analytic-gradient path) and ComputeJacobian (C# 378-496, GMM's
// GetJacobian numerical fallback), including their bounds-aware step logic, plus the
// AvailableLeft/AvailableRight/IsBad helpers only those two use (deliberately deferred
// with them in Phase 4). Every member of the C# class is now ported.
// `compute_jacobian` returns the row-major `Matrix2D`, mirroring the C# `double[,]`
// return (GMM's GetJacobian wraps it in a Matrix at the call site, exactly as the C#
// does with `new Matrix(...)`).
//
// Task T13 (BestFit v2.0.0, 0d6821d..7efa9d0) edge-hardened both adaptive step
// searches:
//   - `adaptive_hessian_diag_step` (ComputeHessian's diagonal step search) is now
//     bounded at `kMaxAdaptiveHessianDiagStepAttempts` (64) attempts instead of an
//     unbounded `while (h <= MaxStep)` loop -- alternating finite flat spots and
//     non-finite larger steps could otherwise retry forever. It now returns the
//     selected step ALONGSIDE the finite forward/backward stencil values it already
//     computed while searching (`HessianDiagStepResult`), so `compute_hessian`'s
//     diagonal-entry loop reads them directly instead of re-evaluating `function` a
//     second time at the same two points. It tracks the last finite step/stencil seen
//     (`last_finite_step` et al.) so a later non-finite excursion can fall back to the
//     best finite result found so far, and throws `std::runtime_error` (C#
//     `InvalidOperationException`) when NO finite central stencil is ever found for a
//     parameter. INTERIOR SMOOTH VALUES ARE IDENTICAL to the pre-T13 port: a normal
//     first attempt that already clears the flat-spot threshold returns on iteration 0
//     with the exact same step/values the old unbounded loop would have found, so
//     every existing interior fixture is untouched by this change.
//   - `compute_jacobian`'s per-parameter step search is bounded the same way
//     (`kMaxAdaptiveJacobianStepAttempts`, 64) and gains a ONE-SIDED FALLBACK: when
//     the central (room-on-both-sides) stencil has one non-finite side but the base
//     value g0 and the OTHER side are both finite, it falls back to that one-sided
//     difference for the column instead of treating the whole attempt as a failure
//     (mirrors the existing "near a bound" one-sided branches, just triggered by a
//     non-finite far side instead of insufficient room). `normalize_step` guards every
//     step-progress comparison against NaN/Inf/non-positive step candidates.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"

namespace corehydro::estimation {

class NumericalDiff {
   public:
    // Default relative step size for finite differences.
    static constexpr double kDefaultRelStep = 1e-4;

    // Default absolute minimum step size.
    static constexpr double kDefaultAbsStep = 1e-8;

    // Maximum step size for flat-spot retry escalation.
    static constexpr double kMaxStep = 1e-2;

    // Relative tolerance for detecting flat spots. If the central difference
    // |f(theta+h) - f(theta-h)| is less than this fraction of the function scale, the
    // derivative is considered effectively zero and the step is escalated.
    static constexpr double kFlatSpotRelTol = 1e-12;

    // Factor by which to multiply the step size on each flat-spot retry.
    static constexpr double kStepGrowthFactor = 4.0;

    // Computes the initial step size for a single parameter:
    // h = max(RelStep * (|theta| + 1), AbsStep). The (|theta| + 1) scale factor
    // matches the Numerics library convention and ensures a minimum step of RelStep
    // (1e-4) even when the parameter value is zero or very small.
    static double initial_step(double parameter_value) {
        return std::max(kDefaultRelStep * (std::fabs(parameter_value) + 1.0), kDefaultAbsStep);
    }

    // Computes initial step sizes for all parameters.
    static std::vector<double> compute_step_sizes(const std::vector<double>& parameters) {
        std::vector<double> h(parameters.size());
        for (std::size_t j = 0; j < parameters.size(); ++j) h[j] = initial_step(parameters[j]);
        return h;
    }

    // Computes pointwise gradients of a vector-valued function via central
    // differences with flat-spot detection and step escalation.
    // pointwise_func maps parameters to an array of n pointwise values (e.g. the
    // model's pointwise data log-likelihood). n is the number of observations, p the
    // number of parameters. Returns an [n][p] matrix where result[i][j] =
    // d f_i / d theta_j, using the central difference formula
    // d f / d theta_j ~= [f(theta + h_j e_j) - f(theta - h_j e_j)] / (2 h_j). When the
    // total change across all observations is below FlatSpotRelTol, the step is
    // multiplied by StepGrowthFactor and retried up to MaxStep.
    static std::vector<std::vector<double>> compute_pointwise_gradients(
        const std::function<std::vector<double>(const std::vector<double>&)>& pointwise_func,
        const std::vector<double>& parameters, int n, int p) {
        std::vector<std::vector<double>> gradients(
            static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(p), 0.0));

        std::vector<double> perturbed = parameters;

        for (int j = 0; j < p; ++j) {
            double h = initial_step(parameters[static_cast<std::size_t>(j)]);
            std::vector<double> forward_ll;
            std::vector<double> backward_ll;
            bool have_result = false;

            while (h <= kMaxStep) {
                perturbed[static_cast<std::size_t>(j)] = parameters[static_cast<std::size_t>(j)] + h;
                forward_ll = pointwise_func(perturbed);

                perturbed[static_cast<std::size_t>(j)] = parameters[static_cast<std::size_t>(j)] - h;
                backward_ll = pointwise_func(perturbed);

                perturbed[static_cast<std::size_t>(j)] = parameters[static_cast<std::size_t>(j)];
                have_result = true;

                if (!is_pointwise_flat(forward_ll, backward_ll, n)) break;

                h *= kStepGrowthFactor;
            }

            double effective_h = std::min(h, kMaxStep);
            if (have_result) {
                for (int i = 0; i < n; ++i)
                    gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                        (forward_ll[static_cast<std::size_t>(i)] - backward_ll[static_cast<std::size_t>(i)]) /
                        (2.0 * effective_h);
            }
        }

        return gradients;
    }

    // Computes the Hessian matrix of a scalar function via central differences with
    // flat-spot detection and step escalation. function is f: R^p -> R. p is the
    // number of parameters. lower_bounds/upper_bounds are optional (an empty vector
    // means "no bound in that direction", matching C#'s null). Returns the p x p
    // Hessian matrix of second partial derivatives.
    // MUTABLE-POINT SEMANTICS (M14, C#-fidelity): the scalar `function` takes a NON-CONST
    // vector, matching C#'s `Func<double[], double>` (arrays are reference types). The shared
    // `perturbed` working array below is reused across evaluations with targeted per-slot
    // resets, exactly like the C# source -- so a mutating function (RMC.BestFit's MixtureModel
    // normalizes the weight entries in place) leaves its write-back in `perturbed` between
    // evaluations, shaping the finite differences identically to upstream. Callables taking
    // `const std::vector<double>&` still convert unchanged.
    static corehydro::numerics::math::linalg::Matrix compute_hessian(
        const std::function<double(std::vector<double>&)>& function,
        const std::vector<double>& parameters, int p, const std::vector<double>& lower_bounds = {},
        const std::vector<double>& upper_bounds = {}) {
        corehydro::numerics::math::linalg::Matrix hessian(p, p);
        std::vector<double> perturbed = parameters;
        double f0 = function(perturbed);

        // Compute effective step sizes per parameter (with flat-spot escalation for
        // the diagonal entries), caching the forward/backward stencil values each
        // search already computed so the diagonal-entry loop below never re-evaluates
        // `function` a second time at the same two points (T13).
        std::vector<double> h(static_cast<std::size_t>(p));
        std::vector<HessianDiagStepResult> diagonal_stencils(static_cast<std::size_t>(p));
        for (int j = 0; j < p; ++j) {
            diagonal_stencils[static_cast<std::size_t>(j)] = adaptive_hessian_diag_step(
                function, parameters, perturbed, j, f0, lower_bounds, upper_bounds);
            h[static_cast<std::size_t>(j)] = diagonal_stencils[static_cast<std::size_t>(j)].step;
        }

        // Diagonal entries: H[j,j] = (f(theta+h) - 2f(theta) + f(theta-h)) / h^2
        for (int j = 0; j < p; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            double fp = diagonal_stencils[sj].forward_value;
            double fm = diagonal_stencils[sj].backward_value;
            hessian(j, j) = (fp - 2.0 * f0 + fm) / (h[sj] * h[sj]);
        }

        // Off-diagonal entries: H[j,k] = (fpp - fpm - fmp + fmm) / (4 h_j h_k)
        for (int j = 0; j < p; ++j) {
            for (int k = j + 1; k < p; ++k) {
                std::size_t sj = static_cast<std::size_t>(j);
                std::size_t sk = static_cast<std::size_t>(k);

                perturbed[sj] = clamp(parameters[sj] + h[sj], j, lower_bounds, upper_bounds);
                perturbed[sk] = clamp(parameters[sk] + h[sk], k, lower_bounds, upper_bounds);
                double fpp = function(perturbed);

                perturbed[sk] = clamp(parameters[sk] - h[sk], k, lower_bounds, upper_bounds);
                double fpm = function(perturbed);

                perturbed[sj] = clamp(parameters[sj] - h[sj], j, lower_bounds, upper_bounds);
                perturbed[sk] = clamp(parameters[sk] + h[sk], k, lower_bounds, upper_bounds);
                double fmp = function(perturbed);

                perturbed[sk] = clamp(parameters[sk] - h[sk], k, lower_bounds, upper_bounds);
                double fmm = function(perturbed);

                double hjk = (fpp - fpm - fmp + fmm) / (4.0 * h[sj] * h[sk]);
                hessian(j, k) = hjk;
                hessian(k, j) = hjk;

                perturbed[sj] = parameters[sj];
                perturbed[sk] = parameters[sk];
            }
        }

        return hessian;
    }

    // Computes the gradient of a scalar function via central differences with flat-spot
    // detection, step escalation, and optional boundary handling (C# ComputeGradient,
    // 282-362; ported with B8). function is f: R^p -> R (same mutable-point signature as
    // compute_hessian above). lower_bounds/upper_bounds are optional (empty = unbounded).
    // Returns the gradient vector where result[j] = df/dtheta_j.
    static std::vector<double> compute_gradient(
        const std::function<double(std::vector<double>&)>& function,
        const std::vector<double>& parameters, const std::vector<double>& lower_bounds = {},
        const std::vector<double>& upper_bounds = {}) {
        int p = static_cast<int>(parameters.size());
        std::vector<double> grad(static_cast<std::size_t>(p), 0.0);
        std::vector<double> perturbed = parameters;
        double f0 = function(perturbed);

        for (int j = 0; j < p; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            double h = initial_step(parameters[sj]);

            while (h <= kMaxStep) {
                double room_left = available_left(parameters, j, lower_bounds);
                double room_right = available_right(parameters, j, upper_bounds);

                double derivative;
                bool success;
                bool is_flat;

                if (room_left >= h && room_right >= h) {
                    double fp = eval_perturbed(function, perturbed, parameters[sj] + h, j,
                                               lower_bounds, upper_bounds);
                    double fm = eval_perturbed(function, perturbed, parameters[sj] - h, j,
                                               lower_bounds, upper_bounds);
                    perturbed[sj] = parameters[sj];
                    success = is_finite(fp) && is_finite(fm);
                    if (success) {
                        derivative = (fp - fm) / (2.0 * h);
                        double diff = std::fabs(fp - fm);
                        double scale = std::max(std::fabs(fp), std::max(std::fabs(fm), 1.0));
                        is_flat = diff < kFlatSpotRelTol * scale;
                    } else {
                        derivative = 0;
                        is_flat = false;
                    }
                } else if (room_right >= h) {
                    double fp = eval_perturbed(function, perturbed, parameters[sj] + h, j,
                                               lower_bounds, upper_bounds);
                    perturbed[sj] = parameters[sj];
                    success = is_finite(fp);
                    if (success) {
                        derivative = (fp - f0) / h;
                        double diff = std::fabs(fp - f0);
                        double scale = std::max(std::fabs(fp), std::max(std::fabs(f0), 1.0));
                        is_flat = diff < kFlatSpotRelTol * scale;
                    } else {
                        derivative = 0;
                        is_flat = false;
                    }
                } else if (room_left >= h) {
                    double fm = eval_perturbed(function, perturbed, parameters[sj] - h, j,
                                               lower_bounds, upper_bounds);
                    perturbed[sj] = parameters[sj];
                    success = is_finite(fm);
                    if (success) {
                        derivative = (f0 - fm) / h;
                        double diff = std::fabs(f0 - fm);
                        double scale = std::max(std::fabs(fm), std::max(std::fabs(f0), 1.0));
                        is_flat = diff < kFlatSpotRelTol * scale;
                    } else {
                        derivative = 0;
                        is_flat = false;
                    }
                } else {
                    h *= 0.5;
                    continue;
                }

                if (!success) {
                    h *= 0.5;
                    continue;
                }
                if (!is_flat) {
                    grad[sj] = derivative;
                    break;
                }
                h *= kStepGrowthFactor;
            }
        }

        return grad;
    }

    // Computes the Jacobian matrix of a vector-valued function via central differences
    // with flat-spot detection, step escalation, and optional boundary handling (C#
    // ComputeJacobian; ported with B8, edge-hardened by T13). function is g: R^p -> R^m;
    // m is the number of output components (rows of the Jacobian). Returns an m x p
    // row-major Matrix2D where J[i][j] = dg_i/dtheta_j (the C# `double[,]` return).
    // T13: the per-parameter search is bounded at kMaxAdaptiveJacobianStepAttempts (64)
    // attempts, and the room-on-both-sides branch now falls back to a ONE-SIDED
    // difference (using g0 and whichever perturbed side is finite) when the central
    // stencil has one non-finite side but the base value and the other side are both
    // finite, rather than treating the whole attempt as a failure.
    static corehydro::numerics::math::linalg::Matrix2D compute_jacobian(
        const std::function<std::vector<double>(std::vector<double>&)>& function,
        const std::vector<double>& parameters, int m,
        const std::vector<double>& lower_bounds = {},
        const std::vector<double>& upper_bounds = {}) {
        int p = static_cast<int>(parameters.size());
        corehydro::numerics::math::linalg::Matrix2D J(
            static_cast<std::size_t>(m), std::vector<double>(static_cast<std::size_t>(p), 0.0));
        std::vector<double> perturbed = parameters;
        std::vector<double> g0 = function(perturbed);
        bool base_is_bad = is_bad(g0, m);

        for (int j = 0; j < p; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            double h = std::min(normalize_step(initial_step(parameters[sj])), kMaxStep);
            std::vector<double> col;
            bool have_col = false;  // stands in for the C# nullable `double[]? col`

            for (int attempt = 0; attempt < kMaxAdaptiveJacobianStepAttempts; ++attempt) {
                h = std::min(normalize_step(h), kMaxStep);
                double room_left = available_left(parameters, j, lower_bounds);
                double room_right = available_right(parameters, j, upper_bounds);

                bool success;
                bool is_flat;

                if (room_left >= h && room_right >= h) {
                    perturbed[sj] = clamp(parameters[sj] + h, j, lower_bounds, upper_bounds);
                    std::vector<double> g_plus = function(perturbed);
                    perturbed[sj] = clamp(parameters[sj] - h, j, lower_bounds, upper_bounds);
                    std::vector<double> g_minus = function(perturbed);
                    perturbed[sj] = parameters[sj];

                    bool plus_is_bad = is_bad(g_plus, m);
                    bool minus_is_bad = is_bad(g_minus, m);
                    if (!plus_is_bad && !minus_is_bad) {
                        success = true;
                        double total_diff = 0, total_scale = 0;
                        col.assign(static_cast<std::size_t>(m), 0.0);
                        have_col = true;
                        double two_h = 2.0 * h;
                        for (int i = 0; i < m; ++i) {
                            std::size_t si = static_cast<std::size_t>(i);
                            col[si] = (g_plus[si] - g_minus[si]) / two_h;
                            total_diff += std::fabs(g_plus[si] - g_minus[si]);
                            total_scale += std::max(std::fabs(g_plus[si]), std::fabs(g_minus[si]));
                        }
                        is_flat = total_diff < kFlatSpotRelTol * std::max(total_scale, 1.0);
                    } else if (!base_is_bad && !plus_is_bad) {
                        // One-sided fallback (T13): the backward side is non-finite, but
                        // the base value and the forward side are both usable.
                        success = true;
                        double total_diff = 0, total_scale = 0;
                        col.assign(static_cast<std::size_t>(m), 0.0);
                        have_col = true;
                        for (int i = 0; i < m; ++i) {
                            std::size_t si = static_cast<std::size_t>(i);
                            col[si] = (g_plus[si] - g0[si]) / h;
                            total_diff += std::fabs(g_plus[si] - g0[si]);
                            total_scale += std::max(std::fabs(g_plus[si]), std::fabs(g0[si]));
                        }
                        is_flat = total_diff < kFlatSpotRelTol * std::max(total_scale, 1.0);
                    } else if (!base_is_bad && !minus_is_bad) {
                        // One-sided fallback (T13): the forward side is non-finite, but
                        // the base value and the backward side are both usable.
                        success = true;
                        double total_diff = 0, total_scale = 0;
                        col.assign(static_cast<std::size_t>(m), 0.0);
                        have_col = true;
                        for (int i = 0; i < m; ++i) {
                            std::size_t si = static_cast<std::size_t>(i);
                            col[si] = (g0[si] - g_minus[si]) / h;
                            total_diff += std::fabs(g0[si] - g_minus[si]);
                            total_scale += std::max(std::fabs(g_minus[si]), std::fabs(g0[si]));
                        }
                        is_flat = total_diff < kFlatSpotRelTol * std::max(total_scale, 1.0);
                    } else {
                        success = false;
                        is_flat = false;
                    }
                } else if (!base_is_bad && room_right >= h) {
                    perturbed[sj] = clamp(parameters[sj] + h, j, lower_bounds, upper_bounds);
                    std::vector<double> g_plus = function(perturbed);
                    perturbed[sj] = parameters[sj];

                    if (is_bad(g_plus, m)) {
                        success = false;
                        is_flat = false;
                    } else {
                        success = true;
                        double total_diff = 0, total_scale = 0;
                        col.assign(static_cast<std::size_t>(m), 0.0);
                        have_col = true;
                        for (int i = 0; i < m; ++i) {
                            std::size_t si = static_cast<std::size_t>(i);
                            col[si] = (g_plus[si] - g0[si]) / h;
                            total_diff += std::fabs(g_plus[si] - g0[si]);
                            total_scale += std::max(std::fabs(g_plus[si]), std::fabs(g0[si]));
                        }
                        is_flat = total_diff < kFlatSpotRelTol * std::max(total_scale, 1.0);
                    }
                } else if (!base_is_bad && room_left >= h) {
                    perturbed[sj] = clamp(parameters[sj] - h, j, lower_bounds, upper_bounds);
                    std::vector<double> g_minus = function(perturbed);
                    perturbed[sj] = parameters[sj];

                    if (is_bad(g_minus, m)) {
                        success = false;
                        is_flat = false;
                    } else {
                        success = true;
                        double total_diff = 0, total_scale = 0;
                        col.assign(static_cast<std::size_t>(m), 0.0);
                        have_col = true;
                        for (int i = 0; i < m; ++i) {
                            std::size_t si = static_cast<std::size_t>(i);
                            col[si] = (g0[si] - g_minus[si]) / h;
                            total_diff += std::fabs(g0[si] - g_minus[si]);
                            total_scale += std::max(std::fabs(g_minus[si]), std::fabs(g0[si]));
                        }
                        is_flat = total_diff < kFlatSpotRelTol * std::max(total_scale, 1.0);
                    }
                } else {
                    double next_h = h * 0.5;
                    if (next_h >= h || next_h < kDefaultAbsStep * 0.01) break;
                    h = next_h;
                    continue;
                }

                if (!success) {
                    double next_h = h * 0.5;
                    if (next_h >= h || next_h < kDefaultAbsStep * 0.01) break;
                    h = next_h;
                    continue;
                }
                if (!is_flat) break;
                if (h >= kMaxStep) break;
                h = std::min(h * kStepGrowthFactor, kMaxStep);
            }

            if (!have_col) {
                // No step size produced a usable Jacobian column. C# flags this via
                // Debug.WriteLine ("column left as zeros"); no C++ logger is wired into
                // this core, so the trace is dropped and the column stays zero.
                col.assign(static_cast<std::size_t>(m), 0.0);
            }
            for (int i = 0; i < m; ++i)
                J[static_cast<std::size_t>(i)][sj] = col[static_cast<std::size_t>(i)];
        }

        return J;
    }

   private:
    // Maximum number of attempts used when searching for a usable Hessian diagonal step
    // (C# `MaxAdaptiveHessianDiagStepAttempts`; T13).
    static constexpr int kMaxAdaptiveHessianDiagStepAttempts = 64;

    // Maximum number of attempts used when searching for a usable Jacobian column step
    // (C# `MaxAdaptiveJacobianStepAttempts`; T13).
    static constexpr int kMaxAdaptiveJacobianStepAttempts = 64;

    // Return type for adaptive_hessian_diag_step: the selected step size and the finite
    // forward/backward stencil values found at that step (C# ValueTuple `(double Step,
    // double ForwardValue, double BackwardValue)`; T13). Letting the caller read these
    // back avoids re-evaluating `function` a second time at the same two points.
    struct HessianDiagStepResult {
        double step;
        double forward_value;
        double backward_value;
    };

    // Normalizes a finite-difference step to a positive finite value, protecting retry
    // loops from NaN, infinity, zero, and underflow values before comparing step
    // progress (C# `NormalizeStep`; T13).
    static double normalize_step(double step) {
        if (!is_finite(step) || step <= 0.0) return kDefaultAbsStep;
        return step;
    }

    // Determines whether the pointwise central difference values indicate a flat spot.
    static bool is_pointwise_flat(const std::vector<double>& forward, const std::vector<double>& backward,
                                   int n) {
        double total_diff = 0.0;
        double total_scale = 0.0;
        for (int i = 0; i < n; ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            if (!is_finite(forward[si]) || !is_finite(backward[si])) return false;
            total_diff += std::fabs(forward[si] - backward[si]);
            total_scale += std::max(std::fabs(forward[si]), std::fabs(backward[si]));
        }
        return total_diff < kFlatSpotRelTol * std::max(total_scale, 1.0);
    }

    // Computes an adaptive step size for the j-th diagonal Hessian entry, escalating
    // through flat spots until curvature is detected. Bounded at
    // kMaxAdaptiveHessianDiagStepAttempts attempts instead of an unbounded loop (T13:
    // alternating finite flat spots and non-finite larger steps could otherwise retry
    // forever), returning the cached forward/backward stencil values alongside the step.
    // Tracks the last finite step/stencil seen so a later non-finite excursion can fall
    // back to the best finite result found so far; throws std::runtime_error (C#
    // InvalidOperationException) when no finite central stencil is ever found. INTERIOR
    // SMOOTH VALUES ARE IDENTICAL to the pre-T13 loop: a first attempt that already
    // clears the flat-spot threshold returns immediately with the same step/values the
    // old unbounded `while` loop would have found.
    static HessianDiagStepResult adaptive_hessian_diag_step(
        const std::function<double(std::vector<double>&)>& function,
        const std::vector<double>& parameters, std::vector<double>& perturbed, int j, double f0,
        const std::vector<double>& lower_bounds, const std::vector<double>& upper_bounds) {
        std::size_t sj = static_cast<std::size_t>(j);
        double h = normalize_step(initial_step(parameters[sj]));
        double last_finite_step = std::numeric_limits<double>::quiet_NaN();
        double last_finite_forward = std::numeric_limits<double>::quiet_NaN();
        double last_finite_backward = std::numeric_limits<double>::quiet_NaN();
        double first_non_finite_upper_step = std::numeric_limits<double>::infinity();

        for (int attempt = 0; attempt < kMaxAdaptiveHessianDiagStepAttempts; ++attempt) {
            h = normalize_step(h);

            if (h > kMaxStep && !is_finite(last_finite_step)) {
                h = kMaxStep;
            } else if (h > kMaxStep) {
                return {last_finite_step, last_finite_forward, last_finite_backward};
            }

            double fp;
            double fm;
            try {
                fp = eval_perturbed(function, perturbed, parameters[sj] + h, j, lower_bounds, upper_bounds);
                fm = eval_perturbed(function, perturbed, parameters[sj] - h, j, lower_bounds, upper_bounds);
            } catch (...) {
                perturbed[sj] = parameters[sj];
                throw;
            }
            perturbed[sj] = parameters[sj];

            if (!is_finite(fp) || !is_finite(fm)) {
                first_non_finite_upper_step = std::min(first_non_finite_upper_step, h);
                double next_h = normalize_step(h * 0.5);
                if (next_h >= h || next_h < kDefaultAbsStep * 0.01) break;
                h = next_h;
                continue;
            }

            last_finite_step = h;
            last_finite_forward = fp;
            last_finite_backward = fm;

            double second_deriv = std::fabs(fp - 2.0 * f0 + fm);
            double scale = std::max(std::fabs(fp), std::max(std::fabs(fm), std::max(std::fabs(f0), 1.0)));

            if (second_deriv >= kFlatSpotRelTol * scale) return {h, fp, fm};

            double grown_h = normalize_step(h * kStepGrowthFactor);
            if (grown_h <= h || grown_h >= first_non_finite_upper_step)
                return {last_finite_step, last_finite_forward, last_finite_backward};

            h = grown_h;
        }

        if (is_finite(last_finite_step))
            return {last_finite_step, last_finite_forward, last_finite_backward};

        throw std::runtime_error("NumericalDiff.Hessian: no finite central stencil found for parameter " +
                                  std::to_string(j) + ".");
    }

    // Evaluates the function with parameter j set to the given value (clamped to
    // bounds). Caller must reset perturbed[j] after use.
    static double eval_perturbed(const std::function<double(std::vector<double>&)>& function,
                                  std::vector<double>& perturbed, double value, int j,
                                  const std::vector<double>& lower_bounds,
                                  const std::vector<double>& upper_bounds) {
        perturbed[static_cast<std::size_t>(j)] = clamp(value, j, lower_bounds, upper_bounds);
        return function(perturbed);
    }

    // Clamps a value to the bounds for parameter j. An empty bounds vector means "no
    // bound in that direction" (matching C#'s nullable double[]?).
    static double clamp(double value, int j, const std::vector<double>& lower_bounds,
                         const std::vector<double>& upper_bounds) {
        std::size_t sj = static_cast<std::size_t>(j);
        if (!lower_bounds.empty() && value < lower_bounds[sj]) value = lower_bounds[sj];
        if (!upper_bounds.empty() && value > upper_bounds[sj]) value = upper_bounds[sj];
        return value;
    }

    // Calculates available space below the current parameter value (C# AvailableLeft;
    // ported with B8 -- used only by compute_gradient/compute_jacobian). An empty bounds
    // vector means "no bound" (C#'s null), giving infinite room.
    static double available_left(const std::vector<double>& parameters, int j,
                                 const std::vector<double>& lower_bounds) {
        return lower_bounds.empty()
                   ? std::numeric_limits<double>::infinity()
                   : parameters[static_cast<std::size_t>(j)] - lower_bounds[static_cast<std::size_t>(j)];
    }

    // Calculates available space above the current parameter value (C# AvailableRight;
    // ported with B8).
    static double available_right(const std::vector<double>& parameters, int j,
                                  const std::vector<double>& upper_bounds) {
        return upper_bounds.empty()
                   ? std::numeric_limits<double>::infinity()
                   : upper_bounds[static_cast<std::size_t>(j)] - parameters[static_cast<std::size_t>(j)];
    }

    // Checks if a vector is missing, mis-sized, or contains any non-finite values (C#
    // IsBad; ported with B8). The C# null check has no analogue for a by-value vector;
    // a wrong length covers the defective-result cases reachable here.
    static bool is_bad(const std::vector<double>& v, int expected_length) {
        if (static_cast<int>(v.size()) != expected_length) return true;
        for (int i = 0; i < expected_length; ++i) {
            if (!is_finite(v[static_cast<std::size_t>(i)])) return true;
        }
        return false;
    }

    // Checks if a value is finite (not NaN and not infinity).
    static bool is_finite(double value) { return std::isfinite(value); }
};

}  // namespace corehydro::estimation
