// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/NumericalDiff.cs @ fc28c0c
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
// bestfit/numerics/distributions/base/univariate_distribution_base.hpp).
//
// Port scope (Task T3): ComputeHessian (called by MLE's DataLogLikelihood path and
// MAP's LogLikelihood path) and ComputePointwiseGradients (called by MLE/MAP for the
// sandwich/influence/Cook's-D diagnostics), plus every private/internal helper those
// two transitively use: InitialStep, ComputeStepSizes, AdaptiveHessianDiagStep,
// EvalPerturbed, Clamp, IsPointwiseFlat, IsFinite. C# `double[]? lowerBounds` /
// `upperBounds` (null meaning "unbounded") are represented here as possibly-empty
// `std::vector<double>` -- an empty vector means "no bound", matching C#'s null.
//
// DEFERRED (severable, tracked for the GeneralizedMethodOfMoments follow-up):
// ComputeGradient (C# ~282-377) and ComputeJacobian (C# ~378-505) are used only by the
// severed GMM estimator; no in-scope estimator calls them, so neither is ported here.
// Note on a brief/source discrepancy: the task brief lists AvailableLeft and
// AvailableRight among the helpers "those two [in-scope methods] need", but the C#
// source shows neither is actually called by ComputeHessian or
// ComputePointwiseGradients -- both are used exclusively inside the deferred
// ComputeGradient/ComputeJacobian (their bounds-aware "how much room is left"
// checks). Per this task's own rule ("C# governs when brief and source conflict") and
// YAGNI, AvailableLeft/AvailableRight are deferred alongside ComputeGradient/
// ComputeJacobian rather than ported speculatively; port them together when the GMM
// follow-up lands.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <vector>

#include "bestfit/numerics/math/linalg/matrix.hpp"

namespace bestfit::estimation {

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
    static bestfit::numerics::math::linalg::Matrix compute_hessian(
        const std::function<double(std::vector<double>&)>& function,
        const std::vector<double>& parameters, int p, const std::vector<double>& lower_bounds = {},
        const std::vector<double>& upper_bounds = {}) {
        bestfit::numerics::math::linalg::Matrix hessian(p, p);
        std::vector<double> perturbed = parameters;
        double f0 = function(perturbed);

        // Compute effective step sizes per parameter (with flat-spot escalation for
        // the diagonal entries).
        std::vector<double> h(static_cast<std::size_t>(p));
        for (int j = 0; j < p; ++j)
            h[static_cast<std::size_t>(j)] = adaptive_hessian_diag_step(
                function, parameters, perturbed, j, f0, lower_bounds, upper_bounds);

        // Diagonal entries: H[j,j] = (f(theta+h) - 2f(theta) + f(theta-h)) / h^2
        for (int j = 0; j < p; ++j) {
            double fp = eval_perturbed(function, perturbed, parameters[static_cast<std::size_t>(j)] + h[static_cast<std::size_t>(j)],
                                        j, lower_bounds, upper_bounds);
            double fm = eval_perturbed(function, perturbed, parameters[static_cast<std::size_t>(j)] - h[static_cast<std::size_t>(j)],
                                        j, lower_bounds, upper_bounds);
            hessian(j, j) =
                (fp - 2.0 * f0 + fm) / (h[static_cast<std::size_t>(j)] * h[static_cast<std::size_t>(j)]);
            perturbed[static_cast<std::size_t>(j)] = parameters[static_cast<std::size_t>(j)];
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

   private:
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
    // through flat spots until curvature is detected.
    static double adaptive_hessian_diag_step(
        const std::function<double(std::vector<double>&)>& function,
        const std::vector<double>& parameters, std::vector<double>& perturbed, int j, double f0,
        const std::vector<double>& lower_bounds, const std::vector<double>& upper_bounds) {
        std::size_t sj = static_cast<std::size_t>(j);
        double h = initial_step(parameters[sj]);

        while (h <= kMaxStep) {
            double fp = eval_perturbed(function, perturbed, parameters[sj] + h, j, lower_bounds, upper_bounds);
            double fm = eval_perturbed(function, perturbed, parameters[sj] - h, j, lower_bounds, upper_bounds);
            perturbed[sj] = parameters[sj];

            if (!is_finite(fp) || !is_finite(fm)) {
                h *= 0.5;
                if (h < kDefaultAbsStep * 0.01) return initial_step(parameters[sj]);
                continue;
            }

            double second_deriv = std::fabs(fp - 2.0 * f0 + fm);
            double scale = std::max(std::fabs(fp), std::max(std::fabs(fm), std::max(std::fabs(f0), 1.0)));

            if (second_deriv >= kFlatSpotRelTol * scale) return h;

            h *= kStepGrowthFactor;
        }

        return std::min(h, kMaxStep);
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

    // Checks if a value is finite (not NaN and not infinity).
    static bool is_finite(double value) { return std::isfinite(value); }
};

}  // namespace bestfit::estimation
