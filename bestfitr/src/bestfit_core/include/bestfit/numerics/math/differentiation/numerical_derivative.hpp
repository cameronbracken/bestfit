// ported from: Numerics/Mathematics/Differentiation/NumericalDerivative.cs @ a2c4dbf
//
// Only the bound-aware adaptive-step Gradient and Hessian overloads, and the private
// helpers they call (AvailableLeft/AvailableRight/ClampInPlace), are ported -- the subset
// HMC.cs's default gradient function (`NumericalDerivative.Gradient((y) =>
// SafeLogLikelihood(y), x.ToArray(), _lowerBounds, _upperBounds)`) and
// Optimizer.ComputeHessian's MAP-init Hessian (`NumericalDerivative.Hessian((x) =>
// ObjectiveFunction(x), BestParameterSet.Values)`, called with no bounds) actually need.
// NOT ported: the unbounded scalar derivative helpers (ForwardDifference/
// BackwardDifference/CentralDifference/Derivative/RiddersMethod) and the CalculateStepSize
// helper only they use, both Jacobian overloads (fixed-step and bound-aware), and IsBad
// (private to the bound-aware Jacobian overload) -- no caller in this port's scope needs
// them; add them if a later task does. Also NOT ported: the "Second Derivatives - Single
// Variable" region's SecondDerivative/SecondDerivativeForward/SecondDerivativeBackward
// (lines ~252-306) -- grep-confirmed zero callers anywhere in Numerics or RMC-BestFit
// (only Test_Differentiation.cs and docs/mathematics/differentiation.md reference them);
// add them if a later task does.
//
// `lowerBounds`/`upperBounds` are nullable `double[]` in C#, defaulting to null
// (unbounded). This port represents "unbounded" as an empty vector (`{}`) rather than a
// pointer/optional, since AvailableLeft/AvailableRight/ClampInPlace only ever check
// null-ness, never index through a partially-null array -- an empty vector is a direct,
// unambiguous stand-in. ArgumentNullException checks on `f`/`theta` are not ported: the
// C++ signatures take a `const ScalarFunction&` and `const std::vector<double>&`, neither
// of which has a null representation to check.
//
// Every function-evaluation attempt inside the backtracking loops is wrapped in try/catch
// exactly where the C# wraps it in try/catch, mirroring "a bad evaluation halves the step
// and retries" fault tolerance even though no distribution ported so far throws from
// pdf/log_pdf/log_likelihood (they return -inf/NaN instead, which the IsFinite checks
// alone already catch) -- a future caller's f could still throw, and the port should
// behave identically if it does.
#pragma once
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::math::differentiation {

using ScalarFunction = std::function<double(const std::vector<double>&)>;

namespace detail {

// Clamps every element of v to [lower_bounds[k], upper_bounds[k]] in place. Either bound
// vector may be empty (unbounded on that side), mirroring the C# nullable `double[]` args.
inline void clamp_in_place(std::vector<double>& v, const std::vector<double>& lower_bounds,
                            const std::vector<double>& upper_bounds) {
    if (lower_bounds.empty() && upper_bounds.empty()) return;
    for (std::size_t k = 0; k < v.size(); ++k) {
        if (!lower_bounds.empty() && v[k] < lower_bounds[k]) v[k] = lower_bounds[k];
        if (!upper_bounds.empty() && v[k] > upper_bounds[k]) v[k] = upper_bounds[k];
    }
}

// Distance from theta[j] to its lower bound, or +infinity if unbounded.
inline double available_left(const std::vector<double>& theta, int j, const std::vector<double>& lower_bounds) {
    return lower_bounds.empty() ? std::numeric_limits<double>::infinity()
                                 : theta[static_cast<std::size_t>(j)] - lower_bounds[static_cast<std::size_t>(j)];
}

// Distance from theta[j] to its upper bound, or +infinity if unbounded.
inline double available_right(const std::vector<double>& theta, int j, const std::vector<double>& upper_bounds) {
    return upper_bounds.empty() ? std::numeric_limits<double>::infinity()
                                 : upper_bounds[static_cast<std::size_t>(j)] - theta[static_cast<std::size_t>(j)];
}

}  // namespace detail

// Computes the gradient of a scalar function f: R^p -> R at theta, with adaptive
// bound-aware steps: central differences when there's room on both sides of theta[j],
// falling back to a one-sided (forward/backward) difference near a bound, halving the step
// and retrying (up to maxBacktrack times) on a non-finite evaluation or a bound-clamped
// step too small to resolve. A component that never succeeds is set to 0. Throws if
// f(theta) itself is not finite.
inline std::vector<double> gradient(const ScalarFunction& f, const std::vector<double>& theta,
                                     const std::vector<double>& lower_bounds = {},
                                     const std::vector<double>& upper_bounds = {}, double rel_step = 1e-5,
                                     double abs_step = 1e-7, int max_backtrack = 5) {
    int p = static_cast<int>(theta.size());
    std::vector<double> grad(static_cast<std::size_t>(p), 0.0);
    std::vector<double> theta0 = theta;
    double f0 = f(theta0);
    if (!bestfit::numerics::is_finite(f0)) throw std::domain_error("f(theta) is not finite.");

    for (int j = 0; j < p; ++j) {
        bool success = false;
        double scale = std::fabs(theta0[static_cast<std::size_t>(j)]) + 1.0;
        double hj = std::max(abs_step, rel_step * scale);

        for (int backtrack = 0; backtrack <= max_backtrack && !success; ++backtrack) {
            double room_left = detail::available_left(theta0, j, lower_bounds);
            double room_right = detail::available_right(theta0, j, upper_bounds);

            bool can_central = room_left >= hj && room_right >= hj;
            bool use_forward = !can_central && room_right >= hj;
            bool use_backward = !can_central && !use_forward && room_left >= hj;

            try {
                if (can_central) {
                    std::vector<double> theta_plus = theta0, theta_minus = theta0;
                    theta_plus[static_cast<std::size_t>(j)] += hj;
                    theta_minus[static_cast<std::size_t>(j)] -= hj;
                    detail::clamp_in_place(theta_plus, lower_bounds, upper_bounds);
                    detail::clamp_in_place(theta_minus, lower_bounds, upper_bounds);

                    double actual_step =
                        theta_plus[static_cast<std::size_t>(j)] - theta_minus[static_cast<std::size_t>(j)];
                    if (std::fabs(actual_step) < 1e-16 * scale) {
                        hj *= 0.5;
                        continue;
                    }

                    double f_plus = f(theta_plus);
                    double f_minus = f(theta_minus);
                    if (!bestfit::numerics::is_finite(f_plus) || !bestfit::numerics::is_finite(f_minus)) {
                        hj *= 0.5;
                        continue;
                    }

                    grad[static_cast<std::size_t>(j)] = (f_plus - f_minus) / actual_step;
                    success = true;
                } else if (use_forward) {
                    std::vector<double> theta_plus = theta0;
                    theta_plus[static_cast<std::size_t>(j)] += hj;
                    detail::clamp_in_place(theta_plus, lower_bounds, upper_bounds);

                    double actual_step = theta_plus[static_cast<std::size_t>(j)] - theta0[static_cast<std::size_t>(j)];
                    if (std::fabs(actual_step) < 1e-16 * scale) {
                        hj *= 0.5;
                        continue;
                    }

                    double f_plus = f(theta_plus);
                    if (!bestfit::numerics::is_finite(f_plus)) {
                        hj *= 0.5;
                        continue;
                    }

                    grad[static_cast<std::size_t>(j)] = (f_plus - f0) / actual_step;
                    success = true;
                } else if (use_backward) {
                    std::vector<double> theta_minus = theta0;
                    theta_minus[static_cast<std::size_t>(j)] -= hj;
                    detail::clamp_in_place(theta_minus, lower_bounds, upper_bounds);

                    double actual_step = theta0[static_cast<std::size_t>(j)] - theta_minus[static_cast<std::size_t>(j)];
                    if (std::fabs(actual_step) < 1e-16 * scale) {
                        hj *= 0.5;
                        continue;
                    }

                    double f_minus = f(theta_minus);
                    if (!bestfit::numerics::is_finite(f_minus)) {
                        hj *= 0.5;
                        continue;
                    }

                    grad[static_cast<std::size_t>(j)] = (f0 - f_minus) / actual_step;
                    success = true;
                } else {
                    hj *= 0.5;
                }
            } catch (...) {
                hj *= 0.5;
            }
        }
        if (!success) grad[static_cast<std::size_t>(j)] = 0.0;
    }
    return grad;
}

// Computes the p x p symmetric Hessian matrix of a scalar function f: R^p -> R at theta,
// with the same adaptive bound-aware stepping strategy as gradient(). Diagonal elements use
// three-point formulas (central/forward/backward); off-diagonal (mixed partial) elements
// use a central four-point stencil when both parameters have room on both sides, otherwise
// a one-sided (forward-forward or backward-backward) three-point stencil. A component that
// never succeeds is set to 0. Final pass averages H[i,j]/H[j,i] to enforce exact symmetry.
// Throws if f(theta) itself is not finite.
inline std::vector<std::vector<double>> hessian(const ScalarFunction& f, const std::vector<double>& theta,
                                                  const std::vector<double>& lower_bounds = {},
                                                  const std::vector<double>& upper_bounds = {},
                                                  double rel_step = 1e-4, double abs_step = 1e-6,
                                                  int max_backtrack = 6) {
    int p = static_cast<int>(theta.size());
    std::vector<std::vector<double>> h(static_cast<std::size_t>(p), std::vector<double>(static_cast<std::size_t>(p), 0.0));
    std::vector<double> x0 = theta;
    double f0 = f(x0);
    if (!bestfit::numerics::is_finite(f0)) throw std::domain_error("f(theta) is not finite.");

    // Diagonal elements: d2f/dtheta_j^2.
    for (int j = 0; j < p; ++j) {
        bool success = false;
        double scale = std::fabs(x0[static_cast<std::size_t>(j)]) + 1.0;
        double hj = std::max(abs_step, rel_step * scale);

        for (int backtrack = 0; backtrack <= max_backtrack && !success; ++backtrack) {
            double room_left = detail::available_left(x0, j, lower_bounds);
            double room_right = detail::available_right(x0, j, upper_bounds);

            bool can_central = room_left >= hj && room_right >= hj;
            bool can_forward = room_right >= 2.0 * hj;
            bool can_backward = room_left >= 2.0 * hj;

            try {
                if (can_central) {
                    std::vector<double> x_plus = x0, x_minus = x0;
                    x_plus[static_cast<std::size_t>(j)] += hj;
                    x_minus[static_cast<std::size_t>(j)] -= hj;
                    detail::clamp_in_place(x_plus, lower_bounds, upper_bounds);
                    detail::clamp_in_place(x_minus, lower_bounds, upper_bounds);

                    double f_plus = f(x_plus);
                    double f_minus = f(x_minus);
                    if (!bestfit::numerics::is_finite(f_plus) || !bestfit::numerics::is_finite(f_minus)) {
                        hj *= 0.5;
                        continue;
                    }

                    double actual_step = x_plus[static_cast<std::size_t>(j)] - x_minus[static_cast<std::size_t>(j)];
                    if (std::fabs(actual_step) < 1e-16 * scale) {
                        hj *= 0.5;
                        continue;
                    }

                    double h_eff = 0.5 * actual_step;
                    h[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] =
                        (f_plus - 2.0 * f0 + f_minus) / (h_eff * h_eff);
                    success = true;
                } else if (can_forward) {
                    std::vector<double> x1 = x0, x2 = x0;
                    x1[static_cast<std::size_t>(j)] += hj;
                    x2[static_cast<std::size_t>(j)] += 2.0 * hj;
                    detail::clamp_in_place(x1, lower_bounds, upper_bounds);
                    detail::clamp_in_place(x2, lower_bounds, upper_bounds);

                    double f1 = f(x1);
                    double f2 = f(x2);
                    if (!bestfit::numerics::is_finite(f1) || !bestfit::numerics::is_finite(f2)) {
                        hj *= 0.5;
                        continue;
                    }

                    double h_eff = x1[static_cast<std::size_t>(j)] - x0[static_cast<std::size_t>(j)];
                    if (std::fabs(h_eff) < 1e-16 * scale) {
                        hj *= 0.5;
                        continue;
                    }

                    h[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = (f0 - 2.0 * f1 + f2) / (h_eff * h_eff);
                    success = true;
                } else if (can_backward) {
                    std::vector<double> x1 = x0, x2 = x0;
                    x1[static_cast<std::size_t>(j)] -= hj;
                    x2[static_cast<std::size_t>(j)] -= 2.0 * hj;
                    detail::clamp_in_place(x1, lower_bounds, upper_bounds);
                    detail::clamp_in_place(x2, lower_bounds, upper_bounds);

                    double f1 = f(x1);
                    double f2 = f(x2);
                    if (!bestfit::numerics::is_finite(f1) || !bestfit::numerics::is_finite(f2)) {
                        hj *= 0.5;
                        continue;
                    }

                    double h_eff = x0[static_cast<std::size_t>(j)] - x1[static_cast<std::size_t>(j)];
                    if (std::fabs(h_eff) < 1e-16 * scale) {
                        hj *= 0.5;
                        continue;
                    }

                    h[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = (f2 - 2.0 * f1 + f0) / (h_eff * h_eff);
                    success = true;
                } else {
                    hj *= 0.5;
                }
            } catch (...) {
                hj *= 0.5;
            }
        }
        if (!success) h[static_cast<std::size_t>(j)][static_cast<std::size_t>(j)] = 0.0;
    }

    // Off-diagonal elements: mixed partials d2f/(dtheta_i dtheta_j), i < j.
    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j) {
            bool success = false;
            double scale_i = std::fabs(x0[static_cast<std::size_t>(i)]) + 1.0;
            double scale_j = std::fabs(x0[static_cast<std::size_t>(j)]) + 1.0;
            double hi = std::max(abs_step, rel_step * scale_i);
            double hj = std::max(abs_step, rel_step * scale_j);

            for (int backtrack = 0; backtrack <= max_backtrack && !success; ++backtrack) {
                double left_i = detail::available_left(x0, i, lower_bounds);
                double right_i = detail::available_right(x0, i, upper_bounds);
                double left_j = detail::available_left(x0, j, lower_bounds);
                double right_j = detail::available_right(x0, j, upper_bounds);

                bool central_i = left_i >= hi && right_i >= hi;
                bool central_j = left_j >= hj && right_j >= hj;

                try {
                    if (central_i && central_j) {
                        // Central 4-point stencil.
                        std::vector<double> x_pp = x0, x_pm = x0, x_mp = x0, x_mm = x0;
                        x_pp[static_cast<std::size_t>(i)] += hi;
                        x_pp[static_cast<std::size_t>(j)] += hj;
                        x_pm[static_cast<std::size_t>(i)] += hi;
                        x_pm[static_cast<std::size_t>(j)] -= hj;
                        x_mp[static_cast<std::size_t>(i)] -= hi;
                        x_mp[static_cast<std::size_t>(j)] += hj;
                        x_mm[static_cast<std::size_t>(i)] -= hi;
                        x_mm[static_cast<std::size_t>(j)] -= hj;

                        detail::clamp_in_place(x_pp, lower_bounds, upper_bounds);
                        detail::clamp_in_place(x_pm, lower_bounds, upper_bounds);
                        detail::clamp_in_place(x_mp, lower_bounds, upper_bounds);
                        detail::clamp_in_place(x_mm, lower_bounds, upper_bounds);

                        double f_pp = f(x_pp);
                        double f_pm = f(x_pm);
                        double f_mp = f(x_mp);
                        double f_mm = f(x_mm);
                        if (!bestfit::numerics::is_finite(f_pp) || !bestfit::numerics::is_finite(f_pm) ||
                            !bestfit::numerics::is_finite(f_mp) || !bestfit::numerics::is_finite(f_mm)) {
                            hi *= 0.5;
                            hj *= 0.5;
                            continue;
                        }

                        double actual_step_i = x_pp[static_cast<std::size_t>(i)] - x_mm[static_cast<std::size_t>(i)];
                        double actual_step_j = x_pp[static_cast<std::size_t>(j)] - x_mm[static_cast<std::size_t>(j)];
                        if (std::fabs(actual_step_i) < 1e-16 * scale_i || std::fabs(actual_step_j) < 1e-16 * scale_j) {
                            hi *= 0.5;
                            hj *= 0.5;
                            continue;
                        }

                        double mixed = (f_pp - f_pm - f_mp + f_mm) / (actual_step_i * actual_step_j);
                        h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = mixed;
                        h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = mixed;
                        success = true;
                    } else {
                        bool forward_forward = right_i >= hi && right_j >= hj;
                        bool backward_backward = left_i >= hi && left_j >= hj;

                        if (forward_forward) {
                            std::vector<double> x_i = x0, x_j = x0, x_ij = x0;
                            x_i[static_cast<std::size_t>(i)] += hi;
                            x_j[static_cast<std::size_t>(j)] += hj;
                            x_ij[static_cast<std::size_t>(i)] += hi;
                            x_ij[static_cast<std::size_t>(j)] += hj;

                            detail::clamp_in_place(x_i, lower_bounds, upper_bounds);
                            detail::clamp_in_place(x_j, lower_bounds, upper_bounds);
                            detail::clamp_in_place(x_ij, lower_bounds, upper_bounds);

                            double f_i = f(x_i);
                            double f_j = f(x_j);
                            double f_ij = f(x_ij);
                            if (!bestfit::numerics::is_finite(f_i) || !bestfit::numerics::is_finite(f_j) ||
                                !bestfit::numerics::is_finite(f_ij)) {
                                hi *= 0.5;
                                hj *= 0.5;
                                continue;
                            }

                            double actual_step_i = x_i[static_cast<std::size_t>(i)] - x0[static_cast<std::size_t>(i)];
                            double actual_step_j = x_j[static_cast<std::size_t>(j)] - x0[static_cast<std::size_t>(j)];
                            if (std::fabs(actual_step_i) < 1e-16 * scale_i ||
                                std::fabs(actual_step_j) < 1e-16 * scale_j) {
                                hi *= 0.5;
                                hj *= 0.5;
                                continue;
                            }

                            double mixed = (f_ij - f_i - f_j + f0) / (actual_step_i * actual_step_j);
                            h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = mixed;
                            h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = mixed;
                            success = true;
                        } else if (backward_backward) {
                            std::vector<double> x_i = x0, x_j = x0, x_ij = x0;
                            x_i[static_cast<std::size_t>(i)] -= hi;
                            x_j[static_cast<std::size_t>(j)] -= hj;
                            x_ij[static_cast<std::size_t>(i)] -= hi;
                            x_ij[static_cast<std::size_t>(j)] -= hj;

                            detail::clamp_in_place(x_i, lower_bounds, upper_bounds);
                            detail::clamp_in_place(x_j, lower_bounds, upper_bounds);
                            detail::clamp_in_place(x_ij, lower_bounds, upper_bounds);

                            double f_i = f(x_i);
                            double f_j = f(x_j);
                            double f_ij = f(x_ij);
                            if (!bestfit::numerics::is_finite(f_i) || !bestfit::numerics::is_finite(f_j) ||
                                !bestfit::numerics::is_finite(f_ij)) {
                                hi *= 0.5;
                                hj *= 0.5;
                                continue;
                            }

                            double actual_step_i = x0[static_cast<std::size_t>(i)] - x_i[static_cast<std::size_t>(i)];
                            double actual_step_j = x0[static_cast<std::size_t>(j)] - x_j[static_cast<std::size_t>(j)];
                            if (std::fabs(actual_step_i) < 1e-16 * scale_i ||
                                std::fabs(actual_step_j) < 1e-16 * scale_j) {
                                hi *= 0.5;
                                hj *= 0.5;
                                continue;
                            }

                            double mixed = (f_ij - f_i - f_j + f0) / (actual_step_i * actual_step_j);
                            h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = mixed;
                            h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = mixed;
                            success = true;
                        } else {
                            hi *= 0.5;
                            hj *= 0.5;
                        }
                    }
                } catch (...) {
                    hi *= 0.5;
                    hj *= 0.5;
                }
            }

            if (!success) {
                h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = 0.0;
                h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = 0.0;
            }
        }
    }

    // Final symmetry enforcement: average the upper/lower triangular parts to reduce
    // finite-difference noise.
    for (int i = 0; i < p; ++i) {
        for (int j = i + 1; j < p; ++j) {
            double average = 0.5 * (h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +
                                     h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)]);
            h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = average;
            h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = average;
        }
    }

    return h;
}

}  // namespace bestfit::numerics::math::differentiation
