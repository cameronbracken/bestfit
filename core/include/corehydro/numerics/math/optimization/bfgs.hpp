// ported from: Numerics/Mathematics/Optimization/Local/BFGS.cs @ 2a0357a
//
// The Broyden-Fletcher-Goldfarb-Shanno (BFGS) local optimizer: an iterative method for
// unconstrained nonlinear optimization that gradually improves an approximation to the
// inverse Hessian of the loss function from gradient evaluations via a generalized secant
// method (Numerical Recipes, Press et al.; see the C# file's references). This is the
// first REAL local Optimizer subclass in the port -- it derives from the ported Optimizer
// base exactly as differential_evolution.hpp does, NOT via the Phase 4
// estimation/support/optimizer_adapters.hpp `detail::` stopgap (which wrapped the
// standalone Phase 0 nelder_mead/brent_search; B5+ optimizers make that pattern
// unnecessary going forward).
//
// Transcription notes:
//
// 1. Gradient member. C# exposes a settable public field `Func<double[], double[]>?
//    Gradient` (also settable via the ctor's optional trailing parameter); null means
//    "use finite differences". Ported as a public std::function member named `gradient`
//    with the same two entry points; an empty std::function is the null state. Every
//    numeric fallback routes through the ported bound-aware
//    differentiation::gradient(f, point) overload -- the same call shape as the C#'s
//    `NumericalDerivative.Gradient(x => Evaluate(x, ref cancel), p)` (function + point,
//    default steps, no bounds) -- with the probe function calling the BASE's evaluate()
//    so function-evaluation counting, best-parameter tracking, and the cancellation
//    cascade all mirror the C# exactly. The lambda copies its (const-ref) probe point
//    into a local because Objective/evaluate take a mutable reference (see
//    optimizer.hpp's MUTABLE-POINT SEMANTICS note). optimize()'s two identical inline
//    ternaries share the small private numerical_gradient() helper; LineSearch/Zoom keep
//    theirs inline because of the cancelFlag round-trip (note 2).
//
// 2. C# LineSearch/Zoom capture quirk. Inside LineSearch/Zoom, `cancel` is a `ref`
//    parameter, which C# lambdas cannot capture -- the C# copies it into a local
//    `cancelFlag`, lets the gradient lambda capture that, then writes it back. C++
//    reference parameters have no such restriction, but the local-copy round-trip is
//    transcribed anyway so the code maps line-for-line onto the C#.
//
// 3. LineSearchArmijo is ported for structural fidelity but is NOT called by optimize()
//    -- exactly as in the C#, where the strong-Wolfe LineSearch superseded it. Its
//    `throw new Exception("Roundoff problem in line search.")` is a genuine C# throw
//    (not a Debug.WriteLine/swallowed guard), so it ports as a real std::runtime_error;
//    if it ever fired it would surface through the base's minimize()/maximize()
//    Failure-status path like any other objective-side exception.
//
// 4. `TOLX` in optimize() is declared-and-unused in the C# too (LineSearchArmijo has its
//    own local TOLX); kept, with a (void) cast to satisfy -Wall/-Wextra.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/differentiation/numerical_derivative.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class BFGS : public Optimizer {
   public:
    // The function for evaluating the gradient of the objective function (C#'s settable
    // public `Gradient` field). Empty (the default) means fall back to finite differences.
    using GradientFunction = std::function<std::vector<double>(const std::vector<double>&)>;

    // Construct a new BFGS optimization method. `gradient` is optional; the default uses
    // finite differences (mirrors the C# ctor's optional trailing parameter).
    BFGS(Objective objective_function, int number_of_parameters,
         std::vector<double> initial_values, std::vector<double> lower_bounds,
         std::vector<double> upper_bounds, GradientFunction gradient_function = nullptr)
        : Optimizer(std::move(objective_function), number_of_parameters) {
        // Check if the length of the initial, lower and upper bounds equal the number of
        // parameters.
        if (static_cast<int>(initial_values.size()) != number_of_parameters ||
            static_cast<int>(lower_bounds.size()) != number_of_parameters ||
            static_cast<int>(upper_bounds.size()) != number_of_parameters) {
            throw ArgumentException(
                "The initial values and lower and upper bounds must be the same length as the "
                "number of parameters.");
        }
        // Check if the initial values are between the lower and upper values.
        for (std::size_t j = 0; j < initial_values.size(); j++) {
            if (upper_bounds[j] < lower_bounds[j]) {
                throw ArgumentException("The upper bound cannot be less than the lower bound.");
            }
            if (initial_values[j] < lower_bounds[j] || initial_values[j] > upper_bounds[j]) {
                throw ArgumentException(
                    "The initial values must be between the upper and lower bounds.");
            }
        }
        initial_values_ = std::move(initial_values);
        lower_bounds_ = std::move(lower_bounds);
        upper_bounds_ = std::move(upper_bounds);
        gradient = std::move(gradient_function);
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The function for evaluating the gradient of the objective function (see the class
    // header note 1; C# public field `Gradient`).
    GradientFunction gradient;

   protected:
    void optimize() override {
        int D = number_of_parameters_;
        double EPS = kDoubleMachineEpsilon;
        double TOLX = 4 * EPS, STPMX = 100.0;
        (void)TOLX;  // declared-and-unused in the C# as well (see class header note 4)
        bool cancel = false, check = false;

        auto p = initial_values_;
        std::vector<double> pnew(static_cast<std::size_t>(D));

        // Calculate the starting function value and gradient, and initialize the inverse
        // Hessian to the unit matrix.
        double fp = evaluate(p, cancel);
        auto g = gradient ? gradient(p) : numerical_gradient(p, cancel);
        std::vector<double> dg(static_cast<std::size_t>(D));
        std::vector<double> hdg(static_cast<std::size_t>(D));
        std::vector<double> xi(static_cast<std::size_t>(D));
        auto hessin = linalg::Matrix(D, D);

        double sum = 0.0;
        for (int i = 0; i < D; i++) {
            for (int j = 0; j < D; j++) hessin(i, j) = 0.0;
            hessin(i, i) = 1.0;
            xi[i] = -g[i];
            sum += p[i] * p[i];
        }

        double fret = 0.0;
        double stpmax = STPMX * std::max(std::sqrt(sum), static_cast<double>(D));

        while (iterations_ < max_iterations) {
            // Perform line search.
            line_search(p, fp, g, xi, pnew, fret, stpmax, check, cancel);
            if (cancel) return;

            // Check convergence.
            if (check_convergence(fp, fret)) {
                update_status(OptimizationStatus::Success);
                return;
            }

            // The new function evaluation occurs in line search; save the function value in
            // fp for the next line search. It is usually safe to ignore the value of check.
            fp = fret;
            for (int i = 0; i < D; i++) {
                xi[i] = pnew[i] - p[i];
                p[i] = pnew[i];
            }

            // Save the old gradient, and get the new gradient.
            for (int i = 0; i < D; i++) dg[i] = g[i];
            g = gradient ? gradient(p) : numerical_gradient(p, cancel);
            if (cancel) return;

            // Compute difference of gradients.
            for (int i = 0; i < D; i++) dg[i] = g[i] - dg[i];

            // And difference times current matrix.
            for (int i = 0; i < D; i++) {
                hdg[i] = 0.0;
                for (int j = 0; j < D; j++) hdg[i] += hessin(i, j) * dg[j];
            }

            // Calculate dot products for the denominators.
            double fac = 0.0, fae = 0.0, sumdg = 0.0, sumxi = 0.0;
            for (int i = 0; i < D; i++) {
                fac += dg[i] * xi[i];
                fae += dg[i] * hdg[i];
                sumdg += sqr(dg[i]);
                sumxi += sqr(xi[i]);
            }

            // Skip update if fac not sufficiently positive.
            if (fac > std::sqrt(EPS * sumdg * sumxi)) {
                fac = 1.0 / fac;
                double fad = 1.0 / fae;
                for (int i = 0; i < D; i++) dg[i] = fac * xi[i] - fad * hdg[i];
                for (int i = 0; i < D; i++) {
                    for (int j = i; j < D; j++) {
                        hessin(i, j) +=
                            fac * xi[i] * xi[j] - fad * hdg[i] * hdg[j] + fae * dg[i] * dg[j];
                        hessin(j, i) = hessin(i, j);
                    }
                }
            }

            for (int i = 0; i < D; i++) {
                xi[i] = 0.0;
                for (int j = 0; j < D; j++) xi[i] -= hessin(i, j) * g[j];
            }

            iterations_ += 1;
        }

        // If we made it to here, the maximum iterations were reached.
        update_status(OptimizationStatus::MaximumIterationsReached);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    // Finite-difference fallback for the gradient: the C#'s
    // `NumericalDerivative.Gradient(x => Evaluate(x, ref cancel), p)` (see class header
    // note 1).
    std::vector<double> numerical_gradient(const std::vector<double>& p, bool& cancel) {
        return differentiation::gradient(
            [this, &cancel](const std::vector<double>& x) {
                auto values = x;  // Objective/evaluate take a mutable reference
                return evaluate(values, cancel);
            },
            p);
    }

    // Auxiliary function for searching a line (Numerical-Recipes backtracking Armijo
    // search). Ported for structural fidelity; NOT called by optimize(), exactly as in the
    // C# (see class header note 3).
    //   xold:   n-dimensional point [0..n-1].
    //   fold:   value of the function at xold.
    //   g:      gradient of function at xold.
    //   p:      a direction to search (C# `ref`; may be rescaled in place).
    //   x:      a new point x[0..n-1] (C# `ref`).
    //   f:      the new function value.
    //   stpmax: limits the length of steps.
    //   check:  false on a normal exit, true when x is too close to xold.
    //   cancel: determines if the solver should be canceled.
    void line_search_armijo(const std::vector<double>& xold, double fold,
                            const std::vector<double>& g, std::vector<double>& p,
                            std::vector<double>& x, double& f, double stpmax, bool& check,
                            bool& cancel) {
        double ALF = 1.0e-4, TOLX = kDoubleMachineEpsilon;
        double a, alam, alam2 = 0.0, alamin, b, disc, f2 = 0.0;
        double rhs1, rhs2, slope = 0.0, sum = 0.0, temp, test, tmplam;
        std::size_t i, n = xold.size();
        check = false;
        for (i = 0; i < n; i++) sum += p[i] * p[i];
        sum = std::sqrt(sum);
        if (sum > stpmax)
            for (i = 0; i < n; i++) p[i] *= stpmax / sum;
        for (i = 0; i < n; i++) slope += g[i] * p[i];
        if (slope == 0.0) return;  // If the slope is zero, it is on a flat spot. Exit the routine.
        if (slope > 0.0) throw std::runtime_error("Roundoff problem in line search.");
        test = 0.0;
        for (i = 0; i < n; i++) {
            temp = std::fabs(p[i]) / std::max(std::fabs(xold[i]), 1.0);
            if (temp > test) test = temp;
        }
        alamin = TOLX / test;
        alam = 1.0;
        for (;;) {
            for (i = 0; i < n; i++) {
                x[i] = xold[i] + alam * p[i];
                // Make sure the parameters are within the bounds.
                x[i] = repair_parameter(x[i], lower_bounds_[i], upper_bounds_[i]);
            }
            f = evaluate(x, cancel);
            if (cancel) return;
            if (alam < alamin) {
                for (i = 0; i < n; i++) x[i] = xold[i];
                check = true;
                return;
            } else if (f <= fold + ALF * alam * slope) {
                return;
            } else {
                if (alam == 1.0) {
                    tmplam = -slope / (2.0 * (f - fold - slope));
                } else {
                    rhs1 = f - fold - alam * slope;
                    rhs2 = f2 - fold - alam2 * slope;
                    a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2);
                    b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2)) /
                        (alam - alam2);
                    if (a == 0.0) {
                        tmplam = -slope / (2.0 * b);
                    } else {
                        disc = b * b - 3.0 * a * slope;
                        if (disc < 0.0)
                            tmplam = 0.5 * alam;
                        else if (b <= 0.0)
                            tmplam = (-b + std::sqrt(disc)) / (3.0 * a);
                        else
                            tmplam = -slope / (b + std::sqrt(disc));
                    }
                    if (tmplam > 0.5 * alam) tmplam = 0.5 * alam;
                }
            }
            alam2 = alam;
            f2 = f;
            alam = std::max(tmplam, 0.1 * alam);
        }
    }

    // Performs a strong Wolfe line search to find a step size that satisfies both the
    // sufficient decrease (Armijo) and curvature conditions.
    //   x0:     the current parameter vector.
    //   f0:     the objective function value at x0.
    //   g0:     the gradient at x0.
    //   p:      the search direction (C# `double[]`, a reference type mutated in place
    //           when rescaled to stpmax -- so a mutable reference here too).
    //   x:      the output parameter vector at the accepted step size.
    //   f:      the objective function value at x.
    //   stpmax: the maximum allowable step length.
    //   check:  returns true if the search failed to find an acceptable step.
    //   cancel: set to true if cancellation is requested during evaluation.
    void line_search(const std::vector<double>& x0, double f0, const std::vector<double>& g0,
                     std::vector<double>& p, std::vector<double>& x, double& f, double stpmax,
                     bool& check, bool& cancel) {
        const double c1 = 1e-4, c2 = 0.9;
        double alpha = 1.0, alpha_prev = 0.0;
        double f_prev = f0;
        double slope0 = sum_product(g0, p);
        std::vector<double> g(p.size());
        std::vector<double> x_temp(p.size());

        // C#: double normP = Math.Sqrt(p.Sum(pi => pi * pi));
        double sum_sq = 0.0;
        for (std::size_t i = 0; i < p.size(); i++) sum_sq += p[i] * p[i];
        double norm_p = std::sqrt(sum_sq);
        if (norm_p > stpmax) {
            double scale = stpmax / norm_p;
            for (std::size_t i = 0; i < p.size(); i++) p[i] *= scale;
        }

        for (int iter = 0; iter < 20; iter++) {
            for (std::size_t i = 0; i < x0.size(); i++) {
                x_temp[i] = x0[i] + alpha * p[i];
                x_temp[i] = repair_parameter(x_temp[i], lower_bounds_[i], upper_bounds_[i]);
            }

            f = evaluate(x_temp, cancel);
            if (cancel) return;

            if (f > f0 + c1 * alpha * slope0 || (iter > 0 && f >= f_prev)) {
                zoom(x0, f0, slope0, p, alpha_prev, alpha, f, x, cancel);
                return;
            }

            // C# local-copy round-trip transcribed as-is (see class header note 2).
            bool cancel_flag = cancel;
            g = gradient ? gradient(x_temp)
                         : differentiation::gradient(
                               [this, &cancel_flag](const std::vector<double>& xg) {
                                   auto values = xg;
                                   return evaluate(values, cancel_flag);
                               },
                               x_temp);
            cancel = cancel_flag;
            if (cancel) return;

            double slope = sum_product(g, p);

            if (std::fabs(slope) <= -c2 * slope0) {
                std::copy(x_temp.begin(), x_temp.end(), x.begin());
                return;
            }

            if (slope >= 0) {
                zoom(x0, f0, slope0, p, alpha, alpha_prev, f, x, cancel);
                return;
            }

            alpha_prev = alpha;
            f_prev = f;
            alpha *= 2.0;
        }

        std::copy(x0.begin(), x0.end(), x.begin());
        check = true;
    }

    // Zoom phase of the strong Wolfe line search: bisection between two step sizes to find
    // an acceptable step satisfying the Wolfe conditions.
    //   x0:        the initial parameter vector.
    //   f0:        the objective function value at x0.
    //   slope0:    the directional derivative at x0 along the search direction.
    //   p:         the search direction vector.
    //   alphaLow:  the lower bound of the step size interval (by value, as in the C#).
    //   alphaHigh: the upper bound of the step size interval (by value, as in the C#).
    //   f:         the objective function value at the final accepted point.
    //   x:         the parameter vector at the final accepted step size.
    //   cancel:    set to true if cancellation is requested during evaluation.
    void zoom(const std::vector<double>& x0, double f0, double slope0,
              const std::vector<double>& p, double alpha_low, double alpha_high, double& f,
              std::vector<double>& x, bool& cancel) {
        const double c1 = 1e-4, c2 = 0.9;
        std::vector<double> g(p.size());
        std::vector<double> x_temp(p.size());

        for (int iter = 0; iter < 20; iter++) {
            double alpha = 0.5 * (alpha_low + alpha_high);
            for (std::size_t i = 0; i < x0.size(); i++) {
                x_temp[i] = x0[i] + alpha * p[i];
                x_temp[i] = repair_parameter(x_temp[i], lower_bounds_[i], upper_bounds_[i]);
            }

            f = evaluate(x_temp, cancel);
            if (cancel) return;

            if (f > f0 + c1 * alpha * slope0) {
                alpha_high = alpha;
            } else {
                // C# local-copy round-trip transcribed as-is (see class header note 2).
                bool cancel_flag = cancel;
                g = gradient ? gradient(x_temp)
                             : differentiation::gradient(
                                   [this, &cancel_flag](const std::vector<double>& xg) {
                                       auto values = xg;
                                       return evaluate(values, cancel_flag);
                                   },
                                   x_temp);
                cancel = cancel_flag;
                if (cancel) return;

                double slope = sum_product(g, p);

                if (std::fabs(slope) <= -c2 * slope0) {
                    std::copy(x_temp.begin(), x_temp.end(), x.begin());
                    return;
                }

                if (slope * (alpha_high - alpha_low) >= 0) alpha_high = alpha_low;

                alpha_low = alpha;
            }
        }

        std::copy(x0.begin(), x0.end(), x.begin());
    }
};

}  // namespace corehydro::numerics::math::optimization
