// ported from: Numerics/Mathematics/Optimization/Local/BrentSearch.cs @ 2a0357a
//             + Numerics/Mathematics/Optimization/Support/Optimizer.cs (base behavior)
//
// Brent's method (golden-section step + parabolic interpolation) for 1D optimization
// (Sprott 1991 / Press et al.). The Optimizer-base best-tracking, minimize/maximize
// sign handling, and max-iterations behavior are folded in here the same way
// nelder_mead.hpp folds them in for the simplex optimizer (see that header for the
// precedent). Function-evaluation counting/limits, parameter-set tracing, and Hessian
// computation are intentionally omitted, matching the nelder_mead.hpp precedent.
// Copula MPL/IFM fit oracles (a later task) replay this exact golden-section/parabolic
// search path, so transcription fidelity of the loop below is the point of this file.
//
// Bracket() (the step-and-expand bracketing helper) was omitted in Phase 0 (no caller
// needed it -- copula fits supply their own [lower, upper] bounds directly); Task B6 adds
// it (plus the best_fitness() accessor) ADDITIVELY for Powell's LineMinimization, which
// calls `Bracket(0.1)` then `Minimize()` then reads `BestParameterSet.Values[0]` /
// `.Fitness` on the C# BrentSearch.
//
// v2.1.4 (upstream-sync Task 11, 651035e "Harden MVNDST status and Brent bracketing"):
// Bracket() gained real geometric growth. The expansion factor `k` was declared but
// UNUSED pre-2.1.4 (every step added the SAME `s`, so this file's old `(void)k;` marker
// was correct at the time -- NOW RETIRED); `k` is applied every unsuccessful iteration
// (`s *= k`), and the termination test loosened from `fc > fb` to `fc >= fb` (the C#
// dropped its now-unused `fa` tracking to match; kept here too since nothing reads it
// past the initial swap). Upstream's measured regression: a minimum 1000 units away
// now brackets in a few dozen evaluations (logarithmic in distance/s) instead of
// thousands (linear). Bracket() also gained: upfront validation of `s` (finite,
// nonzero) and `k` (finite, > 1); NaN-objective guards after every evaluation;
// non-finite-coordinate guards on the initial point and every trial `c`, and on `s`
// itself after each `s *= k`; and an explicit `max_iterations`-bounded loop (the
// pre-2.1.4 C++ used `while (true)` -- an unconditional loop that never terminates for
// a monotone, flat, or NaN-valued objective, since `fc > fb` can then never become
// true; the new bound is a live-hang fix upstream also makes). Every guard below fires
// BEFORE the success path's `lower_bound_`/`upper_bound_` assignment, so a failed
// bracket() call always leaves the existing bounds untouched -- the C# "failed searches
// leave the existing bounds unchanged" contract.
//
// Status-surface divergence (carried from Task 2's review, still applicable here): this
// port has NO OptimizationStatus surface -- BrentSearch here is the deliberately-
// standalone Phase 0 class (see above), not a subclass of the ported Optimizer base (see
// optimizer.hpp). The C# Bracket() routes every failure through `UpdateStatus(status,
// exception)`, which only throws when `ReportFailure` is true (the C# default) and
// otherwise records `Status` and returns without throwing. This port has neither
// `ReportFailure` nor `Status`: every guard below throws UNCONDITIONALLY, matching the
// C# DEFAULT-configuration observable behavior (a bare `BrentSearch(...).Bracket(...)`
// with `ReportFailure` left at its default `true`) -- the only behavior any current
// caller (Powell's LineMinimization) can observe. Two narrower C# behaviors are
// consequently NOT reproduced: (1) `ReportFailure = false` silently swallowing a
// failure and leaving a queryable `Status` -- there is no equivalent query surface
// here, every failure throws; (2) a Bracket()-originated `MaximumIterationsReached`
// exception, when Bracket() is invoked from inside an ENCLOSING C# Optimizer's own
// Optimize() (as Powell's LineMinimization does), is silently swallowed by that
// enclosing optimizer's `catch (ArgumentException ex)` ParamName filter (see
// optimizer.hpp's header) because it shares the `nameof(MaxIterations)` tag -- here it
// propagates as an ordinary thrown C++ exception through Powell::optimize() and out
// through Optimizer::maximize()/minimize()'s `catch (...)`, so it follows THAT
// optimizer's own `report_failure` instead of always being swallowed. Neither
// divergence is reachable by any current fixture: every existing Bracket() caller
// (box_cox.hpp/yeo_johnson.hpp/optimizer_adapters.hpp use only minimize()/maximize()
// with fixed bounds, never bracket(); only powell.hpp's LineMinimization calls
// `bracket(0.1)`) supplies a smooth, bounded-below objective that brackets in a handful
// of iterations -- the pre-2.1.4 C++ `while (true)` never hung on any of them -- so
// max_iterations is never exhausted and every guard stays dormant.
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>

#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class BrentSearch {
   public:
    using Objective = std::function<double(double)>;

    // lower_bound/upper_bound: the inclusive interval containing the optimum.
    BrentSearch(Objective objective, double lower_bound, double upper_bound)
        : objective_(std::move(objective)), lower_bound_(lower_bound), upper_bound_(upper_bound) {
        if (upper_bound < lower_bound)
            throw std::invalid_argument("The upper bound cannot be less than the lower bound.");
    }

    int max_iterations = 10000;
    double relative_tolerance = 1E-8;
    double absolute_tolerance = 1E-8;

    void maximize() {
        function_scale_ = -1;
        clear_and_optimize();
    }
    void minimize() {
        function_scale_ = 1;
        clear_and_optimize();
    }

    // The parameter that produced the best (min/max, per the last maximize()/minimize()
    // call) fitness seen across all evaluations.
    double best_parameter() const { return best_value_; }

    // The fitness at best_parameter(), in the internal sign convention (function_scale *
    // objective, i.e. the raw objective value for minimize()) -- the C# BrentSearch's
    // `BestParameterSet.Fitness`. Added in B6 for Powell's LineMinimization.
    double best_fitness() const { return best_fitness_; }

    // The current interval bounds -- the C# BrentSearch's `LowerBound`/`UpperBound`
    // properties. Added in Task 11 so callers (and the bracket() unit tests) can observe
    // the bracketing interval bracket() leaves behind.
    double lower_bound() const { return lower_bound_; }
    double upper_bound() const { return upper_bound_; }

    // Bracket the objective function minimum (ported from BrentSearch.cs Bracket();
    // see the file header for the v2.1.4 hardening this task ports and the
    // Status-surface divergence it necessarily carries). Steps from the lower bound and
    // expands geometrically (multiplying the step by `k` after every unsuccessful trial)
    // downhill until the function turns back up, then replaces [lower_bound_,
    // upper_bound_] with the bracketing interval. Calls the RAW objective directly --
    // no function-scale flip, no best-tracking -- exactly as the C# calls
    // `ObjectiveFunction(...)` rather than `Evaluate(...)`.
    //   s: starting step size (default 1E-2). Must be finite and nonzero.
    //   k: geometric expansion factor (default 2). Must be finite and greater than one.
    // Throws (unconditionally -- see the file header) if: `s`/`k` fail the checks above;
    // any trial coordinate or the growing step leaves the finite range of `double`; or
    // the objective function returns NaN. A failed call leaves lower_bound()/
    // upper_bound() exactly as they were before the call.
    void bracket(double s = 1E-2, double k = 2.0) {
        if (!corehydro::numerics::is_finite(s) || s == 0.0)
            throw std::invalid_argument("The starting step size must be finite and nonzero.");
        if (!corehydro::numerics::is_finite(k) || k <= 1.0)
            throw std::invalid_argument("The expansion factor must be finite and greater than one.");

        double a = lower_bound_, b = a + s;
        if (!corehydro::numerics::is_finite(a) || !corehydro::numerics::is_finite(b))
            throw std::runtime_error("The initial bracketing coordinates must be finite.");

        double fa = objective_(a);
        double fb = objective_(b);
        if (std::isnan(fa) || std::isnan(fb))
            throw std::runtime_error("The objective function returned NaN while initializing the bracket.");

        if (fb > fa) {
            double temp = a;
            a = b;
            b = temp;
            fb = fa;
            s *= -1;
        }

        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            double c = b + s;
            if (!corehydro::numerics::is_finite(c))
                throw std::runtime_error(
                    "The bracketing search exceeded the finite range of double-precision coordinates.");

            double fc = objective_(c);
            if (std::isnan(fc))
                throw std::runtime_error("The objective function returned NaN while expanding the bracket.");

            if (fc >= fb) {
                lower_bound_ = std::min(a, c);
                upper_bound_ = std::max(a, c);
                return;
            }

            a = b;
            b = c;
            fb = fc;
            s *= k;
            if (!corehydro::numerics::is_finite(s))
                throw std::runtime_error(
                    "The geometric bracketing step exceeded the finite range of double precision.");
        }

        throw std::runtime_error("The maximum number of bracketing iterations was reached.");
    }

   private:
    Objective objective_;
    double lower_bound_;
    double upper_bound_;
    int function_scale_ = 1;
    bool has_best_ = false;
    double best_fitness_ = 0.0;
    double best_value_ = 0.0;

    void clear_and_optimize() {
        has_best_ = false;
        optimize();
    }

    double evaluate(double x) {
        double fitness = function_scale_ * objective_(x);
        if (!has_best_ || fitness <= best_fitness_) {
            best_value_ = x;
            best_fitness_ = fitness;
            has_best_ = true;
        }
        return fitness;
    }

    // Fortran-style sign transfer: |a| if b >= 0 else -|a| (mirrors C# Tools.Sign).
    static double sign(double a, double b) {
        return b >= 0 ? (a >= 0 ? a : -a) : (a >= 0 ? -a : a);
    }

    void optimize() {
        // Golden ratio and a small number which protects against trying to achieve
        // fractional accuracy for a minimum that happens to be exactly zero.
        const double kCGold = 0.381966;
        const double kZeps = corehydro::numerics::kDoubleMachineEpsilon * 1.0e-3;

        double ax = lower_bound_, bx = 0.5 * (upper_bound_ + lower_bound_), cx = upper_bound_;
        double a = (ax < cx ? ax : cx);
        double b = (ax > cx ? ax : cx);
        double x, w, v, fw, fv, fx;
        x = w = v = bx;
        fw = fv = fx = evaluate(x);
        double e = 0.0, d = 0.0;

        for (int i = 1; i <= max_iterations; ++i) {
            double xm = 0.5 * (a + b);
            double tol1 = relative_tolerance * std::fabs(x) + kZeps;
            double tol2 = 2.0 * tol1;
            // Test for done here.
            if (std::fabs(x - xm) <= tol2 - 0.5 * (b - a)) return;

            // Construct a trial parabolic fit.
            if (std::fabs(e) > tol1) {
                double r = (x - w) * (fx - fv);
                double q = (x - v) * (fx - fw);
                double p = (x - v) * q - (x - w) * r;
                q = 2.0 * (q - r);
                if (q > 0.0) p = -p;
                q = std::fabs(q);
                double etemp = e;
                e = d;
                if (std::fabs(p) >= std::fabs(0.5 * q * etemp) || p <= q * (a - x) ||
                    p >= q * (b - x)) {
                    // The above conditions determine the acceptability of the parabolic
                    // fit; here we take the golden section step into the larger of the
                    // two segments.
                    e = (x >= xm ? a - x : b - x);
                    d = kCGold * e;
                } else {
                    d = p / q;
                    double u = x + d;
                    if (u - a < tol2 || b - u < tol2) d = sign(tol1, xm - x);
                }
            } else {
                e = (x >= xm ? a - x : b - x);
                d = kCGold * e;
            }

            double u = (std::fabs(d) >= tol1) ? x + d : x + sign(tol1, d);
            // This is the one function evaluation per iteration.
            double fu = evaluate(u);

            if (fu <= fx) {
                if (u >= x)
                    a = x;
                else
                    b = x;
                v = w;
                w = x;
                x = u;
                fv = fw;
                fw = fx;
                fx = fu;
            } else {
                if (u < x)
                    a = u;
                else
                    b = u;
                if (fu <= fw || w == x) {
                    v = w;
                    w = u;
                    fv = fw;
                    fw = fu;
                } else if (fu <= fv || v == x || v == w) {
                    v = u;
                    fv = fu;
                }
            }
            // Done with housekeeping. Back for another iteration.
        }
        // Max iterations reached: upstream Maximize()/Minimize() swallow this and return
        // best-so-far (see nelder_mead.hpp for the identical convention).
    }
};

}  // namespace corehydro::numerics::math::optimization
