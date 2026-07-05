// ported from: Numerics/Mathematics/Optimization/Local/BrentSearch.cs @ a2c4dbf
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
// `.Fitness` on the C# BrentSearch. No pre-existing line of this oracle-locked file was
// altered other than this comment.
#pragma once
#include <cmath>
#include <functional>
#include <stdexcept>

#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::math::optimization {

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

    // Bracket the objective function minimum (ported from BrentSearch.cs Bracket(),
    // added in B6 for Powell's LineMinimization). Steps from the lower bound and expands
    // downhill until the function turns back up, then replaces [lower_bound_,
    // upper_bound_] with the bracketing interval. Calls the RAW objective directly --
    // no function-scale flip, no best-tracking -- exactly as the C# calls
    // `ObjectiveFunction(...)` rather than `Evaluate(...)`.
    //   s: starting step size (default 1E-2).
    //   k: expansion factor (default 2). Declared-and-unused in the C# too: the C# body
    //      never applies `k`, so the step expands linearly. Transcribed as-is.
    void bracket(double s = 1E-2, double k = 2.0) {
        (void)k;  // unused upstream as well (see above)
        double a = lower_bound_, b = a + s;
        double fa = objective_(a);
        double fb = objective_(b);
        double c, fc, temp;
        if (fb > fa) {
            temp = a;
            a = b;
            b = temp;
            temp = fa;
            fa = fb;
            fb = temp;
            s *= -1;
        }
        while (true) {
            c = b + s;
            fc = objective_(c);
            if (fc > fb) break;
            a = b;
            b = c;
            fa = fb;
            fb = fc;
        }
        if (a < c) {
            lower_bound_ = a;
            upper_bound_ = c;
        } else {
            lower_bound_ = c;
            upper_bound_ = a;
        }
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
        const double kZeps = bestfit::numerics::kDoubleMachineEpsilon * 1.0e-3;

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

}  // namespace bestfit::numerics::math::optimization
