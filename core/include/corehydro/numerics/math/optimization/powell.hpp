// ported from: Numerics/Mathematics/Optimization/Local/Powell.cs @ a2c4dbf
//
// Powell's direction-set optimization method (Numerical Recipes, Press et al.; see the
// C# file's references): minimizes without derivatives by bi-directionally line-searching
// along a maintained set of direction vectors via Brent's method, replacing directions
// with the average displacement as the search proceeds. A REAL `Optimizer` subclass
// deriving from the ported base exactly as bfgs.hpp / differential_evolution.hpp do (NOT
// the Phase 4 estimation/support/optimizer_adapters.hpp `detail::` stopgap).
//
// Transcription notes:
//
// 1. Ctor validation: the C# throws ArgumentOutOfRangeException for the three checks
//    (length; upper < lower; initial within bounds). Per the repo-wide stand-in
//    documented in optimizer.hpp's file header (and used by BFGS), every C#
//    ArgumentException-family throw ports as `ArgumentException` (kind Other) so the
//    base's minimize()/maximize() catch filter behaves exactly like the C#'s.
//
// 2. LineMinimization constructs a standalone `BrentSearch` over the 1-D slice function
//    -- `new BrentSearch(func, 0d, 1d)` with the Powell instance's tolerances, then
//    `Bracket(0.1)`, then `Minimize()` -- exactly as the C#. The ported BrentSearch is
//    the deliberately-standalone Phase 0 class (see brent_search.hpp's header), extended
//    additively in this task with `bracket()` and `best_fitness()`; its one documented
//    shape drift from the C# is that, not deriving from the Optimizer base, it performs
//    NO post-success Hessian computation. In the C#, BrentSearch.Minimize() numerically
//    differentiates the slice function at the line minimum (ComputeHessian defaults
//    true), and those probe calls route through Powell's Evaluate -- so the C# Powell
//    logs a few extra function evaluations per line search and its best-parameter
//    tracking sees the probe points. Neither effect alters the accepted line minimum
//    (the Hessian runs AFTER Brent's loop) and any probe-point "improvement" is O(h^2)
//    around an already-converged 1-D minimum -- orders of magnitude below the 1E-4
//    upstream oracle tolerances, which reproduce (see test_optimizers_local.cpp).
//
// 3. `ximat` (C# `double[,]`) ports as a plain vector-of-vectors, mirroring the C# 2-D
//    array shape rather than reaching for linalg::Matrix.
#pragma once
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::math::optimization {

class Powell : public Optimizer {
   public:
    // Construct a new Powell optimization method.
    //   objective_function:   the objective function to evaluate.
    //   number_of_parameters: the number of parameters in the objective function.
    //   initial_values:       an array of initial values to evaluate.
    //   lower_bounds:         an array of lower bounds (inclusive) of the interval
    //                         containing the optimal point.
    //   upper_bounds:         an array of upper bounds (inclusive) of the interval
    //                         containing the optimal point.
    Powell(Objective objective_function, int number_of_parameters,
           std::vector<double> initial_values, std::vector<double> lower_bounds,
           std::vector<double> upper_bounds)
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
    }

    // An array of initial values to evaluate.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

   protected:
    void optimize() override {
        // Set variables
        int i, j, D = number_of_parameters_, ibig;
        bool cancel = false;
        double t, fret, fp, fptt, delta;
        auto p = initial_values_;
        std::vector<double> pt(static_cast<std::size_t>(D));
        std::vector<double> ptt(static_cast<std::size_t>(D));
        std::vector<double> xi(static_cast<std::size_t>(D));  // Direction vector
        // Set the initial matrix for directions
        // and save the initial point
        std::vector<std::vector<double>> ximat(static_cast<std::size_t>(D),
                                               std::vector<double>(static_cast<std::size_t>(D)));
        for (i = 0; i < D; i++) {
            ximat[i][i] = 1.0;
            pt[i] = p[i];
        }
        // initial function evaluation
        fret = evaluate(p, cancel);
        while (iterations_ < max_iterations) {
            fp = fret;
            ibig = 0;
            delta = 0.0;  // Will be the biggest function decrease.
            // In each iteration, loop over all directions in the set.
            for (i = 0; i < D; i++) {
                // Copy the direction
                for (j = 0; j < D; j++) xi[j] = ximat[j][i];
                fptt = fret;
                fret = line_minimization(p, xi, cancel);
                if (cancel == true) return;
                // And record it if it is the larges decrease so far.
                if (fptt - fret > delta) {
                    delta = fptt - fret;
                    ibig = i + 1;
                }
            }
            // Check convergence
            if (check_convergence(fp, fret)) {
                update_status(OptimizationStatus::Success);
                return;
            }
            // Construct the extrapolated point and save the average direction moved.
            // Save the old starting point.
            for (j = 0; j < D; j++) {
                ptt[j] = 2.0 * p[j] - pt[j];
                xi[j] = p[j] - pt[j];
                pt[j] = p[j];
            }
            // Function evaluated at the extrapolated point
            fptt = evaluate(ptt, cancel);
            if (cancel == true) return;
            if (fptt < fp) {
                t = 2.0 * (fp - 2.0 * fret + fptt) * sqr(fp - fret - delta) -
                    delta * sqr(fp - fptt);
                if (t < 0.0) {
                    // Move to the minimum of the new direction and save the new direction
                    fret = line_minimization(p, xi, cancel);
                    if (cancel == true) return;
                    for (j = 0; j < D; j++) {
                        ximat[j][ibig - 1] = ximat[j][D - 1];
                        ximat[j][D - 1] = xi[j];
                    }
                }
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

    // Auxiliary line minimization routine.
    //   start_point: the initial point (moved to the line minimum in place).
    //   direction:   the initial direction (replaced by the actual displacement).
    //   cancel:      determines if the solver should be canceled.
    double line_minimization(std::vector<double>& start_point, std::vector<double>& direction,
                             bool& cancel) {
        // Line-minimization routine, Given an n-dimensional point p[0..n-1] and an
        // n-dimension direction xi[0..n-1], moves and resets p to where the function of
        // functor func(p) takes on a minimum along the direction xi from p, and replaces
        // xi by the actual vector displacement that p was moved. Also returns the value
        // of func at the return location p. This is actually all accomplished by calling
        // the Brent minimize routine.
        int D = number_of_parameters_;
        // C# copies the ref parameter into a local so the lambda can capture it (a C#
        // ref-capture restriction); transcribed as-is for line-for-line mapping.
        bool c = cancel;
        auto func = [this, &start_point, &direction, D, &c](double alpha) {
            std::vector<double> x(static_cast<std::size_t>(D));
            for (int i = 0; i < D; i++) x[i] = start_point[i] + alpha * direction[i];
            return evaluate(x, c);
        };
        BrentSearch brent(func, 0.0, 1.0);
        brent.relative_tolerance = relative_tolerance;
        brent.absolute_tolerance = absolute_tolerance;
        brent.bracket(0.1);
        brent.minimize();
        cancel = c;
        if (cancel) return std::numeric_limits<double>::quiet_NaN();
        double xmin = brent.best_parameter();
        for (int j = 0; j < number_of_parameters_; j++) {
            direction[j] *= xmin;
            start_point[j] += direction[j];
            // Make sure the parameter is within bounds
            start_point[j] = repair_parameter(start_point[j], lower_bounds_[j], upper_bounds_[j]);
        }
        return brent.best_fitness();
    }
};

}  // namespace corehydro::numerics::math::optimization
