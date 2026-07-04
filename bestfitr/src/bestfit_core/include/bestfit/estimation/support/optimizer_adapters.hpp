// Extracted from: RMC-BestFit/src/RMC.BestFit/Estimation/MaximumLikelihood.cs @ fc28c0c (Phase 4,
// Task T7's original `detail::` adapters, moved verbatim to this shared header in Task T8 so both
// `MaximumLikelihood` and `MaximumAPosteriori` can build their optimizer off the same two classes).
//
// ADAPTER NOTE (real API vs. brief's assumption): the brief describes holding a single
// `std::unique_ptr<Optimizer>` and dispatching `optimize->maximize()/status()/
// best_parameter_set()/function_evaluations()` polymorphically for every method. That is
// exactly how C#'s `Optimizer` base class works (BrentSearch/NelderMead/DifferentialEvolution/
// etc. all derive from it there). In THIS C++ port, only `DifferentialEvolution` actually
// derives from `Optimizer` -- `BrentSearch` and `NelderMead` are deliberately-standalone
// classes with their own `maximize()/minimize()` and `best_parameter()/best_parameters()`
// (see nelder_mead.hpp's header: folding the Optimizer-base machinery in was a documented
// Phase 0 shortcut, left untouched because every existing caller -- normal.hpp, gumbel.hpp,
// competing_risks.hpp, etc. -- calls them directly and never needed polymorphism). Rather than
// refactor those two shared, oracle-locked files (out of scope and risky per their own header),
// this file provides two small internal adapters (`detail::BrentOptimizerAdapter`,
// `detail::NelderMeadOptimizerAdapter`) that derive from `Optimizer` and delegate to the wrapped
// standalone optimizer, so callers can hold one uniform `std::unique_ptr<Optimizer>` exactly as
// the brief (and the C# design) intends. Both adapters assume "ran to completion == Success",
// mirroring every existing caller of BrentSearch/NelderMead in this codebase: neither wrapped
// optimizer reports a distinct "hit max iterations without converging" signal to distinguish
// from convergence.
//
// STATUS FIDELITY (adapter limitation; investigated during review, not guessed): unlike
// DifferentialEvolution -- a real `Optimizer` subclass whose own `optimize()` calls
// `update_status(OptimizationStatus::Success)` on its convergence test but
// `update_status(OptimizationStatus::MaximumIterationsReached)` when its loop instead runs out
// of iterations (differential_evolution.hpp, ~208 vs ~216) -- the two adapters above always
// report `Success`. Confirmed directly against nelder_mead.hpp/brent_search.hpp (read only, not
// modified -- both are oracle-locked): each wrapped solver's private `optimize()` either
// `return`s early on its own internal convergence test or falls through its loop when
// `max_iterations` is exhausted, but NEITHER class exposes that distinction through any public
// member -- no iteration counter, no converged flag, nothing an external caller can read after
// `maximize()`/`minimize()` returns. A faithful fix (reporting `MaximumIterationsReached` when
// the wrapped solver didn't converge) would require adding such a getter to those two files,
// which is out of scope here and would touch oracle-locked code. So this is a documented,
// standing limitation rather than a bug: `status()`/`is_estimated()` can never distinguish
// "converged" from "silently hit max_iterations" on the Brent/NelderMead paths, though the
// returned VALUES are the best point seen either way and are correct/oracle-verified regardless
// of which occurred. The DifferentialEvolution path already distinguishes the two faithfully.
// TODO: log this as a tracked entry in `docs/upstream-csharp-issues.md` (T13) rather than
// re-solving it here.
//
// total_function_evaluations() FIDELITY: for the Brent/NelderMead adapter paths this counts
// objective calls made by this port's own wrapping code -- the wrapped solver's internal
// evaluations plus one extra re-evaluation at the reported best point to recover the
// fitness/sign convention (see the adapter bodies) -- not a line-for-line replay of the C#
// `FunctionEvaluations` counter, so it will generally run one-or-more calls higher than a
// faithful C# count. Exact-count fidelity across languages, if ever required, is owned by T12;
// this port's counts are internally self-consistent and monotonic but not asserted to match C#.
//
// SIGN CONVENTION (verified): `Optimizer::maximize()` sets `function_scale_ = -1`, and
// `Optimizer::evaluate()` stores `fitness = function_scale_ * objective_function_(x)` in
// `best_parameter_set_`. Both adapters replicate this same convention manually (see their
// bodies) since the wrapped BrentSearch/NelderMead each apply their OWN internal function-scale
// flip identically to Optimizer's, so re-evaluating the raw objective at the optimizer's
// reported best point and scaling by `function_scale_` reproduces the same fitness convention.
#pragma once
#include <utility>
#include <vector>

#include "bestfit/numerics/math/optimization/brent_search.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/optimization/support/optimizer.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"

namespace bestfit::estimation::detail {

// Adapts BrentSearch (see this file's header ADAPTER NOTE) to the `Optimizer` interface.
class BrentOptimizerAdapter final : public bestfit::numerics::math::optimization::Optimizer {
    using Base = bestfit::numerics::math::optimization::Optimizer;

   public:
    BrentOptimizerAdapter(Base::Objective objective, int number_of_parameters, double lower_bound,
                          double upper_bound)
        : Base(std::move(objective), number_of_parameters),
          lower_bound_(lower_bound),
          upper_bound_(upper_bound) {}

   protected:
    void optimize() override {
        int evaluations = 0;
        bestfit::numerics::math::optimization::BrentSearch solver(
            [this, &evaluations](double x) {
                ++evaluations;
                return objective_function_(std::vector<double>{x});
            },
            lower_bound_, upper_bound_);
        solver.max_iterations = max_iterations;
        solver.relative_tolerance = relative_tolerance;
        solver.absolute_tolerance = absolute_tolerance;

        if (function_scale_ < 0)
            solver.maximize();
        else
            solver.minimize();

        function_evaluations_ = evaluations;
        std::vector<double> best{solver.best_parameter()};
        double raw = objective_function_(best);
        ++function_evaluations_;
        best_parameter_set_ = bestfit::numerics::math::optimization::ParameterSet(
            best, static_cast<double>(function_scale_) * raw);
        update_status(bestfit::numerics::math::optimization::OptimizationStatus::Success);
    }

   private:
    double lower_bound_;
    double upper_bound_;
};

// Adapts NelderMead (see this file's header ADAPTER NOTE) to the `Optimizer` interface.
class NelderMeadOptimizerAdapter final : public bestfit::numerics::math::optimization::Optimizer {
    using Base = bestfit::numerics::math::optimization::Optimizer;

   public:
    NelderMeadOptimizerAdapter(Base::Objective objective, int number_of_parameters,
                                std::vector<double> initial_values, std::vector<double> lower_bounds,
                                std::vector<double> upper_bounds)
        : Base(std::move(objective), number_of_parameters),
          initial_values_(std::move(initial_values)),
          lower_bounds_(std::move(lower_bounds)),
          upper_bounds_(std::move(upper_bounds)) {}

   protected:
    void optimize() override {
        int evaluations = 0;
        bestfit::numerics::math::optimization::NelderMead solver(
            [this, &evaluations](const std::vector<double>& x) {
                ++evaluations;
                return objective_function_(x);
            },
            number_of_parameters_, initial_values_, lower_bounds_, upper_bounds_);
        solver.max_iterations = max_iterations;
        solver.relative_tolerance = relative_tolerance;
        solver.absolute_tolerance = absolute_tolerance;

        if (function_scale_ < 0)
            solver.maximize();
        else
            solver.minimize();

        function_evaluations_ = evaluations;
        std::vector<double> best = solver.best_parameters();
        double raw = objective_function_(best);
        ++function_evaluations_;
        best_parameter_set_ = bestfit::numerics::math::optimization::ParameterSet(
            best, static_cast<double>(function_scale_) * raw);
        update_status(bestfit::numerics::math::optimization::OptimizationStatus::Success);
    }

   private:
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
};

}  // namespace bestfit::estimation::detail
