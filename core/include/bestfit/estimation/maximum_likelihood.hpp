// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/MaximumLikelihood.cs @ fc28c0c
//
// Estimates model parameters via Maximum Likelihood Estimation (MLE): maximizes the data
// log-likelihood surface exposed by `bestfit::models::ModelBase` using one of the ported
// optimizers, then computes an adaptive Hessian for uncertainty quantification (covariance,
// standard errors, sandwich/robust standard errors, observation influence, Cook's distance,
// AIC/BIC).
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
// refactor those two shared, oracle-locked files (out of scope for this task and risky per
// their own header), this file adds two small internal adapters
// (`detail::BrentOptimizerAdapter`, `detail::NelderMeadOptimizerAdapter`) that derive from
// `Optimizer` and delegate to the wrapped standalone optimizer, so `MaximumLikelihood` can
// hold one uniform `std::unique_ptr<Optimizer>` exactly as the brief (and the C# design)
// intends. Both adapters assume "ran to completion == Success", mirroring every existing
// caller of BrentSearch/NelderMead in this codebase: neither wrapped optimizer reports a
// distinct "hit max iterations without converging" signal to distinguish from convergence.
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
// `best_parameter_set_`. So after a successful `maximize()`, `best_parameter_set().fitness ==
// -data_log_likelihood(best values)`, and this class's `maximum_log_likelihood()` (`=
// is_estimated ? -best_parameter_set.fitness : NaN`) recovers the actual (unscaled) data
// log-likelihood at the optimum -- exactly mirroring the C# invariant
// (`MaximumLogLikelihood => IsEstimated ? -BestParameterSet.Fitness : double.NaN`) with no
// sign adjustment needed. The two adapters replicate this same convention manually (see their
// bodies) since the wrapped BrentSearch/NelderMead each apply their OWN internal function-scale
// flip identically to Optimizer's, so re-evaluating the raw objective at the optimizer's
// reported best point and scaling by `function_scale_` reproduces the same fitness convention.
// The C++-only ctest (test_maximum_likelihood.cpp) asserts this equality directly to 1e-9.
//
// GATING (deliberate Phase-4 sever, matches the task brief): `OptimizationMethod::BFGS`,
// `::Powell`, and `::MultilevelSingleLinkage` are NOT constructed -- none of the three has been
// ported (BFGS/Powell/MLSL are deferred alongside GeneralizedMethodOfMoments, per
// `.claude/PLAN.md`). Selecting one of them throws `std::invalid_argument` from
// `set_up_optimizer()` without ever touching an (nonexistent) optimizer type.
//
// EXCEPTION-TYPE MAPPING for THIS file: C# `ArgumentException`/`InvalidOperationException`
// (state guards: not-yet-estimated, wrong parameter count, null Hessian) -> `std::invalid_argument`;
// C# `ArgumentOutOfRangeException` (numeric range guards: bins < 2, alpha outside (0,1),
// sample size < 1) -> `std::out_of_range`. This mirrors the existing convention elsewhere in
// the port (e.g. model_base.hpp's `set_parameter_values` uses `std::invalid_argument` for its
// C# `ArgumentException`).
//
// Debug.WriteLine NOTE: C# logs a message via `Debug.WriteLine` in three places -- Estimate()'s
// caught Hessian-computation failure, GetCovarianceMatrix()'s "matrix was regularized" notice,
// and the Fisher-inversion failure paths in GetObservationInfluence()/GetCooksDistance(). None
// of those is a compute-path side effect (they are desktop-app trace/diagnostic logging with no
// C++ logger wired into this core), so each becomes a plain comment at the corresponding
// catch/branch site rather than a call to anything -- the catch bodies themselves (falling back
// to a null Hessian / a zero matrix / a zero vector) ARE ported, only the logging text is not.
//
// Matrix-shaped return types: C# `ProfileLikelihood` returns `List<double[,]>` (one Nx2 array
// per parameter) and `ParameterConfidenceIntervals`/`GetObservationInfluence` return `double[,]`.
// This port represents all three as `bestfit::numerics::math::linalg::Matrix` (already used for
// the covariance/correlation matrices below), rather than inventing a separate 2D-array type,
// for one consistent matrix representation across the whole class.
#pragma once
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bestfit/estimation/numerical_diff.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/distributions/chi_squared.hpp"
#include "bestfit/numerics/math/linalg/lu_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/matrix_regularization.hpp"
#include "bestfit/numerics/math/optimization/brent_search.hpp"
#include "bestfit/numerics/math/optimization/differential_evolution.hpp"
#include "bestfit/numerics/math/optimization/nelder_mead.hpp"
#include "bestfit/numerics/math/optimization/support/optimizer.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "bestfit/numerics/math/rootfinding/brent.hpp"
#include "bestfit/numerics/sampling/stratification_bin.hpp"
#include "bestfit/numerics/sampling/stratification_options.hpp"
#include "bestfit/numerics/sampling/stratify.hpp"

namespace bestfit::estimation {

namespace detail {

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

}  // namespace detail

class MaximumLikelihood {
   public:
    using Optimizer = bestfit::numerics::math::optimization::Optimizer;
    using OptimizationStatus = bestfit::numerics::math::optimization::OptimizationStatus;
    using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;
    using Matrix = bestfit::numerics::math::linalg::Matrix;

    // Constructs a new MLE estimator for `model` (held by reference; NOT owned -- mirrors the
    // C# ctor's `Model` property, which merely stores the passed-in reference). C# also
    // null-checks `model`; a C++ reference cannot be null, so that guard has no analogue here.
    explicit MaximumLikelihood(bestfit::models::ModelBase& model,
                                OptimizationMethod method = OptimizationMethod::DifferentialEvolution)
        : model_(model), method_(method) {
        set_up_optimizer();
        is_estimated_ = false;
    }

    // --- Properties ----------------------------------------------------------------------

    OptimizationMethod optimizer_method() const { return method_; }

    // Changing the method clears existing results and rebuilds the optimizer (C# setter, 59-71).
    void set_optimizer_method(OptimizationMethod method) {
        if (method_ != method) {
            method_ = method;
            clear_results();
            set_up_optimizer();
        }
    }

    Optimizer& optimizer() { return *optimizer_; }
    const Optimizer& optimizer() const { return *optimizer_; }

    bool report_failure() const { return report_failure_; }
    void set_report_failure(bool value) {
        report_failure_ = value;
        if (optimizer_) optimizer_->report_failure = value;
    }

    bool compute_hessian() const { return compute_hessian_; }
    void set_compute_hessian(bool value) {
        compute_hessian_ = value;
        if (!value) hessian_.reset();
    }

    OptimizationStatus status() const { return status_; }

    int number_of_parameters() const { return model_.number_of_parameters(); }

    const std::vector<double>& initial_values() const { return initial_values_; }
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    bool is_estimated() const { return is_estimated_; }

    const ParameterSet& best_parameter_set() const { return best_parameter_set_; }

    int total_function_evaluations() const { return total_function_evaluations_; }

    // C# `MaximumLogLikelihood => IsEstimated ? -BestParameterSet.Fitness : double.NaN` --
    // see this file's header SIGN CONVENTION note.
    double maximum_log_likelihood() const {
        return is_estimated_ ? -best_parameter_set_.fitness : std::numeric_limits<double>::quiet_NaN();
    }

    // --- Methods -------------------------------------------------------------------------

    // Estimates the model parameters that maximize the likelihood function (C# 224-268).
    bool estimate() {
        is_estimated_ = false;
        status_ = OptimizationStatus::None;
        total_function_evaluations_ = 0;
        // Reset cached Hessian from any prior successful run so a subsequent failed estimate()
        // doesn't leave get_covariance_matrix() returning stale covariance.
        hessian_.reset();

        try {
            optimizer_->maximize();
            status_ = optimizer_->status();
            if (status_ == OptimizationStatus::Success) {
                is_estimated_ = true;
                best_parameter_set_ = optimizer_->best_parameter_set().clone();
                total_function_evaluations_ = optimizer_->function_evaluations();

                if (compute_hessian_) {
                    // Compute Hessian with adaptive step sizes (flat-spot detection).
                    try {
                        hessian_ = NumericalDiff::compute_hessian(
                            [this](const std::vector<double>& p) { return model_.data_log_likelihood(p); },
                            best_parameter_set_.values, number_of_parameters());
                    } catch (const std::exception&) {
                        // C# logs "Hessian computation failed: ..." via Debug.WriteLine; no C++
                        // logger is wired into this core, so this catch is a silent no-op.
                        hessian_.reset();
                    }
                }

                return true;
            }
        } catch (const std::exception&) {
            status_ = OptimizationStatus::Failure;
            // C# logs "MLE estimation failed: ..." via Debug.WriteLine; silent no-op here.
        }

        return false;
    }

    // Clears all estimation results (C# 273-280).
    void clear_results() {
        is_estimated_ = false;
        status_ = OptimizationStatus::None;
        total_function_evaluations_ = 0;
        best_parameter_set_ = ParameterSet();
        hessian_.reset();
    }

    // Returns the profile likelihood for each model parameter: one Matrix(bins, 2) per
    // parameter, columns [parameter value, log-likelihood] (C# 289-316).
    std::vector<Matrix> profile_likelihood(int bins = 100) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (bins < 2) throw std::out_of_range("Number of bins must be at least 2.");

        std::vector<Matrix> result;
        result.reserve(static_cast<std::size_t>(number_of_parameters()));

        for (int i = 0; i < number_of_parameters(); ++i) {
            const auto& parameter = model_.parameters()[static_cast<std::size_t>(i)];
            auto seq = bestfit::numerics::sampling::Stratify::XValues(
                bestfit::numerics::sampling::StratificationOptions(parameter.lower_bound(),
                                                                     parameter.upper_bound(), bins));
            std::vector<double> parms = best_parameter_set_.values;
            Matrix profile(bins, 2);

            for (int j = 0; j < bins; ++j) {
                double midpoint = seq[static_cast<std::size_t>(j)].midpoint();
                parms[static_cast<std::size_t>(i)] = midpoint;
                profile(j, 0) = midpoint;
                profile(j, 1) = model_.data_log_likelihood(parms);
            }

            result.push_back(std::move(profile));
        }

        return result;
    }

    // Returns parameter confidence intervals from profile likelihood using the chi-squared
    // threshold: an Nx2 Matrix, columns [lower bound, upper bound] (C# 325-375).
    Matrix parameter_confidence_intervals(double alpha = 0.1) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (alpha <= 0 || alpha >= 1) throw std::out_of_range("Alpha must be between 0 and 1.");

        bestfit::numerics::distributions::ChiSquared chi_squared(1);
        double threshold = -best_parameter_set_.fitness - 0.5 * chi_squared.inverse_cdf(1 - alpha);
        Matrix cis(number_of_parameters(), 2);

        for (int i = 0; i < number_of_parameters(); ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            const auto& parameter = model_.parameters()[si];
            std::vector<double> parms = best_parameter_set_.values;

            // Lower limit: check if the lower bound already exceeds the threshold.
            parms[si] = parameter.lower_bound();
            double lower_llh = model_.data_log_likelihood(parms);
            if (lower_llh < threshold) {
                cis(i, 0) = bestfit::numerics::math::rootfinding::solve(
                    [&](double x) {
                        parms[si] = x;
                        return model_.data_log_likelihood(parms) - threshold;
                    },
                    parameter.lower_bound(), best_parameter_set_.values[si]);
            } else {
                cis(i, 0) = parameter.lower_bound();
            }

            // Upper limit: check if the upper bound already exceeds the threshold.
            parms[si] = parameter.upper_bound();
            double upper_llh = model_.data_log_likelihood(parms);
            if (upper_llh < threshold) {
                cis(i, 1) = bestfit::numerics::math::rootfinding::solve(
                    [&](double x) {
                        parms[si] = x;
                        return model_.data_log_likelihood(parms) - threshold;
                    },
                    best_parameter_set_.values[si], parameter.upper_bound());
            } else {
                cis(i, 1) = parameter.upper_bound();
            }
        }

        return cis;
    }

    // Returns the parameter covariance matrix from the inverse of the Fisher Information
    // Matrix (negative Hessian), regularized to be symmetric positive-definite (C# 382-415).
    Matrix get_covariance_matrix() const {
        if (number_of_parameters() < 2)
            throw std::invalid_argument("Cannot compute the covariance matrix with fewer than two parameters.");
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (!hessian_.has_value()) throw std::invalid_argument("The Hessian is null. Estimation may have failed.");

        try {
            Matrix fisher = (*hessian_) * -1.0;
            Matrix covariance = fisher.inverse();
            // C# logs a "matrix regularized" notice via Debug.WriteLine when regularization
            // changes the matrix; no C++ logger is wired in, so that trace is dropped (the
            // regularization itself is still performed).
            return bestfit::numerics::math::linalg::MatrixRegularization::make_symmetric_positive_definite(
                covariance);
        } catch (const std::exception&) {
            return Matrix(number_of_parameters(), number_of_parameters());
        }
    }

    // Returns standard errors from the diagonal of the covariance matrix (C# 422-433).
    std::vector<double> get_standard_errors() const {
        Matrix covariance = get_covariance_matrix();
        std::vector<double> standard_errors(static_cast<std::size_t>(number_of_parameters()));
        for (int i = 0; i < number_of_parameters(); ++i)
            standard_errors[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, covariance(i, i)));
        return standard_errors;
    }

    // Returns the correlation matrix from the covariance matrix (C# 440-455).
    Matrix get_correlation_matrix() const {
        Matrix covariance = get_covariance_matrix();
        Matrix correlation(number_of_parameters(), number_of_parameters());
        for (int i = 0; i < number_of_parameters(); ++i) {
            for (int j = 0; j < number_of_parameters(); ++j) {
                double denom = std::sqrt(covariance(i, i) * covariance(j, j));
                correlation(i, j) = denom > 0 ? covariance(i, j) / denom : 0.0;
            }
        }
        return correlation;
    }

    // Returns the sandwich (robust) covariance matrix Var = H^-1 J H^-1 (C# 480-510).
    Matrix get_sandwich_covariance_matrix() const {
        if (number_of_parameters() < 2)
            throw std::invalid_argument("Cannot compute the covariance matrix with fewer than two parameters.");
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (!hessian_.has_value()) throw std::invalid_argument("The Hessian is null. Estimation may have failed.");

        try {
            Matrix fisher = (*hessian_) * -1.0;
            Matrix fisher_inverse = fisher.inverse();
            Matrix meat = compute_meat_matrix(best_parameter_set_.values);
            Matrix sandwich = fisher_inverse * meat * fisher_inverse;
            return bestfit::numerics::math::linalg::MatrixRegularization::make_symmetric_positive_definite(
                sandwich);
        } catch (const std::exception&) {
            return Matrix(number_of_parameters(), number_of_parameters());
        }
    }

    // Returns robust (sandwich) standard errors (C# 520-531).
    std::vector<double> get_robust_standard_errors() const {
        Matrix sandwich = get_sandwich_covariance_matrix();
        std::vector<double> robust_se(static_cast<std::size_t>(number_of_parameters()));
        for (int i = 0; i < number_of_parameters(); ++i)
            robust_se[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, sandwich(i, i)));
        return robust_se;
    }

    // Returns pointwise observation influence (DFBETAS-like): Nx(NumberOfParameters) Matrix,
    // influence[i,j] = (H^-1 g_i)_j / SE_j, using the data-only Hessian (C# 608-655).
    Matrix get_observation_influence() const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");

        std::vector<double> pointwise_ll = model_.pointwise_data_log_likelihood(best_parameter_set_.values);
        int n = static_cast<int>(pointwise_ll.size());
        auto gradients = compute_pointwise_gradients(best_parameter_set_.values, n);

        if (!hessian_.has_value()) return Matrix(n, number_of_parameters());
        Matrix fisher = (*hessian_) * -1.0;
        std::optional<Matrix> fisher_inverse_opt;
        try {
            fisher_inverse_opt = fisher.inverse();
        } catch (const std::exception&) {
            // C# logs "Fisher matrix inversion failed" via Debug.WriteLine; silent no-op here.
            return Matrix(n, number_of_parameters());
        }
        const Matrix& fisher_inverse = *fisher_inverse_opt;

        Matrix influence(n, number_of_parameters());
        std::vector<double> se = get_standard_errors();

        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < number_of_parameters(); ++j) {
                double infl_j = 0.0;
                for (int k = 0; k < number_of_parameters(); ++k)
                    infl_j += fisher_inverse(j, k) *
                              gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
                influence(i, j) = se[static_cast<std::size_t>(j)] > 0 ? infl_j / se[static_cast<std::size_t>(j)] : 0.0;
            }
        }

        return influence;
    }

    // Returns Cook's distance-like measure D_i = g_i^T H^-1 g_i / p per observation, using the
    // data-only Hessian (C# 657-729).
    std::vector<double> get_cooks_distance() const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");

        std::vector<double> pointwise_ll = model_.pointwise_data_log_likelihood(best_parameter_set_.values);
        int n = static_cast<int>(pointwise_ll.size());
        auto gradients = compute_pointwise_gradients(best_parameter_set_.values, n);

        if (!hessian_.has_value()) return std::vector<double>(static_cast<std::size_t>(n), 0.0);
        Matrix fisher = (*hessian_) * -1.0;
        std::optional<Matrix> fisher_inverse_opt;
        try {
            fisher_inverse_opt = fisher.inverse();
        } catch (const std::exception&) {
            // C# logs "Fisher matrix inversion failed" via Debug.WriteLine; silent no-op here.
            return std::vector<double>(static_cast<std::size_t>(n), 0.0);
        }
        const Matrix& fisher_inverse = *fisher_inverse_opt;

        std::vector<double> cooks_d(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            double quad_form = 0.0;
            for (int j = 0; j < number_of_parameters(); ++j) {
                double tmp = 0.0;
                for (int k = 0; k < number_of_parameters(); ++k)
                    tmp += fisher_inverse(j, k) *
                           gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
                quad_form += gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * tmp;
            }
            cooks_d[static_cast<std::size_t>(i)] = quad_form / number_of_parameters();
        }

        return cooks_d;
    }

    // Computes the Akaike Information Criterion (C# 736-742).
    double get_aic() const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        return bestfit::numerics::data::GoodnessOfFit::aic(number_of_parameters(), maximum_log_likelihood());
    }

    // Computes the Bayesian Information Criterion (C# 744-759).
    double get_bic(int sample_size) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (sample_size < 1) throw std::out_of_range("Sample size must be at least 1.");
        return bestfit::numerics::data::GoodnessOfFit::bic(sample_size, number_of_parameters(),
                                                              maximum_log_likelihood());
    }

   private:
    bestfit::models::ModelBase& model_;
    OptimizationMethod method_;
    std::unique_ptr<Optimizer> optimizer_;
    bool report_failure_ = false;
    bool compute_hessian_ = true;
    std::optional<Matrix> hessian_;
    OptimizationStatus status_ = OptimizationStatus::None;
    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;
    bool is_estimated_ = false;
    ParameterSet best_parameter_set_;
    int total_function_evaluations_ = 0;

    // Builds the optimizer for the selected method (C# 176-218). See this file's header
    // ADAPTER NOTE and GATING comment.
    void set_up_optimizer() {
        initial_values_.clear();
        lower_bounds_.clear();
        upper_bounds_.clear();
        for (const auto& parameter : model_.parameters()) {
            initial_values_.push_back(parameter.value());
            lower_bounds_.push_back(parameter.lower_bound());
            upper_bounds_.push_back(parameter.upper_bound());
        }

        Optimizer::Objective objective = [this](const std::vector<double>& p) {
            return model_.data_log_likelihood(p);
        };

        switch (method_) {
            case OptimizationMethod::Brent:
                if (number_of_parameters() != 1)
                    throw std::invalid_argument("Brent method requires exactly one parameter.");
                optimizer_ = std::make_unique<detail::BrentOptimizerAdapter>(
                    objective, number_of_parameters(), lower_bounds_[0], upper_bounds_[0]);
                break;
            case OptimizationMethod::NelderMead:
                optimizer_ = std::make_unique<detail::NelderMeadOptimizerAdapter>(
                    objective, number_of_parameters(), initial_values_, lower_bounds_, upper_bounds_);
                break;
            case OptimizationMethod::DifferentialEvolution:
                optimizer_ = std::make_unique<bestfit::numerics::math::optimization::DifferentialEvolution>(
                    objective, number_of_parameters(), lower_bounds_, upper_bounds_);
                break;
            case OptimizationMethod::BFGS:
                throw std::invalid_argument(
                    "Optimization method 'BFGS' is not supported in this build; BFGS is deferred "
                    "alongside GeneralizedMethodOfMoments (Phase 4 scope decision).");
            case OptimizationMethod::Powell:
                throw std::invalid_argument(
                    "Optimization method 'Powell' is not supported in this build; Powell is "
                    "deferred alongside GeneralizedMethodOfMoments (Phase 4 scope decision).");
            case OptimizationMethod::MultilevelSingleLinkage:
                throw std::invalid_argument(
                    "Optimization method 'MultilevelSingleLinkage' is not supported in this "
                    "build; MultilevelSingleLinkage is deferred alongside "
                    "GeneralizedMethodOfMoments (Phase 4 scope decision).");
        }

        optimizer_->report_failure = report_failure_;
        optimizer_->record_traces = false;
        optimizer_->compute_hessian = false;  // This class computes its own adaptive Hessian.
    }

    // Computes numerical gradients of pointwise data log-likelihoods via central differences
    // (C# 565-584, delegates to NumericalDiff).
    std::vector<std::vector<double>> compute_pointwise_gradients(const std::vector<double>& parameters,
                                                                    int n) const {
        return NumericalDiff::compute_pointwise_gradients(
            [this](const std::vector<double>& p) { return model_.pointwise_data_log_likelihood(p); },
            parameters, n, number_of_parameters());
    }

    // Computes the "meat" matrix J = sum_i g_i g_i^T for the sandwich estimator (C# 533-563).
    Matrix compute_meat_matrix(const std::vector<double>& parameters) const {
        std::vector<double> pointwise_ll = model_.pointwise_data_log_likelihood(parameters);
        int n = static_cast<int>(pointwise_ll.size());
        auto gradients = compute_pointwise_gradients(parameters, n);

        Matrix meat(number_of_parameters(), number_of_parameters());
        for (int i = 0; i < n; ++i) {
            const std::vector<double>& g = gradients[static_cast<std::size_t>(i)];
            for (int j = 0; j < number_of_parameters(); ++j)
                for (int k = 0; k < number_of_parameters(); ++k)
                    meat(j, k) += g[static_cast<std::size_t>(j)] * g[static_cast<std::size_t>(k)];
        }
        return meat;
    }
};

}  // namespace bestfit::estimation
