// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/MaximumAPosteriori.cs @ fc28c0c
//
// Estimates model parameters via Maximum A Posteriori (MAP) estimation: maximizes the full
// log-likelihood surface (data log-likelihood + prior log-likelihood) exposed by
// `corehydro::models::ModelBase::log_likelihood` using one of the ported optimizers, then computes
// an adaptive Hessian OF THE POSTERIOR for uncertainty quantification (covariance, standard
// errors, observation influence, Cook's distance, AIC/BIC). MAP is structurally near-identical
// to the already-ported `MaximumLikelihood` (Phase 4, Task T7) -- see that file's header for the
// shared adapter/status-fidelity/sign-convention background, restated here only where MAP
// differs.
//
// DIFFERENCES FROM MaximumLikelihood (Task T8 brief; verified against the C# source above):
//   1. The optimizer objective is `Model.LogLikelihood` (data + prior), NOT `DataLogLikelihood`.
//   2. The adaptive Hessian in estimate() is computed from `model_.log_likelihood` (the
//      POSTERIOR Hessian), not the data-only likelihood.
//   3. `maximum_log_likelihood()` keeps the C# member name; it is now the log-POSTERIOR at the
//      optimum (`= is_estimated ? -best_parameter_set.fitness : NaN`, same formula as MLE).
//   4. NO sandwich/robust methods: MAP has no `GetSandwichCovarianceMatrix` /
//      `GetRobustStandardErrors` / `ComputeMeatMatrix` in the C# source, and none are added here.
//   5. `get_observation_influence()` / `get_cooks_distance()` ARE ported and self-contained: they
//      use the POSTERIOR Hessian (from `hessian_`, computed over `log_likelihood`) but the SAME
//      pointwise DATA gradients as MLE (`model_.pointwise_data_log_likelihood` via
//      `compute_pointwise_gradients`) -- this matches the C# source exactly (MaximumAPosteriori.cs
//      566/569/631/634 call `Model.PointwiseDataLogLikelihood`, not a posterior-pointwise
//      variant), so only the Hessian differs from MLE's version of these two methods.
//   6. `set_up_optimizer()` reuses the shared `detail::BrentOptimizerAdapter` /
//      `detail::NelderMeadOptimizerAdapter` from `corehydro/estimation/support/optimizer_adapters.hpp`
//      (extracted from MLE in this same task, Part A). BFGS/Powell/MultilevelSingleLinkage are
//      un-gated exactly as in MLE (Phase 6, Task B7): constructed for real over the full
//      posterior objective, mirroring the C# `SetUpOptimizer`, with MLSL taking
//      `LocalMethod::NelderMead` explicitly per the C# call site.
//
// GATING (Diagnostics layer deferred, Phase 4 scope decision): C#'s `ComputeLeverageDiagnostics()`
// (MaximumAPosteriori.cs:683) returns a `RMC.BestFit.Diagnostics.LeverageDiagnostics`, and the
// Diagnostics layer itself is deferred past Phase 4 (see `.claude/PLAN.md`). This port provides
// `compute_leverage_diagnostics()` as a stub that throws `std::logic_error` unconditionally,
// rather than omitting the method entirely, so callers get a clear compile-time-visible member
// and a clear runtime message instead of a missing-symbol error. This is the ONLY
// Diagnostics-coupled member on this class.
//
// EXCEPTION-TYPE MAPPING for THIS file: same convention as MaximumLikelihood -- C#
// `ArgumentException`/`InvalidOperationException` (state guards: not-yet-estimated, wrong
// parameter count, null Hessian) -> `std::invalid_argument`; C# `ArgumentOutOfRangeException`
// (numeric range guards: bins < 2, alpha outside (0,1), sample size < 1) -> `std::out_of_range`.
//
// Debug.WriteLine NOTE: as in MaximumLikelihood, C# logs via `Debug.WriteLine` in several places
// (Estimate()'s caught Hessian-computation failure, GetCovarianceMatrix()'s "matrix was
// regularized" notice, and the Fisher-inversion failure paths in GetObservationInfluence()/
// GetCooksDistance()). None of those is a compute-path side effect, so each becomes a plain
// comment at the corresponding catch/branch site rather than a call to anything.
//
// Matrix-shaped return types: as in MaximumLikelihood, C# `ProfileLikelihood` returns
// `List<double[,]>` and `ParameterConfidenceIntervals`/`GetObservationInfluence` return
// `double[,]`; this port represents all three as `corehydro::numerics::math::linalg::Matrix`.
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

#include "corehydro/diagnostics/leverage_diagnostics.hpp"
#include "corehydro/estimation/numerical_diff.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "corehydro/estimation/support/optimizer_adapters.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/distributions/chi_squared.hpp"
#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/matrix_regularization.hpp"
#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/differential_evolution.hpp"
#include "corehydro/numerics/math/optimization/mlsl.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/math/optimization/support/local_method.hpp"
#include "corehydro/numerics/math/optimization/support/optimizer.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/math/rootfinding/brent.hpp"
#include "corehydro/numerics/sampling/stratification_bin.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"

namespace corehydro::estimation {

class MaximumAPosteriori {
   public:
    using Optimizer = corehydro::numerics::math::optimization::Optimizer;
    using OptimizationStatus = corehydro::numerics::math::optimization::OptimizationStatus;
    using ParameterSet = corehydro::numerics::math::optimization::ParameterSet;
    using Matrix = corehydro::numerics::math::linalg::Matrix;

    // Constructs a new MAP estimator for `model` (held by reference; NOT owned -- mirrors the
    // C# ctor's `Model` property, which merely stores the passed-in reference). C# also
    // null-checks `model`; a C++ reference cannot be null, so that guard has no analogue here.
    explicit MaximumAPosteriori(corehydro::models::ModelBase& model,
                                 OptimizationMethod method = OptimizationMethod::DifferentialEvolution)
        : model_(model), method_(method) {
        set_up_optimizer();
        is_estimated_ = false;
    }

    // --- Properties ----------------------------------------------------------------------

    OptimizationMethod optimizer_method() const { return method_; }

    // Changing the method clears existing results and rebuilds the optimizer (C# 62-74).
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

    // C# `MaximumLogLikelihood => IsEstimated ? -BestParameterSet.Fitness : double.NaN` -- for
    // MAP this is the log-POSTERIOR (data + prior) at the optimum, since the optimizer's
    // objective was `Model.LogLikelihood` (see set_up_optimizer()). Kept as `fitness` sign
    // convention matching MaximumLikelihood's identical formula.
    double maximum_log_likelihood() const {
        return is_estimated_ ? -best_parameter_set_.fitness : std::numeric_limits<double>::quiet_NaN();
    }

    // --- Methods -------------------------------------------------------------------------

    // Estimates the model parameters that maximize the posterior (C# 226-268).
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
                    // Compute Hessian with adaptive step sizes (flat-spot detection), over the
                    // full posterior log-likelihood (data + prior) -- the POSTERIOR Fisher
                    // information, unlike MLE's data-only Hessian.
                    try {
                        hessian_ = NumericalDiff::compute_hessian(
                            [this](std::vector<double>& p) { return model_.log_likelihood(p); },
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
            // C# logs "MAP estimation failed: ..." via Debug.WriteLine; silent no-op here.
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

    // Returns the profile likelihood (of the POSTERIOR) for each model parameter: one
    // Matrix(bins, 2) per parameter, columns [parameter value, log-posterior] (C# 289-316).
    std::vector<Matrix> profile_likelihood(int bins = 100) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (bins < 2) throw std::out_of_range("Number of bins must be at least 2.");

        std::vector<Matrix> result;
        result.reserve(static_cast<std::size_t>(number_of_parameters()));

        for (int i = 0; i < number_of_parameters(); ++i) {
            const auto& parameter = model_.parameters()[static_cast<std::size_t>(i)];
            auto seq = corehydro::numerics::sampling::Stratify::XValues(
                corehydro::numerics::sampling::StratificationOptions(parameter.lower_bound(),
                                                                     parameter.upper_bound(), bins));
            std::vector<double> parms = best_parameter_set_.values;
            Matrix profile(bins, 2);

            for (int j = 0; j < bins; ++j) {
                double midpoint = seq[static_cast<std::size_t>(j)].midpoint();
                parms[static_cast<std::size_t>(i)] = midpoint;
                profile(j, 0) = midpoint;
                profile(j, 1) = model_.log_likelihood(parms);
            }

            result.push_back(std::move(profile));
        }

        return result;
    }

    // Returns parameter confidence intervals from profile likelihood (of the POSTERIOR) using
    // the chi-squared threshold: an Nx2 Matrix, columns [lower bound, upper bound] (C# 325-375).
    Matrix parameter_confidence_intervals(double alpha = 0.1) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (alpha <= 0 || alpha >= 1) throw std::out_of_range("Alpha must be between 0 and 1.");

        corehydro::numerics::distributions::ChiSquared chi_squared(1);
        double threshold = -best_parameter_set_.fitness - 0.5 * chi_squared.inverse_cdf(1 - alpha);
        Matrix cis(number_of_parameters(), 2);

        for (int i = 0; i < number_of_parameters(); ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            const auto& parameter = model_.parameters()[si];
            std::vector<double> parms = best_parameter_set_.values;

            // Lower limit: check if the lower bound already exceeds the threshold.
            parms[si] = parameter.lower_bound();
            double lower_llh = model_.log_likelihood(parms);
            if (lower_llh < threshold) {
                cis(i, 0) = corehydro::numerics::math::rootfinding::solve(
                    [&](double x) {
                        parms[si] = x;
                        return model_.log_likelihood(parms) - threshold;
                    },
                    parameter.lower_bound(), best_parameter_set_.values[si]);
            } else {
                cis(i, 0) = parameter.lower_bound();
            }

            // Upper limit: check if the upper bound already exceeds the threshold.
            parms[si] = parameter.upper_bound();
            double upper_llh = model_.log_likelihood(parms);
            if (upper_llh < threshold) {
                cis(i, 1) = corehydro::numerics::math::rootfinding::solve(
                    [&](double x) {
                        parms[si] = x;
                        return model_.log_likelihood(parms) - threshold;
                    },
                    best_parameter_set_.values[si], parameter.upper_bound());
            } else {
                cis(i, 1) = parameter.upper_bound();
            }
        }

        return cis;
    }

    // Returns the parameter covariance matrix from the inverse of the Fisher Information
    // Matrix (negative POSTERIOR Hessian), regularized to be symmetric positive-definite
    // (C# 382-415).
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
            return corehydro::numerics::math::linalg::MatrixRegularization::make_symmetric_positive_definite(
                covariance);
        } catch (const std::exception&) {
            return Matrix(number_of_parameters(), number_of_parameters());
        }
    }

    // Returns standard errors from the diagonal of the covariance matrix (C# 413-424).
    std::vector<double> get_standard_errors() const {
        Matrix covariance = get_covariance_matrix();
        std::vector<double> standard_errors(static_cast<std::size_t>(number_of_parameters()));
        for (int i = 0; i < number_of_parameters(); ++i)
            standard_errors[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, covariance(i, i)));
        return standard_errors;
    }

    // Returns the correlation matrix from the covariance matrix (C# 431-446).
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

    // NOTE: unlike MaximumLikelihood, MAP has NO sandwich/robust standard error methods --
    // the C# source defines no `GetSandwichCovarianceMatrix`/`GetRobustStandardErrors`/
    // `ComputeMeatMatrix` for MaximumAPosteriori. Do not add them here (brief difference #4).

    // Returns pointwise observation influence (DFBETAS-like): Nx(NumberOfParameters) Matrix,
    // influence[i,j] = (H^-1 g_i)_j / SE_j, using the POSTERIOR Hessian but the same pointwise
    // DATA gradients as MLE (C# 560-604; C# itself calls PointwiseDataLogLikelihood here, not a
    // posterior-pointwise variant -- see this file's header difference #5).
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
            // C# logs "Posterior Hessian inversion failed in GetObservationInfluence" via
            // Debug.WriteLine; silent no-op here.
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
    // POSTERIOR Hessian but the same pointwise DATA gradients as MLE (C# 625-670; see this
    // file's header difference #5).
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
            // C# logs "Posterior Hessian inversion failed in GetCooksDistance" via
            // Debug.WriteLine; silent no-op here.
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

    // Computes the Akaike Information Criterion at the MAP estimate, using the full posterior
    // log-likelihood (data + prior) (C# 478-484). With uniform/improper-flat priors the prior
    // contribution is constant in theta and this reduces to the conventional MLE-based AIC;
    // with informative priors, direct AIC comparison across analyses with different priors can
    // be misleading (see the C# doc comment for the DIC/WAIC/LOO-CV recommendation).
    double get_aic() const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        return corehydro::numerics::data::GoodnessOfFit::aic(number_of_parameters(), maximum_log_likelihood());
    }

    // Computes the Bayesian Information Criterion at the MAP estimate, using the full posterior
    // log-likelihood (data + prior) (C# 514-522). Same informative-prior caveat as get_aic().
    double get_bic(int sample_size) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (sample_size < 1) throw std::out_of_range("Sample size must be at least 1.");
        return corehydro::numerics::data::GoodnessOfFit::bic(sample_size, number_of_parameters(),
                                                              maximum_log_likelihood());
    }

    // Computes leverage diagnostics from the posterior Hessian (C# 672-691). Delegates to the
    // LeverageDiagnostics(IModel, double[]) constructor, which computes the posterior Hessian
    // numerically at the MAP values (BestParameterSet.Values) -- consistent with the
    // BayesianAnalysis path (D3 un-stub; the Diagnostics layer is now ported).
    corehydro::diagnostics::LeverageDiagnostics compute_leverage_diagnostics() const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        return corehydro::diagnostics::LeverageDiagnostics(model_, best_parameter_set_.values);
    }

   private:
    corehydro::models::ModelBase& model_;
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

    // Builds the optimizer for the selected method (C# 178-220). Objective is the full
    // log-likelihood (data + prior) -- see this file's header difference #1. See
    // maximum_likelihood.hpp's header ADAPTER NOTE and this file's header GATING section.
    void set_up_optimizer() {
        initial_values_.clear();
        lower_bounds_.clear();
        upper_bounds_.clear();
        for (const auto& parameter : model_.parameters()) {
            initial_values_.push_back(parameter.value());
            lower_bounds_.push_back(parameter.lower_bound());
            upper_bounds_.push_back(parameter.upper_bound());
        }

        // Mutable pass-through (M14): mirrors C# handing `Model.LogLikelihood` straight to
        // the optimizer -- a mutating model (MixtureModel) writes back into the optimizer's own
        // working vectors.
        Optimizer::Objective objective = [this](std::vector<double>& p) {
            return model_.log_likelihood(p);
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
                optimizer_ = std::make_unique<corehydro::numerics::math::optimization::DifferentialEvolution>(
                    objective, number_of_parameters(), lower_bounds_, upper_bounds_);
                break;
            case OptimizationMethod::BFGS:
                // The trailing GradientFunction defaults to nullptr = finite differences,
                // matching the C# call that omits the optional gradient (C# 192-195).
                optimizer_ = std::make_unique<corehydro::numerics::math::optimization::BFGS>(
                    objective, number_of_parameters(), initial_values_, lower_bounds_,
                    upper_bounds_);
                break;
            case OptimizationMethod::Powell:
                optimizer_ = std::make_unique<corehydro::numerics::math::optimization::Powell>(
                    objective, number_of_parameters(), initial_values_, lower_bounds_,
                    upper_bounds_);
                break;
            case OptimizationMethod::MultilevelSingleLinkage:
                // C# passes LocalMethod.NelderMead explicitly (C# 208-211); the C++ MLSL ctor
                // default is LocalMethod::BFGS, so the default must not be relied on.
                optimizer_ = std::make_unique<corehydro::numerics::math::optimization::MLSL>(
                    objective, number_of_parameters(), initial_values_, lower_bounds_,
                    upper_bounds_,
                    corehydro::numerics::math::optimization::LocalMethod::NelderMead);
                break;
        }

        optimizer_->report_failure = report_failure_;
        optimizer_->record_traces = false;
        optimizer_->compute_hessian = false;  // This class computes its own adaptive Hessian.
    }

    // Computes numerical gradients of pointwise DATA log-likelihoods via central differences
    // (C# 539-543, delegates to NumericalDiff; same as MLE -- see this file's header
    // difference #5 for why the gradients stay data-only while the Hessian is posterior-wide).
    std::vector<std::vector<double>> compute_pointwise_gradients(const std::vector<double>& parameters,
                                                                    int n) const {
        return NumericalDiff::compute_pointwise_gradients(
            [this](const std::vector<double>& p) { return model_.pointwise_data_log_likelihood(p); },
            parameters, n, number_of_parameters());
    }
};

}  // namespace corehydro::estimation
