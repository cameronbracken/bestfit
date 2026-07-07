// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/GeneralizedMethodOfMoments.cs @ fc28c0c
//
// GMM parameter estimation: minimizes the quadratic form Q(theta) = g(theta)' W g(theta),
// where g(theta) is the sample mean of moment conditions and W is a weighting matrix.
// Supports one-step, two-step, and iterative estimation strategies; default optimizer is
// BFGS (with the analytic GetGradient) plus an automatic NelderMead fallback.
// Reference: Hansen, L.P. (1982). Large Sample Properties of Generalized Method of Moments
// Estimators. Econometrica, 50(4), 1029-1054.
//
// The four file-scope C# delegates (JacobianFunction / PenaltyFunction /
// MomentConditionFunction / PointwiseMomentConditionFunction, C# lines 18-58) live in the
// separate header estimation/gmm_delegates.hpp so B9's IGMMModel can include the aliases
// without this full class (compile-order contract; see that header).
//
// B9 CTOR SLOT -- FILLED (Task B9): the C# class has TWO constructors -- the
// delegate-based one (C# 143, ported below) and an IGMMModel-based one (C# 106) mirroring
// MaximumLikelihood's IModel pattern. The IGMMModel overload and the `model()` accessor
// (C# `Model` property, 241) were added additively in B9 when the IGMMModel type landed
// (B8's existing lines untouched, exactly as the slot note reserved). Like
// MaximumLikelihood's ModelBase& ctor, the model parameter is a NON-OWNING reference (the
// C# ArgumentNullException for a null model is structurally unrepresentable) and the model
// must outlive the estimator; `model()` returns a nullable pointer, null for
// delegate-based construction, mirroring the C# `IGMMModel? Model`. The model-dependent
// branches inside the deferred Influence Diagnostics region remain deferred.
//
// ADAPTER NOTE: like MaximumLikelihood/MaximumAPosteriori (see maximum_likelihood.hpp's
// header), the Brent and NelderMead optimizer branches use the shared Phase 4 adapters in
// estimation/support/optimizer_adapters.hpp (this port's BrentSearch/NelderMead are
// standalone classes, not Optimizer subclasses). All adapter limitations documented there
// apply here too: both adapters always report Success after running to completion, and the
// base `max_function_evaluations` budget is not forwarded into the wrapped standalone
// solvers.
//
// EnableStartPointProbe DEVIATION: C# sets `EnableStartPointProbe = true` on the
// NelderMead branch of SetUpOptimizer, on the first iteration of MinimizeWithFallback, and
// on the BFGS->NelderMead fallback. The ported NelderMead deliberately OMITS the
// start-point probe (a Phase 0 scope decision recorded in nelder_mead.hpp's header;
// disabled by default upstream), so the flag is accepted for structural fidelity and is a
// documented no-op everywhere it appears below.
//
// EXCEPTION-TYPE MAPPING for THIS file (same convention as maximum_likelihood.hpp): C#
// `ArgumentNullException`/`InvalidOperationException` -> `std::invalid_argument`; C#
// `ArgumentOutOfRangeException` -> `std::out_of_range`. C#'s null-forgiving `W!` /
// `this.S!` dereferences (NullReferenceException if violated) port as
// `std::optional::value()` (`std::bad_optional_access` if violated).
//
// Debug.WriteLine / swallowed-exception guards: every C# `Debug.WriteLine` +
// catch-and-continue site (GetCovariance, GetMomentResidualCovariance,
// MinimizeWithFallback's two catches, the profile helpers' optimizer/root-finding
// fallbacks) ports as a silent commented no-throw guard, per the repo convention -- the
// catch BODIES (zero matrix / fallback value / Failure status) are ported, only the trace
// text is not.
//
// DEFERRED -- Influence Diagnostics region (C# 1382-2061): GetObservationInfluence,
// GetCooksDistance, both GetInfluenceDiagnostics overloads, GetLeverageDiagnostics, and
// their private BuildDataComponents/BuildRowToComponentMapping helpers depend on the
// unported RMC.BestFit.Diagnostics layer (InfluenceDiagnostics / ObservationInfluence /
// LeverageDiagnostics) and on the B9+ Bulletin17CDistribution model coupling. Following
// the Phase 4 precedent (MaximumAPosteriori::compute_leverage_diagnostics), they are
// ported as THROWING STUBS (std::logic_error) so callers get a clear member and runtime
// message; none of it is on any oracle path.
//
// SKIPPED (deliberate, one-line note per the brief): the XML (de)serialization region
// (ToXElement/RestoreFromXElement/DeserializeParameterSet/DeserializeMatrix, C# 2446+;
// desktop persistence) and the INotifyPropertyChanged plumbing (PropertyChanged event +
// RaisePropertyChange; the INPC properties are ported as plain get/set accessors that
// keep the C# "only assign when changed" guard).
#pragma once
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bestfit/diagnostics/influence_diagnostics.hpp"
#include "bestfit/diagnostics/leverage_diagnostics.hpp"
#include "bestfit/estimation/gmm_delegates.hpp"
#include "bestfit/estimation/numerical_diff.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/estimation/support/optimizer_adapters.hpp"
#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/i_gmm_model.hpp"
#include "bestfit/numerics/distributions/chi_squared.hpp"
#include "bestfit/numerics/math/linalg/lu_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/matrix_regularization.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"
#include "bestfit/numerics/math/optimization/bfgs.hpp"
#include "bestfit/numerics/math/optimization/differential_evolution.hpp"
#include "bestfit/numerics/math/optimization/mlsl.hpp"
#include "bestfit/numerics/math/optimization/powell.hpp"
#include "bestfit/numerics/math/optimization/support/local_method.hpp"
#include "bestfit/numerics/math/optimization/support/optimizer.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "bestfit/numerics/math/rootfinding/brent.hpp"
#include "bestfit/numerics/sampling/stratification_bin.hpp"
#include "bestfit/numerics/sampling/stratification_options.hpp"
#include "bestfit/numerics/sampling/stratify.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::estimation {

class GeneralizedMethodOfMoments {
   public:
    using Matrix = bestfit::numerics::math::linalg::Matrix;
    using Vector = bestfit::numerics::math::linalg::Vector;
    using Optimizer = bestfit::numerics::math::optimization::Optimizer;
    using OptimizationStatus = bestfit::numerics::math::optimization::OptimizationStatus;
    using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;

    // Identification status for a GMM specification based on the counts of moment
    // conditions (q) and parameters (p) (C# nested enum, 215).
    enum class GMMIdentificationStatus {
        // More parameters than moment conditions (p > q). Requires a penalty function.
        UnderIdentified,
        // Equal parameters and moment conditions (p = q). Exactly identified.
        JustIdentified,
        // Fewer parameters than moment conditions (p < q). Can test specification.
        OverIdentified,
    };

    // Enumeration of GMM estimation strategies (C# nested enum, 228).
    enum class GMMEstimationStrategy {
        // Single-step using the initial weighting matrix (identity or user-supplied).
        OneStep,
        // Two passes: first with initial W, second with W = S^-1.
        TwoStep,
        // Iterative refinement of W = S^-1 until parameter convergence.
        Iterative,
    };

    // ------------------------------------------------------------------------------------
    // Construction (the IGMMModel ctor slot was filled in B9 -- see this file's header)
    // ------------------------------------------------------------------------------------

    // Constructs a new GMM estimation class from an IGMMModel (C# 106). All configuration
    // is read from the model: parameter values become initial values, parameter bounds
    // become optimizer bounds, and the moment condition function and optional delegates
    // are read directly. The model is held as a non-owning pointer and must outlive this
    // estimator (see the file header); an empty MomentConditionFunction throws
    // std::invalid_argument (C# ArgumentException).
    explicit GeneralizedMethodOfMoments(bestfit::models::IGMMModel& model,
                                        OptimizationMethod method = OptimizationMethod::BFGS) {
        model_ = &model;

        moment_condition_function_ = model.moment_condition_function();
        if (!moment_condition_function_)
            throw std::invalid_argument("The model's MomentConditionFunction cannot be null.");
        number_of_parameters_ = model.number_of_parameters();
        number_of_moment_conditions_ = model.number_of_moment_conditions();
        sample_size_ = model.sample_size();

        initial_values_.clear();
        lower_bounds_.clear();
        upper_bounds_.clear();
        for (const auto& p : model.parameters()) {
            initial_values_.push_back(p.value());
            lower_bounds_.push_back(p.lower_bound());
            upper_bounds_.push_back(p.upper_bound());
        }

        jacobian_function_ = model.jacobian_function();
        penalty_function_ = model.penalty_function();
        pointwise_moment_conditions_ = model.pointwise_moment_conditions();

        compute_identification_status();
        optimizer_method_ = method;
    }

    // Constructs a new GMM estimation class from raw delegate functions (C# 143), with the
    // C# validation checks in the C# order.
    GeneralizedMethodOfMoments(MomentConditionFunction moment_condition_function,
                               int number_of_parameters, int number_of_moment_conditions,
                               int sample_size, const std::vector<double>& initial_values,
                               const std::vector<double>& lower_bounds,
                               const std::vector<double>& upper_bounds,
                               std::optional<Matrix> initial_w = std::nullopt,
                               JacobianFunction jacobian_function = nullptr,
                               PenaltyFunction penalty_function = nullptr,
                               PointwiseMomentConditionFunction pointwise_moment_conditions = nullptr) {
        if (!moment_condition_function)
            throw std::invalid_argument("The moment condition function cannot be null.");
        if (static_cast<int>(initial_values.size()) != number_of_parameters ||
            static_cast<int>(lower_bounds.size()) != number_of_parameters ||
            static_cast<int>(upper_bounds.size()) != number_of_parameters)
            throw std::out_of_range(
                "The initial values and lower and upper bounds must be the same length as the "
                "number of parameters.");

        for (std::size_t j = 0; j < initial_values.size(); ++j) {
            if (upper_bounds[j] < lower_bounds[j])
                throw std::out_of_range("The upper bound cannot be less than the lower bound.");
            if (initial_values[j] < lower_bounds[j] || initial_values[j] > upper_bounds[j])
                throw std::out_of_range(
                    "The initial values must be between the upper and lower bounds.");
        }

        moment_condition_function_ = std::move(moment_condition_function);
        number_of_parameters_ = number_of_parameters;
        number_of_moment_conditions_ = number_of_moment_conditions;
        sample_size_ = sample_size;
        initial_values_ = initial_values;
        lower_bounds_ = lower_bounds;
        upper_bounds_ = upper_bounds;
        w_ = std::move(initial_w);
        jacobian_function_ = std::move(jacobian_function);
        penalty_function_ = std::move(penalty_function);
        pointwise_moment_conditions_ = std::move(pointwise_moment_conditions);

        compute_identification_status();
    }

    // ------------------------------------------------------------------------------------
    // Members (C# INPC properties are plain accessors here -- see the header SKIPPED note)
    // ------------------------------------------------------------------------------------

    // The IGMMModel used for estimation, if constructed with one; null for delegate-based
    // construction (C# `IGMMModel? Model`, 241; non-owning -- B9).
    bestfit::models::IGMMModel* model() const { return model_; }

    // The moment condition function.
    const MomentConditionFunction& moment_condition_function() const {
        return moment_condition_function_;
    }

    // The optional analytical Jacobian function (empty = numerical differentiation).
    const JacobianFunction& jacobian_function() const { return jacobian_function_; }

    // The optional penalty function (empty = no penalty).
    const PenaltyFunction& penalty_function() const { return penalty_function_; }

    // The optional pointwise moment condition function for influence diagnostics.
    const PointwiseMomentConditionFunction& pointwise_moment_conditions() const {
        return pointwise_moment_conditions_;
    }

    // The number of model parameters (p).
    int number_of_parameters() const { return number_of_parameters_; }

    // The number of moment conditions (q).
    int number_of_moment_conditions() const { return number_of_moment_conditions_; }

    // The identification status.
    GMMIdentificationStatus identification_status() const { return identification_status_; }

    // An array of initial values to evaluate. NOTE (C# fidelity): the two-step/iterative
    // strategies overwrite this with each pass's best point, exactly as the C# does.
    const std::vector<double>& initial_values() const { return initial_values_; }

    // An array of lower bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& lower_bounds() const { return lower_bounds_; }

    // An array of upper bounds (inclusive) of the interval containing the optimal point.
    const std::vector<double>& upper_bounds() const { return upper_bounds_; }

    // The optimization method used for estimating the model parameters. Default = BFGS.
    OptimizationMethod optimizer_method() const { return optimizer_method_; }
    void set_optimizer_method(OptimizationMethod value) {
        if (optimizer_method_ != value) {
            optimizer_method_ = value;
        }
    }

    // Whether to automatically fall back to NelderMead when BFGS fails. Default = true.
    // Only applies when optimizer_method() is BFGS.
    bool use_fallback_optimizer() const { return use_fallback_optimizer_; }
    void set_use_fallback_optimizer(bool value) {
        if (use_fallback_optimizer_ != value) {
            use_fallback_optimizer_ = value;
        }
    }

    // The GMM estimation strategy. Default = Iterative.
    GMMEstimationStrategy estimation_strategy() const { return estimation_strategy_; }
    void set_estimation_strategy(GMMEstimationStrategy value) {
        if (estimation_strategy_ != value) {
            estimation_strategy_ = value;
        }
    }

    // The total sample size used in GMM estimation.
    int sample_size() const { return sample_size_; }

    // The maximum number of iterations for iterative GMM.
    int max_gmm_iterations() const { return max_gmm_iterations_; }
    void set_max_gmm_iterations(int value) {
        if (max_gmm_iterations_ != value) {
            max_gmm_iterations_ = value;
        }
    }

    // The maximum function evaluations allowed within each GMM iteration.
    int max_function_evaluations() const { return max_function_evaluations_; }
    void set_max_function_evaluations(int value) {
        if (max_function_evaluations_ != value) {
            max_function_evaluations_ = value;
        }
    }

    // The absolute tolerance for convergence.
    double absolute_tolerance() const { return absolute_tolerance_; }
    void set_absolute_tolerance(double value) {
        if (absolute_tolerance_ != value) {
            absolute_tolerance_ = value;
        }
    }

    // The relative tolerance for convergence.
    double relative_tolerance() const { return relative_tolerance_; }
    void set_relative_tolerance(double value) {
        if (relative_tolerance_ != value) {
            relative_tolerance_ = value;
        }
    }

    // The optimizer used to minimize the objective function. Transient: rebuilt each time
    // estimate() runs; null before any estimation (C# `Optimizer { get; private set; } =
    // null!`). Read status() instead for post-estimation logic.
    Optimizer* optimizer() { return optimizer_.get(); }
    const Optimizer* optimizer() const { return optimizer_.get(); }

    // The final OptimizationStatus from the most recent estimation run (captured at the end
    // of minimize_with_fallback so it survives the transient optimizer). None before any
    // estimation and after clear_results().
    OptimizationStatus status() const { return status_; }

    // Whether the model has been successfully estimated.
    bool is_estimated() const { return is_estimated_; }

    // The degrees of freedom for the J-statistic test: max(0, q - p).
    int degree_of_freedom() const {
        return std::max(0, number_of_moment_conditions_ - number_of_parameters_);
    }

    // The moment condition covariance matrix (null before estimation).
    const std::optional<Matrix>& s() const { return s_; }

    // The current GMM weighting matrix (null before estimation unless supplied via the
    // ctor's initial_w).
    const std::optional<Matrix>& w() const { return w_; }

    // The estimated parameter covariance matrix. Valid only after post_process() completes
    // (see the C# remarks: null while estimate() is mid-iteration or before it is called).
    const std::optional<Matrix>& sigma() const { return sigma_; }

    // The best parameter set found during estimation. Initialized to an empty ParameterSet
    // so consumers that bypass is_estimated() get a deterministic empty set.
    const ParameterSet& best_parameter_set() const { return best_parameter_set_; }

    // The J-statistic value for model fit.
    double jstat() const { return jstat_; }

    // The p-value for the J-statistic.
    double jstat_pval() const { return jstat_pval_; }

    // The number of iterations used for iterative estimation.
    int gmm_iterations() const { return gmm_iterations_; }

    // True only when the most recent estimate() run reached the convergence tolerance
    // before max_gmm_iterations() (C# ConvergedWithinTolerance).
    bool converged_within_tolerance() const {
        return is_estimated_ && gmm_iterations_ < max_gmm_iterations_;
    }

    // The total number of function evaluations required to estimate the model.
    int total_function_evaluations() const { return total_function_evaluations_; }

    // The objective function value Q(theta-hat) at the estimated parameters; NaN if the
    // model has not been estimated.
    double objective_function_value() const {
        return is_estimated_ ? q(best_parameter_set_.values)
                             : std::numeric_limits<double>::quiet_NaN();
    }

    // The history of objective function values across GMM iterations (iterative only).
    const std::vector<double>& convergence_history() const { return convergence_history_; }

    // Whether the penalty target parameters are random variables. Default = true.
    // Controls whether the penalty Hessian H = P/n appears in the sandwich meat:
    //   true (default): the penalty target theta_0 is a random variable with its own
    //     sampling variance MSE (e.g. the B17C regional skewness estimate); Meat =
    //     D'WSWD + H, collapsing to the Bayesian-correct weighted MSE when W = S^-1.
    //   false: theta_0 is a deterministic constant (ridge-type regularization); Meat =
    //     D'WSWD only.
    // (C# auto-property with public setter and no INPC -- a plain public member here.)
    bool penalty_is_random = true;

    // ------------------------------------------------------------------------------------
    // Moment Condition Methods
    // ------------------------------------------------------------------------------------

    // Computes the sample mean vector of the moment conditions at the given parameters:
    // delegates to the MomentConditionFunction and extracts only G (C# 550).
    Vector get_g(const std::vector<double>& parameters) const {
        return moment_condition_function_(parameters).G;
    }

    // Computes the GMM objective function Q(theta) (C# 570).
    //   Without penalties: Q = g'Wg, the standard GMM quadratic form.
    //   With penalties (half-quadratic convention): Q = (1/2) g'Wg + penalty, where the
    //   penalty already includes its own 1/2 factor -- the 1/2 here applies only to the
    //   moment quadratic form so gradient/Hessian scaling stays consistent for BFGS. The
    //   argmin is invariant to this positive scaling.
    // Returns std::numeric_limits<double>::max() (C# double.MaxValue) if non-finite.
    double q(const std::vector<double>& parameters) const {
        if (!w_.has_value())
            throw std::invalid_argument(
                "Weighting matrix W must be set before computing Q. Call Estimate() or set W "
                "directly.");

        Vector gtmean = get_g(parameters);
        // C# `gtmean.Multiply(W)` computes W * gtmean (see Vector.Multiply(Matrix)), then
        // multiplies elementwise with gtmean and sums: g'Wg.
        double qv = ((*w_) * gtmean).multiply(gtmean).sum();

        if (penalty_function_) {
            // Half-quadratic convention: Q = (1/2) g'Wg + penalty. The penalty already
            // includes its own 1/2 factor, so the 1/2 here applies only to the moment
            // quadratic form to maintain consistent gradient and Hessian scaling for BFGS.
            qv *= 0.5;
            qv += penalty_function_(parameters);
        }

        return bestfit::numerics::is_finite(qv) ? qv : std::numeric_limits<double>::max();
    }

    // Computes the sample covariance matrix S of the moment conditions, regularized to be
    // symmetric positive definite (C# 600). In two-step/iterative GMM the optimal
    // weighting matrix is W = S^-1.
    Matrix get_s(const std::vector<double>& parameters) const {
        Matrix s = moment_condition_function_(parameters).S;
        s = bestfit::numerics::math::linalg::MatrixRegularization::
            make_symmetric_positive_definite(s);
        return s;
    }

    // Computes the Jacobian matrix D = dg/dtheta of the moment conditions with respect to
    // parameters: a q x p Matrix (C# 616). Uses the analytic delegate if supplied,
    // otherwise NumericalDiff::compute_jacobian with adaptive, bounds-aware step sizes.
    Matrix get_jacobian(const std::vector<double>& parameters) const {
        if (!jacobian_function_) {
            // Use adaptive step sizes with boundary handling. Bounds prevent sigma from
            // going negative and keep gamma within feasible range.
            Vector g0 = get_g(parameters);
            return Matrix(NumericalDiff::compute_jacobian(
                [this](std::vector<double>& x) { return get_g(x).to_array(); }, parameters,
                g0.length(), lower_bounds_, upper_bounds_));
        } else {
            return Matrix(jacobian_function_(parameters));
        }
    }

    // Computes the Hessian matrix of the penalty function with respect to parameters:
    // a p x p Matrix, zero when there is no penalty (C# 659).
    Matrix get_penalty_hessian(const std::vector<double>& parameters) const {
        if (!penalty_function_) return Matrix(static_cast<int>(parameters.size()));

        return NumericalDiff::compute_hessian(
            [this](std::vector<double>& x) { return penalty_function_(x); }, parameters,
            static_cast<int>(parameters.size()), lower_bounds_, upper_bounds_);
    }

    // Computes the gradient of the GMM objective function Q(theta) (C# 685).
    //   With penalties (half-quadratic convention): grad Q = D'Wg + grad penalty, the
    //   exact gradient of Q = (1/2) g'Wg + penalty.
    //   Without penalties: grad Q = D'Wg, proportional to the true gradient 2 D'Wg of
    //   g'Wg; the constant factor does not affect the BFGS optimum.
    Vector get_gradient(const std::vector<double>& parameters) const {
        Vector gt_mean = get_g(parameters);
        Matrix J = get_jacobian(parameters);
        Matrix JT = J.transpose();
        // C# `W!` null-forgiving dereference -> optional::value() (see the header mapping).
        Vector grad = JT * (w_.value() * gt_mean) + get_penalty_gradient(parameters);
        return grad;
    }

    // ------------------------------------------------------------------------------------
    // Covariance Methods
    // ------------------------------------------------------------------------------------

    // Computes the asymptotic covariance matrix of the estimated parameters (C# 729).
    // Bread = D'WD + H (penalty curvature always in the bread); non-sandwich shortcut
    // Sigma = Bread^-1 / n; sandwich meat = D'WSWD (+ H when penalty_is_random); sandwich
    // Sigma = Bread^-1 Meat Bread^-1 / n. Any failure returns a zero matrix (silent guard).
    // NOTE: like the C#, this recomputes and STORES S and W at `parameters` as side
    // effects.
    Matrix get_covariance(const std::vector<double>& parameters, bool sandwich = true) {
        try {
            // 1) S and W -- regularize S before inversion to handle ill-conditioning
            // under heavy censoring (few observations in some moment conditions).
            s_ = get_s(parameters);
            s_ = bestfit::numerics::math::linalg::MatrixRegularization::regularize(*s_);
            w_ = s_->inverse();

            // 2) Jacobian and penalty Hessian.
            Matrix d = get_jacobian(parameters);
            Matrix dT = d.transpose();
            Matrix H = get_penalty_hessian(parameters);

            // 3) Bread: D'WD + H always includes penalty curvature. Regularize before
            // inversion to handle near-singular information matrices.
            Matrix bread = dT * (*w_) * d + H;
            bread = bestfit::numerics::math::linalg::MatrixRegularization::
                make_symmetric_positive_definite(bread);
            Matrix bread_inv = bread.inverse();

            // 4) Non-sandwich shortcut: Sigma = Bread^-1 / n.
            if (!sandwich) {
                bread_inv = divide(bread_inv, static_cast<double>(sample_size_));
                bread_inv = bestfit::numerics::math::linalg::MatrixRegularization::
                    make_symmetric_positive_definite(bread_inv);
                return bread_inv;
            }

            // 5) Sandwich meat: D'WSWD, plus H when penalty parameters are random. When
            // penalty_is_random = true (B17C regional estimates), the penalty target
            // theta_0 is itself a random variable with variance MSE, adding P/n to the
            // meat. When false (deterministic regularization), only data variability
            // contributes.
            Matrix meat = dT * (*w_) * (*s_) * (*w_) * d;
            if (penalty_is_random) meat = meat + H;

            // 6) Sandwich covariance: Sigma = Bread^-1 Meat Bread^-1 / n.
            Matrix covariance = bread_inv * meat * bread_inv;
            covariance = divide(covariance, static_cast<double>(sample_size_));
            covariance = bestfit::numerics::math::linalg::MatrixRegularization::
                make_symmetric_positive_definite(covariance);

            return covariance;
        } catch (const std::exception&) {
            // C# logs "Failed to compute GMM covariance matrix: ..." via Debug.WriteLine;
            // silent no-throw guard here -- fall back to a zero matrix.
            return Matrix(number_of_parameters_, number_of_parameters_);
        }
    }

    // Returns the parameter covariance matrix, preferring the cached Sigma from
    // post_process() (C# 786: `Sigma ?? GetCovariance(...)`).
    Matrix get_covariance_matrix(bool sandwich = true) {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        return sigma_.has_value() ? *sigma_
                                  : get_covariance(best_parameter_set_.values, sandwich);
    }

    // Returns the standard errors of the parameter estimates from the diagonal of the
    // covariance matrix (C# 798).
    std::vector<double> get_standard_errors() {
        Matrix cov = get_covariance_matrix();
        std::vector<double> se(static_cast<std::size_t>(number_of_parameters_));
        for (int i = 0; i < number_of_parameters_; ++i)
            se[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, cov(i, i)));
        return se;
    }

    // Returns the correlation matrix from the covariance matrix (C# 812).
    Matrix get_correlation_matrix() {
        Matrix cov = get_covariance_matrix();
        Matrix corr(number_of_parameters_, number_of_parameters_);
        for (int i = 0; i < number_of_parameters_; ++i) {
            for (int j = 0; j < number_of_parameters_; ++j) {
                double denom = std::sqrt(cov(i, i) * cov(j, j));
                corr(i, j) = denom > 0 ? cov(i, j) / denom : 0;
            }
        }
        return corr;
    }

    // Returns the sandwich (robust) covariance matrix (C# 832).
    Matrix get_sandwich_covariance_matrix() {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        return get_covariance(best_parameter_set_.values, /*sandwich=*/true);
    }

    // Returns the robust standard errors from the sandwich covariance matrix (C# 844).
    std::vector<double> get_robust_standard_errors() {
        Matrix sandwich = get_sandwich_covariance_matrix();
        std::vector<double> se(static_cast<std::size_t>(number_of_parameters_));
        for (int i = 0; i < number_of_parameters_; ++i)
            se[static_cast<std::size_t>(i)] = std::sqrt(std::max(0.0, sandwich(i, i)));
        return se;
    }

    // ------------------------------------------------------------------------------------
    // Profile Methods
    // ------------------------------------------------------------------------------------

    // Returns the profile Q objective for each model parameter: one Matrix(bins, 2) per
    // parameter, columns [parameter value, Q value] (C# 927). trueProfile re-optimizes the
    // nuisance parameters at each grid point; otherwise they stay at their estimates.
    std::vector<Matrix> profile_q(int bins = 100, bool true_profile = true) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (bins < 2) throw std::out_of_range("Number of bins must be at least 2.");

        std::vector<Matrix> list;
        list.reserve(static_cast<std::size_t>(number_of_parameters_));

        for (int i = 0; i < number_of_parameters_; ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            auto seq = bestfit::numerics::sampling::Stratify::XValues(
                bestfit::numerics::sampling::StratificationOptions(lower_bounds_[si],
                                                                   upper_bounds_[si], bins));
            Matrix profile(bins, 2);

            for (int j = 0; j < bins; ++j) {
                double grid_value = seq[static_cast<std::size_t>(j)].midpoint();
                profile(j, 0) = grid_value;
                profile(j, 1) = true_profile ? profile_q_true_at_point(i, grid_value)
                                             : profile_q_conditional_at_point(i, grid_value);
            }

            list.push_back(std::move(profile));
        }

        return list;
    }

    // Returns parameter confidence intervals based on the profile Q statistic using the
    // chi-squared threshold: an Nx2 Matrix, columns [lower bound, upper bound] (C# 974).
    // The Q-space threshold is Q(theta-hat) + chi2_1(1-alpha) / (2n), the half-quadratic
    // convention.
    Matrix profile_confidence_intervals(double alpha = 0.1, bool true_profile = true) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (alpha <= 0 || alpha >= 1) throw std::out_of_range("Alpha must be between 0 and 1.");

        double q_hat = q(best_parameter_set_.values);
        double threshold = q_hat + profile_q_threshold_increment(alpha);
        Matrix cis(number_of_parameters_, 2);

        for (int i = 0; i < number_of_parameters_; ++i) {
            cis(i, 0) = profile_q_find_bound(i, threshold, /*lower=*/true, true_profile);
            cis(i, 1) = profile_q_find_bound(i, threshold, /*lower=*/false, true_profile);
        }

        return cis;
    }

    // Returns parameter percentiles from the profile Q surface by converting to an
    // approximate marginal density via exp(-scale * dQ) -- the Laplace approximation to
    // the marginal posterior under a flat prior -- calibrating the implied variance to the
    // sandwich Sigma, and numerically inverting the CDF (C# 1032). Returns a Matrix of
    // dimension [p x len(percentiles)]. An empty `percentiles` means the C# null default
    // {0.05, 0.25, 0.50, 0.75, 0.95}.
    Matrix profile_percentiles(std::vector<double> percentiles = {}, bool true_profile = true,
                               int bins = 200) const {
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (bins < 10) throw std::out_of_range("Number of bins must be at least 10.");

        if (percentiles.empty()) percentiles = {0.05, 0.25, 0.50, 0.75, 0.95};

        for (std::size_t k = 0; k < percentiles.size(); ++k) {
            if (percentiles[k] <= 0 || percentiles[k] >= 1)
                throw std::out_of_range("Each percentile must be in (0, 1).");
        }

        // Compute profile Q on a fine grid for each parameter.
        std::vector<Matrix> profiles = profile_q(bins, true_profile);
        double q_hat = q(best_parameter_set_.values);
        Matrix result(number_of_parameters_, static_cast<int>(percentiles.size()));

        // Scale factor: n for half-quadratic Q = (1/2)g'Wg + penalty (n*Q ~ -l), n/2 for
        // standard Q = g'Wg (n*Q ~ -2l). Use n/2 as a conservative default that works for
        // both conventions (the shape is what matters, not absolute scale); the sandwich
        // SE calibration below corrects any residual scale mismatch.
        double scale = 0.5 * sample_size_;

        for (int i = 0; i < number_of_parameters_; ++i) {
            const Matrix& profile = profiles[static_cast<std::size_t>(i)];
            int n_bins = profile.number_of_rows();

            // Step 1: log-weights logw[j] = -scale * dQ[j]; subtract q_hat for numerical
            // stability (avoids exp overflow/underflow).
            std::vector<double> theta(static_cast<std::size_t>(n_bins));
            std::vector<double> logw(static_cast<std::size_t>(n_bins));

            for (int j = 0; j < n_bins; ++j) {
                theta[static_cast<std::size_t>(j)] = profile(j, 0);
                logw[static_cast<std::size_t>(j)] = -scale * (profile(j, 1) - q_hat);
            }

            // Step 2: calibrate scale so the implied variance matches the sandwich SE
            // (Sigma[i, i]); the percentile spread stays consistent with the sandwich
            // estimator while the shape comes from the profile Q curve.
            double target_var =
                sigma_.has_value() ? std::max((*sigma_)(i, i), 1e-30) : 0.0;
            if (target_var > 0) {
                double implied_var = compute_implied_variance(theta, logw);
                if (implied_var > 1e-30) {
                    // Var ~ 1/scale, so new_scale = old_scale * impliedVar / targetVar.
                    double scale_adj = implied_var / target_var;
                    for (int j = 0; j < n_bins; ++j)
                        logw[static_cast<std::size_t>(j)] *= scale_adj;
                }
            }

            // Step 3: convert log-weights to a normalized CDF via the trapezoidal rule
            // (shift log-weights so the max is 0 for numerical stability).
            double max_log_w = -std::numeric_limits<double>::infinity();
            for (int j = 0; j < n_bins; ++j)
                if (logw[static_cast<std::size_t>(j)] > max_log_w)
                    max_log_w = logw[static_cast<std::size_t>(j)];

            std::vector<double> w(static_cast<std::size_t>(n_bins));
            for (int j = 0; j < n_bins; ++j)
                w[static_cast<std::size_t>(j)] =
                    std::exp(logw[static_cast<std::size_t>(j)] - max_log_w);

            std::vector<double> cdf(static_cast<std::size_t>(n_bins));
            cdf[0] = 0;
            for (int j = 1; j < n_bins; ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                double dx = theta[sj] - theta[sj - 1];
                cdf[sj] = cdf[sj - 1] + 0.5 * (w[sj - 1] + w[sj]) * dx;
            }

            double total = cdf[static_cast<std::size_t>(n_bins - 1)];
            if (total > 0) {
                for (int j = 0; j < n_bins; ++j) cdf[static_cast<std::size_t>(j)] /= total;
            }

            // Step 4: invert the CDF at each requested percentile via linear interpolation.
            for (std::size_t k = 0; k < percentiles.size(); ++k) {
                double p = percentiles[k];
                result(i, static_cast<int>(k)) = interpolate_cdf_inverse(
                    theta, cdf, p, best_parameter_set_.values[static_cast<std::size_t>(i)]);
            }
        }

        return result;
    }

    // ------------------------------------------------------------------------------------
    // Influence Diagnostics (D4 un-stub -- the Diagnostics layer is now ported)
    // ------------------------------------------------------------------------------------
    //
    // These four methods are NON-const (the C# instance methods use `Sigma ?? GetCovariance(...)`,
    // and `get_covariance` recomputes + stores S/W as a side effect -- see its NOTE). This
    // mirrors the C# fidelity: bread uses the current W, then sigma is taken from the cached
    // `sigma_` or recomputed. The C# InvalidOperationException maps to std::invalid_argument.

    // Pointwise influence of each observation on parameter estimates (DFBETAS-like; C# 1402).
    // IF_ij = (Sigma . D'W g_i)_j / SE_j.
    Matrix get_observation_influence() {
        namespace linalg = bestfit::numerics::math::linalg;
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (!pointwise_moment_conditions_)
            throw std::invalid_argument(
                "Observation influence requires pointwise moment conditions. Provide a "
                "PointwiseMomentConditionFunction.");

        int p = number_of_parameters_;
        int q = number_of_moment_conditions_;
        linalg::Matrix2D gi = pointwise_moment_conditions_(best_parameter_set_.values);
        int n = static_cast<int>(gi.size());

        // Penalized bread B = D'WD + H; breadInv with regularization fallback.
        Matrix d_mat = get_jacobian(best_parameter_set_.values);
        Matrix dt_w = d_mat.transpose() * w_.value();
        Matrix bread = dt_w * d_mat + get_penalty_hessian(best_parameter_set_.values);
        Matrix bread_inv(p, p);
        try {
            bread_inv = bread.inverse();
        } catch (const std::exception&) {
            bread = linalg::MatrixRegularization::make_symmetric_positive_definite(bread);
            bread_inv = bread.inverse();
        }

        // Sigma and standard errors.
        Matrix sigma = sigma_.has_value() ? *sigma_
                                          : get_covariance(best_parameter_set_.values, true);
        std::vector<double> se(static_cast<std::size_t>(p));
        for (int j = 0; j < p; ++j)
            se[static_cast<std::size_t>(j)] = std::sqrt(std::max(0.0, sigma(j, j)));

        Matrix influence(n, p);
        for (int i = 0; i < n; ++i) {
            // DtWg_i = D'W g_i [p]; psi_i = B^{-1} D'W g_i [p].
            std::vector<double> dtwg(static_cast<std::size_t>(p), 0.0);
            for (int j = 0; j < p; ++j)
                for (int k = 0; k < q; ++k)
                    dtwg[static_cast<std::size_t>(j)] +=
                        dt_w(j, k) * gi[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            std::vector<double> psi(static_cast<std::size_t>(p), 0.0);
            for (int j = 0; j < p; ++j)
                for (int k = 0; k < p; ++k)
                    psi[static_cast<std::size_t>(j)] +=
                        bread_inv(j, k) * dtwg[static_cast<std::size_t>(k)];

            for (int j = 0; j < p; ++j) {
                double infl_j = 0;
                for (int k = 0; k < p; ++k)
                    infl_j += sigma(j, k) * psi[static_cast<std::size_t>(k)];
                influence(i, j) = se[static_cast<std::size_t>(j)] > 0
                                      ? infl_j / se[static_cast<std::size_t>(j)]
                                      : 0.0;
            }
        }
        return influence;
    }

    // Cook's distance-like measure for each observation (C# 1486): D_i = psi_i' Sigma psi_i / p,
    // psi_i = B^{-1} D'W g_i.
    std::vector<double> get_cooks_distance() {
        namespace linalg = bestfit::numerics::math::linalg;
        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (!pointwise_moment_conditions_)
            throw std::invalid_argument(
                "Cook's distance requires pointwise moment conditions. Provide a "
                "PointwiseMomentConditionFunction.");

        int p = number_of_parameters_;
        int q = number_of_moment_conditions_;
        linalg::Matrix2D gi = pointwise_moment_conditions_(best_parameter_set_.values);
        int n = static_cast<int>(gi.size());

        Matrix d_mat = get_jacobian(best_parameter_set_.values);
        Matrix dt_w = d_mat.transpose() * w_.value();
        Matrix bread = dt_w * d_mat + get_penalty_hessian(best_parameter_set_.values);
        Matrix bread_inv(p, p);
        try {
            bread_inv = bread.inverse();
        } catch (const std::exception&) {
            bread = linalg::MatrixRegularization::make_symmetric_positive_definite(bread);
            bread_inv = bread.inverse();
        }

        Matrix sigma = sigma_.has_value() ? *sigma_
                                          : get_covariance(best_parameter_set_.values, true);

        std::vector<double> cooks_d(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            std::vector<double> dtwg(static_cast<std::size_t>(p), 0.0);
            for (int j = 0; j < p; ++j)
                for (int k = 0; k < q; ++k)
                    dtwg[static_cast<std::size_t>(j)] +=
                        dt_w(j, k) * gi[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            std::vector<double> psi(static_cast<std::size_t>(p), 0.0);
            for (int j = 0; j < p; ++j)
                for (int k = 0; k < p; ++k)
                    psi[static_cast<std::size_t>(j)] +=
                        bread_inv(j, k) * dtwg[static_cast<std::size_t>(k)];

            double quad_form = 0;
            for (int j = 0; j < p; ++j) {
                double tmp = 0;
                for (int k = 0; k < p; ++k) tmp += sigma(j, k) * psi[static_cast<std::size_t>(k)];
                quad_form += psi[static_cast<std::size_t>(j)] * tmp;
            }
            cooks_d[static_cast<std::size_t>(i)] = quad_form / p;
        }
        return cooks_d;
    }

    // Influence diagnostics using Cook's distance as the influence metric, mapped to ParetoK
    // (C# 1562).
    //
    // BULLETIN17C-COUPLED BRANCH OMITTED (deliberate, C#-faithful for every non-B17C model): the
    // C# body has a `Model is Bulletin17CDistribution` branch (aggregated DataComponent metadata +
    // BuildRowToComponentMapping). The Bulletin17CDistribution coupling is severed from this port
    // (see the DEFERRED header note / D3 leverage precedent), so `dataComponents`/`rowMap` stay
    // null exactly as for any non-B17C IGMMModel and the method falls to the raw per-row path.
    bestfit::diagnostics::InfluenceDiagnostics get_influence_diagnostics() {
        std::vector<double> cooks_d = get_cooks_distance();
        std::vector<bestfit::diagnostics::InfluenceDiagnostics::ObservationInfluence> observations;
        observations.reserve(cooks_d.size());
        for (std::size_t i = 0; i < cooks_d.size(); ++i)
            observations.emplace_back(static_cast<int>(i), cooks_d[i],
                                      std::numeric_limits<double>::quiet_NaN());
        return bestfit::diagnostics::InfluenceDiagnostics(std::move(observations));
    }

    // Influence diagnostics (Cook's distance in ParetoK) with data component metadata for labeling
    // (C# 1770).
    bestfit::diagnostics::InfluenceDiagnostics get_influence_diagnostics(
        const std::vector<bestfit::models::DataComponent>& data_components) {
        std::vector<double> cooks_d = get_cooks_distance();
        int n = std::min(static_cast<int>(cooks_d.size()),
                         static_cast<int>(data_components.size()));
        std::vector<bestfit::diagnostics::InfluenceDiagnostics::ObservationInfluence> observations;
        observations.reserve(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            const bestfit::models::DataComponent& dc = data_components[static_cast<std::size_t>(i)];
            observations.emplace_back(i, cooks_d[static_cast<std::size_t>(i)],
                                      std::numeric_limits<double>::quiet_NaN(), dc.value(),
                                      dc.type(), dc.count(), dc.name());
        }
        return bestfit::diagnostics::InfluenceDiagnostics(std::move(observations));
    }

    // Leverage diagnostics for the GMM estimator (C# 1822; D3 un-stub -- the Diagnostics layer
    // is now ported). Decomposes each observation's leverage into FitInfluence (Cook's D via
    // fullSigma = [Hessian(Q)]^{-1}) and VarianceInfluence (via breadInv = (D'WD + H)^{-1}).
    //
    // BULLETIN17C-COUPLED BRANCHES OMITTED (deliberate, C#-faithful for every non-B17C model):
    // the C# body has two `Model is Bulletin17CDistribution` branches -- (1) the aggregated
    // DataComponent metadata + row-to-component mapping (BuildDataComponents /
    // BuildRowToComponentMapping), and (2) the per-parameter / per-quantile penalty components
    // (ParameterPenalties / QuantilePenalties / LinkController). The Bulletin17CDistribution
    // model coupling is severed from this port (see PLAN / this file's DEFERRED header note),
    // so both branches are unreachable exactly as they are for any non-B17C IGMMModel: data
    // components stay absent (each expanded gi row is its own component with default metadata)
    // and the penalty-component array stays empty. Returns an EMPTY LeverageDiagnostics on the
    // bread regularization-failure branch, mirroring the C#.
    bestfit::diagnostics::LeverageDiagnostics get_leverage_diagnostics() const {
        using LD = bestfit::diagnostics::LeverageDiagnostics;
        namespace linalg = bestfit::numerics::math::linalg;

        if (!is_estimated_) throw std::invalid_argument("The model has not been estimated.");
        if (!pointwise_moment_conditions_)
            throw std::invalid_argument(
                "Leverage diagnostics require pointwise moment conditions.");

        int p = number_of_parameters_;
        int q = number_of_moment_conditions_;

        // 1. Per-observation moment conditions gi [n x q].
        linalg::Matrix2D gi = pointwise_moment_conditions_(best_parameter_set_.values);
        int n = static_cast<int>(gi.size());

        // 2. Penalized bread: B = D'WD + H.
        Matrix d_mat = get_jacobian(best_parameter_set_.values);
        Matrix dt = d_mat.transpose();
        Matrix dt_w = dt * w_.value();
        Matrix bread = dt_w * d_mat + get_penalty_hessian(best_parameter_set_.values);

        Matrix bread_inv(p, p);
        try {
            bread_inv = bread.inverse();
        } catch (const std::exception&) {
            // C# Debug.WriteLine then attempts regularization.
            try {
                bread = linalg::MatrixRegularization::make_symmetric_positive_definite(bread);
                bread_inv = bread.inverse();
            } catch (const std::exception&) {
                // C# returns an empty LeverageDiagnostics on regularization failure.
                return LD();
            }
        }

        // 3. fullSigma = [Hessian(Q)]^{-1}; -Q is the GMM analog of LogLikelihood.
        auto neg_q = [this](std::vector<double>& parms) { return -this->q(parms); };
        Matrix full_hess =
            LD::compute_numerical_hessian_public(neg_q, best_parameter_set_.values, p);
        Matrix full_neg_hess = full_hess * -1.0;
        Matrix full_sigma(p, p);
        try {
            full_sigma = full_neg_hess.inverse();
        } catch (const std::exception&) {
            full_sigma =
                linalg::MatrixRegularization::make_symmetric_positive_definite(full_neg_hess)
                    .inverse();
        }

        // 4. (B17C DataComponent metadata + row mapping omitted -- see this method's header.)
        //    rowMap == null so each row is its own component: nComponents = n, ci = i.
        int n_components = n;
        std::vector<double> agg_fit_influence(static_cast<std::size_t>(n_components), 0.0);
        std::vector<double> agg_variance_influence(static_cast<std::size_t>(n_components), 0.0);
        std::vector<std::vector<double>> agg_dtwg(static_cast<std::size_t>(n_components),
                                                  std::vector<double>(static_cast<std::size_t>(p),
                                                                      0.0));

        // Phase 1: accumulate DtWg vectors per component.
        for (int i = 0; i < n; ++i) {
            std::vector<double> dtwg(static_cast<std::size_t>(p), 0.0);
            for (int j = 0; j < p; ++j)
                for (int k = 0; k < q; ++k)
                    dtwg[static_cast<std::size_t>(j)] +=
                        dt_w(j, k) * gi[static_cast<std::size_t>(i)][static_cast<std::size_t>(k)];
            std::size_t ci = static_cast<std::size_t>(i);  // rowMap == null -> ci = i
            for (int j = 0; j < p; ++j)
                agg_dtwg[ci][static_cast<std::size_t>(j)] += dtwg[static_cast<std::size_t>(j)];
        }

        // Phase 2: quadratic forms on the summed DtWg vectors.
        for (int ci = 0; ci < n_components; ++ci) {
            const std::vector<double>& v = agg_dtwg[static_cast<std::size_t>(ci)];

            // Variance influence: |v' breadInv v| / (n . p).
            double raw_leverage = 0;
            for (int j = 0; j < p; ++j) {
                double tmp = 0;
                for (int k = 0; k < p; ++k)
                    tmp += bread_inv(j, k) * v[static_cast<std::size_t>(k)];
                raw_leverage += v[static_cast<std::size_t>(j)] * tmp;
            }
            agg_variance_influence[static_cast<std::size_t>(ci)] =
                std::fabs(raw_leverage) / (static_cast<double>(n) * p);

            // Fit influence (Cook's D): |v' fullSigma v| / (n^2 . p).
            double fit_influence = 0;
            for (int j = 0; j < p; ++j) {
                double tmp = 0;
                for (int k = 0; k < p; ++k)
                    tmp += full_sigma(j, k) * v[static_cast<std::size_t>(k)];
                fit_influence += v[static_cast<std::size_t>(j)] * tmp;
            }
            agg_fit_influence[static_cast<std::size_t>(ci)] =
                std::fabs(fit_influence) / (static_cast<double>(n) * n * p);
        }

        // Build ObservationLeverage array from the aggregated values (default metadata, since
        // the B17C DataComponent branch is omitted).
        std::vector<LD::ObservationLeverage> observations;
        observations.reserve(static_cast<std::size_t>(n_components));
        for (int i = 0; i < n_components; ++i) {
            std::size_t si = static_cast<std::size_t>(i);
            double leverage = agg_fit_influence[si] + agg_variance_influence[si];
            observations.emplace_back(i, leverage, 0.0, agg_fit_influence[si],
                                      agg_variance_influence[si], 0.0, 0.0, 0.0,
                                      bestfit::models::DataComponentType::Exact, 1, std::nullopt);
        }

        // 5/6. Penalty components omitted (B17C-coupled; see this method's header) -- empty.
        std::vector<LD::PriorComponentLeverage> penalty_components;

        return LD(std::move(observations), std::move(penalty_components), p);
    }

    // ------------------------------------------------------------------------------------
    // Optimization
    // ------------------------------------------------------------------------------------

    // Set default simulation options (C# 2183).
    void set_default_options() {
        set_optimizer_method(OptimizationMethod::BFGS);
        set_use_fallback_optimizer(true);
        set_estimation_strategy(GMMEstimationStrategy::Iterative);
        set_max_gmm_iterations(100);
        set_max_function_evaluations(2000);
        set_absolute_tolerance(1E-8);
        set_relative_tolerance(1E-8);
    }

    // ------------------------------------------------------------------------------------
    // Estimation
    // ------------------------------------------------------------------------------------

    // Estimates the model parameters using the configured GMM estimation strategy
    // (C# 2292). Throws when the problem is under-identified without a penalty function,
    // or when using one-step estimation on an over-identified problem.
    bool estimate() {
        is_estimated_ = false;

        // Validation.
        if (identification_status_ == GMMIdentificationStatus::UnderIdentified &&
            !penalty_function_)
            throw std::invalid_argument(
                "The GMM problem is under-identified and cannot be estimated without a penalty "
                "function.");
        if (identification_status_ == GMMIdentificationStatus::OverIdentified &&
            estimation_strategy_ == GMMEstimationStrategy::OneStep)
            throw std::invalid_argument(
                "The GMM problem is over-identified, so you cannot use the one-step estimation "
                "method. Use TwoStep or Iterative instead.");

        gmm_iterations_ = 0;
        total_function_evaluations_ = 0;
        convergence_history_.clear();

        // Set up the weighting matrix.
        if (!w_.has_value()) w_ = Matrix::identity(number_of_moment_conditions_);

        if (estimation_strategy_ == GMMEstimationStrategy::OneStep)
            estimate_one_step();
        else if (estimation_strategy_ == GMMEstimationStrategy::TwoStep)
            estimate_two_step();
        else if (estimation_strategy_ == GMMEstimationStrategy::Iterative)
            estimate_iterative();

        // Check if estimation succeeded (empty values = C# `Values == null || Length == 0`).
        if (!best_parameter_set_.values.empty()) {
            is_estimated_ = true;
            return true;
        }

        return false;
    }

    // Post-process estimation results to compute the model covariance and, optionally,
    // Hansen's J-statistic J = n g' V^-1 g ~ chi2(q - p) (C# 2336). The p-value is NaN
    // unless the model is over-identified (DoF > 0).
    void post_process(bool use_sandwich = true, bool compute_jstat = false) {
        // Compute covariance.
        sigma_ = get_covariance(best_parameter_set_.values, use_sandwich);
        if (!compute_jstat) return;

        // Compute the J-statistic. C# `gtMean * Vinv * gtMean` is Vector*Matrix (= Vinv *
        // gtMean, see Vector.Multiply(Matrix)) followed by an elementwise Vector*Vector;
        // `.Sum()` then yields g' V^-1 g.
        Vector gt_mean = get_g(best_parameter_set_.values);
        Matrix V = get_moment_residual_covariance(best_parameter_set_.values);
        Matrix v_inv = V.inverse();
        jstat_ = (v_inv * gt_mean).multiply(gt_mean).sum();

        // Compute the p-value (only valid if over-identified).
        bestfit::numerics::distributions::ChiSquared chi_squared(degree_of_freedom());
        jstat_pval_ = degree_of_freedom() > 0
                          ? 1.0 - chi_squared.cdf(jstat_)
                          : std::numeric_limits<double>::quiet_NaN();
    }

    // ------------------------------------------------------------------------------------
    // Utility Methods
    // ------------------------------------------------------------------------------------

    // Clear results (C# 2370).
    void clear_results() {
        is_estimated_ = false;
        status_ = OptimizationStatus::None;
        gmm_iterations_ = 0;
        total_function_evaluations_ = 0;
        jstat_ = std::numeric_limits<double>::quiet_NaN();
        jstat_pval_ = std::numeric_limits<double>::quiet_NaN();
        w_.reset();
        s_.reset();
        sigma_.reset();
        best_parameter_set_ = ParameterSet();
        convergence_history_.clear();
    }

    // Validates whether the GMM inputs are valid for estimation (C# 2390; the C# `out
    // List<string> errors` is an output reference parameter here).
    bool is_valid(std::vector<std::string>& errors) const {
        bool valid = true;
        errors.clear();

        if (identification_status_ == GMMIdentificationStatus::UnderIdentified &&
            !penalty_function_) {
            valid = false;
            errors.push_back("The GMM problem is under-identified and cannot be estimated.");
        }
        if (identification_status_ == GMMIdentificationStatus::OverIdentified &&
            estimation_strategy_ == GMMEstimationStrategy::OneStep) {
            valid = false;
            errors.push_back(
                "The GMM problem is over-identified, so you cannot use the one-step estimation "
                "method.");
        }
        if (max_gmm_iterations_ < 1 || max_gmm_iterations_ > 1000) {
            valid = false;
            errors.push_back("The GMM max iterations must be between 1 and 1,000.");
        }
        if (absolute_tolerance_ < 1E-15) {
            valid = false;
            errors.push_back("The absolute tolerance must be greater than 1E-15.");
        }
        if (relative_tolerance_ < 1E-15) {
            valid = false;
            errors.push_back("The relative tolerance must be greater than 1E-15.");
        }
        return valid;
    }

    // Clone the GMM configuration into a fresh, unestimated instance (C# 2427).
    GeneralizedMethodOfMoments clone() const {
        return GeneralizedMethodOfMoments(moment_condition_function_, number_of_parameters_,
                                          number_of_moment_conditions_, sample_size_,
                                          initial_values_, lower_bounds_, upper_bounds_, w_,
                                          jacobian_function_, penalty_function_,
                                          pointwise_moment_conditions_);
    }

   private:
    // Computes the identification status from parameter and moment condition counts
    // (C# 187).
    void compute_identification_status() {
        if (number_of_parameters_ > number_of_moment_conditions_)
            identification_status_ = GMMIdentificationStatus::UnderIdentified;
        else if (number_of_parameters_ == number_of_moment_conditions_)
            identification_status_ = GMMIdentificationStatus::JustIdentified;
        else
            identification_status_ = GMMIdentificationStatus::OverIdentified;
    }

    // Computes the gradient vector of the penalty function with respect to parameters:
    // a p x 1 vector, zero when there is no penalty (C# 641).
    Vector get_penalty_gradient(const std::vector<double>& parameters) const {
        if (!penalty_function_) return Vector(static_cast<int>(parameters.size()));

        std::vector<double> gradient = NumericalDiff::compute_gradient(
            [this](std::vector<double>& x) { return penalty_function_(x); }, parameters,
            lower_bounds_, upper_bounds_);
        return Vector(gradient);
    }

    // Computes the residual covariance matrix of the moment conditions after projection
    // (C# 863): V = S - D(D'S^-1 D)^-1 D' for two-step/iterative GMM, or
    // V = (I - P) S (I - P)' with P = D(D'D)^-1 D' for one-step; both divided by n. Feeds
    // the J-statistic. Inversion failures fall back to a zero matrix (silent guard).
    Matrix get_moment_residual_covariance(const std::vector<double>& parameters) {
        Matrix d = get_jacobian(parameters);
        Matrix dT = d.transpose();
        // C# `this.S!` null-forgiving dereference -> optional::value().
        const Matrix current_s = s_.value();

        try {
            if (estimation_strategy_ == GMMEstimationStrategy::TwoStep ||
                estimation_strategy_ == GMMEstimationStrategy::Iterative) {
                Matrix s_inv = current_s.inverse();
                Matrix middle = (dT * s_inv * d).inverse();
                Matrix projection = d * middle * dT;
                Matrix result = current_s - projection;
                return divide(result, static_cast<double>(sample_size_));
            } else {
                Matrix I = Matrix::identity(current_s.number_of_rows());
                Matrix middle = (dT * d).inverse();
                Matrix projection = d * middle * dT;
                Matrix residual = I - projection;
                Matrix result = residual * current_s * residual.transpose();
                return divide(result, static_cast<double>(sample_size_));
            }
        } catch (const std::exception&) {
            // C# logs "Matrix inversion failed in GetMomentResidualCovariance" via
            // Debug.WriteLine; silent no-throw guard here.
            return Matrix(current_s.number_of_rows(), current_s.number_of_rows());
        }
    }

    // --- Profile helpers ------------------------------------------------------------------

    // Computes the Q-space threshold increment for a significance level (C# 1208): under
    // efficient GMM with the half-quadratic convention, n * 2 * dQ ~ chi2(1), so
    // dQ = chi2_1(1 - alpha) / (2n).
    double profile_q_threshold_increment(double alpha) const {
        bestfit::numerics::distributions::ChiSquared chi_squared(1);
        return chi_squared.inverse_cdf(1 - alpha) / (2.0 * sample_size_);
    }

    // Evaluates the true profile Q at a fixed value of parameter i by re-optimizing the
    // nuisance parameters (C# 1220): BrentSearch for a 1-D reduced problem, NelderMead for
    // multi-D (both via the shared Phase 4 adapters -- see this file's header ADAPTER
    // NOTE). Optimizer failure falls back to evaluating at the current nuisance values.
    double profile_q_true_at_point(int param_index, double fixed_value) const {
        int p = number_of_parameters_;

        if (p == 1) return q({fixed_value});

        // Build the reduced-dimension objective: optimize over theta_{-i}.
        int p_reduced = p - 1;
        std::vector<double> init_reduced(static_cast<std::size_t>(p_reduced));
        std::vector<double> lb_reduced(static_cast<std::size_t>(p_reduced));
        std::vector<double> ub_reduced(static_cast<std::size_t>(p_reduced));

        int r = 0;
        for (int j = 0; j < p; ++j) {
            if (j == param_index) continue;
            std::size_t sr = static_cast<std::size_t>(r);
            std::size_t sj = static_cast<std::size_t>(j);
            init_reduced[sr] = best_parameter_set_.values[sj];
            lb_reduced[sr] = lower_bounds_[sj];
            ub_reduced[sr] = upper_bounds_[sj];
            ++r;
        }

        // Reduced objective: pack fixed_value into the full parameter vector.
        auto reduced_q = [this, p, param_index, fixed_value](const std::vector<double>& reduced_params) {
            std::vector<double> full_params(static_cast<std::size_t>(p));
            std::size_t ri = 0;
            for (int j = 0; j < p; ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                if (j == param_index)
                    full_params[sj] = fixed_value;
                else
                    full_params[sj] = reduced_params[ri++];
            }
            return q(full_params);
        };

        if (p_reduced == 1) {
            // Use Brent for the 1-D reduced optimization (C# BrentSearch with
            // ReportFailure = false).
            detail::BrentOptimizerAdapter brent(
                [&reduced_q](std::vector<double>& x) { return reduced_q(x); }, 1,
                lb_reduced[0], ub_reduced[0]);
            brent.absolute_tolerance = absolute_tolerance_;
            brent.relative_tolerance = relative_tolerance_;
            brent.report_failure = false;

            try {
                brent.minimize();
                if (brent.status() == OptimizationStatus::Success)
                    return brent.best_parameter_set().fitness;
            } catch (const std::exception&) {
                // C# logs "Profile Q Brent optimization failed" via Debug.WriteLine;
                // silent no-throw guard here.
            }

            // Fallback: evaluate at the current nuisance values.
            return reduced_q(init_reduced);
        } else {
            // Use NelderMead for the multi-dimensional reduced optimization. C# also sets
            // EnableStartPointProbe implicitly false here (not set); the ported NelderMead
            // has no probe regardless (see this file's header deviation note).
            detail::NelderMeadOptimizerAdapter nm(
                [&reduced_q](std::vector<double>& x) { return reduced_q(x); }, p_reduced,
                init_reduced, lb_reduced, ub_reduced);
            nm.max_function_evaluations = std::max(500, max_function_evaluations_ / 2);
            nm.absolute_tolerance = absolute_tolerance_;
            nm.relative_tolerance = relative_tolerance_;
            nm.report_failure = false;
            nm.record_traces = false;
            nm.compute_hessian = false;

            try {
                nm.minimize();
                if (nm.status() == OptimizationStatus::Success)
                    return nm.best_parameter_set().fitness;
            } catch (const std::exception&) {
                // C# logs "Profile Q NelderMead optimization failed" via Debug.WriteLine;
                // silent no-throw guard here.
            }

            // Fallback: evaluate at the current nuisance values.
            return reduced_q(init_reduced);
        }
    }

    // Evaluates the conditional profile Q at a fixed value of parameter i, keeping all
    // other parameters at their estimated values (C# 1318).
    double profile_q_conditional_at_point(int param_index, double fixed_value) const {
        std::vector<double> parms = best_parameter_set_.values;
        parms[static_cast<std::size_t>(param_index)] = fixed_value;
        return q(parms);
    }

    // Finds where the profile Q crosses a threshold on one side of the estimate, using
    // Brent root-finding (C# 1335). If the bound does not exceed the threshold the CI
    // extends to the bound; root-finding failure also falls back to the bound.
    double profile_q_find_bound(int param_index, double threshold, bool lower,
                                bool true_profile) const {
        std::size_t si = static_cast<std::size_t>(param_index);
        double theta_hat = best_parameter_set_.values[si];
        double lb = lower_bounds_[si];
        double ub = upper_bounds_[si];

        double search_lo, search_hi;
        if (lower) {
            search_lo = lb;
            search_hi = theta_hat;
        } else {
            search_lo = theta_hat;
            search_hi = ub;
        }

        // Check that the bound actually exceeds the threshold; if not, the CI extends to
        // the bound.
        double q_at_bound = true_profile
                                ? profile_q_true_at_point(param_index, lower ? search_lo : search_hi)
                                : profile_q_conditional_at_point(param_index,
                                                                 lower ? search_lo : search_hi);

        if (q_at_bound < threshold) {
            return lower ? lb : ub;
        }

        try {
            // C# Brent.Solve(..., tolerance: 1e-6, reportFailure: false); maxIterations
            // keeps the shared default (1000).
            return bestfit::numerics::math::rootfinding::solve(
                [&](double x) {
                    double q_val = true_profile ? profile_q_true_at_point(param_index, x)
                                                : profile_q_conditional_at_point(param_index, x);
                    return q_val - threshold;
                },
                search_lo, search_hi, /*tolerance=*/1e-6, /*max_iterations=*/1000,
                /*report_failure=*/false);
        } catch (const std::exception&) {
            // C# logs "Profile Q Brent root-finding failed" via Debug.WriteLine; silent
            // no-throw guard here.
            return lower ? lb : ub;
        }
    }

    // Computes the variance implied by a log-weighted density on a grid (C# 1135).
    static double compute_implied_variance(const std::vector<double>& theta,
                                           const std::vector<double>& logw) {
        int n = static_cast<int>(theta.size());

        // Shift for numerical stability.
        double max_lw = -std::numeric_limits<double>::infinity();
        for (int j = 0; j < n; ++j)
            if (logw[static_cast<std::size_t>(j)] > max_lw)
                max_lw = logw[static_cast<std::size_t>(j)];

        // Compute mean and variance via the trapezoidal rule.
        double sum_w = 0, sum_wx = 0, sum_wx2 = 0;
        for (int j = 1; j < n; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            double dx = theta[sj] - theta[sj - 1];
            double w0 = std::exp(logw[sj - 1] - max_lw);
            double w1 = std::exp(logw[sj] - max_lw);
            double w_avg = 0.5 * (w0 + w1);
            double x_avg = 0.5 * (theta[sj - 1] + theta[sj]);
            double x2_avg = 0.5 * (theta[sj - 1] * theta[sj - 1] + theta[sj] * theta[sj]);

            sum_w += w_avg * dx;
            sum_wx += w_avg * x_avg * dx;
            sum_wx2 += w_avg * x2_avg * dx;
        }

        if (sum_w <= 0) return 0;
        double mean = sum_wx / sum_w;
        double var = sum_wx2 / sum_w - mean * mean;
        return std::max(var, 0.0);
    }

    // Linearly interpolates the inverse CDF at a given probability level (C# 1174).
    static double interpolate_cdf_inverse(const std::vector<double>& theta,
                                          const std::vector<double>& cdf, double p,
                                          double fallback) {
        int n = static_cast<int>(theta.size());

        // If p is outside the CDF range, return the boundary.
        if (p <= cdf[0]) return theta[0];
        if (p >= cdf[static_cast<std::size_t>(n - 1)]) return theta[static_cast<std::size_t>(n - 1)];

        // Find the bracketing interval and interpolate.
        for (int j = 1; j < n; ++j) {
            std::size_t sj = static_cast<std::size_t>(j);
            if (cdf[sj] >= p) {
                double d_cdf = cdf[sj] - cdf[sj - 1];
                if (d_cdf < 1e-30) return theta[sj];
                double t = (p - cdf[sj - 1]) / d_cdf;
                return theta[sj - 1] + t * (theta[sj] - theta[sj - 1]);
            }
        }

        return fallback;
    }

    // --- Optimization helpers ---------------------------------------------------------------

    // Configures the optimizer based on the selected optimization method (C# 2068). All
    // six OptimizationMethod branches mirror the C# constructions (BFGS gets the analytic
    // get_gradient; MLSL gets LocalMethod::NelderMead explicitly since the C++ ctor default
    // is BFGS; the C# NelderMead branch's `EnableStartPointProbe = true` is a documented
    // no-op -- see this file's header deviation note).
    void set_up_optimizer() {
        Optimizer::Objective objective = [this](std::vector<double>& parms) { return q(parms); };

        if (optimizer_method_ == OptimizationMethod::Brent) {
            // C# `new BrentSearch(x => Q(new[] { x }), LowerBounds[0], UpperBounds[0])`;
            // the C# BrentSearch base is a 1-parameter optimizer.
            optimizer_ = std::make_unique<detail::BrentOptimizerAdapter>(
                objective, 1, lower_bounds_[0], upper_bounds_[0]);
        } else if (optimizer_method_ == OptimizationMethod::BFGS) {
            optimizer_ = std::make_unique<bestfit::numerics::math::optimization::BFGS>(
                objective, number_of_parameters_, initial_values_, lower_bounds_,
                upper_bounds_,
                [this](const std::vector<double>& x) { return get_gradient(x).to_array(); });
        } else if (optimizer_method_ == OptimizationMethod::NelderMead) {
            optimizer_ = std::make_unique<detail::NelderMeadOptimizerAdapter>(
                objective, number_of_parameters_, initial_values_, lower_bounds_,
                upper_bounds_);
            // C# `{ EnableStartPointProbe = true }`: no-op in this port (header note).
        } else if (optimizer_method_ == OptimizationMethod::Powell) {
            optimizer_ = std::make_unique<bestfit::numerics::math::optimization::Powell>(
                objective, number_of_parameters_, initial_values_, lower_bounds_,
                upper_bounds_);
        } else if (optimizer_method_ == OptimizationMethod::DifferentialEvolution) {
            optimizer_ =
                std::make_unique<bestfit::numerics::math::optimization::DifferentialEvolution>(
                    objective, number_of_parameters_, lower_bounds_, upper_bounds_);
        } else if (optimizer_method_ == OptimizationMethod::MultilevelSingleLinkage) {
            optimizer_ = std::make_unique<bestfit::numerics::math::optimization::MLSL>(
                objective, number_of_parameters_, initial_values_, lower_bounds_,
                upper_bounds_, bestfit::numerics::math::optimization::LocalMethod::NelderMead);
        }

        optimizer_->max_function_evaluations = max_function_evaluations_;
        optimizer_->absolute_tolerance = absolute_tolerance_;
        optimizer_->relative_tolerance = relative_tolerance_;
        optimizer_->report_failure = false;
        optimizer_->record_traces = false;
        optimizer_->compute_hessian = false;
    }

    // Runs the primary optimizer with the optional BFGS-to-NelderMead fallback (C# 2108).
    // enable_start_point_probe mirrors the C# parameter shape; it is a documented no-op in
    // this port (header deviation note).
    bool minimize_with_fallback(bool enable_start_point_probe = true) {
        // Run the primary optimizer.
        set_up_optimizer();
        // C# `if (Optimizer is NelderMead nm) nm.EnableStartPointProbe =
        // enableStartPointProbe;` -- no-op (no start-point probe in the ported NelderMead).
        (void)enable_start_point_probe;

        bool primary_succeeded = false;
        try {
            optimizer_->minimize();
            status_ = optimizer_->status();
            primary_succeeded = optimizer_->status() == OptimizationStatus::Success;
        } catch (const std::exception&) {
            status_ = OptimizationStatus::Failure;
            // C# logs "Primary optimizer threw exception" via Debug.WriteLine; silent
            // no-throw guard here.
        }

        total_function_evaluations_ += optimizer_->function_evaluations();

        if (primary_succeeded) return true;

        // Fallback only if enabled and the primary was BFGS.
        if (!use_fallback_optimizer_ || optimizer_method_ != OptimizationMethod::BFGS)
            return false;

        // C# logs "BFGS failed, falling back to NelderMead with start-point probe."

        // Use the BFGS best point if it found anything reasonable, otherwise InitialValues.
        std::vector<double> fallback_initials =
            (!optimizer_->best_parameter_set().values.empty() &&
             bestfit::numerics::is_finite(optimizer_->best_parameter_set().fitness) &&
             optimizer_->best_parameter_set().fitness < std::numeric_limits<double>::max())
                ? optimizer_->best_parameter_set().values
                : initial_values_;

        auto fallback = std::make_unique<detail::NelderMeadOptimizerAdapter>(
            [this](std::vector<double>& parms) { return q(parms); }, number_of_parameters_,
            fallback_initials, lower_bounds_, upper_bounds_);
        // C# `{ EnableStartPointProbe = enableStartPointProbe, ... }` -- probe is a no-op.
        fallback->max_function_evaluations = max_function_evaluations_;
        fallback->absolute_tolerance = absolute_tolerance_;
        fallback->relative_tolerance = relative_tolerance_;
        fallback->report_failure = false;
        fallback->record_traces = false;
        fallback->compute_hessian = false;

        try {
            fallback->minimize();
            status_ = fallback->status();
        } catch (const std::exception&) {
            status_ = OptimizationStatus::Failure;
            // C# logs "NelderMead fallback threw exception" via Debug.WriteLine; silent
            // no-throw guard here.
            return false;
        }

        total_function_evaluations_ += fallback->function_evaluations();

        if (fallback->status() == OptimizationStatus::Success) {
            optimizer_ = std::move(fallback);
            return true;
        }

        return false;
    }

    // --- Estimation strategies --------------------------------------------------------------

    // Performs the one-step optimization (C# 2201).
    void estimate_one_step() {
        if (!minimize_with_fallback()) return;
        best_parameter_set_ = optimizer_->best_parameter_set().clone();
    }

    // Performs the two-step optimization using the updated weighting matrix (C# 2211).
    void estimate_two_step() {
        gmm_iterations_ = 0;
        total_function_evaluations_ = 0;

        // Perform the 1st optimization step.
        if (!minimize_with_fallback()) return;

        // Update results.
        best_parameter_set_ = optimizer_->best_parameter_set().clone();
        s_ = get_s(best_parameter_set_.values);
        w_ = s_->inverse();

        // Perform the 2nd optimization step.
        initial_values_ = optimizer_->best_parameter_set().values;
        if (!minimize_with_fallback()) return;
        best_parameter_set_ = optimizer_->best_parameter_set().clone();
    }

    // Performs the 'iterative method' that improves the weighting matrix until parameter
    // convergence (C# 2235). NOTE (C# fidelity): a loop that exhausts max_gmm_iterations
    // leaves gmm_iterations_ == max_gmm_iterations_ + 1 (the for-loop increment), which is
    // what makes converged_within_tolerance() false for a best-effort run.
    void estimate_iterative() {
        convergence_history_.clear();

        // Perform the 1st optimization step.
        if (!minimize_with_fallback(/*enable_start_point_probe=*/true)) return;

        // Update results.
        best_parameter_set_ = optimizer_->best_parameter_set().clone();
        s_ = get_s(best_parameter_set_.values);
        w_ = s_->inverse();

        initial_values_ = best_parameter_set_.values;
        std::vector<double> old_values = best_parameter_set_.values;
        double old_q = q(old_values);
        convergence_history_.push_back(old_q);

        for (gmm_iterations_ = 2; gmm_iterations_ <= max_gmm_iterations_; ++gmm_iterations_) {
            // Subsequent iterations: no start-point probe needed.
            if (!minimize_with_fallback(/*enable_start_point_probe=*/false)) break;

            // Update results.
            best_parameter_set_ = optimizer_->best_parameter_set().clone();
            std::vector<double> new_values = best_parameter_set_.values;
            s_ = get_s(best_parameter_set_.values);
            w_ = s_->inverse();

            double new_q = q(new_values);
            convergence_history_.push_back(new_q);

            // Check convergence: absolute parameter distance OR relative objective change.
            double distance = bestfit::numerics::distance(new_values, old_values);
            double rel_change = std::fabs(new_q - old_q) / (std::fabs(old_q) + 1e-15);

            if (distance < absolute_tolerance_ || rel_change < relative_tolerance_) {
                best_parameter_set_ = optimizer_->best_parameter_set().clone();
                return;
            }

            initial_values_ = new_values;
            old_values = new_values;
            old_q = new_q;
        }
    }

    // Elementwise scalar division, standing in for the C# `Matrix / double` operator
    // (`covariance /= (double)SampleSize` etc.), which the ported Matrix omits.
    static Matrix divide(const Matrix& m, double scalar) {
        Matrix result(m.number_of_rows(), m.number_of_columns());
        for (int i = 0; i < m.number_of_rows(); ++i)
            for (int j = 0; j < m.number_of_columns(); ++j) result(i, j) = m(i, j) / scalar;
        return result;
    }

    // --- Fields (C# Members region backing fields + auto-properties) --------------------------

    bestfit::models::IGMMModel* model_ = nullptr;  // C# `IGMMModel? Model` (241; non-owning, B9)
    MomentConditionFunction moment_condition_function_;
    JacobianFunction jacobian_function_;
    PenaltyFunction penalty_function_;
    PointwiseMomentConditionFunction pointwise_moment_conditions_;

    int number_of_parameters_ = 0;
    int number_of_moment_conditions_ = 0;
    int sample_size_ = 0;
    GMMIdentificationStatus identification_status_ = GMMIdentificationStatus::JustIdentified;

    std::vector<double> initial_values_;
    std::vector<double> lower_bounds_;
    std::vector<double> upper_bounds_;

    OptimizationMethod optimizer_method_ = OptimizationMethod::BFGS;
    GMMEstimationStrategy estimation_strategy_ = GMMEstimationStrategy::Iterative;
    int max_gmm_iterations_ = 100;
    double absolute_tolerance_ = 1E-8;
    double relative_tolerance_ = 1E-8;
    int max_function_evaluations_ = 2000;
    bool use_fallback_optimizer_ = true;

    std::unique_ptr<Optimizer> optimizer_;  // transient; null before estimation
    OptimizationStatus status_ = OptimizationStatus::None;
    bool is_estimated_ = false;

    std::optional<Matrix> s_;
    std::optional<Matrix> w_;
    std::optional<Matrix> sigma_;
    ParameterSet best_parameter_set_;
    double jstat_ = 0.0;
    double jstat_pval_ = 0.0;
    int gmm_iterations_ = 0;
    int total_function_evaluations_ = 0;
    std::vector<double> convergence_history_;
};

}  // namespace bestfit::estimation
