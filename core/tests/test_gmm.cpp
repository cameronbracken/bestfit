// Standalone tests for bestfit::estimation::GeneralizedMethodOfMoments (Phase 6, Task B8).
//
// PART 1 transcribes BOTH structural upstream test files, values and tolerances unaltered:
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/ModelEstimation/GeneralizedMethodOfMomentsTests.cs
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/ModelEstimation/GeneralizedMethodOfMomentsExpandedTests.cs
// Those files are STRUCTURAL ONLY (no Estimate() numeric oracles -- real estimation parity
// lives in the absent RMC.BestFit.Verification project). Skipped upstream methods (see the
// task report for reasons):
//   - Test_Constructor_NullModel_Throws (IGMMModel ctor arrives in B9)
//   - MaxGMMIterations_SetSameValue_DoesNotRaisePropertyChanged and
//     RelativeTolerance_SetNewValue_RaisesPropertyChanged (INPC/PropertyChanged not ported;
//     the properties are plain accessors here)
//
// PART 2 is a SUPPLEMENT the upstream suite lacks: a delegate-based just-identified toy GMM
// problem with a closed-form solution, exercising Estimate() (all strategies), the covariance
// stack, profile confidence intervals, the Q conventions, and the newly ported support surface
// (NumericalDiff::compute_gradient/compute_jacobian, EigenValueDecomposition,
// MatrixRegularization::regularize, tools distance).
//
// TOY PROBLEM DERIVATION (documented per the brief):
//   Moment conditions g(theta) = (theta_1 - c1, theta_2 - c2) with constants c1 = 5, c2 = 3
//   (the "match mean/variance of a known dataset" moments in canonical form: for data with
//   sample mean c1 and sample second moment c2, the just-identified GMM estimator solves
//   g(theta) = 0 exactly). The moment-condition covariance is supplied as the fixed SPD matrix
//       S = [[0.5, 0.1], [0.1, 0.3]],   sample size n = 25.
//   Analytic results:
//     - Optimum: g(theta*) = 0  =>  theta* = (c1, c2) = (5, 3), for ANY weighting matrix W,
//       so OneStep (W = I), TwoStep, and Iterative (W = S^-1) all share the same optimum.
//     - Jacobian: D = dg/dtheta = I (supplied analytically for the covariance tests).
//     - Covariance (no penalty, W = S^-1): bread = D'WD = S^-1, meat = D'WSWD = S^-1, so
//       Sigma = bread^-1 * meat * bread^-1 / n = S * S^-1 * S / n = S / n  (sandwich), and the
//       non-sandwich form bread^-1 / n = S / n is identical for a just-identified problem.
//     - Standard errors: se_i = sqrt(S_ii / n); correlation = S_12 / sqrt(S_11 S_22).
//     - Profile CI (no penalty => Q = g'Wg, NO half factor; threshold increment is
//       chi2_1(1-alpha) / (2n) per the C# half-quadratic threshold convention): the true
//       profile of the quadratic form over theta_{-i} partials out to
//       Q_prof(theta_i) = (theta_i - c_i)^2 / S_ii (Schur complement identity for W = S^-1),
//       so the CI half-width is delta_i = sqrt(chi2_1(1-alpha) * S_ii / (2n)).
//       The conditional profile fixes theta_{-i} at the optimum:
//       Q_cond(theta_i) = (theta_i - c_i)^2 * W_ii, so delta_i = sqrt(chi2/(2n)/W_ii).
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/estimation/generalized_method_of_moments.hpp"
#include "bestfit/estimation/gmm_delegates.hpp"
#include "bestfit/estimation/numerical_diff.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/numerics/distributions/chi_squared.hpp"
#include "bestfit/numerics/math/linalg/eigenvalue_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/matrix_regularization.hpp"
#include "bestfit/numerics/tools.hpp"
#include "check.hpp"

using bestfit::estimation::GeneralizedMethodOfMoments;
using bestfit::estimation::MomentConditionResult;
using bestfit::estimation::NumericalDiff;
using bestfit::estimation::OptimizationMethod;
using bestfit::numerics::distributions::ChiSquared;
using bestfit::numerics::math::linalg::EigenValueDecomposition;
using bestfit::numerics::math::linalg::Matrix;
using bestfit::numerics::math::linalg::Matrix2D;
using bestfit::numerics::math::linalg::MatrixRegularization;
using bestfit::numerics::math::linalg::Vector;
using OptimizationStatus = bestfit::numerics::math::optimization::OptimizationStatus;
using GMMIdentificationStatus = GeneralizedMethodOfMoments::GMMIdentificationStatus;
using GMMEstimationStrategy = GeneralizedMethodOfMoments::GMMEstimationStrategy;

namespace {

// ======================================================================================
// PART 1 -- upstream structural test transcriptions
// ======================================================================================

// Trivial moment-condition function used by the delegate-based constructor tests. Returns a
// zero G vector and zero S matrix sized to the requested counts (upstream StubMomentConditions).
MomentConditionResult stub_moment_conditions(const std::vector<double>& parameters) {
    int len = static_cast<int>(parameters.size());
    return {Vector(len), Matrix(len, len)};
}

// Helper that builds a stub GMM instance with the requested (parameters, moments) shape
// (upstream MakeStubGmm: init 0.5, bounds [0, 1], sample size 100).
GeneralizedMethodOfMoments make_stub_gmm(int parameters, int moments) {
    std::vector<double> init(static_cast<std::size_t>(parameters), 0.5);
    std::vector<double> lower(static_cast<std::size_t>(parameters), 0.0);
    std::vector<double> upper(static_cast<std::size_t>(parameters), 1.0);
    return GeneralizedMethodOfMoments(stub_moment_conditions, parameters, moments, 100, init,
                                      lower, upper);
}

// upstream Test_DelegateConstructor_PopulatesShape_AndUnestimatedState
void test_delegate_constructor_populates_shape_and_unestimated_state() {
    std::vector<double> init{0.5, 1.0};
    std::vector<double> lower{0.0, 0.0};
    std::vector<double> upper{1.0, 2.0};

    GeneralizedMethodOfMoments gmm(stub_moment_conditions, 2, 3, 100, init, lower, upper);

    CHECK_EQ(gmm.number_of_parameters(), 2);
    CHECK_EQ(gmm.number_of_moment_conditions(), 3);
    CHECK_EQ(gmm.sample_size(), 100);
    CHECK_TRUE(gmm.initial_values() == init);
    CHECK_TRUE(gmm.lower_bounds() == lower);
    CHECK_TRUE(gmm.upper_bounds() == upper);
    CHECK_TRUE(!gmm.is_estimated());
    CHECK_TRUE(gmm.status() == OptimizationStatus::None);
}

// upstream Test_DelegateConstructor_NullMomentConditionFunction_Throws
void test_delegate_constructor_null_moment_condition_function_throws() {
    CHECK_THROWS(GeneralizedMethodOfMoments(nullptr, 1, 1, 10, {0.0}, {0.0}, {1.0}));
}

// upstream Test_DelegateConstructor_MismatchedArrayLength_Throws
void test_delegate_constructor_mismatched_array_length_throws() {
    // initial values length 1, but numberOfParameters = 2
    CHECK_THROWS(GeneralizedMethodOfMoments(stub_moment_conditions, 2, 2, 10, {0.0},
                                            {0.0, 0.0}, {1.0, 1.0}));
}

// upstream Test_DelegateConstructor_UpperLessThanLower_Throws
void test_delegate_constructor_upper_less_than_lower_throws() {
    CHECK_THROWS(GeneralizedMethodOfMoments(stub_moment_conditions, 1, 1, 10, {0.5}, {1.0},
                                            {0.0}));  // upper < lower
}

// upstream Test_DelegateConstructor_InitialOutsideBounds_Throws
void test_delegate_constructor_initial_outside_bounds_throws() {
    CHECK_THROWS(GeneralizedMethodOfMoments(stub_moment_conditions, 1, 1, 10, {5.0}, {0.0},
                                            {1.0}));  // 5.0 outside [0, 1]
}

// upstream Test_IdentificationStatus_ParametersEqualMoments_IsJustIdentified
// upstream Test_IdentificationStatus_MomentsGreaterThanParameters_IsOverIdentified
// upstream Test_IdentificationStatus_MomentsLessThanParameters_IsUnderIdentified
void test_identification_status() {
    CHECK_TRUE(make_stub_gmm(2, 2).identification_status() ==
               GMMIdentificationStatus::JustIdentified);
    CHECK_TRUE(make_stub_gmm(2, 4).identification_status() ==
               GMMIdentificationStatus::OverIdentified);
    CHECK_TRUE(make_stub_gmm(3, 2).identification_status() ==
               GMMIdentificationStatus::UnderIdentified);
}

// upstream Test_OptimizerMethod_ReflectsConstructorChoice (C# object-initializer syntax sets
// the property right after construction; ported as a setter call)
void test_optimizer_method_reflects_constructor_choice() {
    auto gmm = make_stub_gmm(1, 1);
    gmm.set_optimizer_method(OptimizationMethod::NelderMead);
    CHECK_TRUE(gmm.optimizer_method() == OptimizationMethod::NelderMead);
}

// upstream Test_MaxGMMIterations_RoundTrips
void test_max_gmm_iterations_round_trips() {
    auto gmm = make_stub_gmm(1, 1);
    gmm.set_max_gmm_iterations(25);
    CHECK_EQ(gmm.max_gmm_iterations(), 25);
}

// upstream Test_ObjectiveFunctionValue_BeforeEstimation_IsNaN
void test_objective_function_value_before_estimation_is_nan() {
    auto gmm = make_stub_gmm(1, 1);
    CHECK_TRUE(std::isnan(gmm.objective_function_value()));
}

// upstream (Expanded) UseFallbackOptimizer_SetterRoundTrips
void test_use_fallback_optimizer_setter_round_trips() {
    auto gmm = make_stub_gmm(2, 2);
    gmm.set_use_fallback_optimizer(false);
    CHECK_TRUE(!gmm.use_fallback_optimizer());
    gmm.set_use_fallback_optimizer(true);
    CHECK_TRUE(gmm.use_fallback_optimizer());
}

// upstream (Expanded) EstimationStrategy_SetterRoundTrips_ForEveryValue
void test_estimation_strategy_setter_round_trips_for_every_value() {
    auto gmm = make_stub_gmm(2, 2);
    for (auto s : {GMMEstimationStrategy::OneStep, GMMEstimationStrategy::TwoStep,
                   GMMEstimationStrategy::Iterative}) {
        gmm.set_estimation_strategy(s);
        CHECK_TRUE(gmm.estimation_strategy() == s);
    }
}

// upstream (Expanded) MaxFunctionEvaluations_SetterRoundTrips
void test_max_function_evaluations_setter_round_trips() {
    auto gmm = make_stub_gmm(2, 2);
    gmm.set_max_function_evaluations(5000);
    CHECK_EQ(gmm.max_function_evaluations(), 5000);
}

// upstream (Expanded) AbsoluteTolerance_SetterRoundTrips (upstream tolerance 1e-15)
void test_absolute_tolerance_setter_round_trips() {
    auto gmm = make_stub_gmm(2, 2);
    gmm.set_absolute_tolerance(1e-9);
    CHECK_NEAR(gmm.absolute_tolerance(), 1e-9, 1e-15);
}

// upstream (Expanded) RelativeTolerance_SetterRoundTrips (upstream tolerance 1e-15)
void test_relative_tolerance_setter_round_trips() {
    auto gmm = make_stub_gmm(2, 2);
    gmm.set_relative_tolerance(1e-7);
    CHECK_NEAR(gmm.relative_tolerance(), 1e-7, 1e-15);
}

// upstream (Expanded) IsEstimated_BeforeEstimation_IsFalse
void test_is_estimated_before_estimation_is_false() {
    CHECK_TRUE(!make_stub_gmm(2, 2).is_estimated());
}

// upstream (Expanded) JStat_BeforeEstimation_IsZero (JStat is a default double = 0; the
// meaningful invariant is on IsEstimated)
void test_jstat_before_estimation_is_zero() {
    auto gmm = make_stub_gmm(2, 2);
    CHECK_TRUE(!gmm.is_estimated());
    CHECK_EQ(gmm.jstat(), 0.0);
}

// upstream (Expanded) S_W_Sigma_BeforeEstimation_AreNull (C# null -> empty std::optional)
void test_s_w_sigma_before_estimation_are_null() {
    auto gmm = make_stub_gmm(2, 2);
    CHECK_TRUE(!gmm.s().has_value());
    CHECK_TRUE(!gmm.w().has_value());
    CHECK_TRUE(!gmm.sigma().has_value());
}

// upstream (Expanded) TotalFunctionEvaluations_BeforeEstimation_IsZero
void test_total_function_evaluations_before_estimation_is_zero() {
    CHECK_EQ(make_stub_gmm(2, 2).total_function_evaluations(), 0);
}

// upstream (Expanded) GMMIterations_BeforeEstimation_IsZero
void test_gmm_iterations_before_estimation_is_zero() {
    CHECK_EQ(make_stub_gmm(2, 2).gmm_iterations(), 0);
}

// upstream (Expanded) ConvergedWithinTolerance_BeforeEstimation_IsFalse
void test_converged_within_tolerance_before_estimation_is_false() {
    CHECK_TRUE(!make_stub_gmm(2, 2).converged_within_tolerance());
}

// upstream (Expanded) ConvergenceHistory_BeforeEstimation_IsEmpty
void test_convergence_history_before_estimation_is_empty() {
    CHECK_TRUE(make_stub_gmm(2, 2).convergence_history().empty());
}

// upstream (Expanded) BestParameterSet_BeforeEstimation_IsNotNull (C# asserts a non-null
// deterministic empty ParameterSet; here ParameterSet is a value type whose "unset" state is
// an empty values vector -- see parameter_set.hpp's file header)
void test_best_parameter_set_before_estimation_is_deterministic_empty() {
    auto gmm = make_stub_gmm(2, 2);
    CHECK_TRUE(gmm.best_parameter_set().values.empty());
    CHECK_EQ(gmm.best_parameter_set().fitness, 0.0);
}

// upstream (Expanded) DegreeOfFreedom_OverIdentified_EqualsMomentsMinusParameters,
// DegreeOfFreedom_JustIdentified_IsZero, DegreeOfFreedom_UnderIdentified_IsZero
void test_degree_of_freedom() {
    CHECK_EQ(make_stub_gmm(2, 5).degree_of_freedom(), 3);
    CHECK_EQ(make_stub_gmm(2, 2).degree_of_freedom(), 0);
    CHECK_EQ(make_stub_gmm(4, 2).degree_of_freedom(), 0);
}

// upstream (Expanded) OptimizerMethod_TransitionsThroughEveryValue
void test_optimizer_method_transitions_through_every_value() {
    auto gmm = make_stub_gmm(2, 2);
    for (auto method : {OptimizationMethod::BFGS, OptimizationMethod::NelderMead,
                        OptimizationMethod::Powell, OptimizationMethod::MultilevelSingleLinkage,
                        OptimizationMethod::DifferentialEvolution}) {
        gmm.set_optimizer_method(method);
        CHECK_TRUE(gmm.optimizer_method() == method);
    }
}

// ======================================================================================
// PART 2 -- SUPPLEMENT: closed-form toy GMM problem + newly ported support surface
// (see the file-header derivation)
// ======================================================================================

constexpr double kC1 = 5.0;
constexpr double kC2 = 3.0;
constexpr int kToyN = 25;

Matrix toy_s() {
    Matrix s(2, 2);
    s(0, 0) = 0.5;
    s(0, 1) = 0.1;
    s(1, 0) = 0.1;
    s(1, 1) = 0.3;
    return s;
}

MomentConditionResult toy_moments(const std::vector<double>& theta) {
    Vector g(2);
    g[0] = theta[0] - kC1;
    g[1] = theta[1] - kC2;
    return {g, toy_s()};
}

Matrix2D identity_jacobian(const std::vector<double>& /*theta*/) {
    return {{1.0, 0.0}, {0.0, 1.0}};
}

// Toy GMM with the analytic identity Jacobian supplied (exact D = I in the covariance stack).
GeneralizedMethodOfMoments make_toy_gmm_analytic_jacobian() {
    return GeneralizedMethodOfMoments(toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0},
                                      {10.0, 10.0}, std::nullopt, identity_jacobian);
}

// Toy GMM relying on the numerical Jacobian path.
GeneralizedMethodOfMoments make_toy_gmm_numerical_jacobian() {
    return GeneralizedMethodOfMoments(toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0},
                                      {10.0, 10.0});
}

void test_toy_estimate_iterative_recovers_analytic_optimum() {
    auto gmm = make_toy_gmm_analytic_jacobian();

    CHECK_TRUE(gmm.estimate());
    CHECK_TRUE(gmm.is_estimated());
    CHECK_TRUE(gmm.status() == OptimizationStatus::Success);
    // Optimizer-dependent point estimates: rel 1e-6 (brief's 1e-5..1e-8 band).
    CHECK_NEAR(gmm.best_parameter_set().values[0], kC1, 5e-6);
    CHECK_NEAR(gmm.best_parameter_set().values[1], kC2, 5e-6);
    // Iterative bookkeeping: W update (I -> S^-1) does not move a just-identified optimum,
    // so the loop converges at its first re-optimization (GMMIterations == 2).
    CHECK_EQ(gmm.gmm_iterations(), 2);
    CHECK_TRUE(gmm.converged_within_tolerance());
    CHECK_EQ(static_cast<int>(gmm.convergence_history().size()), 2);
    CHECK_TRUE(gmm.total_function_evaluations() > 0);
    CHECK_TRUE(gmm.w().has_value());
    CHECK_TRUE(gmm.s().has_value());
    // Q at the optimum is ~0 (g(theta*) = 0).
    CHECK_NEAR(gmm.objective_function_value(), 0.0, 1e-12);
}

void test_toy_covariance_stack_matches_analytic() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_TRUE(gmm.estimate());
    gmm.post_process();  // sandwich covariance, no J-stat

    // Sigma = S / n (derivation in the file header).
    const Matrix s = toy_s();
    CHECK_TRUE(gmm.sigma().has_value());
    const Matrix& sigma = *gmm.sigma();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(sigma(i, j), s(i, j) / kToyN, 1e-8);

    // Standard errors: sqrt(S_ii / n).
    std::vector<double> se = gmm.get_standard_errors();
    CHECK_NEAR(se[0], std::sqrt(s(0, 0) / kToyN), 1e-7);
    CHECK_NEAR(se[1], std::sqrt(s(1, 1) / kToyN), 1e-7);

    // Correlation: S_12 / sqrt(S_11 * S_22).
    Matrix corr = gmm.get_correlation_matrix();
    CHECK_NEAR(corr(0, 0), 1.0, 1e-9);
    CHECK_NEAR(corr(1, 1), 1.0, 1e-9);
    CHECK_NEAR(corr(0, 1), s(0, 1) / std::sqrt(s(0, 0) * s(1, 1)), 1e-7);

    // Sandwich covariance and robust SEs coincide with the efficient ones for a
    // just-identified problem (meat == bread == S^-1).
    Matrix sandwich = gmm.get_sandwich_covariance_matrix();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(sandwich(i, j), s(i, j) / kToyN, 1e-8);
    std::vector<double> robust = gmm.get_robust_standard_errors();
    CHECK_NEAR(robust[0], se[0], 1e-9);
    CHECK_NEAR(robust[1], se[1], 1e-9);

    // Non-sandwich form (bread^-1 / n) is also S / n here. GetCovarianceMatrix would return
    // the cached Sigma (C# `Sigma ?? GetCovariance(...)`), so call get_covariance directly.
    Matrix nonsandwich = gmm.get_covariance(gmm.best_parameter_set().values, false);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(nonsandwich(i, j), s(i, j) / kToyN, 1e-8);
}

void test_toy_profile_confidence_intervals_match_analytic() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_TRUE(gmm.estimate());
    gmm.post_process();

    const Matrix s = toy_s();
    double chi2 = ChiSquared(1).inverse_cdf(0.9);  // alpha = 0.1

    // True profile: delta_i = sqrt(chi2 * S_ii / (2n)).
    Matrix cis = gmm.profile_confidence_intervals(0.1, true);
    for (int i = 0; i < 2; ++i) {
        double c = i == 0 ? kC1 : kC2;
        double delta = std::sqrt(chi2 * s(i, i) / (2.0 * kToyN));
        CHECK_NEAR(cis(i, 0), c - delta, 1e-4);
        CHECK_NEAR(cis(i, 1), c + delta, 1e-4);
    }

    // Conditional profile: delta_i = sqrt(chi2 / (2n) / W_ii), with W read off the estimator
    // (W = S^-1 after iterative estimation, up to the SPD regularization ridge).
    Matrix cond = gmm.profile_confidence_intervals(0.1, false);
    for (int i = 0; i < 2; ++i) {
        double c = i == 0 ? kC1 : kC2;
        double delta = std::sqrt(chi2 / (2.0 * kToyN) / (*gmm.w())(i, i));
        CHECK_NEAR(cond(i, 0), c - delta, 1e-4);
        CHECK_NEAR(cond(i, 1), c + delta, 1e-4);
    }

    // Guards: alpha outside (0, 1) throws; unestimated estimator throws.
    CHECK_THROWS(gmm.profile_confidence_intervals(1.5));
    auto fresh = make_toy_gmm_analytic_jacobian();
    CHECK_THROWS(fresh.profile_confidence_intervals(0.1));
}

void test_toy_profile_percentiles_shape_and_symmetry() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_TRUE(gmm.estimate());
    gmm.post_process();

    Matrix pct = gmm.profile_percentiles();  // default {0.05, 0.25, 0.50, 0.75, 0.95}
    CHECK_EQ(pct.number_of_rows(), 2);
    CHECK_EQ(pct.number_of_columns(), 5);
    for (int i = 0; i < 2; ++i) {
        double c = i == 0 ? kC1 : kC2;
        // Symmetric quadratic profile: the median tracks the point estimate (grid-resolution
        // limited: bins = 200 over [0, 10] is a 0.05 step).
        CHECK_NEAR(pct(i, 2), c, 0.05);
        for (int k = 1; k < 5; ++k) CHECK_TRUE(pct(i, k) > pct(i, k - 1));
    }
    CHECK_THROWS(gmm.profile_percentiles({}, true, 5));    // bins < 10
    CHECK_THROWS(gmm.profile_percentiles({0.5, 1.5}));     // percentile outside (0, 1)
}

void test_toy_profile_q_shape() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_TRUE(gmm.estimate());
    std::vector<Matrix> profiles = gmm.profile_q(20, false);
    CHECK_EQ(static_cast<int>(profiles.size()), 2);
    CHECK_EQ(profiles[0].number_of_rows(), 20);
    CHECK_EQ(profiles[0].number_of_columns(), 2);
    // The conditional profile at a grid point reproduces Q directly.
    std::vector<double> parms = gmm.best_parameter_set().values;
    parms[0] = profiles[0](0, 0);
    CHECK_NEAR(profiles[0](0, 1), gmm.q(parms), 1e-12);
    CHECK_THROWS(gmm.profile_q(1));  // bins < 2
}

void test_q_conventions_with_and_without_penalty() {
    // W = I supplied through the ctor's initialW so Q is computable pre-estimation.
    GeneralizedMethodOfMoments gmm(toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0},
                                   {10.0, 10.0}, Matrix::identity(2));
    // Without a penalty: Q = g'Wg. At theta = (6, 3): g = (1, 0) so Q = 1.
    CHECK_NEAR(gmm.q({6.0, 3.0}), 1.0, 1e-12);

    // With a penalty: Q = (1/2) g'Wg + penalty (half-quadratic convention).
    GeneralizedMethodOfMoments pen(
        toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0}, {10.0, 10.0}, Matrix::identity(2),
        nullptr, [](const std::vector<double>&) { return 0.7; });
    CHECK_NEAR(pen.q({6.0, 3.0}), 0.5 * 1.0 + 0.7, 1e-12);
}

void test_q_throws_when_w_is_null() {
    auto gmm = make_toy_gmm_analytic_jacobian();  // no initialW, not estimated -> W null
    CHECK_THROWS(gmm.q({5.0, 3.0}));
}

void test_q_nonfinite_returns_double_max() {
    auto nan_moments = [](const std::vector<double>& theta) -> MomentConditionResult {
        Vector g(static_cast<int>(theta.size()));
        g[0] = std::numeric_limits<double>::quiet_NaN();
        return {g, Matrix(static_cast<int>(theta.size()), static_cast<int>(theta.size()))};
    };
    GeneralizedMethodOfMoments gmm(nan_moments, 2, 2, 10, {0.5, 0.5}, {0.0, 0.0}, {1.0, 1.0},
                                   Matrix::identity(2));
    CHECK_EQ(gmm.q({0.5, 0.5}), std::numeric_limits<double>::max());
}

void test_get_g_and_get_s() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    Vector g = gmm.get_g({6.0, 5.0});
    CHECK_NEAR(g[0], 1.0, 1e-15);
    CHECK_NEAR(g[1], 2.0, 1e-15);
    // GetS regularizes to SPD; the toy S is already SPD so only a tiny ridge is added.
    Matrix s = gmm.get_s({6.0, 5.0});
    const Matrix expected = toy_s();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(s(i, j), expected(i, j), 1e-8);
}

void test_get_jacobian_numerical_matches_analytic() {
    auto gmm = make_toy_gmm_numerical_jacobian();
    Matrix J = gmm.get_jacobian({5.0, 3.0});
    CHECK_NEAR(J(0, 0), 1.0, 1e-9);
    CHECK_NEAR(J(0, 1), 0.0, 1e-9);
    CHECK_NEAR(J(1, 0), 0.0, 1e-9);
    CHECK_NEAR(J(1, 1), 1.0, 1e-9);
}

void test_get_gradient_matches_analytic() {
    // W = I: gradient = D'Wg = g for D = I. At theta = (6, 5): g = (1, 2).
    GeneralizedMethodOfMoments gmm(toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0},
                                   {10.0, 10.0}, Matrix::identity(2), identity_jacobian);
    Vector grad = gmm.get_gradient({6.0, 5.0});
    CHECK_NEAR(grad[0], 1.0, 1e-10);
    CHECK_NEAR(grad[1], 2.0, 1e-10);

    // With a quadratic penalty p(theta) = (theta_1 - 5)^2 the gradient gains (2(theta_1-5), 0).
    GeneralizedMethodOfMoments pen(
        toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0}, {10.0, 10.0}, Matrix::identity(2),
        identity_jacobian,
        [](const std::vector<double>& t) { return (t[0] - 5.0) * (t[0] - 5.0); });
    Vector pgrad = pen.get_gradient({6.0, 3.0});
    CHECK_NEAR(pgrad[0], 1.0 + 2.0, 1e-6);
    CHECK_NEAR(pgrad[1], 0.0, 1e-6);
}

void test_get_penalty_hessian() {
    GeneralizedMethodOfMoments pen(
        toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0}, {10.0, 10.0}, std::nullopt, nullptr,
        [](const std::vector<double>& t) { return (t[0] - 5.0) * (t[0] - 5.0); });
    Matrix h = pen.get_penalty_hessian({5.0, 3.0});
    CHECK_NEAR(h(0, 0), 2.0, 1e-6);
    CHECK_NEAR(h(0, 1), 0.0, 1e-6);
    CHECK_NEAR(h(1, 0), 0.0, 1e-6);
    CHECK_NEAR(h(1, 1), 0.0, 1e-6);

    // No penalty -> zero matrix.
    auto plain = make_toy_gmm_analytic_jacobian();
    Matrix z = plain.get_penalty_hessian({5.0, 3.0});
    CHECK_EQ(z(0, 0), 0.0);
    CHECK_EQ(z(1, 1), 0.0);
}

void test_toy_one_step_and_two_step_recover_optimum() {
    for (auto strategy : {GMMEstimationStrategy::OneStep, GMMEstimationStrategy::TwoStep}) {
        auto gmm = make_toy_gmm_analytic_jacobian();
        gmm.set_estimation_strategy(strategy);
        CHECK_TRUE(gmm.estimate());
        CHECK_TRUE(gmm.is_estimated());
        CHECK_NEAR(gmm.best_parameter_set().values[0], kC1, 5e-6);
        CHECK_NEAR(gmm.best_parameter_set().values[1], kC2, 5e-6);
    }
}

void test_toy_alternate_optimizers_recover_optimum() {
    // NelderMead / Powell / DifferentialEvolution / MLSL on the same toy problem (BFGS is
    // the default, already covered). Optimizer-dependent tolerance: abs 1e-4.
    for (auto method : {OptimizationMethod::NelderMead, OptimizationMethod::Powell,
                        OptimizationMethod::DifferentialEvolution,
                        OptimizationMethod::MultilevelSingleLinkage}) {
        auto gmm = make_toy_gmm_analytic_jacobian();
        gmm.set_optimizer_method(method);
        CHECK_TRUE(gmm.estimate());
        CHECK_NEAR(gmm.best_parameter_set().values[0], kC1, 1e-4);
        CHECK_NEAR(gmm.best_parameter_set().values[1], kC2, 1e-4);
    }
}

void test_toy_brent_single_parameter() {
    // 1-parameter just-identified problem for the Brent branch: g = theta - 5, S = [[0.2]].
    auto moments = [](const std::vector<double>& theta) -> MomentConditionResult {
        Vector g(1);
        g[0] = theta[0] - 5.0;
        Matrix s(1, 1);
        s(0, 0) = 0.2;
        return {g, s};
    };
    GeneralizedMethodOfMoments gmm(moments, 1, 1, 10, {2.0}, {0.0}, {10.0});
    gmm.set_optimizer_method(OptimizationMethod::Brent);
    CHECK_TRUE(gmm.estimate());
    CHECK_NEAR(gmm.best_parameter_set().values[0], 5.0, 1e-5);
}

void test_estimate_guards() {
    // Under-identified without a penalty function -> throws.
    auto under = make_stub_gmm(3, 2);
    CHECK_THROWS(under.estimate());

    // Over-identified + OneStep -> throws.
    auto over = make_stub_gmm(2, 4);
    over.set_estimation_strategy(GMMEstimationStrategy::OneStep);
    CHECK_THROWS(over.estimate());
}

void test_under_identified_with_penalty_estimates() {
    // p = 2, q = 1: g = (theta_1 - 5) with S = [[0.2]]; the penalty (theta_2 - 3)^2 pins the
    // otherwise-unidentified second parameter. Optimum: (5, 3).
    auto moments = [](const std::vector<double>& theta) -> MomentConditionResult {
        Vector g(1);
        g[0] = theta[0] - 5.0;
        Matrix s(1, 1);
        s(0, 0) = 0.2;
        return {g, s};
    };
    GeneralizedMethodOfMoments gmm(
        moments, 2, 1, 10, {2.0, 8.0}, {0.0, 0.0}, {10.0, 10.0}, std::nullopt, nullptr,
        [](const std::vector<double>& t) { return (t[1] - 3.0) * (t[1] - 3.0); });
    CHECK_TRUE(gmm.identification_status() == GMMIdentificationStatus::UnderIdentified);
    CHECK_TRUE(gmm.estimate());
    CHECK_NEAR(gmm.best_parameter_set().values[0], 5.0, 1e-5);
    CHECK_NEAR(gmm.best_parameter_set().values[1], 3.0, 1e-5);
}

void test_penalty_is_random_toggle_in_sandwich_meat() {
    // With a penalty, PenaltyIsRandom controls whether H = penalty Hessian enters the meat:
    //   Sigma_random - Sigma_fixed = breadInv * H * breadInv / n
    // (both share bread = D'WD + H). Expected difference built from the ported linalg pieces
    // with the analytic H = [[2, 0], [0, 0]] and D = I.
    auto make = []() {
        return GeneralizedMethodOfMoments(
            toy_moments, 2, 2, kToyN, {2.0, 8.0}, {0.0, 0.0}, {10.0, 10.0}, std::nullopt,
            identity_jacobian,
            [](const std::vector<double>& t) { return (t[0] - 5.0) * (t[0] - 5.0); });
    };
    auto gmm = make();
    CHECK_TRUE(gmm.estimate());
    std::vector<double> best = gmm.best_parameter_set().values;

    gmm.penalty_is_random = true;
    Matrix sigma_random = gmm.get_covariance(best, true);
    gmm.penalty_is_random = false;
    Matrix sigma_fixed = gmm.get_covariance(best, true);

    Matrix s_reg = MatrixRegularization::regularize(gmm.get_s(best));
    Matrix w = s_reg.inverse();
    Matrix h(2, 2);
    h(0, 0) = 2.0;
    Matrix bread = MatrixRegularization::make_symmetric_positive_definite(w + h);
    Matrix bread_inv = bread.inverse();
    Matrix expected_diff = (bread_inv * h * bread_inv) * (1.0 / kToyN);

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            CHECK_NEAR(sigma_random(i, j) - sigma_fixed(i, j), expected_diff(i, j), 1e-6);
    CHECK_TRUE(sigma_random(0, 0) > sigma_fixed(0, 0));
}

void test_covariance_matrix_guards_and_failure_fallback() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_THROWS(gmm.get_covariance_matrix());          // not estimated
    CHECK_THROWS(gmm.get_sandwich_covariance_matrix()); // not estimated
    CHECK_TRUE(gmm.estimate());
    // Sigma not yet set (PostProcess not called): GetCovarianceMatrix computes on demand.
    CHECK_TRUE(!gmm.sigma().has_value());
    Matrix cov = gmm.get_covariance_matrix();
    const Matrix s = toy_s();
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(cov(i, j), s(i, j) / kToyN, 1e-8);

    // A throwing moment function inside GetCovariance -> caught -> zero matrix (silent guard).
    auto throwing_moments = [](const std::vector<double>&) -> MomentConditionResult {
        throw std::runtime_error("boom");
    };
    GeneralizedMethodOfMoments bad(throwing_moments, 2, 2, 10, {0.5, 0.5}, {0.0, 0.0},
                                   {1.0, 1.0});
    Matrix zero = bad.get_covariance({0.5, 0.5});
    CHECK_EQ(zero.number_of_rows(), 2);
    CHECK_EQ(zero(0, 0), 0.0);
    CHECK_EQ(zero(1, 1), 0.0);
}

void test_influence_diagnostics_stubs_throw() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_TRUE(gmm.estimate());
    CHECK_THROWS(gmm.get_observation_influence());
    CHECK_THROWS(gmm.get_cooks_distance());
    CHECK_THROWS(gmm.get_influence_diagnostics());
    CHECK_THROWS(gmm.get_leverage_diagnostics());
}

void test_clear_results_resets_state() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    CHECK_TRUE(gmm.estimate());
    gmm.post_process();
    CHECK_TRUE(gmm.is_estimated());

    gmm.clear_results();
    CHECK_TRUE(!gmm.is_estimated());
    CHECK_TRUE(gmm.status() == OptimizationStatus::None);
    CHECK_EQ(gmm.gmm_iterations(), 0);
    CHECK_EQ(gmm.total_function_evaluations(), 0);
    CHECK_TRUE(std::isnan(gmm.jstat()));
    CHECK_TRUE(std::isnan(gmm.jstat_pval()));
    CHECK_TRUE(!gmm.w().has_value());
    CHECK_TRUE(!gmm.s().has_value());
    CHECK_TRUE(!gmm.sigma().has_value());
    CHECK_TRUE(gmm.best_parameter_set().values.empty());
    CHECK_TRUE(gmm.convergence_history().empty());
}

void test_is_valid_reports_errors() {
    auto gmm = make_stub_gmm(2, 2);
    std::vector<std::string> errors;
    CHECK_TRUE(gmm.is_valid(errors));
    CHECK_TRUE(errors.empty());

    gmm.set_max_gmm_iterations(0);
    gmm.set_absolute_tolerance(1e-16);
    gmm.set_relative_tolerance(1e-16);
    errors.clear();
    CHECK_TRUE(!gmm.is_valid(errors));
    CHECK_EQ(static_cast<int>(errors.size()), 3);

    auto under = make_stub_gmm(3, 2);
    errors.clear();
    CHECK_TRUE(!under.is_valid(errors));
    CHECK_EQ(static_cast<int>(errors.size()), 1);

    auto over = make_stub_gmm(2, 4);
    over.set_estimation_strategy(GMMEstimationStrategy::OneStep);
    errors.clear();
    CHECK_TRUE(!over.is_valid(errors));
    CHECK_EQ(static_cast<int>(errors.size()), 1);
}

void test_clone_copies_configuration() {
    auto gmm = make_toy_gmm_analytic_jacobian();
    auto copy = gmm.clone();
    CHECK_EQ(copy.number_of_parameters(), 2);
    CHECK_EQ(copy.number_of_moment_conditions(), 2);
    CHECK_EQ(copy.sample_size(), kToyN);
    CHECK_TRUE(copy.initial_values() == gmm.initial_values());
    CHECK_TRUE(copy.lower_bounds() == gmm.lower_bounds());
    CHECK_TRUE(copy.upper_bounds() == gmm.upper_bounds());
    CHECK_TRUE(!copy.is_estimated());
}

void test_set_default_options() {
    auto gmm = make_stub_gmm(2, 2);
    gmm.set_optimizer_method(OptimizationMethod::Powell);
    gmm.set_use_fallback_optimizer(false);
    gmm.set_estimation_strategy(GMMEstimationStrategy::OneStep);
    gmm.set_max_gmm_iterations(7);
    gmm.set_max_function_evaluations(50);
    gmm.set_absolute_tolerance(1e-3);
    gmm.set_relative_tolerance(1e-3);

    gmm.set_default_options();

    CHECK_TRUE(gmm.optimizer_method() == OptimizationMethod::BFGS);
    CHECK_TRUE(gmm.use_fallback_optimizer());
    CHECK_TRUE(gmm.estimation_strategy() == GMMEstimationStrategy::Iterative);
    CHECK_EQ(gmm.max_gmm_iterations(), 100);
    CHECK_EQ(gmm.max_function_evaluations(), 2000);
    CHECK_NEAR(gmm.absolute_tolerance(), 1e-8, 1e-20);
    CHECK_NEAR(gmm.relative_tolerance(), 1e-8, 1e-20);
}

// --- Newly ported support surface -----------------------------------------------------

void test_numerical_diff_compute_gradient() {
    // f(x) = x0^2 + 3 x0 x1: grad at (1, 2) = (2 + 6, 3) = (8, 3). Central differences are
    // exact for quadratics up to roundoff.
    auto f = [](std::vector<double>& x) { return x[0] * x[0] + 3.0 * x[0] * x[1]; };
    std::vector<double> grad = NumericalDiff::compute_gradient(f, {1.0, 2.0});
    CHECK_NEAR(grad[0], 8.0, 1e-8);
    CHECK_NEAR(grad[1], 3.0, 1e-8);

    // Bounds-aware one-sided path: x0 pinned at its lower bound forces a forward difference,
    // exact for the linear f = 2 x0 + x1.
    auto lin = [](std::vector<double>& x) { return 2.0 * x[0] + x[1]; };
    std::vector<double> g2 =
        NumericalDiff::compute_gradient(lin, {0.0, 5.0}, {0.0, 0.0}, {10.0, 10.0});
    CHECK_NEAR(g2[0], 2.0, 1e-10);
    CHECK_NEAR(g2[1], 1.0, 1e-10);
}

void test_numerical_diff_compute_jacobian() {
    // g(x) = (x0 + 2 x1, x0 * x1): J at (1, 2) = [[1, 2], [2, 1]] (bilinear -> central
    // differences exact up to roundoff).
    auto g = [](std::vector<double>& x) {
        return std::vector<double>{x[0] + 2.0 * x[1], x[0] * x[1]};
    };
    Matrix2D J = NumericalDiff::compute_jacobian(g, {1.0, 2.0}, 2);
    CHECK_NEAR(J[0][0], 1.0, 1e-8);
    CHECK_NEAR(J[0][1], 2.0, 1e-8);
    CHECK_NEAR(J[1][0], 2.0, 1e-8);
    CHECK_NEAR(J[1][1], 1.0, 1e-8);

    // Boundary: parameter at its upper bound forces a backward difference, exact for 3 x0.
    auto lin = [](std::vector<double>& x) { return std::vector<double>{3.0 * x[0]}; };
    Matrix2D Jb = NumericalDiff::compute_jacobian(lin, {10.0}, 1, {0.0}, {10.0});
    CHECK_NEAR(Jb[0][0], 3.0, 1e-10);
}

void test_eigenvalue_decomposition() {
    // [[2, 1], [1, 2]]: one Jacobi rotation (theta = pi/4) gives eigenvalues (1, 3).
    Matrix a(2, 2);
    a(0, 0) = 2.0;
    a(0, 1) = 1.0;
    a(1, 0) = 1.0;
    a(1, 1) = 2.0;
    EigenValueDecomposition eig(a);
    CHECK_NEAR(eig.eigen_values()[0], 1.0, 1e-12);
    CHECK_NEAR(eig.eigen_values()[1], 3.0, 1e-12);
    // A v_i = lambda_i v_i for each eigenvector column.
    for (int i = 0; i < 2; ++i) {
        for (int r = 0; r < 2; ++r) {
            double av = a(r, 0) * eig.eigen_vectors()(0, i) + a(r, 1) * eig.eigen_vectors()(1, i);
            CHECK_NEAR(av, eig.eigen_values()[i] * eig.eigen_vectors()(r, i), 1e-12);
        }
    }
    // Dutilleul effective sample size: (1 + 3)^2 / (1 + 9) = 1.6.
    CHECK_NEAR(eig.effective_sample_size(), 1.6, 1e-12);

    // Guards: non-square and non-symmetric inputs throw.
    CHECK_THROWS(EigenValueDecomposition(Matrix(2, 3)));
    Matrix ns(2, 2);
    ns(0, 1) = 1.0;  // asymmetric
    CHECK_THROWS(EigenValueDecomposition(ns));
}

void test_matrix_regularization_regularize() {
    // An SPD matrix passes through (eigenvalues inside [floor, cap]).
    Matrix s = toy_s();
    Matrix r = MatrixRegularization::regularize(s);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(r(i, j), s(i, j), 1e-9);

    // Negative eigenvalue floored: diag(2, -1) has eigenvalue trace 1, so
    // floor = 1e-6 * (1/2) = 5e-7; median(lambda) = 0.5 -> cap = 25 (inactive).
    Matrix neg(2, 2);
    neg(0, 0) = 2.0;
    neg(1, 1) = -1.0;
    Matrix rn = MatrixRegularization::regularize(neg);
    CHECK_NEAR(rn(0, 0), 2.0, 1e-12);
    CHECK_NEAR(rn(1, 1), 5e-7, 1e-12);
    CHECK_NEAR(rn(0, 1), 0.0, 1e-12);

    // Non-square input throws.
    CHECK_THROWS(MatrixRegularization::regularize(Matrix(2, 3)));
}

void test_tools_distance() {
    CHECK_NEAR(bestfit::numerics::distance({1.0, 2.0}, {4.0, 6.0}), 5.0, 1e-15);
    CHECK_NEAR(bestfit::numerics::distance({1.0}, {1.0}), 0.0, 1e-15);
}

}  // namespace

int main() {
    // PART 1: upstream structural transcriptions.
    test_delegate_constructor_populates_shape_and_unestimated_state();
    test_delegate_constructor_null_moment_condition_function_throws();
    test_delegate_constructor_mismatched_array_length_throws();
    test_delegate_constructor_upper_less_than_lower_throws();
    test_delegate_constructor_initial_outside_bounds_throws();
    test_identification_status();
    test_optimizer_method_reflects_constructor_choice();
    test_max_gmm_iterations_round_trips();
    test_objective_function_value_before_estimation_is_nan();
    test_use_fallback_optimizer_setter_round_trips();
    test_estimation_strategy_setter_round_trips_for_every_value();
    test_max_function_evaluations_setter_round_trips();
    test_absolute_tolerance_setter_round_trips();
    test_relative_tolerance_setter_round_trips();
    test_is_estimated_before_estimation_is_false();
    test_jstat_before_estimation_is_zero();
    test_s_w_sigma_before_estimation_are_null();
    test_total_function_evaluations_before_estimation_is_zero();
    test_gmm_iterations_before_estimation_is_zero();
    test_converged_within_tolerance_before_estimation_is_false();
    test_convergence_history_before_estimation_is_empty();
    test_best_parameter_set_before_estimation_is_deterministic_empty();
    test_degree_of_freedom();
    test_optimizer_method_transitions_through_every_value();

    // PART 2: toy problem + support surface supplement.
    test_toy_estimate_iterative_recovers_analytic_optimum();
    test_toy_covariance_stack_matches_analytic();
    test_toy_profile_confidence_intervals_match_analytic();
    test_toy_profile_percentiles_shape_and_symmetry();
    test_toy_profile_q_shape();
    test_q_conventions_with_and_without_penalty();
    test_q_throws_when_w_is_null();
    test_q_nonfinite_returns_double_max();
    test_get_g_and_get_s();
    test_get_jacobian_numerical_matches_analytic();
    test_get_gradient_matches_analytic();
    test_get_penalty_hessian();
    test_toy_one_step_and_two_step_recover_optimum();
    test_toy_alternate_optimizers_recover_optimum();
    test_toy_brent_single_parameter();
    test_estimate_guards();
    test_under_identified_with_penalty_estimates();
    test_penalty_is_random_toggle_in_sandwich_meat();
    test_covariance_matrix_guards_and_failure_fallback();
    test_influence_diagnostics_stubs_throw();
    test_clear_results_resets_state();
    test_is_valid_reports_errors();
    test_clone_copies_configuration();
    test_set_default_options();

    test_numerical_diff_compute_gradient();
    test_numerical_diff_compute_jacobian();
    test_eigenvalue_decomposition();
    test_matrix_regularization_regularize();
    test_tools_distance();

    return bftest::summary("gmm");
}
