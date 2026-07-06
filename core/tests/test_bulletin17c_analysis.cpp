// Structural / behavioral tests for bestfit::analyses::Bulletin17CAnalysis (A7): the
// point-estimate GMM fit + the DEFAULT MultivariateNormal uncertainty-quantification path.
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/Univariate/Bulletin17CAnalysisTests.cs
// that exercise the ported A7 surface (construction, the default point estimator, the
// UncertaintyMethod enum, ClearResults lifecycle, Validate, unestimated-getter nulls). Per the
// Phase-8 policy there are NO seeded GMM/MVN oracle NUMBERS here -- the exact fitted parameters,
// covariance, and CI widths depend on the seeded optimizer + MVN stream and are the A11 dotnet
// emitter's job. A real run() over a small flood frame is included but only its STRUCTURAL /
// finite / monotone properties are asserted, never exact quantile numbers.
//
// DELIBERATE DEVIATION exercised here (documented in the header + report): the shipped default
// UncertaintyMethod is MultivariateNormal, NOT the C# ctor's LinkedMultivariateNormal (which is
// deferred to Phase 9 and would throw). So the C# test
// Constructor_DefaultUncertaintyMethod_IsLinkedMultivariateNormal is transcribed as
// test_default_uncertainty_method_is_multivariate_normal (asserting MVN).
//
// SKIPPED C# test methods (reasons in the report):
//   * XmlSerialization_RoundTrip_PreservesConfiguration, ToXElement_HasExpectedRootName,
//     Constructor_NullXmlOrModel_Throws -- XML (de)serialization surface, dropped project-wide.
//   * UncertaintyMethod_SettingDifferentValue_RaisesPropertyChange,
//     UncertaintyMethod_SettingSameValue_DoesNotRaisePropertyChange -- INotifyPropertyChanged;
//     no notification system in this port. (The state change itself -- setting a new method
//     clears results -- is a Phase-9 concern since the LinkedMVN default path is deferred.)
//   * All the *Link* / GammaWeds* / PearsonScaleLink / PositiveParameterLink reflection tests --
//     they probe the LinkedMultivariateNormal link-builder helpers, which are DEFERRED to Phase 9.
//   * CohnConfidenceIntervalResult_* -- the Cohn CI DTO lands in A9.
#include <cmath>
#include <memory>
#include <tuple>
#include <vector>

#include "bestfit/analyses/support/bootstrap_diagnostics.hpp"
#include "bestfit/analyses/support/cohn_confidence_interval_result.hpp"
#include "bestfit/analyses/univariate/bulletin17c_analysis.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/data_frame/data_collections/exact_series.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "bestfit/numerics/distributions/student_t.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "check.hpp"

// Friend accessor for the private A8 acceleration_constants() method (C# private; not called on
// the shipped Bootstrap path -- see the header). Declared a friend in the analysis header so the
// C++-only determinism ctest can reach it without widening the public API.
namespace bestfit::analyses {
struct Bulletin17CAnalysisTestAccess {
    static std::vector<double> acceleration_constants(Bulletin17CAnalysis& a,
                                                      const std::vector<double>& theta_hats) {
        return a.acceleration_constants(theta_hats);
    }

    // A9 Cohn CI building blocks (private in the class; reached here without widening the API).
    static std::vector<double> clamp_for_covariance(const std::vector<double>& p) {
        return Bulletin17CAnalysis::clamp_for_covariance(p);
    }
    static std::vector<double> clamp_for_quantile(const std::vector<double>& p) {
        return Bulletin17CAnalysis::clamp_for_quantile(p);
    }
    static double weighted_covariance(const std::vector<double>& x, const std::vector<double>& y,
                                      const std::vector<double>& w) {
        return Bulletin17CAnalysis::weighted_covariance(x, y, w);
    }
    static std::tuple<double, double, double, double> cohn_adjusted_student_t_ci(
        double q_hat, double var_q, double cov_q_se, double var_se, double confidence_level) {
        return Bulletin17CAnalysis::cohn_adjusted_student_t_ci(q_hat, var_q, cov_q_se, var_se,
                                                              confidence_level);
    }
    static void enforce_monotonicity(std::vector<double>& lower_ci, std::vector<double>& upper_ci,
                                     int n_prob) {
        Bulletin17CAnalysis::enforce_monotonicity(lower_ci, upper_ci, n_prob);
    }
    static std::pair<std::vector<std::vector<double>>, std::vector<double>> build_quadrature_grid(
        Bulletin17CAnalysis& a, const std::vector<double>& mean,
        const bestfit::numerics::math::linalg::Matrix& covariance, int dimension,
        int n_nodes_per_dim) {
        return a.build_quadrature_grid(mean, covariance, dimension, n_nodes_per_dim);
    }
};
}  // namespace bestfit::analyses

using bestfit::analyses::BootstrapDiagnostics;
using bestfit::analyses::Bulletin17CAnalysis;
using bestfit::analyses::Bulletin17CAnalysisTestAccess;
using bestfit::analyses::UncertaintyMethod;
using bestfit::models::Bulletin17CDistribution;
using bestfit::models::DataFrame;
using bestfit::models::ExactSeries;
using PET = bestfit::estimation::PointEstimateType;
using UDT = bestfit::numerics::distributions::UnivariateDistributionType;

namespace {

// Deterministic flood-like fixture, mirroring the C# inline LogNormal(8,0.4) draws. The ported
// Mersenne Twister is bit-exact, so these 50 values match the C# InlineFloodData exactly.
std::vector<double> flood_data() {
    return bestfit::numerics::distributions::LogNormal(8.0, 0.4).generate_random_values(50, 12345);
}

std::unique_ptr<Bulletin17CDistribution> make_lp3_model() {
    DataFrame df;
    df.set_exact_series(ExactSeries(flood_data()));
    df.calculate_plotting_positions();
    return std::make_unique<Bulletin17CDistribution>(std::move(df), UDT::LogPearsonTypeIII);
}

// ---- Constructor wires all required state (C# Constructor_WithModel_InitializesAllRequiredState) ----
void test_constructor_initializes_state() {
    auto model = make_lp3_model();
    Bulletin17CDistribution* raw = model.get();
    Bulletin17CAnalysis analysis(std::move(model));

    CHECK_TRUE(&analysis.bulletin17c_distribution() == raw);  // same model object
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    CHECK_TRUE(analysis.gmm() == nullptr);                    // GMM created lazily in run()
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
}

// ---- Default point estimator is PosteriorMode (C# Constructor_DefaultPointEstimator_...) ----
void test_default_point_estimator_is_posterior_mode() {
    Bulletin17CAnalysis analysis(make_lp3_model());
    CHECK_TRUE(analysis.bayesian_analysis().point_estimator() == PET::PosteriorMode);
}

// ---- Default uncertainty method is MultivariateNormal (SHIPPED DEVIATION from C# LinkedMVN) ----
void test_default_uncertainty_method_is_multivariate_normal() {
    Bulletin17CAnalysis analysis(make_lp3_model());
    CHECK_TRUE(analysis.uncertainty_method() == UncertaintyMethod::MultivariateNormal);
}

// ---- Null model throws (C# Constructor_NullModel_Throws) ----
void test_null_model_throws() {
    CHECK_THROWS(Bulletin17CAnalysis(std::unique_ptr<Bulletin17CDistribution>(nullptr)));
}

// ---- UncertaintyMethod enum pins exactly four members in C# order (C# UncertaintyMethod_EnumHasFourMembers) ----
void test_uncertainty_method_enum_has_four_members() {
    CHECK_EQ(static_cast<int>(UncertaintyMethod::MultivariateNormal), 0);
    CHECK_EQ(static_cast<int>(UncertaintyMethod::LinkedMultivariateNormal), 1);
    CHECK_EQ(static_cast<int>(UncertaintyMethod::Bootstrap), 2);
    CHECK_EQ(static_cast<int>(UncertaintyMethod::BiasCorrectedBootstrap), 3);
}

// ---- ClearResults resets to un-estimated, preserves the model (C# ClearResults_...) ----
void test_clear_results_resets() {
    auto model = make_lp3_model();
    Bulletin17CDistribution* raw = model.get();
    Bulletin17CAnalysis analysis(std::move(model));

    analysis.clear_results();

    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    CHECK_TRUE(analysis.gmm() == nullptr);
    CHECK_TRUE(&analysis.bulletin17c_distribution() == raw);
}

// ---- Validate is valid for a good LP3 fixture (C# Validate_GoodFixture_IsValid) ----
void test_validate_good_fixture_is_valid() {
    Bulletin17CAnalysis analysis(make_lp3_model());
    CHECK_TRUE(analysis.validate().is_valid);
}

// ---- Validate propagates a model failure -- no DataFrame (C# Validate_InvalidModel_...) ----
void test_validate_invalid_model_propagates() {
    Bulletin17CAnalysis analysis(std::make_unique<Bulletin17CDistribution>());  // default: no frame
    auto result = analysis.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(result.validation_messages.size() > 0);
}

// ---- Getters return null before estimation (C# GetDistribution / GetPointEstimateDistribution) ----
void test_getters_null_when_unestimated() {
    Bulletin17CAnalysis analysis(make_lp3_model());
    CHECK_TRUE(analysis.get_distribution(0) == nullptr);
    CHECK_TRUE(analysis.get_point_estimate_distribution() == nullptr);
    CHECK_TRUE(analysis.get_point_estimate_distribution(PET::PosteriorMean) == nullptr);
}

// ---- A real run() through the DEFAULT MVN path: structural / finite / monotone only ----
void test_run_multivariate_normal_structural() {
    auto analysis = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
    analysis->bayesian_analysis().set_output_length(200);  // keep the MVN draw count small/fast
    analysis->run();

    CHECK_TRUE(analysis->is_estimated());
    CHECK_TRUE(analysis->gmm() != nullptr);

    const auto* results = analysis->analysis_results();
    CHECK_TRUE(results != nullptr);
    if (results != nullptr) {
        std::size_t n = analysis->probability_ordinates().count();
        CHECK_EQ(results->mode_curve.size(), n);
        CHECK_EQ(results->confidence_intervals.size(), n);
        // Finite goodness-of-fit scalars.
        CHECK_TRUE(std::isfinite(results->aic));
        CHECK_TRUE(std::isfinite(results->bic));
        CHECK_TRUE(std::isfinite(results->rmse));
        CHECK_TRUE(std::isfinite(results->erl));
        // Mode curve is a quantile curve. The default ProbabilityOrdinates hold ASCENDING
        // exceedance probabilities, so mode_curve[i] = inverse_cdf(1 - ordinate[i]) DESCENDS.
        for (std::size_t i = 1; i < n; ++i)
            CHECK_TRUE(results->mode_curve[i] <= results->mode_curve[i - 1]);
        // Each CI brackets the mode curve.
        for (std::size_t i = 0; i < n; ++i) {
            CHECK_TRUE(results->confidence_intervals[i][0] <= results->mode_curve[i]);
            CHECK_TRUE(results->confidence_intervals[i][1] >= results->mode_curve[i]);
        }
    }

    // Positive-path getters resolve to live distributions once estimated.
    CHECK_TRUE(analysis->get_distribution(0) != nullptr);
    CHECK_TRUE(analysis->get_point_estimate_distribution() != nullptr);
}

// ---- The deferred dispatch arms throw a clear error (documented scope) ----
void test_deferred_uncertainty_methods_throw() {
    {
        auto a = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
        a->set_uncertainty_method(UncertaintyMethod::LinkedMultivariateNormal);
        CHECK_THROWS(a->run());  // deferred to Phase 9
    }
    {
        auto a = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
        a->set_uncertainty_method(UncertaintyMethod::BiasCorrectedBootstrap);
        CHECK_THROWS(a->run());  // deferred to Phase 9
    }
}

// ============================ A8: parametric bootstrap + jackknife ============================

// ---- BootstrapDiagnostics counters (transcribed from BootstrapDiagnosticsTests.cs, non-XML) ----
void test_bootstrap_diagnostics_defaults_all_zero() {
    BootstrapDiagnostics d;
    CHECK_EQ(d.total_replicates(), 0);
    CHECK_EQ(d.failed_replicates(), 0);
    CHECK_EQ(d.valid_replicates(), 0);
    CHECK_EQ(d.total_retries(), 0);
    CHECK_EQ(d.total_function_evaluations(), 0);
    CHECK_EQ(d.pivot_rejections(), 0);
    CHECK_EQ(d.mahalanobis_rejections(), 0);
}

void test_bootstrap_diagnostics_rates_zero_when_no_replicates() {
    BootstrapDiagnostics d;
    CHECK_EQ(d.failure_rate(), 0.0);
    CHECK_EQ(d.average_retries(), 0.0);
    CHECK_EQ(d.average_function_evaluations(), 0.0);
    CHECK_EQ(d.pivot_rejection_rate(), 0.0);
    CHECK_EQ(d.mahalanobis_rejection_rate(), 0.0);
}

void test_bootstrap_diagnostics_valid_equals_total_minus_failed() {
    BootstrapDiagnostics d;
    d.set_total_replicates(100);
    for (int i = 0; i < 7; i++) d.increment_failed();
    CHECK_EQ(d.valid_replicates(), 93);
}

void test_bootstrap_diagnostics_failure_rate() {
    BootstrapDiagnostics d;
    d.set_total_replicates(200);
    for (int i = 0; i < 50; i++) d.increment_failed();
    CHECK_NEAR(d.failure_rate(), 0.25, 1e-12);
}

void test_bootstrap_diagnostics_add_retries_accumulates() {
    BootstrapDiagnostics d;
    d.set_total_replicates(10);
    d.add_retries(3);
    d.add_retries(7);
    CHECK_EQ(d.total_retries(), 10);
    CHECK_NEAR(d.average_retries(), 1.0, 1e-12);
}

void test_bootstrap_diagnostics_add_function_evaluations_accumulates() {
    BootstrapDiagnostics d;
    d.set_total_replicates(4);
    d.add_function_evaluations(20);
    d.add_function_evaluations(20);
    CHECK_EQ(d.total_function_evaluations(), 40);
    CHECK_NEAR(d.average_function_evaluations(), 10.0, 1e-12);
}

void test_bootstrap_diagnostics_pivot_rejection() {
    BootstrapDiagnostics d;
    d.set_total_replicates(100);
    d.increment_pivot_rejection();
    d.increment_pivot_rejection();
    CHECK_EQ(d.pivot_rejections(), 2);
    CHECK_NEAR(d.pivot_rejection_rate(), 0.02, 1e-12);
}

void test_bootstrap_diagnostics_mahalanobis_rejection() {
    BootstrapDiagnostics d;
    d.set_total_replicates(50);
    d.increment_mahalanobis_rejection();
    CHECK_EQ(d.mahalanobis_rejections(), 1);
    CHECK_NEAR(d.mahalanobis_rejection_rate(), 0.02, 1e-12);
}

// ---- AccelerationConstants determinism (C++-only): deterministic given a fixed frame, finite,
//      length p (no PRNG in the jackknife). ----
void test_acceleration_constants_deterministic() {
    auto analysis = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
    analysis->run();  // MVN default fits the GMM point estimate
    CHECK_TRUE(analysis->is_estimated());
    CHECK_TRUE(analysis->gmm() != nullptr);

    std::vector<double> theta = analysis->gmm()->best_parameter_set().values;
    int p = analysis->bulletin17c_distribution().number_of_parameters();

    std::vector<double> a1 = Bulletin17CAnalysisTestAccess::acceleration_constants(*analysis, theta);
    std::vector<double> a2 = Bulletin17CAnalysisTestAccess::acceleration_constants(*analysis, theta);

    CHECK_EQ(static_cast<int>(a1.size()), p);
    CHECK_EQ(static_cast<int>(a2.size()), p);
    for (int i = 0; i < p; ++i) {
        CHECK_TRUE(std::isfinite(a1[static_cast<std::size_t>(i)]));
        CHECK_EQ(a1[static_cast<std::size_t>(i)], a2[static_cast<std::size_t>(i)]);  // deterministic
    }
}

// ---- A real run() through the Bootstrap UQ path: structural / finite / monotone + diagnostics. ----
void test_run_bootstrap_structural() {
    auto analysis = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
    analysis->set_uncertainty_method(UncertaintyMethod::Bootstrap);
    const int b = 30;  // small replicate count keeps the re-fit loop fast
    analysis->bayesian_analysis().set_output_length(b);
    analysis->run();

    CHECK_TRUE(analysis->is_estimated());
    CHECK_TRUE(analysis->gmm() != nullptr);

    // Diagnostics populated with the replicate accounting identity.
    const auto* diag = analysis->bootstrap_results();
    CHECK_TRUE(diag != nullptr);
    if (diag != nullptr) {
        CHECK_EQ(diag->total_replicates(), b);
        CHECK_EQ(diag->valid_replicates() + diag->failed_replicates(), b);
    }

    const auto* results = analysis->analysis_results();
    CHECK_TRUE(results != nullptr);
    if (results != nullptr) {
        std::size_t n = analysis->probability_ordinates().count();
        CHECK_EQ(results->mode_curve.size(), n);
        CHECK_EQ(results->confidence_intervals.size(), n);
        for (std::size_t i = 0; i < n; ++i) CHECK_TRUE(std::isfinite(results->mode_curve[i]));
        // Mode curve DESCENDS on ascending exceedance ordinates (same as the MVN test).
        for (std::size_t i = 1; i < n; ++i)
            CHECK_TRUE(results->mode_curve[i] <= results->mode_curve[i - 1]);
        // Each CI brackets the mode curve.
        for (std::size_t i = 0; i < n; ++i) {
            CHECK_TRUE(results->confidence_intervals[i][0] <= results->mode_curve[i]);
            CHECK_TRUE(results->confidence_intervals[i][1] >= results->mode_curve[i]);
        }
    }
}

// ============================ A9: Cohn-style delta-method confidence intervals ============================

using bestfit::analyses::CohnConfidenceIntervalResult;
using bestfit::numerics::math::linalg::Matrix;
using TA = Bulletin17CAnalysisTestAccess;

// ---- ClampForCovariance: sigma floor 1e-10, |gamma| clamp to 1.5, 0.063 floor sign-preserving ----
void test_clamp_for_covariance() {
    // sigma floored, gamma clamped to +1.5.
    auto c1 = TA::clamp_for_covariance({5.0, -3.0, 2.0});
    CHECK_NEAR(c1[0], 5.0, 1e-15);       // mu untouched
    CHECK_NEAR(c1[1], 1e-10, 1e-20);     // sigma floored to 1e-10
    CHECK_NEAR(c1[2], 1.5, 1e-15);       // gamma clamped to +1.5
    // gamma clamped to -1.5.
    auto c2 = TA::clamp_for_covariance({5.0, 1.0, -2.0});
    CHECK_NEAR(c2[2], -1.5, 1e-15);
    // 0.063 floor with sign preserved.
    auto c3 = TA::clamp_for_covariance({5.0, 1.0, 0.01});
    CHECK_NEAR(c3[2], 0.063, 1e-15);
    auto c4 = TA::clamp_for_covariance({5.0, 1.0, -0.02});
    CHECK_NEAR(c4[2], -0.063, 1e-15);
    // in-range gamma left alone; positive sigma left alone.
    auto c5 = TA::clamp_for_covariance({5.0, 0.4, 0.5});
    CHECK_NEAR(c5[1], 0.4, 1e-15);
    CHECK_NEAR(c5[2], 0.5, 1e-15);
}

// ---- ClampForQuantile: sigma floor only; gamma left UNCLAMPED ----
void test_clamp_for_quantile() {
    auto q1 = TA::clamp_for_quantile({5.0, -3.0, 2.0});
    CHECK_NEAR(q1[1], 1e-10, 1e-20);  // sigma floored
    CHECK_NEAR(q1[2], 2.0, 1e-15);    // gamma UNCLAMPED (distinct from covariance clamp)
    auto q2 = TA::clamp_for_quantile({5.0, 1.0, 0.01});
    CHECK_NEAR(q2[2], 0.01, 1e-15);   // tiny gamma NOT floored
}

// ---- WeightedCovariance: uniform reduces to population weighted covariance; wSum<=0 -> 0 ----
void test_weighted_covariance() {
    // uniform weights, x == y: population variance of {1,2,3} = 2/3.
    CHECK_NEAR(TA::weighted_covariance({1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 1.0, 1.0}),
               2.0 / 3.0, 1e-12);
    // anti-correlated cross covariance = -2/3.
    CHECK_NEAR(TA::weighted_covariance({1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}, {1.0, 1.0, 1.0}),
               -2.0 / 3.0, 1e-12);
    // non-uniform weights: xbar = 2.5, cov = 0.25*2.25 + 0.75*0.25 = 0.75.
    CHECK_NEAR(TA::weighted_covariance({1.0, 3.0}, {1.0, 3.0}, {0.25, 0.75}), 0.75, 1e-12);
    // wSum <= 0 -> 0.
    CHECK_EQ(TA::weighted_covariance({1.0, 2.0}, {1.0, 2.0}, {0.0, 0.0}), 0.0);
}

// ---- CohnAdjustedStudentTCI: degenerate branches + a symmetric (beta1 == 0) normal case ----
void test_cohn_adjusted_student_t_ci() {
    // varQ <= 0 degenerate branch -> (qHat, qHat, 0, nuMin=5).
    {
        auto [lo, hi, b1, nu] = TA::cohn_adjusted_student_t_ci(50.0, 0.0, 1.0, 1.0, 0.90);
        CHECK_NEAR(lo, 50.0, 1e-15);
        CHECK_NEAR(hi, 50.0, 1e-15);
        CHECK_EQ(b1, 0.0);
        CHECK_NEAR(nu, 5.0, 1e-15);
    }
    {  // negative varQ hits the same branch.
        auto [lo, hi, b1, nu] = TA::cohn_adjusted_student_t_ci(7.0, -1.0, 1.0, 1.0, 0.90);
        CHECK_NEAR(lo, 7.0, 1e-15);
        CHECK_NEAR(hi, 7.0, 1e-15);
        CHECK_EQ(b1, 0.0);
        CHECK_NEAR(nu, 5.0, 1e-15);
    }
    // varSEgivenQ <= 0 -> nu forced to 1000 (before the max(nu,5)); beta1 = covQSE/varQ.
    {
        auto [lo, hi, b1, nu] = TA::cohn_adjusted_student_t_ci(100.0, 4.0, 2.0, 1.0, 0.90);
        (void)lo;
        (void)hi;
        CHECK_NEAR(b1, 0.5, 1e-15);         // covQSE/varQ = 2/4
        CHECK_NEAR(nu, 1000.0, 1e-12);      // varSEgivenQ = 1 - 4/4 = 0 -> 1000
    }
    // Normal, symmetric case: beta1 = 0, nu = 0.5*varQ/varSEgivenQ = 2 -> clamped to nuMin = 5.
    {
        double var_q = 4.0, cov_q_se = 0.0, var_se = 1.0, cl = 0.90;
        auto [lo, hi, b1, nu] = TA::cohn_adjusted_student_t_ci(100.0, var_q, cov_q_se, var_se, cl);
        CHECK_EQ(b1, 0.0);
        CHECK_NEAR(nu, 5.0, 1e-15);
        double t = bestfit::numerics::distributions::StudentT(5.0).inverse_cdf((1.0 + cl) / 2.0);
        double se_q = std::sqrt(var_q);
        CHECK_NEAR(hi, 100.0 + se_q * t, 1e-9);   // beta1 == 0 -> symmetric interval
        CHECK_NEAR(lo, 100.0 - se_q * t, 1e-9);
    }
}

// ---- EnforceMonotonicity: backward sweep makes both bounds non-increasing with index ----
void test_enforce_monotonicity() {
    std::vector<double> lo = {1.0, 5.0, 3.0, 9.0};  // ascending-AEP: should be non-increasing
    std::vector<double> hi = {2.0, 4.0, 8.0, 6.0};
    TA::enforce_monotonicity(lo, hi, 4);
    // backward sweep from i=2..0: each = max(self, next).
    // lo: start {1,5,3,9}; i=2 -> max(3,9)=9; i=1 -> max(5,9)=9; i=0 -> max(1,9)=9 => {9,9,9,9}.
    CHECK_NEAR(lo[0], 9.0, 1e-15);
    CHECK_NEAR(lo[1], 9.0, 1e-15);
    CHECK_NEAR(lo[2], 9.0, 1e-15);
    CHECK_NEAR(lo[3], 9.0, 1e-15);
    // hi: {2,4,8,6}; i=2 -> max(8,6)=8; i=1 -> max(4,8)=8; i=0 -> max(2,8)=8 => {8,8,8,6}.
    CHECK_NEAR(hi[0], 8.0, 1e-15);
    CHECK_NEAR(hi[1], 8.0, 1e-15);
    CHECK_NEAR(hi[2], 8.0, 1e-15);
    CHECK_NEAR(hi[3], 6.0, 1e-15);
}

// Helper: extract dimension d as a column vector across all grid nodes.
std::vector<double> grid_column(const std::vector<std::vector<double>>& grid, int d) {
    std::vector<double> col;
    col.reserve(grid.size());
    for (const auto& node : grid) col.push_back(node[static_cast<std::size_t>(d)]);
    return col;
}

// ---- BuildQuadratureGrid / CohnCholesky / BuildGridFromCholesky: Gaussian-quadrature exactness ----
//      2^p nodes, weights sum to 1, weighted centroid == mean, weighted covariance reproduces input.
void test_build_quadrature_grid_reproduces_moments() {
    Bulletin17CAnalysis analysis(make_lp3_model());  // build_quadrature_grid uses no GMM state

    // --- p = 2, diagonal covariance (exercises the Gamma-node scale dimension d=1). ---
    {
        std::vector<double> mean = {0.0, 2.0};  // mean[1] > 0 -> Gamma quadrature for sigma
        Matrix cov(2, 2);
        cov(0, 0) = 1.0;
        cov(1, 1) = 0.25;
        auto [grid, weights] = TA::build_quadrature_grid(analysis, mean, cov, 2, 2);
        CHECK_EQ(static_cast<int>(grid.size()), 4);  // 2^2
        double wsum = 0.0;
        for (double w : weights) wsum += w;
        CHECK_NEAR(wsum, 1.0, 1e-12);
        auto c0 = grid_column(grid, 0);
        auto c1 = grid_column(grid, 1);
        // Weighted centroid == mean.
        double m0 = 0.0, m1 = 0.0;
        for (std::size_t i = 0; i < grid.size(); ++i) {
            m0 += weights[i] * c0[i];
            m1 += weights[i] * c1[i];
        }
        CHECK_NEAR(m0, 0.0, 1e-9);
        CHECK_NEAR(m1, 2.0, 1e-9);
        // Weighted covariance reproduces the input covariance (first two moments exact).
        CHECK_NEAR(TA::weighted_covariance(c0, c0, weights), 1.0, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c1, c1, weights), 0.25, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c0, c1, weights), 0.0, 1e-6);
    }

    // --- p = 3, diagonal covariance. ---
    {
        std::vector<double> mean = {0.0, 2.0, 0.1};
        Matrix cov(3, 3);
        cov(0, 0) = 1.0;
        cov(1, 1) = 0.25;
        cov(2, 2) = 0.09;
        auto [grid, weights] = TA::build_quadrature_grid(analysis, mean, cov, 3, 2);
        CHECK_EQ(static_cast<int>(grid.size()), 8);  // 2^3
        double wsum = 0.0;
        for (double w : weights) wsum += w;
        CHECK_NEAR(wsum, 1.0, 1e-12);
        auto c0 = grid_column(grid, 0);
        auto c1 = grid_column(grid, 1);
        auto c2 = grid_column(grid, 2);
        CHECK_NEAR(TA::weighted_covariance(c0, c0, weights), 1.0, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c1, c1, weights), 0.25, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c2, c2, weights), 0.09, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c0, c2, weights), 0.0, 1e-6);
    }

    // --- p = 2, CORRELATED covariance (exercises the Cholesky cross-loading). ---
    {
        std::vector<double> mean = {0.0, 3.0};
        Matrix cov(2, 2);
        cov(0, 0) = 1.0;
        cov(1, 1) = 0.5;
        cov(0, 1) = 0.3;
        cov(1, 0) = 0.3;
        auto [grid, weights] = TA::build_quadrature_grid(analysis, mean, cov, 2, 2);
        auto c0 = grid_column(grid, 0);
        auto c1 = grid_column(grid, 1);
        CHECK_NEAR(TA::weighted_covariance(c0, c0, weights), 1.0, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c1, c1, weights), 0.5, 1e-6);
        CHECK_NEAR(TA::weighted_covariance(c0, c1, weights), 0.3, 1e-6);  // cross term reproduced
    }
}

// ---- compute_cohn_style_confidence_intervals: null when unestimated ----
void test_compute_cohn_style_null_when_unestimated() {
    Bulletin17CAnalysis analysis(make_lp3_model());
    CHECK_TRUE(!analysis.compute_cohn_style_confidence_intervals().has_value());
}

// ---- compute_cohn_style_confidence_intervals: end-to-end structural SMOKE on the LP3 fixture ----
void test_compute_cohn_style_structural() {
    auto analysis = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
    analysis->bayesian_analysis().set_output_length(200);
    analysis->run();
    CHECK_TRUE(analysis->is_estimated());

    auto ci = analysis->compute_cohn_style_confidence_intervals();
    CHECK_TRUE(ci.has_value());
    if (ci.has_value()) {
        std::size_t n = analysis->probability_ordinates().count();
        CHECK_EQ(ci->exceedance_probabilities.size(), n);
        CHECK_EQ(ci->point_estimates.size(), n);
        CHECK_EQ(ci->lower_ci.size(), n);
        CHECK_EQ(ci->upper_ci.size(), n);
        CHECK_EQ(ci->beta1.size(), n);
        CHECK_EQ(ci->nu.size(), n);
        CHECK_EQ(ci->quantile_variance.size(), n);
        CHECK_NEAR(ci->confidence_level, analysis->bayesian_analysis().credible_interval_width(),
                   1e-12);
        for (std::size_t i = 0; i < n; ++i) {
            CHECK_NEAR(ci->exceedance_probabilities[i], analysis->probability_ordinates()[i], 1e-15);
            CHECK_TRUE(std::isfinite(ci->point_estimates[i]));
            CHECK_TRUE(std::isfinite(ci->lower_ci[i]));
            CHECK_TRUE(std::isfinite(ci->upper_ci[i]));
            CHECK_TRUE(std::isfinite(ci->beta1[i]));
            CHECK_TRUE(std::isfinite(ci->nu[i]));
            CHECK_TRUE(ci->nu[i] >= 5.0);                     // nuMin floor
            CHECK_TRUE(ci->lower_ci[i] <= ci->upper_ci[i]);   // preserved through monotonicity
        }
        // Both bounds non-increasing with index (ascending-AEP ordinates), post-EnforceMonotonicity.
        for (std::size_t i = 1; i < n; ++i) {
            CHECK_TRUE(ci->lower_ci[i] <= ci->lower_ci[i - 1]);
            CHECK_TRUE(ci->upper_ci[i] <= ci->upper_ci[i - 1]);
        }
    }
}

}  // namespace

int main() {
    test_constructor_initializes_state();
    test_default_point_estimator_is_posterior_mode();
    test_default_uncertainty_method_is_multivariate_normal();
    test_null_model_throws();
    test_uncertainty_method_enum_has_four_members();
    test_clear_results_resets();
    test_validate_good_fixture_is_valid();
    test_validate_invalid_model_propagates();
    test_getters_null_when_unestimated();
    test_run_multivariate_normal_structural();
    test_deferred_uncertainty_methods_throw();

    // A8: parametric bootstrap + jackknife acceleration
    test_bootstrap_diagnostics_defaults_all_zero();
    test_bootstrap_diagnostics_rates_zero_when_no_replicates();
    test_bootstrap_diagnostics_valid_equals_total_minus_failed();
    test_bootstrap_diagnostics_failure_rate();
    test_bootstrap_diagnostics_add_retries_accumulates();
    test_bootstrap_diagnostics_add_function_evaluations_accumulates();
    test_bootstrap_diagnostics_pivot_rejection();
    test_bootstrap_diagnostics_mahalanobis_rejection();
    test_acceleration_constants_deterministic();
    test_run_bootstrap_structural();

    // A9: Cohn-style delta-method confidence intervals
    test_clamp_for_covariance();
    test_clamp_for_quantile();
    test_weighted_covariance();
    test_cohn_adjusted_student_t_ci();
    test_enforce_monotonicity();
    test_build_quadrature_grid_reproduces_moments();
    test_compute_cohn_style_null_when_unestimated();
    test_compute_cohn_style_structural();

    return bftest::summary("bulletin17c_analysis");
}
