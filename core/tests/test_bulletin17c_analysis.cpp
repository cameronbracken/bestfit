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
#include <vector>

#include "bestfit/analyses/univariate/bulletin17c_analysis.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/data_frame/data_collections/exact_series.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "check.hpp"

using bestfit::analyses::Bulletin17CAnalysis;
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

// ---- The deferred / A8 dispatch arms throw a clear error (documented scope) ----
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
    {
        auto a = std::make_unique<Bulletin17CAnalysis>(make_lp3_model());
        a->set_uncertainty_method(UncertaintyMethod::Bootstrap);
        CHECK_THROWS(a->run());  // ships in A8
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

    return bftest::summary("bulletin17c_analysis");
}
