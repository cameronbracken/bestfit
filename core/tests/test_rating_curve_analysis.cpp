// Structural / behavioral tests for bestfit::analyses::RatingCurveAnalysis (X3).
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/RatingCurve/RatingCurveAnalysisTests.cs
//   RMC.BestFit.Tests/RatingCurve/RatingCurveAnalysisGridReprocessTests.cs
// (both @ fc28c0c). There are NO numeric MCMC oracles here (per the Phase-10 policy; the
// seeded end-to-end run lands via the X12 emitter). The tests cover: construction, default
// stage-bin derivation, validation of the stage grid, ClearResults, and the grid-reprocess-
// vs-clear behavior on a fresh (unestimated) analysis. Hardcoded oracles in this C++-only
// ctest are correct (public-API oracle values otherwise live in fixtures/*.json).
//
// SKIPPED C# test methods (WPF/serialization/threading -- no numerical content):
//   - XmlSerialization_* / Constructor_WithNullXElement_* / ToXElement_* : the XML ctor +
//     ToXElement are a project-wide non-port.
//   - MinStage/MaxStage/StageBins/UseDefaultStageBins _Change_RaisesPropertyChanged,
//     *_SetSameValue_DoesNotRaisePropertyChanged, RatingCurveStageDataChange_*,
//     RatingCurveNumberOfSegmentsChange_ClearsResults : INotifyPropertyChanged cascades; no
//     notification system in this port. The reprocess-vs-clear decisions those handlers encode
//     are exercised here by driving the grid setters directly, exactly as the grid-reprocess
//     tests do.
//   - RunAsync_RaisesAnalysisStarting/CompletedEvent, RunAsync_WhenCanceled_*, CancelAnalysis_* :
//     events + cancellation dropped.
//   - RunAsync_WithInvalidConfiguration_ThrowsInvalidOperationException : ported as run()
//     throwing std::runtime_error on an invalid grid (asserted below).
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "bestfit/analyses/rating_curve/rating_curve_analysis.hpp"
#include "bestfit/models/rating_curve/rating_curve.hpp"
#include "bestfit/numerics/data/time_series/time_series.hpp"
#include "check.hpp"

using bestfit::analyses::RatingCurveAnalysis;
using bestfit::models::RatingCurve;
using bestfit::numerics::data::TimeInterval;
using bestfit::numerics::data::TimeSeries;

namespace {

// C# s_trueParams = { 0.5, 1.0, 2.0, 0.05 } -- [xi, log10(alpha), beta, sigma].
const std::vector<double>& true_params() {
    static const std::vector<double> v = {0.5, 1.0, 2.0, 0.05};
    return v;
}

// C# MakeSingleSegmentData: synthetic (stage, discharge) over stage in [1, 10], seed 12345.
RatingCurve::SyntheticData make_single_segment_data(int sample_size = 200) {
    RatingCurve seed;
    seed.set_use_default_flat_priors(false);
    seed.set_parameter_values(true_params());
    return seed.generate_synthetic_data(sample_size, 1.0, 10.0, 12345);
}

std::unique_ptr<RatingCurve> make_test_rating_curve(int number_of_segments = 1) {
    RatingCurve::SyntheticData d = make_single_segment_data();
    return std::make_unique<RatingCurve>(d.stage_data, d.discharge_data, number_of_segments);
}

std::unique_ptr<RatingCurveAnalysis> make_test_analysis(int number_of_segments = 1) {
    return std::make_unique<RatingCurveAnalysis>(make_test_rating_curve(number_of_segments));
}

double series_max(const TimeSeries& s) {
    double m = std::numeric_limits<double>::lowest();
    for (int i = 0; i < s.count(); ++i) {
        double v = s[i].value();
        if (!std::isnan(v) && v > m) m = v;
    }
    return m;
}

bool any_contains(const std::vector<std::string>& msgs, const char* needle) {
    for (const auto& m : msgs)
        if (m.find(needle) != std::string::npos) return true;
    return false;
}

// ---- Constructor_WithRatingCurve_InitializesCorrectly ----
void test_constructor_initializes() {
    RatingCurveAnalysis analysis(make_test_rating_curve());
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    // BayesianAnalysis is over the same model.
    CHECK_TRUE(&analysis.bayesian_analysis().model() == &analysis.rating_curve());
}

// ---- Constructor_WithNullRatingCurve_ThrowsArgumentNullException ----
void test_null_rating_curve_throws() {
    CHECK_THROWS(RatingCurveAnalysis(std::unique_ptr<RatingCurve>{}));
}

// ---- Constructor_DefaultStageBins_InitializedFromDataRange ----
void test_default_stage_bins_from_data_range() {
    RatingCurveAnalysis analysis(make_test_rating_curve());
    CHECK_TRUE(analysis.stage_bins() > 0);
    CHECK_EQ(analysis.stage_bins(), 100);
    CHECK_TRUE(analysis.min_stage() < analysis.max_stage());
}

// ---- Constructor_WithMultiSegmentRatingCurve_InitializesCorrectly + various segment counts ----
void test_constructor_segment_counts() {
    RatingCurveAnalysis analysis2(make_test_rating_curve(2));
    CHECK_EQ(analysis2.rating_curve().number_of_segments(), 2);
    for (int segments = 1; segments <= 3; ++segments) {
        std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis(segments);
        CHECK_EQ(a->rating_curve().number_of_segments(), segments);
    }
}

// ---- Constructor_WithEmptyStageData_SetsDefaultStageBins ----
void test_empty_stage_data_defaults() {
    RatingCurveAnalysis analysis(std::make_unique<RatingCurve>());
    CHECK_EQ(analysis.min_stage(), 0.0);
    CHECK_EQ(analysis.max_stage(), 100.0);
    CHECK_EQ(analysis.stage_bins(), 100);
}

// ---- DefaultStageBins_HasPaddingFromDataRange ----
void test_default_stage_bins_padding() {
    RatingCurve::SyntheticData d = make_single_segment_data();
    double data_min = d.stage_data.min_value();
    double data_max = series_max(d.stage_data);
    double range = data_max - data_min;
    RatingCurveAnalysis analysis(
        std::make_unique<RatingCurve>(d.stage_data, d.discharge_data, 1));
    CHECK_NEAR(analysis.min_stage(), data_min - 0.1 * range, 1e-6);
    CHECK_NEAR(analysis.max_stage(), data_max + 0.1 * range, 1e-6);
}

// ---- Validate_WithValidConfiguration_ReturnsValid + various segment counts ----
void test_validate_valid() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    CHECK_TRUE(a->validate().is_valid);
    for (int segments = 1; segments <= 3; ++segments) {
        std::unique_ptr<RatingCurveAnalysis> s = make_test_analysis(segments);
        CHECK_TRUE(s->validate().is_valid);
    }
}

// ---- Constructor_WithMinimalData_InitializesCorrectly (>=10 pairs) ----
void test_minimal_data_valid() {
    RatingCurve seed;
    seed.set_use_default_flat_priors(false);
    seed.set_parameter_values(true_params());
    RatingCurve::SyntheticData d = seed.generate_synthetic_data(15, 1.0, 10.0, 12345);
    RatingCurveAnalysis analysis(
        std::make_unique<RatingCurve>(d.stage_data, d.discharge_data, 1));
    CHECK_TRUE(analysis.validate().is_valid);
}

// ---- Validate_WithNaNMinStage_ReturnsInvalid ----
void test_validate_nan_min_stage() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_min_stage(std::numeric_limits<double>::quiet_NaN());
    auto result = a->validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "Minimum stage"));
}

// ---- Validate_WithNaNMaxStage_ReturnsInvalid ----
void test_validate_nan_max_stage() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_max_stage(std::numeric_limits<double>::quiet_NaN());
    auto result = a->validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "Maximum stage"));
}

// ---- Validate_WithInvertedStageRange_ReturnsInvalid ----
void test_validate_inverted_range() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_min_stage(10.0);
    a->set_max_stage(5.0);
    auto result = a->validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "less than"));
}

// ---- Validate_WithInvalidStageBins_ReturnsInvalid (5, 1001) ----
void test_validate_invalid_stage_bins() {
    for (int bins : {5, 1001}) {
        std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
        a->set_stage_bins(bins);
        auto result = a->validate();
        CHECK_TRUE(!result.is_valid);
        CHECK_TRUE(any_contains(result.validation_messages, "stage bins"));
    }
}

// ---- Validate_WithValidStageBinsAtBoundary_ReturnsValid (10, 1000) ----
void test_validate_boundary_stage_bins() {
    for (int bins : {10, 1000}) {
        std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
        a->set_stage_bins(bins);
        CHECK_TRUE(a->validate().is_valid);
    }
}

// ---- ClearResults_ResetsAllResults + ClearResults_ClearsBayesianAnalysisResults ----
void test_clear_results() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->clear_results();
    CHECK_TRUE(!a->is_estimated());
    CHECK_TRUE(a->analysis_results() == nullptr);
    CHECK_TRUE(!a->bayesian_analysis().is_estimated());
}

// ---- StageBins_DefaultInitialization_HasDefaultValues ----
void test_stage_bins_default_values() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    CHECK_EQ(a->stage_bins(), 100);
    CHECK_TRUE(a->use_default_stage_bins());
}

// ---- StageBins_CanBeModified ----
void test_stage_bins_can_be_modified() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_min_stage(1.0);
    a->set_max_stage(20.0);
    a->set_stage_bins(200);
    CHECK_NEAR(a->min_stage(), 1.0, 1e-10);
    CHECK_NEAR(a->max_stage(), 20.0, 1e-10);
    CHECK_EQ(a->stage_bins(), 200);
}

// ---- UseDefaultStageBins_SetToTrue_ResetsFromData ----
void test_use_default_stage_bins_reset() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_min_stage(-100.0);
    a->set_max_stage(100.0);
    a->set_stage_bins(25);
    a->set_use_default_stage_bins(true);
    CHECK_EQ(a->stage_bins(), 100);
    CHECK_TRUE(a->min_stage() > -100.0);
    CHECK_TRUE(a->max_stage() < 100.0);
}

// ---- Grid reprocess: MinStage/MaxStage/StageBins change on fresh analysis does not populate
//      results or clear the (empty) MCMC fit. (C# *_ChangeOnFreshAnalysis_DoesNotClearMcmc) ----
void test_grid_change_on_fresh_analysis() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_min_stage(1.0);
    a->set_max_stage(100.0);
    a->set_stage_bins(600);
    CHECK_TRUE(a->analysis_results() == nullptr);
    CHECK_TRUE(!a->is_estimated());
    CHECK_TRUE(!a->bayesian_analysis().results().has_value());
}

// ---- ClearUncertaintyAnalysisResults_PreservesMcmcAndIsEstimated ----
void test_clear_uncertainty_results_safe_on_fresh() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->clear_uncertainty_analysis_results();
    CHECK_TRUE(a->analysis_results() == nullptr);
    CHECK_TRUE(!a->is_estimated());
    CHECK_TRUE(!a->bayesian_analysis().results().has_value());
}

// ---- BayesianAnalysis_CanBeConfigured ----
void test_bayesian_analysis_can_be_configured() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->bayesian_analysis().set_iterations(10000);
    a->bayesian_analysis().set_warmup_iterations(2000);
    a->bayesian_analysis().set_thinning_interval(10);
    a->bayesian_analysis().set_credible_interval_width(0.90);
    CHECK_EQ(a->bayesian_analysis().iterations(), 10000);
    CHECK_EQ(a->bayesian_analysis().warmup_iterations(), 2000);
    CHECK_EQ(a->bayesian_analysis().thinning_interval(), 10);
    CHECK_NEAR(a->bayesian_analysis().credible_interval_width(), 0.90, 1e-10);
}

// ---- run() throws on an invalid grid (C# RunAsync_WithInvalidConfiguration) ----
void test_run_invalid_configuration_throws() {
    std::unique_ptr<RatingCurveAnalysis> a = make_test_analysis();
    a->set_use_default_stage_bins(false);
    a->set_min_stage(100.0);
    a->set_max_stage(50.0);
    CHECK_THROWS(a->run());
}

}  // namespace

int main() {
    test_constructor_initializes();
    test_null_rating_curve_throws();
    test_default_stage_bins_from_data_range();
    test_constructor_segment_counts();
    test_empty_stage_data_defaults();
    test_default_stage_bins_padding();
    test_validate_valid();
    test_minimal_data_valid();
    test_validate_nan_min_stage();
    test_validate_nan_max_stage();
    test_validate_inverted_range();
    test_validate_invalid_stage_bins();
    test_validate_boundary_stage_bins();
    test_clear_results();
    test_stage_bins_default_values();
    test_stage_bins_can_be_modified();
    test_use_default_stage_bins_reset();
    test_grid_change_on_fresh_analysis();
    test_clear_uncertainty_results_safe_on_fresh();
    test_bayesian_analysis_can_be_configured();
    test_run_invalid_configuration_throws();

    return bftest::summary("rating_curve_analysis");
}
