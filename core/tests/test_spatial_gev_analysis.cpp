// Structural / behavioral tests for bestfit::analyses::SpatialGEVAnalysis (X4).
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/SpatialExtremes/SpatialGEVAnalysisTests.cs @ fc28c0c
// There are NO numeric MCMC oracles here (per the Phase-10 policy; the seeded end-to-end run
// lands via the X12 emitter). The tests cover: construction (incl. the null guard), BayesianAnalysis
// model identity, default probability ordinates, validation (valid / minimal-2-site / missing-data),
// ClearResults state transitions, the not-estimated guards on GetSiteQuantiles /
// PredictAtUngaugedLocation, site-weight handling, the IBayesianAnalysis / AnalysisBase interface
// membership, and the CrossValidationResults nullability. Hardcoded oracles in this C++-only ctest
// are correct (public-API oracle values otherwise live in fixtures/*.json).
//
// The inline at-site data does NOT reproduce the C# `new Random(12345)` .NET stream (the core has no
// System.Random port); these structural asserts check dimensions / validity / not-estimated guards
// only, never specific sampled values, so any finite positive 30x5 grid suffices. The core's
// bit-exact MersenneTwister drives GEV inverse-CDF sampling to build a realistic grid.
//
// SKIPPED C# test methods (WPF/serialization/threading -- no numerical content), each dropped with
// its ported-surface counterpart:
//   - XmlSerialization_RoundTrip_* / XmlSerialization_WithBayesianSettings_* /
//     XmlSerialization_PreservesIsEstimatedFlag / ToXElement_CreatesValidXmlStructure /
//     Constructor_WithNullXElement_* / Constructor_XmlWithNullSpatialGEV_* : the XML ctor +
//     ToXElement are a project-wide non-port.
//   - ClearResults_RaisesPropertyChangedEvents / ProbabilityOrdinates_Change_ClearsResults /
//     ModelPropertyChange_PropagatesPropertyChanged : INotifyPropertyChanged cascades; no
//     notification system in this port (explicit-invalidation replaces the handlers).
//   - CancelAnalysis_WhenNotRunning_* / CancelAnalysis_CancelsBayesianSimulation : cancellation
//     dropped.
#include <memory>
#include <vector>

#include "bestfit/analyses/spatial_extremes/spatial_gev_analysis.hpp"
#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_bayesian_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/trend_functions/general_linear_function.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

using bestfit::analyses::AnalysisBase;
using bestfit::analyses::IBayesianAnalysis;
using bestfit::analyses::SpatialGEVAnalysis;
using bestfit::models::spatial_extremes::SpatialGEV;
using bestfit::models::trend_functions::GeneralLinearFunction;
using bestfit::numerics::distributions::GeneralizedExtremeValue;
using bestfit::numerics::sampling::MersenneTwister;

using Grid = std::vector<std::vector<double>>;

namespace {

// Builds a realistic [n_obs][n_sites] grid of GEV(10000, 2500, 0) draws (bit-exact MersenneTwister
// -> GEV inverse-CDF). Values are illustrative only; the structural tests never compare them.
Grid make_at_site_data(int n_obs, int n_sites, std::uint32_t seed) {
    MersenneTwister mt(seed);
    GeneralizedExtremeValue gev(10000.0, 2500.0, 0.0);
    Grid data(static_cast<std::size_t>(n_obs),
              std::vector<double>(static_cast<std::size_t>(n_sites), 0.0));
    for (int site = 0; site < n_sites; ++site)
        for (int year = 0; year < n_obs; ++year)
            data[static_cast<std::size_t>(year)][static_cast<std::size_t>(site)] =
                gev.inverse_cdf(mt.next_double());
    return data;
}

Grid test_coordinates() {
    return {{0.0, 0.0}, {10.0, 5.0}, {22.0, 8.0}, {35.0, 12.0}, {50.0, 15.0}};
}

// 5-site x 30-observation default model (C# CreateTestSpatialGEV).
std::unique_ptr<SpatialGEV> make_test_spatial_gev() {
    return std::make_unique<SpatialGEV>(make_at_site_data(30, 5, 12345), test_coordinates(),
                                        GeneralLinearFunction("Location"),
                                        GeneralLinearFunction("Scale"),
                                        GeneralLinearFunction("Shape"));
}

// 2-site x 20-observation minimal model (C# CreateMinimalSpatialGEV).
std::unique_ptr<SpatialGEV> make_minimal_spatial_gev() {
    return std::make_unique<SpatialGEV>(make_at_site_data(20, 2, 54321),
                                        Grid{{0.0, 0.0}, {10.0, 0.0}},
                                        GeneralLinearFunction("Location"),
                                        GeneralLinearFunction("Scale"),
                                        GeneralLinearFunction("Shape"));
}

// 5-site x 30-observation model with 5 NaN cells (C# CreateSpatialGEVWithMissingData).
std::unique_ptr<SpatialGEV> make_spatial_gev_with_missing() {
    Grid data = make_at_site_data(30, 5, 12345);
    data[5][2] = std::numeric_limits<double>::quiet_NaN();
    data[10][0] = std::numeric_limits<double>::quiet_NaN();
    data[15][4] = std::numeric_limits<double>::quiet_NaN();
    data[20][1] = std::numeric_limits<double>::quiet_NaN();
    data[25][3] = std::numeric_limits<double>::quiet_NaN();
    return std::make_unique<SpatialGEV>(std::move(data), test_coordinates(),
                                        GeneralLinearFunction("Location"),
                                        GeneralLinearFunction("Scale"),
                                        GeneralLinearFunction("Shape"));
}

// ---- Constructor_WithSpatialGEV_InitializesCorrectly (C# 163) ----
void test_constructor_initializes() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    CHECK_TRUE(analysis.site_results() == nullptr);
    CHECK_TRUE(analysis.cross_validation_results() == nullptr);
}

// ---- Constructor_WithNullSpatialGEV_ThrowsArgumentNullException (C# 183) ----
void test_null_model_throws() {
    CHECK_THROWS(SpatialGEVAnalysis(std::unique_ptr<SpatialGEV>{}));
}

// ---- Constructor_BayesianAnalysis_HasCorrectModel (C# 192) ----
void test_bayesian_has_correct_model() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    const bestfit::models::ModelBase* model_ptr =
        static_cast<const bestfit::models::ModelBase*>(&analysis.spatial_gev());
    CHECK_TRUE(&analysis.bayesian_analysis().model() == model_ptr);
}

// ---- Constructor_InitializesDefaultProbabilityOrdinates (C# 206) ----
void test_default_probability_ordinates() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
}

// ---- Validate_WithValidConfiguration / _WithMinimalSites / _WithMissingData (C# 361/389/403) ----
void test_validate() {
    SpatialGEVAnalysis valid(make_test_spatial_gev());
    CHECK_TRUE(valid.validate().is_valid);
    SpatialGEVAnalysis minimal(make_minimal_spatial_gev());
    CHECK_TRUE(minimal.validate().is_valid);
    SpatialGEVAnalysis missing(make_spatial_gev_with_missing());
    CHECK_TRUE(missing.validate().is_valid);
}

// ---- ClearResults_ResetsAllResults + ClearResults_ClearsBayesianAnalysisResults (C# 421/438) ----
void test_clear_results() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    CHECK_TRUE(analysis.site_results() == nullptr);
    CHECK_TRUE(analysis.cross_validation_results() == nullptr);
    CHECK_TRUE(!analysis.bayesian_analysis().is_estimated());
    CHECK_TRUE(!analysis.bayesian_analysis().results().has_value());
}

// ---- GetSiteQuantiles_WhenNotEstimated_ThrowsInvalidOperationException (C# 559) ----
void test_get_site_quantiles_not_estimated_throws() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    std::vector<double> probs = {0.5, 0.1, 0.01};
    CHECK_THROWS(analysis.get_site_quantiles(0, probs));
}

// ---- PredictAtUngaugedLocation_WhenNotEstimated_Throws + _HasAccessibleCoordinates (C# 585/599) ----
void test_predict_at_ungauged_not_estimated_throws() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    std::vector<double> coords = {25.0, 10.0};
    std::vector<double> probs = {0.5, 0.1, 0.01};
    CHECK_THROWS(analysis.predict_at_ungauged_location(coords, {}, probs));
    // Coordinates on the underlying model are accessible.
    CHECK_EQ(analysis.spatial_gev().coordinates().size(), static_cast<std::size_t>(5));
}

// ---- SiteWeights_ZeroWeight_ExcludesSite (C# 695) ----
void test_site_weight_zero_excludes() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    analysis.spatial_gev().site_weights()[0] = 0.0;
    CHECK_EQ(analysis.spatial_gev().site_weights()[0], 0.0);
}

// ---- Analysis_ImplementsIBayesianAnalysis (C# 947) ----
void test_implements_ibayesian_analysis() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    CHECK_TRUE(dynamic_cast<IBayesianAnalysis*>(&analysis) != nullptr);
}

// ---- Analysis_InheritsFromAnalysisBase (C# 960) ----
void test_inherits_from_analysis_base() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    CHECK_TRUE(dynamic_cast<AnalysisBase*>(&analysis) != nullptr);
}

// ---- CrossValidationResults_InitiallyNull + ClearResults_ClearsCrossValidationResults (C# 977/989) ----
void test_cross_validation_results_null() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    CHECK_TRUE(analysis.cross_validation_results() == nullptr);
    analysis.clear_results();
    CHECK_TRUE(analysis.cross_validation_results() == nullptr);
}

// ---- SpatialGEV-surface asserts: site/observation count, coordinates, link functions (C# 714/726/738/754) ----
void test_spatial_gev_surface() {
    SpatialGEVAnalysis analysis(make_test_spatial_gev());
    CHECK_EQ(analysis.spatial_gev().sites(), 5);
    CHECK_EQ(analysis.spatial_gev().observations(), 30);
    CHECK_EQ(analysis.spatial_gev().coordinates().size(), static_cast<std::size_t>(5));
    CHECK_EQ(analysis.spatial_gev().coordinates()[0].size(), static_cast<std::size_t>(2));
    // Link functions default to log-link and are configurable.
    CHECK_TRUE(analysis.spatial_gev().use_log_link_for_location());
    CHECK_TRUE(analysis.spatial_gev().use_log_link_for_scale());
    analysis.spatial_gev().set_use_log_link_for_location(false);
    analysis.spatial_gev().set_use_log_link_for_scale(false);
    CHECK_TRUE(!analysis.spatial_gev().use_log_link_for_location());
    CHECK_TRUE(!analysis.spatial_gev().use_log_link_for_scale());
}

}  // namespace

int main() {
    test_constructor_initializes();
    test_null_model_throws();
    test_bayesian_has_correct_model();
    test_default_probability_ordinates();
    test_validate();
    test_clear_results();
    test_get_site_quantiles_not_estimated_throws();
    test_predict_at_ungauged_not_estimated_throws();
    test_site_weight_zero_excludes();
    test_implements_ibayesian_analysis();
    test_inherits_from_analysis_base();
    test_cross_validation_results_null();
    test_spatial_gev_surface();

    return bftest::summary("spatial_gev_analysis");
}
