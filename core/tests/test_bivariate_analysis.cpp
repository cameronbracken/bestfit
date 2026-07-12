// Structural / behavioral tests for corehydro::analyses::BivariateAnalysis (X3).
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/Bivariate/BivariateAnalysisTests.cs
//   RMC.BestFit.Tests/Bivariate/BivariateAnalysisXYOrdinatesReprocessTests.cs
// (both @ fc28c0c). There are NO numeric MCMC oracles here (per the Phase-10 policy; the
// seeded end-to-end run lands via the X12 emitter). The tests cover: construction (per copula
// type), BayesianAnalysis model identity, validation, ClearResults, XYOrdinates get/set, the
// point-estimator knob, and the XYOrdinates-reprocess-vs-clear behavior on a fresh
// (unestimated) analysis. Hardcoded oracles in this C++-only ctest are correct (public-API
// oracle values otherwise live in fixtures/*.json).
//
// SKIPPED C# test methods (WPF/serialization/threading -- no numerical content):
//   - XmlSerialization_* / Constructor_WithNullXElement_* / ToXElement_* : the XML ctor +
//     ToXElement are a project-wide non-port.
//   - XYOrdinates_Change_RaisesPropertyChanged, ClearResults_RaisesPropertyChangedForAnalysisResults,
//     CopulaTypeChange_ClearsResults, BayesianAnalysisPropertyChange_PropagatesCorrectly,
//     ModelParameterChange_ClearsResults, XYOrdinates_Change_ClearsResults [Ignore] :
//     INotifyPropertyChanged cascades; no notification system in this port. The reprocess-vs-clear
//     decisions those handlers encode are exercised here by driving the setters directly.
//   - CancelAnalysis_* : cancellation dropped.
//   - XYOrdinatesChange_EstimatedAnalysis_PreservesMcmcResults : requires the SetCustomMCMCResults
//     injection + numeric reprocess; the numeric path lands with the X12 emitter. The
//     preserve-on-fresh (no-clear) invariant is asserted here.
#include <memory>
#include <vector>

#include "corehydro/analyses/bivariate/bivariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/bivariate_distribution/bivariate_distribution.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/distributions/copulas/clayton_copula.hpp"
#include "corehydro/numerics/distributions/copulas/frank_copula.hpp"
#include "corehydro/numerics/distributions/copulas/gumbel_copula.hpp"
#include "corehydro/numerics/distributions/copulas/normal_copula.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::analyses::BivariateAnalysis;
using corehydro::estimation::BayesianAnalysis;
using corehydro::models::BivariateDistribution;
using corehydro::models::DataFrame;
using corehydro::models::ExactSeries;
using corehydro::models::UnivariateDistributionModel;
using corehydro::numerics::distributions::Gumbel;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::distributions::copulas::ClaytonCopula;
using corehydro::numerics::distributions::copulas::CopulaType;
using corehydro::numerics::distributions::copulas::FrankCopula;
using corehydro::numerics::distributions::copulas::GumbelCopula;
using corehydro::numerics::distributions::copulas::NormalCopula;

namespace {

constexpr int kFixtureSize = 200;

// C# InlineXData / InlineYData: Normal(100,15).GenerateRandomValues(200,12345) /
// Gumbel(50,15).GenerateRandomValues(200,67890). Bit-exact Mersenne Twister reproduces them.
const std::vector<double>& inline_x_data() {
    static const std::vector<double> v = Normal(100.0, 15.0).generate_random_values(kFixtureSize, 12345);
    return v;
}
const std::vector<double>& inline_y_data() {
    static const std::vector<double> v = Gumbel(50.0, 15.0).generate_random_values(kFixtureSize, 67890);
    return v;
}

// Owns the marginals for the lifetime of the analysis (the BivariateDistribution holds
// non-owning pointers into them, and the analysis owns the distribution). Declared so the
// analysis is destroyed BEFORE the marginals (reverse member-destruction order).
struct Fixture {
    std::unique_ptr<UnivariateDistributionModel> marginal_x;
    std::unique_ptr<UnivariateDistributionModel> marginal_y;
    std::unique_ptr<BivariateAnalysis> analysis;
};

// C# CreateMarginals: `new UnivariateDistribution(df, type)` -- the model ctor auto-estimates
// the marginal parameters from the data (MLE initials), so no explicit SetParameterValues is
// needed (mirroring the C# test, which relies on the same ctor auto-fit).
std::unique_ptr<UnivariateDistributionModel> make_marginal(UnivariateDistributionType type,
                                                           const std::vector<double>& data, int n) {
    std::vector<double> slice(data.begin(), data.begin() + n);
    DataFrame df;
    df.set_exact_series(ExactSeries(slice));
    return std::make_unique<UnivariateDistributionModel>(std::move(df), type);
}

// C# CreateTestAnalysis(CreateTestBivariateDistribution(copulaType, count)) + NumberOfChains=4.
Fixture make_test_analysis(CopulaType copula_type, int count = 100) {
    Fixture f;
    f.marginal_x = make_marginal(UnivariateDistributionType::Normal, inline_x_data(), count);
    f.marginal_y = make_marginal(UnivariateDistributionType::Gumbel, inline_y_data(), count);
    auto bd = std::make_unique<BivariateDistribution>(*f.marginal_x, *f.marginal_y, copula_type);
    f.analysis = std::make_unique<BivariateAnalysis>(std::move(bd));
    f.analysis->bayesian_analysis().set_number_of_chains(4);
    return f;
}

Fixture make_test_analysis(int count = 100) { return make_test_analysis(CopulaType::Normal, count); }

// C# CreateTestXYOrdinates: 5 (x, y) pairs.
std::vector<BivariateAnalysis::XYOrdinate> make_test_xy_ordinates() {
    return {{80, 40}, {100, 50}, {120, 60}, {140, 70}, {160, 80}};
}

// ---- Constructor_WithBivariateDistribution_InitializesCorrectly ----
void test_constructor_initializes() {
    Fixture f = make_test_analysis();
    CHECK_TRUE(f.analysis->xy_ordinates().size() >= 1);
    CHECK_TRUE(!f.analysis->is_estimated());
    CHECK_TRUE(f.analysis->analysis_results() == nullptr);
    // BayesianAnalysis is over the same model.
    CHECK_TRUE(&f.analysis->bayesian_analysis().model() == &f.analysis->bivariate_distribution());
}

// ---- Constructor_WithNullDistribution_ThrowsArgumentNullException ----
void test_null_distribution_throws() {
    CHECK_THROWS(BivariateAnalysis(std::unique_ptr<BivariateDistribution>{}));
}

// ---- Constructor_WithVariousCopulaTypes_InitializesCorrectly + copula-instance checks ----
void test_constructor_copula_types() {
    Fixture fn = make_test_analysis(CopulaType::Normal);
    CHECK_TRUE(dynamic_cast<const NormalCopula*>(&fn.analysis->bivariate_distribution().copula()) != nullptr);
    Fixture fc = make_test_analysis(CopulaType::Clayton);
    CHECK_TRUE(dynamic_cast<const ClaytonCopula*>(&fc.analysis->bivariate_distribution().copula()) != nullptr);
    Fixture ff = make_test_analysis(CopulaType::Frank);
    CHECK_TRUE(dynamic_cast<const FrankCopula*>(&ff.analysis->bivariate_distribution().copula()) != nullptr);
    Fixture fg = make_test_analysis(CopulaType::Gumbel);
    CHECK_TRUE(dynamic_cast<const GumbelCopula*>(&fg.analysis->bivariate_distribution().copula()) != nullptr);
    // CopulaType round-trips.
    CHECK_TRUE(fg.analysis->bivariate_distribution().copula_type() == CopulaType::Gumbel);
}

// ---- BivariateDistribution_Property_ReturnsCorrectReference ----
void test_distribution_property_reference() {
    Fixture f = make_test_analysis();
    // Identity: BayesianAnalysis model IS the analysis's bivariate distribution.
    CHECK_TRUE(&f.analysis->bivariate_distribution() ==
               static_cast<const BivariateDistribution*>(&f.analysis->bayesian_analysis().model()));
}

// ---- Validate_WithValidConfiguration_ReturnsValid + copula types + small samples ----
void test_validate_valid() {
    Fixture f = make_test_analysis();
    CHECK_TRUE(f.analysis->validate().is_valid);
    for (CopulaType ct : {CopulaType::Normal, CopulaType::Clayton, CopulaType::Frank, CopulaType::Gumbel}) {
        Fixture c = make_test_analysis(ct);
        CHECK_TRUE(c.analysis->validate().is_valid);
    }
    // Small-sample edge cases (30, 50).
    Fixture s30 = make_test_analysis(30);
    CHECK_TRUE(s30.analysis->validate().is_valid);
    Fixture s50 = make_test_analysis(50);
    CHECK_TRUE(s50.analysis->validate().is_valid);
}

// ---- ClearResults_ResetsAllResults + ClearResults_ClearsBayesianAnalysisResults ----
void test_clear_results() {
    Fixture f = make_test_analysis();
    f.analysis->clear_results();
    CHECK_TRUE(!f.analysis->is_estimated());
    CHECK_TRUE(f.analysis->analysis_results() == nullptr);
    CHECK_TRUE(!f.analysis->bayesian_analysis().is_estimated());
    CHECK_TRUE(!f.analysis->bayesian_analysis().results().has_value());
}

// ---- XYOrdinates_DefaultInitialization_HasDefaultValues ----
void test_xy_ordinates_default() {
    Fixture f = make_test_analysis();
    CHECK_TRUE(f.analysis->xy_ordinates().size() >= 1);
}

// ---- XYOrdinates_CanBeSet ----
void test_xy_ordinates_can_be_set() {
    Fixture f = make_test_analysis();
    f.analysis->set_xy_ordinates(make_test_xy_ordinates());
    CHECK_EQ(f.analysis->xy_ordinates().size(), static_cast<std::size_t>(5));
}

// ---- AnalysisResults_BeforeEstimation_IsNull + IsEstimated_BeforeEstimation_IsFalse ----
void test_before_estimation_state() {
    Fixture f = make_test_analysis();
    CHECK_TRUE(f.analysis->analysis_results() == nullptr);
    CHECK_TRUE(!f.analysis->is_estimated());
}

// ---- BayesianAnalysis_PointEstimator_CanBeSet (Mean / MAP) ----
void test_point_estimator_can_be_set() {
    using corehydro::estimation::PointEstimateType;
    Fixture f = make_test_analysis();
    f.analysis->bayesian_analysis().set_point_estimator(PointEstimateType::PosteriorMean);
    CHECK_TRUE(f.analysis->bayesian_analysis().point_estimator() == PointEstimateType::PosteriorMean);
    f.analysis->bayesian_analysis().set_point_estimator(PointEstimateType::PosteriorMode);
    CHECK_TRUE(f.analysis->bayesian_analysis().point_estimator() == PointEstimateType::PosteriorMode);
}

// ---- BayesianAnalysis_SettingsCanBeConfigured ----
void test_bayesian_settings_configured() {
    Fixture f = make_test_analysis();
    f.analysis->bayesian_analysis().set_iterations(10000);
    f.analysis->bayesian_analysis().set_warmup_iterations(2000);
    f.analysis->bayesian_analysis().set_thinning_interval(2);
    CHECK_EQ(f.analysis->bayesian_analysis().iterations(), 10000);
    CHECK_EQ(f.analysis->bayesian_analysis().warmup_iterations(), 2000);
    CHECK_EQ(f.analysis->bayesian_analysis().thinning_interval(), 2);
}

// ---- XYOrdinates change on a fresh analysis does not populate results / clear the (empty) fit ----
// (C# XYOrdinates_ChangeOnFreshAnalysis_DoesNotClearMcmc) ----
void test_xy_ordinates_change_fresh() {
    Fixture f = make_test_analysis();
    f.analysis->set_xy_ordinates(make_test_xy_ordinates());
    CHECK_TRUE(f.analysis->analysis_results() == nullptr);
    CHECK_TRUE(!f.analysis->is_estimated());
    CHECK_TRUE(!f.analysis->bayesian_analysis().results().has_value());
}

// ---- ClearFrequencyAnalysisResults_PreservesMcmcAndIsEstimated ----
void test_clear_frequency_results_safe_on_fresh() {
    Fixture f = make_test_analysis();
    f.analysis->clear_frequency_analysis_results();
    CHECK_TRUE(f.analysis->analysis_results() == nullptr);
    CHECK_TRUE(!f.analysis->is_estimated());
    CHECK_TRUE(!f.analysis->bayesian_analysis().results().has_value());
}

// ---- Analysis_WithDifferentMarginalTypes_ValidatesCorrectly (Normal-X, GEV-Y) ----
void test_different_marginal_types() {
    Fixture f;
    f.marginal_x = make_marginal(UnivariateDistributionType::Normal, inline_x_data(), 100);
    f.marginal_y = make_marginal(UnivariateDistributionType::GeneralizedExtremeValue, inline_y_data(), 100);
    auto bd = std::make_unique<BivariateDistribution>(*f.marginal_x, *f.marginal_y, CopulaType::Normal);
    f.analysis = std::make_unique<BivariateAnalysis>(std::move(bd));
    f.analysis->bayesian_analysis().set_number_of_chains(4);
    CHECK_TRUE(f.analysis->validate().is_valid);
}

}  // namespace

int main() {
    test_constructor_initializes();
    test_null_distribution_throws();
    test_constructor_copula_types();
    test_distribution_property_reference();
    test_validate_valid();
    test_clear_results();
    test_xy_ordinates_default();
    test_xy_ordinates_can_be_set();
    test_before_estimation_state();
    test_point_estimator_can_be_set();
    test_bayesian_settings_configured();
    test_xy_ordinates_change_fresh();
    test_clear_frequency_results_safe_on_fresh();
    test_different_marginal_types();

    return chtest::summary("bivariate_analysis");
}
