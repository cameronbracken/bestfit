// Structural / behavioral tests for the univariate-FAMILY analyses (D1):
//   corehydro::analyses::MixtureAnalysis, PointProcessAnalysis, CompetingRiskAnalysis.
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/Univariate/MixtureAnalysisTests.cs
//   RMC.BestFit.Tests/Univariate/PointProcessAnalysisTests.cs
//   RMC.BestFit.Tests/Univariate/CompetingRiskAnalysisTests.cs
// There are NO numeric MCMC oracles for these families (the real parity lives in the absent
// RMC.BestFit.Verification project); every assertion here is a config-default, a
// model-property, a return-null-when-unestimated, or a Validate() check. Data fixtures are
// generated from the ported bit-exact Mersenne Twister (deterministic).
//
// SKIPPED / ADAPTED C# test methods (reasons in task-D1-report.md):
//   * XmlSerialization_RoundTrip_*, Constructor_WithNullXElement_*, XmlSerialization_WithBayesianSettings_*,
//     ToXElement_CreatesValidXmlStructure -- XML (de)serialization surface, dropped project-wide.
//   * ProbabilityOrdinates_Change_RaisesPropertyChanged, ModelPropertyChange_ClearsResults --
//     INotifyPropertyChanged / INotifyCollectionChanged; no notification system in this port.
//   * CancelAnalysis_WhenNotRunning_DoesNotThrow -- CancelAnalysis is WPF cancellation plumbing,
//     dropped in AnalysisBase.
//   * RunAsync_RaisesAnalysisStarting/CompletedEvent -- run-lifecycle events, dropped in AnalysisBase.
//   * IUnivariateAnalysis_*_IsAccessible -- adapted to a single interface-upcast check per family.
#include <memory>
#include <vector>

#include "corehydro/analyses/univariate/competing_risk_analysis.hpp"
#include "corehydro/analyses/univariate/mixture_analysis.hpp"
#include "corehydro/analyses/univariate/point_process_analysis.hpp"
#include "corehydro/analyses/support/i_univariate_analysis.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/data_collections/exact_series.hpp"
#include "corehydro/models/univariate_distribution/competing_risks_model.hpp"
#include "corehydro/models/univariate_distribution/mixture_model.hpp"
#include "corehydro/models/univariate_distribution/point_process_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/generalized_extreme_value.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::analyses::CompetingRiskAnalysis;
using corehydro::analyses::IUnivariateAnalysis;
using corehydro::analyses::MixtureAnalysis;
using corehydro::analyses::PointProcessAnalysis;
using corehydro::models::CompetingRisksModel;
using corehydro::models::DataFrame;
using corehydro::models::ExactSeries;
using corehydro::models::MixtureModel;
using corehydro::models::PointProcessModel;
using UDT = corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

// ---- Deterministic fixtures (ported Mersenne Twister is bit-exact) ----------------------

DataFrame bimodal_frame() {
    // A Normal(5500, 600) "snowmelt" component plus a Normal(20000, 3000) "rainfall" component.
    std::vector<double> low = corehydro::numerics::distributions::Normal(5500.0, 600.0)
                                  .generate_random_values(10, 12345);
    std::vector<double> high = corehydro::numerics::distributions::Normal(20000.0, 3000.0)
                                   .generate_random_values(10, 67890);
    std::vector<double> all = low;
    all.insert(all.end(), high.begin(), high.end());
    DataFrame df;
    df.set_exact_series(ExactSeries(all));
    df.calculate_plotting_positions();
    return df;
}

DataFrame gumbel_frame(double location, double scale, int size) {
    std::vector<double> v = corehydro::numerics::distributions::Gumbel(location, scale)
                                .generate_random_values(size, 12345);
    DataFrame df;
    df.set_exact_series(ExactSeries(v));
    df.calculate_plotting_positions();
    return df;
}

std::unique_ptr<MixtureModel> make_mixture_model(
    const std::vector<UDT>& types = {UDT::Normal, UDT::Normal}) {
    return std::make_unique<MixtureModel>(bimodal_frame(), types);
}

std::unique_ptr<PointProcessModel> make_point_process_model() {
    auto model = std::make_unique<PointProcessModel>();
    model->set_data_frame(gumbel_frame(20000.0, 3000.0, 30));
    model->set_threshold(14000.0);
    return model;
}

std::unique_ptr<CompetingRisksModel> make_competing_risks_model(
    const std::vector<UDT>& types = {UDT::Gumbel, UDT::Gumbel}) {
    return std::make_unique<CompetingRisksModel>(gumbel_frame(15000.0, 4000.0, 30), types);
}

// =========================================================================================
// MixtureAnalysis
// =========================================================================================

// C# Constructor_WithMixtureModel_InitializesCorrectly
void test_mixture_constructor_initializes() {
    MixtureModel* raw = nullptr;
    auto model = make_mixture_model();
    raw = model.get();
    MixtureAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.mixture_distribution() == raw);
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Constructor_WithNullModel_ThrowsArgumentNullException
void test_mixture_null_model_throws() {
    CHECK_THROWS(MixtureAnalysis(std::unique_ptr<MixtureModel>{}));
}

// C# Constructor_BayesianAnalysis_HasCorrectModel
void test_mixture_bayesian_has_correct_model() {
    auto model = make_mixture_model();
    MixtureModel* raw = model.get();
    MixtureAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

// C# Validate_WithValidConfiguration_ReturnsValid / Validate_PropagatesModelValidation
void test_mixture_validate_valid() {
    MixtureAnalysis analysis(make_mixture_model());
    CHECK_TRUE(analysis.validate().is_valid);
}

// C# Validate_WithGEVComponents_ReturnsValid
void test_mixture_validate_gev_valid() {
    MixtureAnalysis analysis(make_mixture_model(
        {UDT::GeneralizedExtremeValue, UDT::GeneralizedExtremeValue}));
    CHECK_TRUE(analysis.validate().is_valid);
}

// C# ClearResults_ResetsAllResults
void test_mixture_clear_results() {
    MixtureAnalysis analysis(make_mixture_model());
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# GetDistribution_WhenNotEstimated_ReturnsNull / GetPointEstimateDistribution_WhenNotEstimated
void test_mixture_getters_null_when_unestimated() {
    MixtureAnalysis analysis(make_mixture_model());
    CHECK_TRUE(analysis.get_distribution(0) == nullptr);
    CHECK_TRUE(analysis.get_point_estimate_distribution() == nullptr);
}

// C# Constructor_WithVariousComponentTypes_InitializesCorrectly (component type match)
void test_mixture_component_types() {
    for (UDT t : {UDT::Normal, UDT::LogNormal, UDT::Gumbel, UDT::GeneralizedExtremeValue}) {
        MixtureAnalysis analysis(make_mixture_model({t, t}));
        CHECK_TRUE(analysis.mixture_distribution().mixture()->component(0).type() == t);
    }
}

// C# Constructor_WithVariousComponentCounts_InitializesCorrectly / Constructor_WithSingleComponent
void test_mixture_component_counts() {
    for (int count : {1, 2, 3}) {
        std::vector<UDT> types(static_cast<std::size_t>(count), UDT::Normal);
        MixtureAnalysis analysis(make_mixture_model(types));
        CHECK_EQ(analysis.mixture_distribution().mixture()->component_count(), count);
    }
}

// C# ProbabilityOrdinates_DefaultInitialization_HasDefaultValues / ProbabilityOrdinates_CanBeModified
void test_mixture_probability_ordinates() {
    MixtureAnalysis analysis(make_mixture_model());
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
    analysis.probability_ordinates().clear();
    analysis.probability_ordinates().add(0.5);
    analysis.probability_ordinates().add(0.1);
    analysis.probability_ordinates().add(0.01);
    CHECK_EQ(analysis.probability_ordinates().count(), static_cast<std::size_t>(3));
    CHECK_TRUE(analysis.probability_ordinates()[0] == 0.5);
}

// C# BayesianAnalysis_HasDefaultSettings
void test_mixture_bayesian_defaults() {
    MixtureAnalysis analysis(make_mixture_model());
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
    CHECK_TRUE(analysis.bayesian_analysis().warmup_iterations() >= 0);
}

// C# IUnivariateAnalysis_* (interface upcast is usable)
void test_mixture_interface_accessible() {
    MixtureAnalysis analysis(make_mixture_model());
    IUnivariateAnalysis& iface = analysis;
    CHECK_TRUE(iface.probability_ordinates().count() > 0);
    CHECK_TRUE(iface.get_distribution(0) == nullptr);
}

// =========================================================================================
// PointProcessAnalysis
// =========================================================================================

// C# Constructor_WithModel_InitializesCorrectly
void test_pp_constructor_initializes() {
    auto model = make_point_process_model();
    PointProcessModel* raw = model.get();
    PointProcessAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.point_process() == raw);
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Constructor_WithNullModel_ThrowsArgumentNullException
void test_pp_null_model_throws() {
    CHECK_THROWS(PointProcessAnalysis(std::unique_ptr<PointProcessModel>{}));
}

// C# Constructor_BayesianAnalysis_HasCorrectModel
void test_pp_bayesian_has_correct_model() {
    auto model = make_point_process_model();
    PointProcessModel* raw = model.get();
    PointProcessAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

// C# Validate_WithValidConfiguration_ReturnsValid
void test_pp_validate_valid() {
    PointProcessAnalysis analysis(make_point_process_model());
    CHECK_TRUE(analysis.validate().is_valid);
}

// C# ClearResults_ResetsAllResults
void test_pp_clear_results() {
    PointProcessAnalysis analysis(make_point_process_model());
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# GetDistribution_WhenNotEstimated_ReturnsNull / GetPointEstimateDistribution_WhenNotEstimated
void test_pp_getters_null_when_unestimated() {
    PointProcessAnalysis analysis(make_point_process_model());
    CHECK_TRUE(analysis.get_distribution(0) == nullptr);
    CHECK_TRUE(analysis.get_point_estimate_distribution() == nullptr);
}

// C# Threshold_CanBeSet / ObservationPeriod_CanBeSet
void test_pp_threshold_and_years_settable() {
    PointProcessAnalysis analysis(make_point_process_model());
    analysis.point_process().set_threshold(15000.0);
    CHECK_TRUE(analysis.point_process().threshold() == 15000.0);
    analysis.point_process().set_total_years(365.25);
    CHECK_NEAR(analysis.point_process().total_years(), 365.25, 0.01);
}

// C# ProbabilityOrdinates_DefaultInitialization_HasDefaultValues
void test_pp_probability_ordinates_default() {
    PointProcessAnalysis analysis(make_point_process_model());
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
}

// C# BayesianAnalysis_HasDefaultSettings
void test_pp_bayesian_defaults() {
    PointProcessAnalysis analysis(make_point_process_model());
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
}

// C# IUnivariateAnalysis_ProbabilityOrdinates_IsAccessible
void test_pp_interface_accessible() {
    PointProcessAnalysis analysis(make_point_process_model());
    IUnivariateAnalysis& iface = analysis;
    CHECK_TRUE(iface.probability_ordinates().count() > 0);
}

// =========================================================================================
// CompetingRiskAnalysis
// =========================================================================================

// C# Constructor_WithModel_InitializesCorrectly
void test_cr_constructor_initializes() {
    auto model = make_competing_risks_model();
    CompetingRisksModel* raw = model.get();
    CompetingRiskAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.competing_risks_distribution() == raw);
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# Constructor_WithNullModel_ThrowsArgumentNullException
void test_cr_null_model_throws() {
    CHECK_THROWS(CompetingRiskAnalysis(std::unique_ptr<CompetingRisksModel>{}));
}

// C# Constructor_BayesianAnalysis_HasCorrectModel
void test_cr_bayesian_has_correct_model() {
    auto model = make_competing_risks_model();
    CompetingRisksModel* raw = model.get();
    CompetingRiskAnalysis analysis(std::move(model));
    CHECK_TRUE(&analysis.bayesian_analysis().model() == raw);
}

// C# Validate_WithValidConfiguration_ReturnsValid
void test_cr_validate_valid() {
    CompetingRiskAnalysis analysis(make_competing_risks_model());
    CHECK_TRUE(analysis.validate().is_valid);
}

// C# Validate_WithGEVComponents_ReturnsValid
void test_cr_validate_gev_valid() {
    CompetingRiskAnalysis analysis(make_competing_risks_model(
        {UDT::GeneralizedExtremeValue, UDT::GeneralizedExtremeValue}));
    CHECK_TRUE(analysis.validate().is_valid);
}

// C# ClearResults_ResetsAllResults
void test_cr_clear_results() {
    CompetingRiskAnalysis analysis(make_competing_risks_model());
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
}

// C# GetDistribution_WhenNotEstimated_ReturnsNull / GetPointEstimateDistribution_WhenNotEstimated
void test_cr_getters_null_when_unestimated() {
    CompetingRiskAnalysis analysis(make_competing_risks_model());
    CHECK_TRUE(analysis.get_distribution(0) == nullptr);
    CHECK_TRUE(analysis.get_point_estimate_distribution() == nullptr);
}

// C# Constructor_WithVariousComponentCounts_InitializesCorrectly (component count match)
void test_cr_component_counts() {
    for (int count : {2, 3}) {
        std::vector<UDT> types(static_cast<std::size_t>(count), UDT::Gumbel);
        CompetingRiskAnalysis analysis(make_competing_risks_model(types));
        CHECK_EQ(analysis.competing_risks_distribution().competing_risks()->component_count(),
                 count);
    }
}

// C# ProbabilityOrdinates_DefaultInitialization_HasDefaultValues / ProbabilityOrdinates_CanBeModified
void test_cr_probability_ordinates() {
    CompetingRiskAnalysis analysis(make_competing_risks_model());
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);
    analysis.probability_ordinates().clear();
    analysis.probability_ordinates().add(0.5);
    analysis.probability_ordinates().add(0.1);
    analysis.probability_ordinates().add(0.01);
    CHECK_EQ(analysis.probability_ordinates().count(), static_cast<std::size_t>(3));
}

// C# BayesianAnalysis_HasDefaultSettings
void test_cr_bayesian_defaults() {
    CompetingRiskAnalysis analysis(make_competing_risks_model());
    CHECK_TRUE(analysis.bayesian_analysis().iterations() > 0);
}

// C# IUnivariateAnalysis_ProbabilityOrdinates_IsAccessible
void test_cr_interface_accessible() {
    CompetingRiskAnalysis analysis(make_competing_risks_model());
    IUnivariateAnalysis& iface = analysis;
    CHECK_TRUE(iface.probability_ordinates().count() > 0);
}

}  // namespace

int main() {
    // Mixture
    test_mixture_constructor_initializes();
    test_mixture_null_model_throws();
    test_mixture_bayesian_has_correct_model();
    test_mixture_validate_valid();
    test_mixture_validate_gev_valid();
    test_mixture_clear_results();
    test_mixture_getters_null_when_unestimated();
    test_mixture_component_types();
    test_mixture_component_counts();
    test_mixture_probability_ordinates();
    test_mixture_bayesian_defaults();
    test_mixture_interface_accessible();

    // PointProcess
    test_pp_constructor_initializes();
    test_pp_null_model_throws();
    test_pp_bayesian_has_correct_model();
    test_pp_validate_valid();
    test_pp_clear_results();
    test_pp_getters_null_when_unestimated();
    test_pp_threshold_and_years_settable();
    test_pp_probability_ordinates_default();
    test_pp_bayesian_defaults();
    test_pp_interface_accessible();

    // CompetingRisk
    test_cr_constructor_initializes();
    test_cr_null_model_throws();
    test_cr_bayesian_has_correct_model();
    test_cr_validate_valid();
    test_cr_validate_gev_valid();
    test_cr_clear_results();
    test_cr_getters_null_when_unestimated();
    test_cr_component_counts();
    test_cr_probability_ordinates();
    test_cr_bayesian_defaults();
    test_cr_interface_accessible();

    return chtest::summary("univariate_family_analyses");
}
