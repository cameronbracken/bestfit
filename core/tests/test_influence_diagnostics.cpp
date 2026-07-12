// Standalone C++-only ctest for corehydro::diagnostics::InfluenceDiagnostics and
// corehydro::diagnostics::PriorInfluenceDiagnostics (Phase 9a, Task D4).
//
// Oracle for behavior is the C# source itself
// (upstream/RMC-BestFit/src/RMC.BestFit/Diagnostics/{InfluenceDiagnostics,
// PriorInfluenceDiagnostics}.cs @ fc28c0c) and their unit-test transcriptions
// (.../RMC.BestFit.Tests/Diagnostics/{InfluenceDiagnostics,PriorInfluenceDiagnostics}Tests.cs).
// Per the validation model, internal-support types get C++-only ctests transcribing the upstream
// test oracles plus deterministic checks; the exact seeded Pareto-k / prior-influence numeric
// values (which need a live MCMC fit) land as D6 emitter fixtures, NOT here.
//
// SKIPPED from the upstream InfluenceDiagnostics test file:
//   - Constructor_NullObservations_Throws, Constructor_FromArrays_NullArgs_Throw: C++ vectors are
//     never null (no analogue).
//   - XmlSerialization_RoundTrip_*, XmlConstructor_NullXml_Throws: no XElement port in this core.
// SKIPPED from the upstream PriorInfluenceDiagnostics test file:
//   - Constructor_Components_Null_Throws, Constructor_Xml_Null_Throws: no null vector / no
//     XElement port.
//   - IsPriorInfluential_{LowRatio,HighRatio,RatioEqualsThreshold}: the C# constructs these via
//     the XElement ctor (CreateDiagnosticsForTesting injects Total* directly). With no XElement
//     port, the only literal path leaves Total* at 0 (covered by
//     test_prior_constructor_components_totals_remain_default); the non-zero-ratio boundary cases
//     require a live MCMC fit and are D6 emitter fixtures.
//   - XmlRoundTrip_*, PriorComponentSummary_XmlRoundTrip_*, PriorComponentSummary_ToString_*: XML
//     / presentation-only, no XElement / ToString port.
#include <climits>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "corehydro/diagnostics/influence_diagnostics.hpp"
#include "corehydro/diagnostics/prior_influence_diagnostics.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "check.hpp"

using corehydro::diagnostics::InfluenceDiagnostics;
using corehydro::diagnostics::ParetoKCategory;
using corehydro::diagnostics::PriorInfluenceDiagnostics;
using corehydro::models::DataComponent;
using corehydro::models::DataComponentType;
using corehydro::models::PriorComponentType;

using ObservationInfluence = InfluenceDiagnostics::ObservationInfluence;
using PriorComponentSummary = PriorInfluenceDiagnostics::PriorComponentSummary;

namespace {

// --- Helpers (mirroring the C# Obs / MakeSummary test helpers) ------------------------------

ObservationInfluence obs(int index, double k, double elpd = -2.0) {
    return ObservationInfluence(index, k, elpd, static_cast<double>(index),
                                DataComponentType::Exact, 1);
}

PriorComponentSummary make_summary(std::string name, PriorComponentType type, double mean,
                                   double sd = 0.1, double min = -5.0, double max = 0.0) {
    return PriorComponentSummary(std::move(name), type, mean, sd, min, max);
}

// ============================================================================================
// InfluenceDiagnostics
// ============================================================================================

// C# DefaultConstructor_RollupsAreNaN_AndCountIsZero (34).
void test_influence_default_ctor_rollups_nan_count_zero() {
    InfluenceDiagnostics diag;

    CHECK_EQ(diag.count(), 0);
    CHECK_TRUE(std::isnan(diag.mean_pareto_k()));
    CHECK_TRUE(std::isnan(diag.max_pareto_k()));
    CHECK_EQ(diag.count_pareto_k_above_05(), 0);
    CHECK_EQ(diag.count_pareto_k_above_07(), 0);
    CHECK_EQ(diag.count_pareto_k_above_10(), 0);
}

// C# Constructor_FromObservations_ComputesSummaryStatistics (66).
void test_influence_from_observations_summary_stats() {
    std::vector<ObservationInfluence> arr{
        obs(0, 0.2), obs(1, 0.4), obs(2, 0.6), obs(3, 0.8), obs(4, 1.2)};

    InfluenceDiagnostics diag(arr);

    CHECK_EQ(diag.count(), 5);
    CHECK_NEAR(diag.mean_pareto_k(), (0.2 + 0.4 + 0.6 + 0.8 + 1.2) / 5.0, 1e-12);
    CHECK_NEAR(diag.max_pareto_k(), 1.2, 1e-12);
    CHECK_EQ(diag.count_pareto_k_above_05(), 3);  // 0.6, 0.8, 1.2
    CHECK_EQ(diag.count_pareto_k_above_07(), 2);  // 0.8, 1.2
    CHECK_EQ(diag.count_pareto_k_above_10(), 1);  // 1.2
}

// C# Constructor_AllNaN_LeavesRollupsAsNaN (93).
void test_influence_all_nan_rollups_nan() {
    std::vector<ObservationInfluence> arr{obs(0, std::nan("")), obs(1, std::nan(""))};

    InfluenceDiagnostics diag(arr);

    CHECK_TRUE(std::isnan(diag.mean_pareto_k()));
    CHECK_TRUE(std::isnan(diag.max_pareto_k()));
}

// C# Constructor_FromArrays_LengthMismatch_Throws (116).
void test_influence_from_arrays_length_mismatch_throws() {
    std::vector<double> pareto_k{0.1, 0.2, 0.3};
    std::vector<double> elpd{-1.0, -2.0};

    CHECK_THROWS(InfluenceDiagnostics(pareto_k, elpd));
}

// C# Constructor_FromArrays_BuildsObservations (142).
void test_influence_from_arrays_builds_observations() {
    std::vector<double> pareto_k{0.3, 0.6, 0.8};
    std::vector<double> elpd{-1.0, -1.5, -2.0};

    InfluenceDiagnostics diag(pareto_k, elpd);

    CHECK_EQ(diag.count(), 3);
    for (int i = 0; i < diag.count(); ++i) {
        CHECK_EQ(diag[i].index(), i);
        CHECK_NEAR(diag[i].pareto_k(), pareto_k[static_cast<std::size_t>(i)], 1e-15);
        CHECK_NEAR(diag[i].elpd_loo(), elpd[static_cast<std::size_t>(i)], 1e-15);
    }
}

// C# Constructor_FromArrays_DataComponentsLengthMismatch_Throws (163).
void test_influence_from_arrays_datacomponents_mismatch_throws() {
    std::vector<double> pareto_k{0.3, 0.6};
    std::vector<double> elpd{-1.0, -1.5};
    std::vector<DataComponent> data_components{DataComponent(0, -1.0, 100.0)};

    CHECK_THROWS(InfluenceDiagnostics(pareto_k, elpd, data_components));
}

// C# ProportionProblematic_EmptyDiagnostics_IsZero (184).
void test_influence_proportion_problematic_empty_is_zero() {
    InfluenceDiagnostics diag;
    CHECK_NEAR(diag.proportion_problematic(), 0.0, 0.0);
}

// C# IsReliable_AnyVeryBadObs_IsFalse (196).
void test_influence_is_reliable_any_very_bad_is_false() {
    std::vector<ObservationInfluence> arr;
    for (int i = 0; i < 99; ++i) arr.push_back(obs(i, 0.1));
    arr.push_back(obs(99, 1.5));

    InfluenceDiagnostics diag(arr);
    CHECK_TRUE(!diag.is_reliable());
}

// C# IsReliable_AllGood_IsTrue (214).
void test_influence_is_reliable_all_good_is_true() {
    std::vector<ObservationInfluence> arr;
    for (int i = 0; i < 100; ++i) arr.push_back(obs(i, 0.2));

    InfluenceDiagnostics diag(arr);
    CHECK_TRUE(diag.is_reliable());
}

// C# GetMostInfluentialObservations_OrdersByDescendingParetoK (232).
void test_influence_get_most_influential_orders_descending() {
    std::vector<ObservationInfluence> arr{obs(0, 0.2), obs(1, 0.9), obs(2, 0.5), obs(3, 0.1)};

    InfluenceDiagnostics diag(arr);
    auto top = diag.get_most_influential_observations();

    CHECK_NEAR(top[0].pareto_k(), 0.9, 1e-12);
    CHECK_NEAR(top[1].pareto_k(), 0.5, 1e-12);
    CHECK_NEAR(top[2].pareto_k(), 0.2, 1e-12);
    CHECK_NEAR(top[3].pareto_k(), 0.1, 1e-12);
}

// C# GetMostInfluentialObservations_RespectsTopN (255).
void test_influence_get_most_influential_respects_top_n() {
    std::vector<ObservationInfluence> arr{obs(0, 0.1), obs(1, 0.4), obs(2, 0.7), obs(3, 1.1)};

    InfluenceDiagnostics diag(arr);
    auto top2 = diag.get_most_influential_observations(2);

    CHECK_EQ(static_cast<int>(top2.size()), 2);
    CHECK_NEAR(top2[0].pareto_k(), 1.1, 1e-12);
    CHECK_NEAR(top2[1].pareto_k(), 0.7, 1e-12);
}

// C# GetProblematicObservations_DefaultThreshold_FiltersAndOrders (275).
void test_influence_get_problematic_default_threshold() {
    std::vector<ObservationInfluence> arr{obs(0, 0.2), obs(1, 0.6), obs(2, 0.8), obs(3, 1.2)};

    InfluenceDiagnostics diag(arr);
    auto problematic = diag.get_problematic_observations();

    CHECK_EQ(static_cast<int>(problematic.size()), 2);
    CHECK_NEAR(problematic[0].pareto_k(), 1.2, 1e-12);
    CHECK_NEAR(problematic[1].pareto_k(), 0.8, 1e-12);
}

// C# GetReliabilitySummary_AllBranches_ReturnExpectedKeyword (300).
void test_influence_reliability_summary_all_branches() {
    CHECK_TRUE(InfluenceDiagnostics().get_reliability_summary().find("No observations") !=
               std::string::npos);

    std::vector<ObservationInfluence> all_good{obs(0, 0.1), obs(1, 0.2)};
    CHECK_TRUE(InfluenceDiagnostics(all_good).get_reliability_summary().find("GOOD") !=
               std::string::npos);

    std::vector<ObservationInfluence> ok{obs(0, 0.5), obs(1, 0.6)};
    CHECK_TRUE(InfluenceDiagnostics(ok).get_reliability_summary().find("OK") != std::string::npos);

    std::vector<ObservationInfluence> caution;
    for (int i = 0; i < 98; ++i) caution.push_back(obs(i, 0.1));
    caution.push_back(obs(98, 0.8));
    caution.push_back(obs(99, 0.9));
    CHECK_TRUE(InfluenceDiagnostics(caution).get_reliability_summary().find("CAUTION") !=
               std::string::npos);

    std::vector<ObservationInfluence> unreliable{obs(0, 1.5)};
    CHECK_TRUE(InfluenceDiagnostics(unreliable).get_reliability_summary().find("UNRELIABLE") !=
               std::string::npos);
}

// C# Indexer_ReturnsObservationByIndex (338).
void test_influence_indexer() {
    std::vector<ObservationInfluence> arr{obs(0, 0.1), obs(1, 0.5), obs(2, 0.9)};
    InfluenceDiagnostics diag(arr);

    CHECK_NEAR(diag[0].pareto_k(), 0.1, 1e-12);
    CHECK_NEAR(diag[1].pareto_k(), 0.5, 1e-12);
    CHECK_NEAR(diag[2].pareto_k(), 0.9, 1e-12);
}

// C# ParetoKCategory boundary classifications (Category property body, C# 497): the 0.5 / 0.7 /
// 1.0 edges are exact and half-open (Good < 0.5 <= OK < 0.7 <= Bad < 1.0 <= VeryBad).
void test_influence_category_boundaries() {
    CHECK_TRUE(obs(0, 0.49).category() == ParetoKCategory::Good);
    CHECK_TRUE(obs(0, 0.5).category() == ParetoKCategory::OK);
    CHECK_TRUE(obs(0, 0.69).category() == ParetoKCategory::OK);
    CHECK_TRUE(obs(0, 0.7).category() == ParetoKCategory::Bad);
    CHECK_TRUE(obs(0, 0.99).category() == ParetoKCategory::Bad);
    CHECK_TRUE(obs(0, 1.0).category() == ParetoKCategory::VeryBad);
    CHECK_TRUE(obs(0, 1.5).category() == ParetoKCategory::VeryBad);
}

// ============================================================================================
// PriorInfluenceDiagnostics
// ============================================================================================

// C# Constructor_Default_EmptyComponentsArray (84).
void test_prior_default_ctor_empty_components() {
    PriorInfluenceDiagnostics diag;

    CHECK_EQ(static_cast<int>(diag.components().size()), 0);
    CHECK_EQ(diag.count(), 0);
}

// C# Constructor_Components_StoresArray_TotalsRemainDefault (124): the (components) ctor does not
// derive Total* from the component means; ratio is 0 and the prior is not flagged influential.
void test_prior_constructor_components_totals_remain_default() {
    std::vector<PriorComponentSummary> comps{
        make_summary("param_prior", PriorComponentType::ParameterPrior, -2.0),
        make_summary("jeffreys", PriorComponentType::JeffreysScalePrior, -0.5)};

    PriorInfluenceDiagnostics diag(comps);

    CHECK_EQ(static_cast<int>(diag.components().size()), 2);
    CHECK_EQ(diag.count(), 2);
    CHECK_TRUE(diag.components()[0].name() == "param_prior");
    CHECK_TRUE(diag.components()[1].name() == "jeffreys");
    CHECK_NEAR(diag.total_prior_log_likelihood(), 0.0, 1e-10);
    CHECK_NEAR(diag.total_data_log_likelihood(), 0.0, 1e-10);
    CHECK_NEAR(diag.prior_to_data_ratio(), 0.0, 1e-10);
    CHECK_TRUE(!diag.is_prior_influential());
}

// C# Constructor_Components_Empty_RatioIsZero (150).
void test_prior_constructor_components_empty_ratio_zero() {
    PriorInfluenceDiagnostics diag(std::vector<PriorComponentSummary>{});

    CHECK_EQ(diag.count(), 0);
    CHECK_NEAR(diag.prior_to_data_ratio(), 0.0, 1e-10);
}

// C# GetComponentsByType_FiltersCorrectly (227).
void test_prior_get_components_by_type_filters() {
    std::vector<PriorComponentSummary> comps{
        make_summary("param1", PriorComponentType::ParameterPrior, -1.0),
        make_summary("jeffreys", PriorComponentType::JeffreysScalePrior, -0.5),
        make_summary("param2", PriorComponentType::ParameterPrior, -0.8)};
    PriorInfluenceDiagnostics diag(comps);

    CHECK_EQ(static_cast<int>(diag.get_components_by_type(PriorComponentType::ParameterPrior).size()),
             2);
    CHECK_EQ(static_cast<int>(
                 diag.get_components_by_type(PriorComponentType::JeffreysScalePrior).size()),
             1);
    CHECK_EQ(static_cast<int>(diag.get_components_by_type(PriorComponentType::QuantilePrior).size()),
             0);
}

// C# GetComponentsByType_EmptyDiagnostics_ReturnsEmpty (250).
void test_prior_get_components_by_type_empty() {
    PriorInfluenceDiagnostics diag;
    CHECK_EQ(static_cast<int>(diag.get_components_by_type(PriorComponentType::ParameterPrior).size()),
             0);
}

// C# GetMostConstrainingComponents_SortedAscending (266): sorted by mean LL ascending.
void test_prior_get_most_constraining_sorted_ascending() {
    std::vector<PriorComponentSummary> comps{
        make_summary("mild", PriorComponentType::ParameterPrior, -0.5),
        make_summary("strong", PriorComponentType::ParameterPrior, -5.0),
        make_summary("medium", PriorComponentType::ParameterPrior, -2.0)};
    PriorInfluenceDiagnostics diag(comps);

    auto sorted = diag.get_most_constraining_components();

    CHECK_TRUE(sorted[0].name() == "strong");
    CHECK_TRUE(sorted[1].name() == "medium");
    CHECK_TRUE(sorted[2].name() == "mild");
}

// C# GetMostConstrainingComponents_TopN_LimitsResult (287).
void test_prior_get_most_constraining_top_n() {
    std::vector<PriorComponentSummary> comps{
        make_summary("a", PriorComponentType::ParameterPrior, -1.0),
        make_summary("b", PriorComponentType::ParameterPrior, -2.0),
        make_summary("c", PriorComponentType::ParameterPrior, -3.0)};
    PriorInfluenceDiagnostics diag(comps);

    CHECK_EQ(static_cast<int>(diag.get_most_constraining_components(2).size()), 2);
}

// C# GetMostConstrainingComponents_EmptyDiagnostics_ReturnsEmpty (306).
void test_prior_get_most_constraining_empty() {
    PriorInfluenceDiagnostics diag;
    CHECK_EQ(static_cast<int>(diag.get_most_constraining_components().size()), 0);
}

// C# GetContributionByType_AggregatesCorrectly (322).
void test_prior_get_contribution_by_type_aggregates() {
    std::vector<PriorComponentSummary> comps{
        make_summary("p1", PriorComponentType::ParameterPrior, -1.0),
        make_summary("p2", PriorComponentType::ParameterPrior, -2.0),
        make_summary("j", PriorComponentType::JeffreysScalePrior, -0.5)};
    PriorInfluenceDiagnostics diag(comps);

    auto contributions = diag.get_contribution_by_type();

    CHECK_TRUE(contributions.count(PriorComponentType::ParameterPrior) == 1);
    CHECK_TRUE(contributions.count(PriorComponentType::JeffreysScalePrior) == 1);
    CHECK_NEAR(contributions[PriorComponentType::ParameterPrior], -3.0, 1e-10);
    CHECK_NEAR(contributions[PriorComponentType::JeffreysScalePrior], -0.5, 1e-10);
}

// C# GetContributionByType_EmptyDiagnostics_ReturnsEmptyDictionary (344).
void test_prior_get_contribution_by_type_empty() {
    PriorInfluenceDiagnostics diag;
    CHECK_EQ(static_cast<int>(diag.get_contribution_by_type().size()), 0);
}

// C# GetSummary_NoComponents_ReturnsNoDataMessage (360).
void test_prior_get_summary_no_components() {
    PriorInfluenceDiagnostics diag;
    CHECK_TRUE(diag.get_summary().find("No prior components") != std::string::npos);
}

// C# GetSummary_WithComponents_ContainsSummaryHeader (373).
void test_prior_get_summary_with_components_header() {
    std::vector<PriorComponentSummary> comps{
        make_summary("p", PriorComponentType::ParameterPrior, -2.0)};
    PriorInfluenceDiagnostics diag(comps);

    CHECK_TRUE(diag.get_summary().find("Prior Influence Summary") != std::string::npos);
}

// C# Indexer_ReturnsCorrectComponent (392).
void test_prior_indexer() {
    std::vector<PriorComponentSummary> comps{
        make_summary("first", PriorComponentType::ParameterPrior, -1.0),
        make_summary("second", PriorComponentType::QuantilePrior, -3.0)};
    PriorInfluenceDiagnostics diag(comps);

    CHECK_TRUE(diag[0].name() == "first");
    CHECK_TRUE(diag[1].name() == "second");
}

// C# PriorComponentSummary_Constructor_SetsAllProperties (474).
void test_prior_component_summary_ctor_sets_all() {
    PriorComponentSummary summary("mu_prior", PriorComponentType::ParameterPrior, -3.5, 0.4, -6.0,
                                  -1.0);

    CHECK_TRUE(summary.name() == "mu_prior");
    CHECK_TRUE(summary.type() == PriorComponentType::ParameterPrior);
    CHECK_NEAR(summary.mean_log_likelihood(), -3.5, 1e-10);
    CHECK_NEAR(summary.standard_deviation(), 0.4, 1e-10);
    CHECK_NEAR(summary.min_log_likelihood(), -6.0, 1e-10);
    CHECK_NEAR(summary.max_log_likelihood(), -1.0, 1e-10);
}

// C# PriorComponentSummary_CoefficientOfVariation_ComputedCorrectly (491): CV = |sd/mean|.
void test_prior_component_summary_cv_computed() {
    PriorComponentSummary summary("p", PriorComponentType::ParameterPrior, -4.0, 2.0, -8.0, 0.0);
    CHECK_NEAR(summary.coefficient_of_variation(), 0.5, 1e-10);
}

// C# PriorComponentSummary_CoefficientOfVariation_ZeroMean_ReturnsZero (503).
void test_prior_component_summary_cv_zero_mean() {
    PriorComponentSummary summary("p", PriorComponentType::ParameterPrior, 0.0, 1.0, -1.0, 1.0);
    CHECK_NEAR(summary.coefficient_of_variation(), 0.0, 1e-10);
}

}  // namespace

int main() {
    test_influence_default_ctor_rollups_nan_count_zero();
    test_influence_from_observations_summary_stats();
    test_influence_all_nan_rollups_nan();
    test_influence_from_arrays_length_mismatch_throws();
    test_influence_from_arrays_builds_observations();
    test_influence_from_arrays_datacomponents_mismatch_throws();
    test_influence_proportion_problematic_empty_is_zero();
    test_influence_is_reliable_any_very_bad_is_false();
    test_influence_is_reliable_all_good_is_true();
    test_influence_get_most_influential_orders_descending();
    test_influence_get_most_influential_respects_top_n();
    test_influence_get_problematic_default_threshold();
    test_influence_reliability_summary_all_branches();
    test_influence_indexer();
    test_influence_category_boundaries();

    test_prior_default_ctor_empty_components();
    test_prior_constructor_components_totals_remain_default();
    test_prior_constructor_components_empty_ratio_zero();
    test_prior_get_components_by_type_filters();
    test_prior_get_components_by_type_empty();
    test_prior_get_most_constraining_sorted_ascending();
    test_prior_get_most_constraining_top_n();
    test_prior_get_most_constraining_empty();
    test_prior_get_contribution_by_type_aggregates();
    test_prior_get_contribution_by_type_empty();
    test_prior_get_summary_no_components();
    test_prior_get_summary_with_components_header();
    test_prior_indexer();
    test_prior_component_summary_ctor_sets_all();
    test_prior_component_summary_cv_computed();
    test_prior_component_summary_cv_zero_mean();

    return chtest::summary("influence_diagnostics");
}
