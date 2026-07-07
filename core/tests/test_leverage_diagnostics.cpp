// Standalone C++-only ctest for bestfit::diagnostics::LeverageDiagnostics (Phase 9a, Task D3).
//
// Oracle for behavior is the C# source itself
// (upstream/RMC-BestFit/src/RMC.BestFit/Diagnostics/LeverageDiagnostics.cs @ fc28c0c) and its
// unit-test transcription
// (upstream/RMC-BestFit/src/RMC.BestFit.Tests/Diagnostics/LeverageDiagnosticsTests.cs). Per the
// validation model, internal-support types get C++-only ctests transcribing the upstream test
// oracles plus deterministic checks; the full numeric leverage / Hessian / gen-var values (the
// optimizer-dependent MAP-point quantities) land as D6 emitter fixtures, NOT here. The upstream
// test file states its computational tests "belong in the Verification project" (absent from
// this checkout), so there are NO analytic oracles for the numerical-Hessian / gen-var statics
// to transcribe -- this ctest covers only the pre-computed arithmetic, the nested-struct field
// round-trips, the top-N / summary helpers, and a deterministic structural self-check of the
// fitting path.
//
// SKIPPED from the upstream test file (all XML, no XElement port in this core): XmlRoundTrip_*,
// Constructor_XmlNull_UsesEmptyState, ObservationLeverage_XmlRoundTrip_*,
// PriorComponentLeverage_XmlRoundTrip_* / _ToXElement_ElementNameIsCorrect.
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "bestfit/diagnostics/leverage_diagnostics.hpp"
#include "bestfit/estimation/maximum_a_posteriori.hpp"
#include "bestfit/estimation/optimization_method.hpp"
#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/prior_component.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "check.hpp"

using bestfit::diagnostics::LeverageDiagnostics;
using bestfit::estimation::MaximumAPosteriori;
using bestfit::estimation::OptimizationMethod;
using bestfit::models::DataComponentType;
using bestfit::models::PriorComponentType;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::distributions::UnivariateDistributionType;

using ObservationLeverage = LeverageDiagnostics::ObservationLeverage;
using PriorComponentLeverage = LeverageDiagnostics::PriorComponentLeverage;

namespace {

std::vector<double> sample_data() {
    return {12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600};
}

// --- Empty constructor (C# Constructor_Default_InitializesEmptyState, 27) ------------------
void test_constructor_default_initializes_empty_state() {
    LeverageDiagnostics diag;

    CHECK_EQ(static_cast<int>(diag.observations().size()), 0);
    CHECK_EQ(static_cast<int>(diag.prior_components().size()), 0);
    CHECK_EQ(diag.number_of_parameters(), 0);
    CHECK_EQ(diag.count(), 0);
    CHECK_NEAR(diag.total_leverage(), 0.0, 0.0);
    CHECK_NEAR(diag.total_observation_leverage(), 0.0, 0.0);
    CHECK_NEAR(diag.total_prior_leverage(), 0.0, 0.0);
    CHECK_NEAR(diag.total_fit_influence(), 0.0, 0.0);
    CHECK_NEAR(diag.total_variance_influence(), 0.0, 0.0);
}

// --- Pre-computed constructor (C# 52) ------------------------------------------------------
void test_precomputed_one_observation_sets_totals() {
    ObservationLeverage obs(0, 0.8, 0, 0.5, 0.3, 0, 0, 100.0, DataComponentType::Exact, 1);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs},
                             std::vector<PriorComponentLeverage>{}, 2);

    CHECK_EQ(static_cast<int>(diag.observations().size()), 1);
    CHECK_EQ(static_cast<int>(diag.prior_components().size()), 0);
    CHECK_EQ(diag.number_of_parameters(), 2);
    CHECK_NEAR(diag.total_observation_leverage(), 0.8, 1e-10);
    CHECK_NEAR(diag.total_prior_leverage(), 0.0, 1e-10);
    CHECK_NEAR(diag.total_leverage(), 0.8, 1e-10);
    CHECK_NEAR(diag.total_fit_influence(), 0.5, 1e-10);
    CHECK_NEAR(diag.total_variance_influence(), 0.3, 1e-10);
}

// C# Constructor_PreComputed_NullObservations_UsesEmptyArray (79): the C# null arrays coalesce
// to empty; the C++ analogue passes empty vectors.
void test_precomputed_empty_vectors_uses_empty() {
    LeverageDiagnostics diag(std::vector<ObservationLeverage>{},
                             std::vector<PriorComponentLeverage>{}, 3);

    CHECK_EQ(static_cast<int>(diag.observations().size()), 0);
    CHECK_EQ(static_cast<int>(diag.prior_components().size()), 0);
    CHECK_EQ(diag.number_of_parameters(), 3);
}

// C# Constructor_PreComputed_MultipleObservations_SumsCorrectly (93).
void test_precomputed_multiple_observations_sums() {
    ObservationLeverage obs1(0, 1.0, 0, 0.6, 0.4, 0, 0, 50.0, DataComponentType::Exact);
    ObservationLeverage obs2(1, 0.5, 0, 0.2, 0.3, 0, 0, 75.0, DataComponentType::Exact);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs1, obs2},
                             std::vector<PriorComponentLeverage>{}, 2);

    CHECK_NEAR(diag.total_observation_leverage(), 1.5, 1e-10);
    CHECK_NEAR(diag.total_leverage(), 1.5, 1e-10);
    CHECK_NEAR(diag.total_fit_influence(), 0.8, 1e-10);
    CHECK_NEAR(diag.total_variance_influence(), 0.7, 1e-10);
}

// C# Constructor_PreComputed_UpdatesPercentages (115): 25% / 75% back-fill.
void test_precomputed_updates_percentages() {
    ObservationLeverage obs1(0, 1.0, 0, 0.6, 0.4, 0, 0, 50.0, DataComponentType::Exact);
    ObservationLeverage obs2(1, 3.0, 0, 1.5, 1.5, 0, 0, 75.0, DataComponentType::Exact);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs1, obs2},
                             std::vector<PriorComponentLeverage>{}, 2);

    CHECK_NEAR(diag.observations()[0].percent_of_total(), 25.0, 1e-8);
    CHECK_NEAR(diag.observations()[1].percent_of_total(), 75.0, 1e-8);
}

// C# Constructor_PreComputed_WithPriorComponent_SetsPriorTotal (137).
void test_precomputed_with_prior_component_sets_prior_total() {
    PriorComponentLeverage prior("mu_prior", PriorComponentType::ParameterPrior, 0.4, 0, 0.2, 0.2,
                                 0, 0);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{},
                             std::vector<PriorComponentLeverage>{prior}, 2);

    CHECK_NEAR(diag.total_prior_leverage(), 0.4, 1e-10);
    CHECK_NEAR(diag.total_leverage(), 0.4, 1e-10);
}

// --- GetMostInfluentialObservations (C# 248/273/292) ---------------------------------------
void test_get_most_influential_returns_top_n_descending() {
    ObservationLeverage obs1(0, 0.3, 0, 0.2, 0.1, 0, 0, 10.0, DataComponentType::Exact);
    ObservationLeverage obs2(1, 1.5, 0, 0.8, 0.7, 0, 0, 20.0, DataComponentType::Exact);
    ObservationLeverage obs3(2, 0.7, 0, 0.4, 0.3, 0, 0, 30.0, DataComponentType::Exact);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs1, obs2, obs3},
                             std::vector<PriorComponentLeverage>{}, 2);

    auto top2 = diag.get_most_influential_observations(2);

    CHECK_EQ(static_cast<int>(top2.size()), 2);
    CHECK_NEAR(top2[0].leverage(), 1.5, 1e-10);
    CHECK_NEAR(top2[1].leverage(), 0.7, 1e-10);
}

void test_get_most_influential_topn_larger_than_count() {
    ObservationLeverage obs(0, 1.0, 0, 0.5, 0.5, 0, 0, 100.0, DataComponentType::Exact);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs},
                             std::vector<PriorComponentLeverage>{}, 2);

    auto result = diag.get_most_influential_observations(100);

    CHECK_EQ(static_cast<int>(result.size()), 1);
}

void test_get_most_influential_empty() {
    LeverageDiagnostics diag;
    auto result = diag.get_most_influential_observations(5);

    CHECK_EQ(static_cast<int>(result.size()), 0);
}

// --- GetSummary (C# 308/321) ---------------------------------------------------------------
void test_get_summary_empty_returns_no_data_message() {
    LeverageDiagnostics diag;
    std::string summary = diag.get_summary();

    CHECK_TRUE(summary.find("No leverage diagnostics") != std::string::npos);
}

void test_get_summary_with_data_contains_parameter_count() {
    ObservationLeverage obs(0, 1.0, 100.0, 0.6, 0.4, 60.0, 40.0, 50.0, DataComponentType::Exact);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs},
                             std::vector<PriorComponentLeverage>{}, 3);

    std::string summary = diag.get_summary();
    CHECK_TRUE(summary.find("p = 3") != std::string::npos);
}

// --- Nested struct field round-trips (C# 343 / 412) ----------------------------------------
void test_observation_leverage_ctor_sets_all_properties() {
    ObservationLeverage obs(5, 2.1, 42.0, 1.2, 0.9, 57.1, 42.9, 3500.0, DataComponentType::Interval,
                            3, std::string("historic"));

    CHECK_EQ(obs.index(), 5);
    CHECK_NEAR(obs.leverage(), 2.1, 1e-10);
    CHECK_NEAR(obs.percent_of_total(), 42.0, 1e-10);
    CHECK_NEAR(obs.fit_influence(), 1.2, 1e-10);
    CHECK_NEAR(obs.variance_influence(), 0.9, 1e-10);
    CHECK_NEAR(obs.percent_fit_of_total(), 57.1, 1e-10);
    CHECK_NEAR(obs.percent_variance_of_total(), 42.9, 1e-10);
    CHECK_NEAR(obs.value(), 3500.0, 1e-10);
    CHECK_TRUE(obs.data_type() == DataComponentType::Interval);
    CHECK_EQ(obs.count(), 3);
    CHECK_TRUE(obs.name().has_value());
    CHECK_TRUE(*obs.name() == "historic");
}

void test_prior_component_leverage_ctor_sets_all_properties() {
    PriorComponentLeverage prior("jeffreys_scale", PriorComponentType::JeffreysScalePrior, 0.7,
                                 35.0, 0.4, 0.3, 20.0, 15.0);

    CHECK_TRUE(prior.name() == "jeffreys_scale");
    CHECK_TRUE(prior.type() == PriorComponentType::JeffreysScalePrior);
    CHECK_NEAR(prior.leverage(), 0.7, 1e-10);
    CHECK_NEAR(prior.percent_of_total(), 35.0, 1e-10);
    CHECK_NEAR(prior.fit_influence(), 0.4, 1e-10);
    CHECK_NEAR(prior.variance_influence(), 0.3, 1e-10);
    CHECK_NEAR(prior.percent_fit_of_total(), 20.0, 1e-10);
    CHECK_NEAR(prior.percent_variance_of_total(), 15.0, 1e-10);
}

// --- Count property (C# 475) ---------------------------------------------------------------
void test_count_reflects_observation_count() {
    ObservationLeverage obs1(0, 0.5, 0, 0.3, 0.2, 0, 0, 1.0, DataComponentType::Exact);
    ObservationLeverage obs2(1, 0.5, 0, 0.3, 0.2, 0, 0, 2.0, DataComponentType::Exact);

    LeverageDiagnostics diag(std::vector<ObservationLeverage>{obs1, obs2},
                             std::vector<PriorComponentLeverage>{}, 1);

    CHECK_EQ(diag.count(), 2);
}

// --- Deterministic structural self-check of the fitting path -------------------------------
// Fit a tiny Normal model, then build LeverageDiagnostics(model, mapValues) at the MAP point
// and assert structural invariants that hold regardless of the exact optimizer output. Exact
// leverage numbers are D6 emitter fixtures, not here.
void test_fitting_path_structural_invariants() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    MaximumAPosteriori map(model, OptimizationMethod::NelderMead);
    CHECK_TRUE(map.estimate());

    std::vector<double> map_values = map.best_parameter_set().values;
    LeverageDiagnostics diag(model, map_values);

    // One leverage ordinate per data observation.
    CHECK_EQ(static_cast<int>(diag.observations().size()),
             static_cast<int>(sample_data().size()));
    // NumberOfParameters mirrors the model.
    CHECK_EQ(diag.number_of_parameters(), model.number_of_parameters());
    // Additive total: Total == Observation + Prior.
    CHECK_NEAR(diag.total_leverage(),
               diag.total_observation_leverage() + diag.total_prior_leverage(), 1e-9);

    // When total leverage is nonzero, all components' PercentOfTotal sum to ~100.
    if (diag.total_leverage() > 0) {
        double pct_sum = 0;
        for (const auto& o : diag.observations()) pct_sum += o.percent_of_total();
        for (const auto& pc : diag.prior_components()) pct_sum += pc.percent_of_total();
        CHECK_NEAR(pct_sum, 100.0, 1e-6);
    }
}

}  // namespace

int main() {
    test_constructor_default_initializes_empty_state();
    test_precomputed_one_observation_sets_totals();
    test_precomputed_empty_vectors_uses_empty();
    test_precomputed_multiple_observations_sums();
    test_precomputed_updates_percentages();
    test_precomputed_with_prior_component_sets_prior_total();
    test_get_most_influential_returns_top_n_descending();
    test_get_most_influential_topn_larger_than_count();
    test_get_most_influential_empty();
    test_get_summary_empty_returns_no_data_message();
    test_get_summary_with_data_contains_parameter_count();
    test_observation_leverage_ctor_sets_all_properties();
    test_prior_component_leverage_ctor_sets_all_properties();
    test_count_reflects_observation_count();
    test_fitting_path_structural_invariants();

    return bftest::summary("leverage_diagnostics");
}
