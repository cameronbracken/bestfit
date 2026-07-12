// Structural / behavioral tests for corehydro::analyses::UnivariateAnalysis (A5).
//
// These transcribe the STRUCTURAL C# tests from
//   RMC.BestFit.Tests/Univariate/UnivariateAnalysisTests.cs
//   RMC.BestFit.Tests/Univariate/UnivariateAnalysisPositivePathReprocessTests.cs
// There are NO numeric MCMC oracles here (per the Phase-8 policy; the seeded end-to-end run
// lands via the A11 emitter). Instead we inject a synthetic MCMCResults through
// BayesianAnalysis::set_custom_mcmc_results (the port of C#'s SetCustomMCMCResults) to flip the
// analysis into the estimated state WITHOUT running a chain, then drive the reprocess code path
// (CreateFrequencyAnalysisResults) directly -- exactly what the C# reprocess tests do.
#include <array>
#include <cmath>
#include <memory>
#include <vector>

#include "corehydro/analyses/univariate/univariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "check.hpp"

using corehydro::analyses::UnivariateAnalysis;
using corehydro::estimation::BayesianAnalysis;
using corehydro::models::UnivariateDistributionModel;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::math::optimization::ParameterSet;
using corehydro::numerics::sampling::mcmc::MCMCResults;

namespace {

// The 10-value exact record from UnivariateAnalysisPositivePathReprocessTests.cs.
const std::vector<double>& reprocess_values() {
    static const std::vector<double> v = {12500, 15300, 8900,  22100, 18700,
                                          14200, 9800,  28500, 17400, 11600};
    return v;
}

// The 30-value exact record from UnivariateAnalysisTests.cs CreateTestDataFrame.
const std::vector<double>& structural_values() {
    static const std::vector<double> v = {
        12500, 15300, 8900,  22100, 18700, 14200, 9800,  28500, 17400, 11600,
        19200, 13800, 25600, 10500, 16900, 21300, 14700, 8200,  23800, 15900,
        12100, 27400, 19800, 11200, 16400, 20600, 13200, 9400,  24900, 17800};
    return v;
}

double population_mean(const std::vector<double>& v) {
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double population_std(const std::vector<double>& v) {
    double m = population_mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / static_cast<double>(v.size()));
}

// C# BuildSyntheticResults analogue: a synthetic Normal(mu, sigma) posterior with a
// deterministic spread (no RNG needed -- these tests have no numeric oracle, they only need
// a non-degenerate spread so the credible band has positive width). MAP = (mean, std).
MCMCResults build_synthetic_results(double mean, double std, int sample_size = 1000) {
    std::vector<ParameterSet> output;
    output.reserve(static_cast<std::size_t>(sample_size));
    for (int i = 0; i < sample_size; ++i) {
        double t = sample_size == 1 ? 0.5
                                    : static_cast<double>(i) / static_cast<double>(sample_size - 1);
        double mu = mean + 0.1 * std * (2.0 * t - 1.0);  // mean +/- 0.1*std
        double sigma = std * (0.9 + 0.2 * t);            // [0.9, 1.1] * std, strictly positive
        output.emplace_back(std::vector<double>{mu, sigma}, 0.0);
    }
    ParameterSet map(std::vector<double>{mean, std}, 0.0);
    return MCMCResults(map, std::move(output), 0.10);
}

std::unique_ptr<UnivariateDistributionModel> make_normal_model(const std::vector<double>& values) {
    return std::make_unique<UnivariateDistributionModel>(UnivariateDistributionType::Normal, values);
}

// Builds a Normal UnivariateAnalysis and injects synthetic MCMC results so it reports estimated
// without running a chain (mirrors C# CreateInjectedAnalysis). Returned by unique_ptr because the
// analysis is deliberately non-movable (its BayesianAnalysis holds a reference into the owned
// model).
std::unique_ptr<UnivariateAnalysis> make_injected_analysis(int sample_size = 1000) {
    const std::vector<double>& values = reprocess_values();
    double mean = population_mean(values);
    double std = population_std(values);

    auto model = make_normal_model(values);
    model->set_parameter_values({mean, std});

    auto analysis = std::make_unique<UnivariateAnalysis>(std::move(model));
    // OutputLength sizes the sampled-distribution loop; must match the synthetic Output count.
    analysis->bayesian_analysis().set_output_length(sample_size);
    analysis->bayesian_analysis().set_custom_mcmc_results(
        build_synthetic_results(mean, std, sample_size), /*skip_information_criteria=*/true);
    return analysis;
}

// ---- Constructor: initializes correctly (C# Constructor_WithDistribution_InitializesCorrectly) ----
void test_constructor_initializes() {
    UnivariateAnalysis analysis(make_normal_model(structural_values()));
    CHECK_TRUE(analysis.probability_ordinates().count() > 0);  // 25 defaults
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    CHECK_TRUE(analysis.chronology_analysis_results() == nullptr);
    // BayesianAnalysis is over the same model.
    CHECK_TRUE(&analysis.bayesian_analysis().model() == &analysis.univariate_distribution());
}

// ---- Null-guard (C# Constructor_WithNullDistribution_ThrowsArgumentNullException) ----
void test_null_distribution_throws() {
    CHECK_THROWS(UnivariateAnalysis(std::unique_ptr<UnivariateDistributionModel>{}));
}

// ---- Validate returns valid for a properly configured analysis (C# Validate_*_ReturnsValid) ----
void test_validate_valid() {
    UnivariateAnalysis analysis(make_normal_model(structural_values()));
    CHECK_TRUE(analysis.validate().is_valid);
}

// ---- ClearResults resets all results (C# ClearResults_ResetsAllResults) ----
void test_clear_results() {
    UnivariateAnalysis analysis(make_normal_model(structural_values()));
    analysis.clear_results();
    CHECK_TRUE(!analysis.is_estimated());
    CHECK_TRUE(analysis.analysis_results() == nullptr);
    CHECK_TRUE(analysis.chronology_analysis_results() == nullptr);
}

// ---- get_distribution / get_point_estimate_distribution null when unestimated ----
void test_getters_null_when_unestimated() {
    UnivariateAnalysis analysis(make_normal_model(structural_values()));
    CHECK_TRUE(analysis.get_distribution(0) == nullptr);
    CHECK_TRUE(analysis.get_point_estimate_distribution() == nullptr);
}

// ---- Frequency reprocess populates AnalysisResults + preserves the MCMC Results reference ----
// (C# ProbabilityOrdinatesChange_EstimatedAnalysis_PreservesResultsReference)
void test_frequency_reprocess_preserves_results_reference() {
    std::unique_ptr<UnivariateAnalysis> analysis = make_injected_analysis();
    CHECK_TRUE(analysis->bayesian_analysis().is_estimated());
    CHECK_TRUE(analysis->bayesian_analysis().results().has_value());
    const MCMCResults* results_before = &analysis->bayesian_analysis().results().value();

    analysis->probability_ordinates().add(0.001);
    analysis->create_frequency_analysis_results();

    CHECK_TRUE(analysis->analysis_results() != nullptr);
    // The MCMC Results object must be the SAME instance (the chain is untouched by a reprocess).
    CHECK_TRUE(&analysis->bayesian_analysis().results().value() == results_before);
    CHECK_TRUE(analysis->bayesian_analysis().is_estimated());
    // One CI per ordinate.
    CHECK_EQ(analysis->analysis_results()->confidence_intervals.size(),
             analysis->probability_ordinates().count());
    // Mode curve is one value per ordinate too.
    CHECK_EQ(analysis->analysis_results()->mode_curve.size(),
             analysis->probability_ordinates().count());
}

// ---- 95% credible band is wider than the 90% band (C# CredibleIntervalWidthChange_WidensCI*) ----
void test_credible_band_widens() {
    std::unique_ptr<UnivariateAnalysis> analysis = make_injected_analysis();

    // Build the frequency results at the default 90% credible interval width.
    analysis->create_frequency_analysis_results();
    CHECK_TRUE(analysis->analysis_results() != nullptr);
    std::vector<std::array<double, 2>> ci90 = analysis->analysis_results()->confidence_intervals;

    // Widen to 95%; the derived frequency band must widen (the chain is unchanged).
    analysis->bayesian_analysis().set_credible_interval_width(0.95);
    analysis->create_frequency_analysis_results();
    CHECK_TRUE(analysis->analysis_results() != nullptr);
    const std::vector<std::array<double, 2>>& ci95 = analysis->analysis_results()->confidence_intervals;

    CHECK_EQ(ci95.size(), ci90.size());
    double total_width_90 = 0.0;
    double total_width_95 = 0.0;
    bool all_brackets = true;
    for (std::size_t i = 0; i < ci95.size(); ++i) {
        double w90 = ci90[i][1] - ci90[i][0];
        double w95 = ci95[i][1] - ci95[i][0];
        total_width_90 += w90;
        total_width_95 += w95;
        // 95% band brackets the 90% band at every ordinate.
        if (ci95[i][0] > ci90[i][0] + 1e-9 || ci95[i][1] < ci90[i][1] - 1e-9) all_brackets = false;
    }
    CHECK_TRUE(all_brackets);
    CHECK_TRUE(total_width_95 > total_width_90);
}

// ---- get_distribution / get_point_estimate_distribution return a live distribution once
//      estimated (C# GetDistribution / GetPointEstimateDistribution positive path) ----
void test_getters_after_injection() {
    std::unique_ptr<UnivariateAnalysis> analysis = make_injected_analysis();
    // With injected results the bayesian analysis is estimated, so the getters resolve.
    auto* d0 = analysis->get_distribution(0);
    CHECK_TRUE(d0 != nullptr);
    auto* pe = analysis->get_point_estimate_distribution();
    CHECK_TRUE(pe != nullptr);
    // Point estimate (PosteriorMean by default) matches MAP=(mean,std) shape: 2 parameters.
    CHECK_EQ(pe->get_parameters().size(), static_cast<std::size_t>(2));
}

}  // namespace

int main() {
    test_constructor_initializes();
    test_null_distribution_throws();
    test_validate_valid();
    test_clear_results();
    test_getters_null_when_unestimated();
    test_frequency_reprocess_preserves_results_reference();
    test_credible_band_widens();
    test_getters_after_injection();

    return chtest::summary("univariate_analysis");
}
