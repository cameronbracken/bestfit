// Standalone C++-only ctest for the bestfit::diagnostics predictive-check layer (Phase 10,
// Task X10): PredictiveSummary, PredictiveCheckResults, PriorPredictiveCheck,
// PosteriorPredictiveCheck.
//
// Oracle for behavior is the C# source itself
// (upstream/RMC-BestFit/src/RMC.BestFit/Diagnostics/{PredictiveSummary,PredictiveCheckResults,
// PriorPredictiveCheck,PosteriorPredictiveCheck}.cs @ fc28c0c) and its unit-test transcriptions
// (upstream/RMC-BestFit/src/RMC.BestFit.Tests/Diagnostics/{PredictiveCheckResultsTests,
// PosteriorPredictiveCheckTests,PosteriorPredictiveCheckExpandedTests,PriorPredictiveCheckTests,
// PriorPredictiveCheckExpandedTests}.cs). Per the validation model, internal-support types get
// C++-only ctests transcribing the upstream test oracles. The upstream test files state their
// computational verification tests "belong in RMC.BestFit.Verification" (absent from this
// checkout); the transcribed methods below are the structural / seeded-reproducibility oracles,
// asserted verbatim.
//
// SKIPPED from the upstream test files (documented in the task report):
//   - Test_Constructor_NullModel_Throws (both check classes): the C++ ctor takes the model by
//     const reference, which is never null; the C# ArgumentNullException guard has no C++
//     analogue (same rationale the influence_diagnostics port documents).
//   - Test_Constructor_NullSamples_Throws / Test_Constructor_NullObservedData_Throws: the C++
//     ctor takes std::vector arguments, which are never null (only empty); the empty cases are
//     covered by the *_Empty*_Throws tests below.
//   - Constructor_NonSimulatableModel_Throws: the C++ check classes are templates constrained by
//     static_assert to ISimulatable<std::vector<double>> + ModelBase, so a non-simulatable model
//     is a COMPILE-TIME error, not a runtime throw.
#include <cmath>
#include <functional>
#include <limits>
#include <vector>

#include "bestfit/diagnostics/posterior_predictive_check.hpp"
#include "bestfit/diagnostics/predictive_check_results.hpp"
#include "bestfit/diagnostics/predictive_summary.hpp"
#include "bestfit/diagnostics/prior_predictive_check.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "check.hpp"

using bestfit::diagnostics::PosteriorPredictiveCheck;
using bestfit::diagnostics::PredictiveCheckResults;
using bestfit::diagnostics::PredictiveSummary;
using bestfit::diagnostics::PriorPredictiveCheck;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::distributions::UnivariateDistributionType;
using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;

namespace {

// --- Shared fixtures mirroring the upstream Make* helpers -----------------------------------

UnivariateDistributionModel make_normal_model() {
    return UnivariateDistributionModel(
        UnivariateDistributionType::Normal,
        std::vector<double>{12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600});
}

// PosteriorPredictiveCheckTests.MakePosteriorSamples
std::vector<ParameterSet> make_posterior_samples_structural(int count) {
    std::vector<ParameterSet> samples;
    for (int i = 0; i < count; ++i)
        samples.emplace_back(std::vector<double>{100.0 + 0.01 * i, 15.0 + 0.005 * i}, 0.0);
    return samples;
}

// PosteriorPredictiveCheckExpandedTests.MakePosteriorSamples (drifted around data mean/SD)
std::vector<ParameterSet> make_posterior_samples(int count) {
    std::vector<ParameterSet> samples;
    for (int i = 0; i < count; ++i)
        samples.emplace_back(std::vector<double>{16500.0 + 5.0 * i, 6000.0 + 1.0 * i}, 0.0);
    return samples;
}

std::vector<double> make_observed_data() {
    return {12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600};
}

// ============================================================================================
// PredictiveCheckResultsTests
// ============================================================================================

void test_results_constructor_default_all_pvalues_are_nan() {
    PredictiveCheckResults r;
    CHECK_TRUE(std::isnan(r.mean_p_value));
    CHECK_TRUE(std::isnan(r.sd_p_value));
    CHECK_TRUE(std::isnan(r.skewness_p_value));
    CHECK_TRUE(std::isnan(r.min_p_value));
    CHECK_TRUE(std::isnan(r.max_p_value));
}

void test_results_constructor_default_number_of_replicates_is_zero() {
    PredictiveCheckResults r;
    CHECK_EQ(r.number_of_replicates, 0);
}

void test_results_properties_set_and_get() {
    PredictiveCheckResults r;
    r.number_of_replicates = 1000;
    r.mean_p_value = 0.52;
    r.sd_p_value = 0.48;
    r.skewness_p_value = 0.31;
    r.min_p_value = 0.07;
    r.max_p_value = 0.93;
    CHECK_EQ(r.number_of_replicates, 1000);
    CHECK_NEAR(r.mean_p_value, 0.52, 1e-10);
    CHECK_NEAR(r.sd_p_value, 0.48, 1e-10);
    CHECK_NEAR(r.skewness_p_value, 0.31, 1e-10);
    CHECK_NEAR(r.min_p_value, 0.07, 1e-10);
    CHECK_NEAR(r.max_p_value, 0.93, 1e-10);
}

void test_results_has_potential_misfit_good_fit_returns_false() {
    PredictiveCheckResults r;
    r.mean_p_value = 0.50;
    r.sd_p_value = 0.45;
    r.skewness_p_value = 0.55;
    r.min_p_value = 0.48;
    r.max_p_value = 0.52;
    CHECK_TRUE(!r.has_potential_misfit());
}

void test_results_has_potential_misfit_low_mean_returns_true() {
    PredictiveCheckResults r;
    r.mean_p_value = 0.02;
    r.sd_p_value = 0.50;
    r.skewness_p_value = 0.50;
    r.min_p_value = 0.50;
    r.max_p_value = 0.50;
    CHECK_TRUE(r.has_potential_misfit());
}

void test_results_has_potential_misfit_high_max_returns_true() {
    PredictiveCheckResults r;
    r.mean_p_value = 0.50;
    r.sd_p_value = 0.50;
    r.skewness_p_value = 0.50;
    r.min_p_value = 0.50;
    r.max_p_value = 0.98;
    CHECK_TRUE(r.has_potential_misfit());
}

void test_results_has_potential_misfit_low_sd_returns_true() {
    PredictiveCheckResults r;
    r.mean_p_value = 0.50;
    r.sd_p_value = 0.01;
    r.skewness_p_value = 0.50;
    r.min_p_value = 0.50;
    r.max_p_value = 0.50;
    CHECK_TRUE(r.has_potential_misfit());
}

void test_results_has_potential_misfit_custom_threshold() {
    PredictiveCheckResults r;
    r.mean_p_value = 0.07;
    r.sd_p_value = 0.50;
    r.skewness_p_value = 0.50;
    r.min_p_value = 0.50;
    r.max_p_value = 0.50;
    CHECK_TRUE(!r.has_potential_misfit(0.05));
    CHECK_TRUE(r.has_potential_misfit(0.10));
}

void test_results_has_potential_misfit_skew_at_lower_threshold_returns_true() {
    PredictiveCheckResults r;
    r.mean_p_value = 0.50;
    r.sd_p_value = 0.50;
    r.skewness_p_value = 0.04;
    r.min_p_value = 0.50;
    r.max_p_value = 0.50;
    CHECK_TRUE(r.has_potential_misfit());
}

void test_results_has_potential_misfit_nan_does_not_throw() {
    PredictiveCheckResults r;  // all NaN by default
    CHECK_TRUE(!r.has_potential_misfit());
}

// ============================================================================================
// PredictiveSummaryTests
// ============================================================================================

void test_summary_constructor_default_all_quantile_arrays_empty() {
    PredictiveSummary s;
    CHECK_EQ(static_cast<int>(s.mean_quantiles.size()), 0);
    CHECK_EQ(static_cast<int>(s.sd_quantiles.size()), 0);
    CHECK_EQ(static_cast<int>(s.min_quantiles.size()), 0);
    CHECK_EQ(static_cast<int>(s.max_quantiles.size()), 0);
}

void test_summary_constructor_default_number_of_valid_draws_is_zero() {
    PredictiveSummary s;
    CHECK_EQ(s.number_of_valid_draws, 0);
}

void test_summary_properties_set_and_get() {
    std::vector<double> mean_q{90.0, 95.0, 100.0, 105.0, 110.0};
    std::vector<double> sd_q{8.0, 10.0, 12.0, 14.0, 18.0};
    std::vector<double> min_q{40.0, 50.0, 60.0, 70.0, 80.0};
    std::vector<double> max_q{120.0, 130.0, 140.0, 150.0, 175.0};
    PredictiveSummary s;
    s.number_of_valid_draws = 500;
    s.mean_quantiles = mean_q;
    s.sd_quantiles = sd_q;
    s.min_quantiles = min_q;
    s.max_quantiles = max_q;
    CHECK_EQ(s.number_of_valid_draws, 500);
    CHECK_TRUE(s.mean_quantiles == mean_q);
    CHECK_TRUE(s.sd_quantiles == sd_q);
    CHECK_TRUE(s.min_quantiles == min_q);
    CHECK_TRUE(s.max_quantiles == max_q);
}

void test_summary_mean_quantiles_length5_reflects_convention() {
    PredictiveSummary s;
    s.mean_quantiles = {85.0, 94.0, 100.0, 106.0, 120.0};
    CHECK_EQ(static_cast<int>(s.mean_quantiles.size()), 5);
    CHECK_TRUE(s.mean_quantiles[0] < s.mean_quantiles[4]);
}

// ============================================================================================
// PosteriorPredictiveCheckTests (structural ctor)
// ============================================================================================

void test_post_constructor_parameter_set_list_succeeds() {
    auto model = make_normal_model();
    auto samples = make_posterior_samples_structural(50);
    std::vector<double> observed{12500, 15300, 8900};
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, samples, observed);
    CHECK_TRUE(&check.model() == &model);
}

void test_post_constructor_empty_samples_throws() {
    auto model = make_normal_model();
    CHECK_THROWS((PosteriorPredictiveCheck<UnivariateDistributionModel>(
        model, std::vector<ParameterSet>{}, std::vector<double>{1, 2, 3})));
}

void test_post_constructor_empty_observed_throws() {
    auto model = make_normal_model();
    CHECK_THROWS((PosteriorPredictiveCheck<UnivariateDistributionModel>(
        model, make_posterior_samples(10), std::vector<double>{})));
}

// ============================================================================================
// PosteriorPredictiveCheckExpandedTests
// ============================================================================================

void test_post_generate_replicates_at_most_number_of_replicates() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(50),
                                                                make_observed_data());
    check.set_seed(42);
    auto replicates = check.generate_replicates(20);
    CHECK_TRUE(static_cast<int>(replicates.size()) <= 20);
}

void test_post_generate_replicates_length_equals_observed() {
    auto model = make_normal_model();
    auto observed = make_observed_data();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(40),
                                                                observed);
    check.set_seed(1);
    auto replicates = check.generate_replicates(15);
    for (const auto& rep : replicates)
        CHECK_EQ(static_cast<int>(rep.size()), static_cast<int>(observed.size()));
}

void test_post_generate_replicates_same_seed_reproducible() {
    auto model_a = make_normal_model();
    auto model_b = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check_a(model_a, make_posterior_samples(30),
                                                                  make_observed_data());
    PosteriorPredictiveCheck<UnivariateDistributionModel> check_b(model_b, make_posterior_samples(30),
                                                                  make_observed_data());
    check_a.set_seed(123);
    check_b.set_seed(123);
    auto reps_a = check_a.generate_replicates(10);
    auto reps_b = check_b.generate_replicates(10);
    CHECK_EQ(static_cast<int>(reps_a.size()), static_cast<int>(reps_b.size()));
    // Reproducible seeded stream: identical replicate contents, not just counts.
    for (std::size_t i = 0; i < reps_a.size(); ++i) CHECK_TRUE(reps_a[i] == reps_b[i]);
}

void test_post_generate_replicates_zero_throws() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(10),
                                                                make_observed_data());
    CHECK_THROWS(check.generate_replicates(0));
}

void test_post_generate_replicates_negative_throws() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(10),
                                                                make_observed_data());
    CHECK_THROWS(check.generate_replicates(-5));
}

void test_post_compute_pvalue_in_unit_interval() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(40),
                                                                make_observed_data());
    check.set_seed(7);
    double p = check.compute_p_value(
        [](const std::vector<double>& d) { return d.empty() ? 0.0 : d[0]; }, 20);
    CHECK_TRUE(p >= 0.0 && p <= 1.0);
}

void test_post_compute_pvalue_null_test_statistic_throws() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(10),
                                                                make_observed_data());
    std::function<double(const std::vector<double>&)> null_fn;
    CHECK_THROWS(check.compute_p_value(null_fn, 5));
}

void test_post_compute_pvalue_uses_provided_test_statistic() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(30),
                                                                make_observed_data());
    check.set_seed(11);
    int call_count = 0;
    double p = check.compute_p_value(
        [&call_count](const std::vector<double>& d) {
            ++call_count;
            double sum = 0.0;
            for (double x : d) sum += x;
            return sum;
        },
        10);
    CHECK_TRUE(call_count > 0);
    CHECK_TRUE(p >= 0.0 && p <= 1.0);
}

void test_post_compute_common_pvalues_populates_all_five() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(60),
                                                                make_observed_data());
    check.set_seed(200);
    auto r = check.compute_common_p_values(40);
    if (r.number_of_replicates > 0) {
        CHECK_TRUE(r.mean_p_value >= 0 && r.mean_p_value <= 1);
        CHECK_TRUE(r.sd_p_value >= 0 && r.sd_p_value <= 1);
        CHECK_TRUE(r.skewness_p_value >= 0 && r.skewness_p_value <= 1);
        CHECK_TRUE(r.min_p_value >= 0 && r.min_p_value <= 1);
        CHECK_TRUE(r.max_p_value >= 0 && r.max_p_value <= 1);
    }
}

void test_post_compute_common_pvalues_at_most_requested() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(50),
                                                                make_observed_data());
    check.set_seed(33);
    auto r = check.compute_common_p_values(25);
    CHECK_TRUE(r.number_of_replicates <= 25);
}

void test_post_compute_summary_returns_non_null() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(30),
                                                                make_observed_data());
    check.set_seed(17);
    auto s = check.compute_summary(15);
    if (s.number_of_valid_draws > 0) {
        CHECK_EQ(static_cast<int>(s.mean_quantiles.size()), 5);
        CHECK_EQ(static_cast<int>(s.sd_quantiles.size()), 5);
        CHECK_EQ(static_cast<int>(s.min_quantiles.size()), 5);
        CHECK_EQ(static_cast<int>(s.max_quantiles.size()), 5);
    }
}

void test_post_compute_summary_valid_draws_at_most_requested() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(40),
                                                                make_observed_data());
    check.set_seed(88);
    auto s = check.compute_summary(20);
    CHECK_TRUE(s.number_of_valid_draws <= 20);
}

void test_post_seed_setter_round_trips() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(5),
                                                                make_observed_data());
    check.set_seed(777);
    CHECK_EQ(check.seed(), 777);
}

void test_post_default_seed_is_12345() {
    auto model = make_normal_model();
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(5),
                                                                make_observed_data());
    CHECK_EQ(check.seed(), 12345);
}

void test_post_number_of_posterior_samples_matches() {
    auto model = make_normal_model();
    auto samples = make_posterior_samples(33);
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, samples, make_observed_data());
    CHECK_EQ(check.number_of_posterior_samples(), 33);
}

void test_post_sample_size_matches_observed_length() {
    auto model = make_normal_model();
    std::vector<double> observed{1, 2, 3, 4, 5, 6, 7};
    PosteriorPredictiveCheck<UnivariateDistributionModel> check(model, make_posterior_samples(5),
                                                                observed);
    CHECK_EQ(check.sample_size(), 7);
}

// ============================================================================================
// PriorPredictiveCheckTests (structural)
// ============================================================================================

void test_prior_constructor_with_simulatable_model_succeeds() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    CHECK_TRUE(&check.model() == &model);
}

void test_prior_defaults_number_of_draws_and_seed() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    CHECK_EQ(check.number_of_draws(), 1000);
    CHECK_EQ(check.seed(), 12345);
}

void test_prior_number_of_draws_setter_round_trips() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(500);
    CHECK_EQ(check.number_of_draws(), 500);
}

void test_prior_number_of_draws_less_than_one_throws() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    CHECK_THROWS(check.set_number_of_draws(0));
}

void test_prior_seed_setter_round_trips() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_seed(99);
    CHECK_EQ(check.seed(), 99);
}

// ============================================================================================
// PriorPredictiveCheckExpandedTests
// ============================================================================================

void test_prior_sample_from_priors_returns_requested_number() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(50);
    check.set_seed(42);
    auto samples = check.sample_from_priors();
    CHECK_EQ(static_cast<int>(samples.size()), 50);
}

void test_prior_sample_from_priors_fixed_seed_reproducible() {
    auto model_a = make_normal_model();
    auto model_b = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check_a(model_a);
    PriorPredictiveCheck<UnivariateDistributionModel> check_b(model_b);
    check_a.set_number_of_draws(30);
    check_a.set_seed(12345);
    check_b.set_number_of_draws(30);
    check_b.set_seed(12345);
    auto samples_a = check_a.sample_from_priors();
    auto samples_b = check_b.sample_from_priors();
    CHECK_EQ(static_cast<int>(samples_a.size()), static_cast<int>(samples_b.size()));
    for (std::size_t i = 0; i < samples_a.size(); ++i)
        for (std::size_t j = 0; j < samples_a[i].values.size(); ++j)
            CHECK_NEAR(samples_a[i].values[j], samples_b[i].values[j], 1e-12);
}

void test_prior_sample_from_priors_values_match_parameter_count() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(5);
    check.set_seed(1);
    auto samples = check.sample_from_priors();
    int expected = model.number_of_parameters();
    for (const auto& s : samples) CHECK_EQ(static_cast<int>(s.values.size()), expected);
}

void test_prior_sample_from_priors_different_seeds_differ() {
    auto model_a = make_normal_model();
    auto model_b = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check_a(model_a);
    PriorPredictiveCheck<UnivariateDistributionModel> check_b(model_b);
    check_a.set_number_of_draws(20);
    check_a.set_seed(1);
    check_b.set_number_of_draws(20);
    check_b.set_seed(2);
    auto samples_a = check_a.sample_from_priors();
    auto samples_b = check_b.sample_from_priors();
    bool any_different = false;
    for (std::size_t i = 0; i < samples_a.size() && !any_different; ++i)
        for (std::size_t j = 0; j < samples_a[i].values.size() && !any_different; ++j)
            if (std::fabs(samples_a[i].values[j] - samples_b[i].values[j]) > 1e-10)
                any_different = true;
    CHECK_TRUE(any_different);
}

void test_prior_sample_from_priors_clamps_to_bounds() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(100);
    check.set_seed(7);
    auto samples = check.sample_from_priors();
    for (std::size_t i = 0; i < samples.size(); ++i)
        for (std::size_t j = 0; j < static_cast<std::size_t>(model.number_of_parameters()); ++j) {
            const auto& p = model.parameters()[j];
            CHECK_TRUE(samples[i].values[j] >= p.lower_bound());
            CHECK_TRUE(samples[i].values[j] <= p.upper_bound());
        }
}

void test_prior_generate_predictive_produces_requested_size() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(20);
    check.set_seed(100);
    auto datasets = check.generate_prior_predictive(25);
    for (const auto& d : datasets) CHECK_EQ(static_cast<int>(d.size()), 25);
}

void test_prior_generate_predictive_count_within_number_of_draws() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(15);
    check.set_seed(5);
    auto datasets = check.generate_prior_predictive(10);
    CHECK_TRUE(static_cast<int>(datasets.size()) <= 15);
}

void test_prior_generate_predictive_sample_size_zero_throws() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(5);
    check.set_seed(1);
    CHECK_THROWS(check.generate_prior_predictive(0));
}

void test_prior_generate_predictive_negative_sample_size_throws() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(5);
    check.set_seed(1);
    CHECK_THROWS(check.generate_prior_predictive(-1));
}

void test_prior_generate_predictive_same_seed_reproducible() {
    auto model_a = make_normal_model();
    auto model_b = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check_a(model_a);
    PriorPredictiveCheck<UnivariateDistributionModel> check_b(model_b);
    check_a.set_number_of_draws(20);
    check_a.set_seed(999);
    check_b.set_number_of_draws(20);
    check_b.set_seed(999);
    auto datasets_a = check_a.generate_prior_predictive(5);
    auto datasets_b = check_b.generate_prior_predictive(5);
    CHECK_EQ(static_cast<int>(datasets_a.size()), static_cast<int>(datasets_b.size()));
    for (std::size_t i = 0; i < datasets_a.size(); ++i) CHECK_TRUE(datasets_a[i] == datasets_b[i]);
}

void test_prior_compute_summary_returns_non_null() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(20);
    check.set_seed(50);
    auto s = check.compute_summary(10);
    (void)s;  // non-null by value; presence checked structurally below
    CHECK_TRUE(s.number_of_valid_draws >= 0);
}

void test_prior_compute_summary_valid_draws_at_most_number_of_draws() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(30);
    check.set_seed(50);
    auto s = check.compute_summary(8);
    CHECK_TRUE(s.number_of_valid_draws <= 30);
}

void test_prior_compute_summary_quantile_arrays_have_five_entries() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(50);
    check.set_seed(22);
    auto s = check.compute_summary(10);
    if (s.number_of_valid_draws > 0) {
        CHECK_EQ(static_cast<int>(s.mean_quantiles.size()), 5);
        CHECK_EQ(static_cast<int>(s.sd_quantiles.size()), 5);
        CHECK_EQ(static_cast<int>(s.min_quantiles.size()), 5);
        CHECK_EQ(static_cast<int>(s.max_quantiles.size()), 5);
    }
}

void test_prior_model_returns_same_instance() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    CHECK_TRUE(&check.model() == &model);
}

void test_prior_number_of_draws_set_to_one_does_not_throw() {
    auto model = make_normal_model();
    PriorPredictiveCheck<UnivariateDistributionModel> check(model);
    check.set_number_of_draws(1);
    CHECK_EQ(check.number_of_draws(), 1);
}

}  // namespace

int main() {
    // PredictiveCheckResults
    test_results_constructor_default_all_pvalues_are_nan();
    test_results_constructor_default_number_of_replicates_is_zero();
    test_results_properties_set_and_get();
    test_results_has_potential_misfit_good_fit_returns_false();
    test_results_has_potential_misfit_low_mean_returns_true();
    test_results_has_potential_misfit_high_max_returns_true();
    test_results_has_potential_misfit_low_sd_returns_true();
    test_results_has_potential_misfit_custom_threshold();
    test_results_has_potential_misfit_skew_at_lower_threshold_returns_true();
    test_results_has_potential_misfit_nan_does_not_throw();

    // PredictiveSummary
    test_summary_constructor_default_all_quantile_arrays_empty();
    test_summary_constructor_default_number_of_valid_draws_is_zero();
    test_summary_properties_set_and_get();
    test_summary_mean_quantiles_length5_reflects_convention();

    // PosteriorPredictiveCheck (structural ctor)
    test_post_constructor_parameter_set_list_succeeds();
    test_post_constructor_empty_samples_throws();
    test_post_constructor_empty_observed_throws();

    // PosteriorPredictiveCheck (expanded)
    test_post_generate_replicates_at_most_number_of_replicates();
    test_post_generate_replicates_length_equals_observed();
    test_post_generate_replicates_same_seed_reproducible();
    test_post_generate_replicates_zero_throws();
    test_post_generate_replicates_negative_throws();
    test_post_compute_pvalue_in_unit_interval();
    test_post_compute_pvalue_null_test_statistic_throws();
    test_post_compute_pvalue_uses_provided_test_statistic();
    test_post_compute_common_pvalues_populates_all_five();
    test_post_compute_common_pvalues_at_most_requested();
    test_post_compute_summary_returns_non_null();
    test_post_compute_summary_valid_draws_at_most_requested();
    test_post_seed_setter_round_trips();
    test_post_default_seed_is_12345();
    test_post_number_of_posterior_samples_matches();
    test_post_sample_size_matches_observed_length();

    // PriorPredictiveCheck (structural)
    test_prior_constructor_with_simulatable_model_succeeds();
    test_prior_defaults_number_of_draws_and_seed();
    test_prior_number_of_draws_setter_round_trips();
    test_prior_number_of_draws_less_than_one_throws();
    test_prior_seed_setter_round_trips();

    // PriorPredictiveCheck (expanded)
    test_prior_sample_from_priors_returns_requested_number();
    test_prior_sample_from_priors_fixed_seed_reproducible();
    test_prior_sample_from_priors_values_match_parameter_count();
    test_prior_sample_from_priors_different_seeds_differ();
    test_prior_sample_from_priors_clamps_to_bounds();
    test_prior_generate_predictive_produces_requested_size();
    test_prior_generate_predictive_count_within_number_of_draws();
    test_prior_generate_predictive_sample_size_zero_throws();
    test_prior_generate_predictive_negative_sample_size_throws();
    test_prior_generate_predictive_same_seed_reproducible();
    test_prior_compute_summary_returns_non_null();
    test_prior_compute_summary_valid_draws_at_most_number_of_draws();
    test_prior_compute_summary_quantile_arrays_have_five_entries();
    test_prior_model_returns_same_instance();
    test_prior_number_of_draws_set_to_one_does_not_throw();

    return bftest::summary("predictive_checks");
}
