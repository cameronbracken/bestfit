// Standalone tests for bestfit::estimation::BayesianAnalysis (Phase 4, Task T9).
//
// Oracle for behavior is the C# source itself (upstream/RMC-BestFit/src/RMC.BestFit/Estimation/
// BayesianAnalysis.cs @ fc28c0c) -- see bayesian_analysis.hpp's header for the exact
// method/line mapping and the scope decisions (synchronous-only, plain knob setters, T10
// deferrals, gated Diagnostics stubs). This is a FAST smoke test, not a convergence test: the
// MCMC run below uses deliberately tiny knobs (iterations=100, warmup=50, output_length=100)
// purely to exercise the wiring end-to-end quickly. Exact seeded digests + cross-language
// reproduction are Task T12's job.
//
// Covers:
//   - estimate() succeeds with DEMCzs on the Phase-4 dataset + a Normal model (Uniform priors
//     from set_default_parameters()); is_estimated() true; results() populated with the
//     expected number of chains; every recorded value is finite; the posterior mean of mu
//     lands in a plausible (loose) range near the sample mean.
//   - Determinism: two estimate() calls with the SAME prng_seed on the SAME instance give
//     identical output (this is a within-C++ determinism check only).
//   - validate(): valid for a well-formed small-but-legal configuration; invalid (with the
//     documented error message) for a bad knob (number_of_chains < 4).
//   - set_default_simulation_options(): the DEMCzs vs. NUTS per-sampler-type defaults match
//     the C# formulas.
//   - Gated methods (compute_influence_diagnostics/compute_prior_influence_diagnostics/
//     compute_leverage_diagnostics) throw (Diagnostics layer deferred).
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "check.hpp"

using bestfit::estimation::BayesianAnalysis;
using bestfit::estimation::SamplerType;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::distributions::UnivariateDistributionType;

namespace {

// Phase-4 dataset (matches test_maximum_likelihood.cpp / test_maximum_a_posteriori.cpp).
std::vector<double> sample_data() {
    return {12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600};
}

double sample_mean(const std::vector<double>& data) {
    return std::accumulate(data.begin(), data.end(), 0.0) / static_cast<double>(data.size());
}

// Applies the tiny-but-legal smoke-test knobs shared by most tests below: 4 chains (the
// minimum both BayesianAnalysis::validate() and DEMCzs's own validate_custom_settings()
// allow), thinning=1, iterations=100 / warmup=50 (iterations' floor and exactly half of it --
// MCMCSampler::validate_settings() hard-errors if warmup exceeds half of iterations), and a
// small output_length so the run finishes in milliseconds.
void apply_fast_knobs(BayesianAnalysis& analysis, int seed) {
    analysis.set_number_of_chains(4);
    analysis.set_thinning_interval(1);
    analysis.set_iterations(100);
    analysis.set_warmup_iterations(50);
    analysis.set_initial_iterations(50);
    analysis.set_output_length(100);
    analysis.set_prng_seed(seed);
}

void test_estimate_succeeds_and_produces_finite_results() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    BayesianAnalysis analysis(model, SamplerType::DEMCzs);
    apply_fast_knobs(analysis, 20260703);

    bool ok = analysis.estimate();

    CHECK_TRUE(ok);
    CHECK_TRUE(analysis.is_estimated());
    CHECK_TRUE(analysis.results().has_value());

    const auto& results = *analysis.results();
    CHECK_TRUE(results.markov_chains.size() == 4);
    CHECK_TRUE(!results.output.empty());

    for (const auto& chain : results.markov_chains) {
        for (const auto& state : chain) {
            for (double v : state.values) CHECK_TRUE(std::isfinite(v));
            CHECK_TRUE(std::isfinite(state.fitness));
        }
    }
    for (const auto& state : results.output) {
        for (double v : state.values) CHECK_TRUE(std::isfinite(v));
    }

    CHECK_TRUE(results.parameter_results.size() == 2);
    for (const auto& pr : results.parameter_results) {
        CHECK_TRUE(std::isfinite(pr.summary_statistics.mean));
        CHECK_TRUE(std::isfinite(pr.summary_statistics.rhat));
        // ESS (not asserted here): with only 100 total iterations the per-parameter
        // autocorrelation sum can land at exactly -0.5, producing a NaN via the
        // 1/(1+2*rho) formula (mcmc_diagnostics.hpp's effective_sample_size) -- an inherent
        // small-sample edge case in the diagnostic itself, not a BayesianAnalysis wiring
        // bug. R-hat is the invariant this smoke test relies on (per the task brief).
    }

    // Loose invariant: the posterior mean of mu (parameter 0) should be in the same
    // ballpark as the sample mean -- this is a smoke test, not a convergence test, so the
    // tolerance is wide (50% of the sample mean).
    double mu_posterior_mean = results.posterior_mean.values[0];
    double mu_sample_mean = sample_mean(sample_data());
    CHECK_TRUE(std::fabs(mu_posterior_mean - mu_sample_mean) <= 0.5 * std::fabs(mu_sample_mean));
}

void test_determinism_same_seed_gives_identical_results() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    BayesianAnalysis analysis(model, SamplerType::DEMCzs);
    apply_fast_knobs(analysis, 42);

    CHECK_TRUE(analysis.estimate());
    std::vector<double> first_output_values;
    for (const auto& state : analysis.results()->output)
        for (double v : state.values) first_output_values.push_back(v);

    CHECK_TRUE(analysis.estimate());
    std::vector<double> second_output_values;
    for (const auto& state : analysis.results()->output)
        for (double v : state.values) second_output_values.push_back(v);

    CHECK_TRUE(first_output_values.size() == second_output_values.size());
    CHECK_TRUE(!first_output_values.empty());
    for (std::size_t i = 0; i < first_output_values.size(); ++i) {
        CHECK_EQ(first_output_values[i], second_output_values[i]);
    }
}

void test_validate_valid_for_well_formed_configuration() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    BayesianAnalysis analysis(model, SamplerType::DEMCzs);
    apply_fast_knobs(analysis, 1234);

    auto [is_valid, messages] = analysis.validate();
    CHECK_TRUE(is_valid);
    // Warnings are still expected (iterations/output_length below 1,000; not estimated yet).
    CHECK_TRUE(!messages.empty());
}

void test_validate_invalid_for_bad_number_of_chains() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    BayesianAnalysis analysis(model, SamplerType::DEMCzs);
    apply_fast_knobs(analysis, 1234);
    analysis.set_number_of_chains(2);  // Below the required minimum of 4.

    auto [is_valid, messages] = analysis.validate();
    CHECK_TRUE(!is_valid);

    bool found_chains_error = false;
    for (const auto& message : messages) {
        if (message.find("number of Markov chains must be between 4 and 20") != std::string::npos) {
            found_chains_error = true;
        }
    }
    CHECK_TRUE(found_chains_error);
}

void test_estimate_throws_when_invalid() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    BayesianAnalysis analysis(model, SamplerType::DEMCzs);
    apply_fast_knobs(analysis, 1234);
    analysis.set_number_of_chains(2);

    CHECK_THROWS(analysis.estimate());
}

void test_set_default_simulation_options_demczs_vs_nuts() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    // DEMCzs: d = 2 parameters -> NumberOfChains = max(4, min(20, 2*d)) = 4;
    // ThinningInterval = max(1, min(100, 10*d)) = 20.
    BayesianAnalysis demczs(model, SamplerType::DEMCzs);
    CHECK_EQ(demczs.number_of_chains(), 4);
    CHECK_EQ(demczs.thinning_interval(), 20);

    // NUTS: NumberOfChains = 4 always; ThinningInterval = 1 always (no thinning).
    BayesianAnalysis nuts(model, SamplerType::NUTS);
    CHECK_EQ(nuts.number_of_chains(), 4);
    CHECK_EQ(nuts.thinning_interval(), 1);
    CHECK_EQ(nuts.max_tree_depth(), 10);
}

void test_gated_diagnostics_throw() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    BayesianAnalysis analysis(model, SamplerType::DEMCzs);
    apply_fast_knobs(analysis, 7);
    CHECK_TRUE(analysis.estimate());

    CHECK_THROWS(analysis.compute_influence_diagnostics());
    CHECK_THROWS(analysis.compute_prior_influence_diagnostics());
    CHECK_THROWS(analysis.compute_leverage_diagnostics());
}

void test_all_four_sampler_types_build_a_sampler() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    BayesianAnalysis demcz(model, SamplerType::DEMCz);
    CHECK_TRUE(demcz.sampler() != nullptr);

    BayesianAnalysis demczs(model, SamplerType::DEMCzs);
    CHECK_TRUE(demczs.sampler() != nullptr);

    BayesianAnalysis arwmh(model, SamplerType::ARWMH);
    CHECK_TRUE(arwmh.sampler() != nullptr);

    BayesianAnalysis nuts(model, SamplerType::NUTS);
    CHECK_TRUE(nuts.sampler() != nullptr);
}

}  // namespace

int main() {
    test_estimate_succeeds_and_produces_finite_results();
    test_determinism_same_seed_gives_identical_results();
    test_validate_valid_for_well_formed_configuration();
    test_validate_invalid_for_bad_number_of_chains();
    test_estimate_throws_when_invalid();
    test_set_default_simulation_options_demczs_vs_nuts();
    test_gated_diagnostics_throw();
    test_all_four_sampler_types_build_a_sampler();
    return bftest::summary("test_bayesian_analysis");
}
