// C++-only MCMC tests that don't fit the fixture-driven `mcmc_sampler` kind's declarative
// {construct, assertions} shape -- see fixtures/README.md's "special_function"/C++-only
// convention and the P3.6 task brief.
//
// The first two cases are transcribed from Test_SNIS.cs's two invalid-weight tests
// (Test_SNIS_AllInvalidWeights_ThrowsInvalidOperationException,
// Test_SNIS_MixedFiniteAndInvalidWeights_NormalizesFiniteWeights). They are intentionally
// NOT fixture cases: their log-likelihood closures are inline C# lambdas with no declarative
// encoding in this port's closed named-function registries (`x < 0.5 ? 0 : -inf` has no
// counterpart in model_registry.hpp, which only builds full Bayesian models, not raw
// closures), and the mixed-weights test's own C# assertions are explicitly ORDER-INSENSITIVE
// (sum/found-zero/found-positive/no-NaN/no-Inf over the whole MarkovChains[0] list) precisely
// because -Infinity fitness ties make any particular sort order irreproducible across
// languages -- there is no literal chain digest to lock here, unlike RWMH/ARWMH/Gibbs.
//
// The third case is transcribed from Test_HMC.cs's
// Test_HMC_NonFiniteGradient_DoesNotCrash: a smoke test, not a numeric-reproduction fixture
// (the C# test itself asserts only "does not throw" + "MarkovChains is non-empty", no
// numeric literal), so it belongs here for the same "no declarative encoding" reason as the
// SNIS cases above -- it also exercises a code path (`HMC::chain_iteration`'s
// `catch (const std::domain_error&)`, see hmc.hpp's file header for the ArithmeticException
// mapping) that neither of `hmc.json`'s two numeric-reproduction cases is guaranteed to hit
// on every run.
//
// The remaining cases (P3.9) are transcribed from Test_MCMCDiagnostics.cs and
// Test_MCMCResults_Recompute.cs. Neither fits the fixture-driven shape either, for the same
// underlying reason: both suites assert INEQUALITIES/invariants over deterministically-seeded
// synthetic data (Rhat thresholds and monotonic ordering; alpha-independent-vs-alpha-dependent
// summary-statistic preservation), not single expected literals a `special_function` case's one
// `"value"` assertion could hold. `MCMCDiagnostics.MinimumSampleSize` -- the one deterministic
// scalar method on this file's C# source -- is NOT here; it has its own `special_function`
// fixture (`fixtures/special_functions/mcmc_diagnostics.json`).
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <vector>

#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/sampling/mcmc/hmc.hpp"
#include "corehydro/numerics/sampling/mcmc/snis.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_diagnostics.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"

namespace mcmc = corehydro::numerics::sampling::mcmc;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::Uniform;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::sampling::MersenneTwister;

namespace {

// Test_SNIS_AllInvalidWeights_ThrowsInvalidOperationException: every draw's log-likelihood is
// -infinity, so every importance weight is non-finite -- SNIS::sample() must fail fast rather
// than silently normalize garbage.
void test_all_invalid_weights_throws() {
    std::vector<std::shared_ptr<UnivariateDistributionBase>> priors{std::make_shared<Uniform>(0.0, 1.0)};
    mcmc::SNIS sampler(priors, [](const std::vector<double>&) { return -std::numeric_limits<double>::infinity(); });
    sampler.set_iterations(100);
    sampler.output_length = 100;
    sampler.set_prng_seed(12345);

    CHECK_THROWS(sampler.sample());
}

// Test_SNIS_MixedFiniteAndInvalidWeights_NormalizesFiniteWeights: half the parameter space
// (x[0] < 0.5) is finite (logLH = 0), the other half is -infinity. Asserts the
// order-insensitive properties the C# test asserts over the RAW MarkovChains[0] list (every
// draw, not the resampled Output): no NaN/Infinity weights, the finite weights sum to 1 (they
// are already normalized posterior weights at this point), and at least one exactly-zero
// weight (the non-finite draws, mapped to 0 by the normalization loop) and at least one
// strictly-positive weight (the finite draws) both occur.
void test_mixed_finite_and_invalid_weights_normalizes() {
    std::vector<std::shared_ptr<UnivariateDistributionBase>> priors{std::make_shared<Uniform>(0.0, 1.0)};
    mcmc::SNIS sampler(priors, [](const std::vector<double>& x) {
        return x[0] < 0.5 ? 0.0 : -std::numeric_limits<double>::infinity();
    });
    sampler.set_iterations(100);
    sampler.output_length = 100;
    sampler.set_prng_seed(12345);

    sampler.sample();

    double sum = 0.0;
    bool found_zero_weight = false;
    bool found_positive_weight = false;
    for (const auto& ps : sampler.markov_chains()[0]) {
        CHECK_TRUE(!std::isnan(ps.weight));
        CHECK_TRUE(!std::isinf(ps.weight));
        sum += ps.weight;
        found_zero_weight |= (ps.weight == 0.0);
        found_positive_weight |= (ps.weight > 0.0);
    }

    CHECK_TRUE(found_zero_weight);
    CHECK_TRUE(found_positive_weight);
    CHECK_NEAR(sum, 1.0, 1e-12);
}

// Test_HMC_NonFiniteGradient_DoesNotCrash: narrow priors (mu in [-100, 100], sigma in [0.01,
// 50]) and a large step size/step count make it easy for leapfrog to drift mu far from the
// data while sigma is small, so `Normal::log_likelihood` legitimately returns -infinity for
// some leapfrog probes. `HMC::chain_iteration`'s `catch (const std::domain_error&)` must
// absorb the resulting non-finite-gradient failure (see hmc.hpp's file header) and return the
// unchanged state rather than letting the exception propagate out of `sample()`.
void test_hmc_non_finite_gradient_does_not_crash() {
    auto mu_prior = std::make_shared<Uniform>(-100.0, 100.0);
    auto sigma_prior = std::make_shared<Uniform>(0.01, 50.0);
    std::vector<std::shared_ptr<UnivariateDistributionBase>> priors{mu_prior, sigma_prior};

    std::vector<double> data{10.0, 12.0, 11.0, 13.0, 9.0, 14.0, 10.5, 11.5};

    mcmc::HMC sampler(
        priors,
        [&data](const std::vector<double>& x) {
            Normal dist(x[0], x[1]);
            return dist.log_likelihood(data);
        },
        std::nullopt, /*step_size=*/1.0, /*steps=*/20);
    sampler.set_number_of_chains(2);
    sampler.set_warmup_iterations(100);
    sampler.set_iterations(200);

    // This should complete without throwing.
    sampler.sample();

    CHECK_TRUE(!sampler.markov_chains().empty());
    CHECK_TRUE(!sampler.markov_chains()[0].empty());
}

// ---------------------------------------------------------------------------------------
// Test_MCMCDiagnostics.cs: Test_GelmanRubin_WithWarmup / Test_GelmanRubin_EdgeCases.
// ---------------------------------------------------------------------------------------

// Test_GelmanRubin_WithWarmup: three chains of 200 single-parameter draws, each seeded from
// its own MersenneTwister, drift-shifted for the first 50 iterations (chain1 +5, chain2 -5,
// chain3 +3) then converging to a shared [-1, 1) uniform draw. Without discarding that
// dispersed warmup period, R-hat must read as clearly non-converged (> 1.1); dropping the
// first 50 draws (warmup=50) must bring R-hat back under 1.1 AND strictly below the
// no-warmup value (warmup can only improve, never worsen, R-hat here).
void test_gelman_rubin_with_warmup() {
    MersenneTwister rng1(42);
    MersenneTwister rng2(123);
    MersenneTwister rng3(456);
    const int chain_length = 200;

    std::vector<mcmc::ParameterSet> chain1, chain2, chain3;
    chain1.reserve(chain_length);
    chain2.reserve(chain_length);
    chain3.reserve(chain_length);

    for (int i = 0; i < chain_length; ++i) {
        double drift1 = i < 50 ? 5.0 : 0.0;
        double drift2 = i < 50 ? -5.0 : 0.0;
        double drift3 = i < 50 ? 3.0 : 0.0;

        chain1.emplace_back(std::vector<double>{rng1.next_double() * 2.0 - 1.0 + drift1}, 0.0);
        chain2.emplace_back(std::vector<double>{rng2.next_double() * 2.0 - 1.0 + drift2}, 0.0);
        chain3.emplace_back(std::vector<double>{rng3.next_double() * 2.0 - 1.0 + drift3}, 0.0);
    }

    std::vector<std::vector<mcmc::ParameterSet>> chains{chain1, chain2, chain3};

    // Without warmup, R-hat should be high (chains have different initial distributions).
    auto rhat_no_warmup = mcmc::gelman_rubin(chains, 0);
    CHECK_TRUE(rhat_no_warmup[0] > 1.1);

    // With warmup=50, R-hat should be close to 1.0 (converged portion only).
    auto rhat_with_warmup = mcmc::gelman_rubin(chains, 50);
    CHECK_TRUE(rhat_with_warmup[0] < 1.1);

    // Warmup should improve R-hat (make it closer to 1.0, i.e. strictly smaller here since
    // both values are >= 1.0).
    CHECK_TRUE(rhat_with_warmup[0] < rhat_no_warmup[0]);
}

// Test_GelmanRubin_EdgeCases: a single chain has no between-chain variance to compare
// against, so GelmanRubin must return NaN rather than dividing by zero or throwing.
void test_gelman_rubin_edge_cases() {
    std::vector<mcmc::ParameterSet> chain;
    chain.reserve(10);
    for (int i = 0; i < 10; ++i) chain.emplace_back(std::vector<double>{1.0}, 0.0);
    std::vector<std::vector<mcmc::ParameterSet>> single_chain{chain};

    auto result = mcmc::gelman_rubin(single_chain);
    CHECK_TRUE(std::isnan(result[0]));
}

// ---------------------------------------------------------------------------------------
// Test_MCMCResults_Recompute.cs: the RecomputeParameterResults(alpha) contract -- recomputing
// at a new alpha must update the credible-interval percentiles (LowerCI/UpperCI) while
// preserving alpha-independent diagnostics (Rhat/ESS/Autocorrelation) AND the underlying
// chain output (Output/MAP).
// ---------------------------------------------------------------------------------------

// StandardNormal: Box-Muller transform, single sample (mirrors the C# test's private helper
// exactly, including the u1 < 1e-300 floor that guards std::log(0)).
double standard_normal(MersenneTwister& rng) {
    double u1 = rng.next_double();
    double u2 = rng.next_double();
    if (u1 < 1e-300) u1 = 1e-300;
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * corehydro::numerics::kPi * u2);
}

// BuildResults: builds a minimal MCMCResults via the (map, parameterSets, alpha) ctor.
// Three parameters (mu, sigma, tau) sampled from synthetic distributions, seeded from
// MersenneTwister(2026) exactly as the C# test does. 1000 samples is enough for smooth
// percentile estimates.
mcmc::MCMCResults build_results(double alpha) {
    MersenneTwister rng(2026);
    std::vector<mcmc::ParameterSet> output;
    output.reserve(1000);
    for (int i = 0; i < 1000; ++i) {
        // Param 0: ~ Normal(10, 1).
        double p0 = 10.0 + standard_normal(rng);
        // Param 1: ~ Normal(2, 0.5).
        double p1 = 2.0 + 0.5 * standard_normal(rng);
        // Param 2: ~ Uniform(-1, 1).
        double p2 = -1.0 + 2.0 * rng.next_double();
        output.emplace_back(std::vector<double>{p0, p1, p2}, 0.0);
    }
    mcmc::ParameterSet map(std::vector<double>{10.0, 2.0, 0.0}, 0.0);
    return mcmc::MCMCResults(map, output, alpha);
}

// Snapshot of the alpha-independent diagnostics SeedDiagnostics seeds and Recompute must
// preserve.
struct SeededDiagnostics {
    std::vector<double> rhats;
    std::vector<double> esss;
    std::vector<std::vector<std::array<double, 2>>> acfs;
};

// SeedDiagnostics: seed each parameter's alpha-independent diagnostics with sentinel values
// (normally populated by the MCMCSampler-driven ctor; set directly here to verify the
// snapshot/restore in isolation, exactly as the C# test does).
//
// C#'s Autocorrelation is a general `double[,]`, and this test seeds a `[1, 5]` shape (1 row,
// 5 columns) purely to exercise reference/value preservation -- unrelated to production code's
// fixed `[51, 2]` populator shape. This port's `autocorrelation` field is `std::vector<
// std::array<double, 2>>` (a fixed 2-column-per-row representation, matching every REAL
// populator in this port -- see mcmc_diagnostics.hpp's own header note), so this port seeds a
// 3-row, fixed-2-column vector instead of a 1x5 array: same intent (arbitrary
// alpha-independent per-parameter data, distinct per parameter index, preserved verbatim
// across `recompute_parameter_results`), fitted to the port's actual field shape.
SeededDiagnostics seed_diagnostics(mcmc::MCMCResults& results) {
    std::size_t n = results.parameter_results.size();
    SeededDiagnostics seeded;
    seeded.rhats.resize(n);
    seeded.esss.resize(n);
    seeded.acfs.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        double di = static_cast<double>(i);
        seeded.rhats[i] = 1.0 + 0.01 * (di + 1.0);  // 1.01, 1.02, 1.03
        seeded.esss[i] = 800.0 + 10.0 * di;         // 800, 810, 820
        seeded.acfs[i] = {{1.0, 0.5 - 0.1 * di}, {0.25, 0.125}, {0.0625, 0.0}};
        results.parameter_results[i].summary_statistics.rhat = seeded.rhats[i];
        results.parameter_results[i].summary_statistics.ess = seeded.esss[i];
        results.parameter_results[i].autocorrelation = seeded.acfs[i];
    }
    return seeded;
}

// Recompute_PreservesOutputReference: `output`'s underlying buffer must not be touched by
// recompute_parameter_results -- there is no C++ analog of C#'s `Assert.AreSame` reference
// check (std::vector always has value semantics; see parameter_set.hpp's clone() header note
// for the same C#-reference-vs-C++-value-semantics distinction), so this checks the buffer
// pointer is unchanged (no reallocation/copy occurred) in addition to size/values.
void test_recompute_preserves_output_reference() {
    auto results = build_results(0.10);
    const mcmc::ParameterSet* output_ptr = results.output.data();
    std::size_t output_size = results.output.size();

    results.recompute_parameter_results(0.05);

    CHECK_TRUE(results.output.data() == output_ptr);
    CHECK_EQ(results.output.size(), output_size);
    CHECK_EQ(results.output.size(), static_cast<std::size_t>(1000));
}

// Recompute_PreservesMAP.
void test_recompute_preserves_map() {
    auto results = build_results(0.10);
    std::vector<double> values_before = results.map.values;
    double fitness_before = results.map.fitness;

    results.recompute_parameter_results(0.05);

    CHECK_EQ(results.map.values.size(), values_before.size());
    for (std::size_t i = 0; i < values_before.size(); ++i) {
        CHECK_NEAR(results.map.values[i], values_before[i], 1e-12);
    }
    CHECK_NEAR(results.map.fitness, fitness_before, 1e-12);
}

// Recompute_PreservesRhat.
void test_recompute_preserves_rhat() {
    auto results = build_results(0.10);
    auto seeded = seed_diagnostics(results);

    results.recompute_parameter_results(0.05);

    for (std::size_t i = 0; i < results.parameter_results.size(); ++i) {
        CHECK_NEAR(results.parameter_results[i].summary_statistics.rhat, seeded.rhats[i], 1e-12);
    }
}

// Recompute_PreservesESS.
void test_recompute_preserves_ess() {
    auto results = build_results(0.10);
    auto seeded = seed_diagnostics(results);

    results.recompute_parameter_results(0.05);

    for (std::size_t i = 0; i < results.parameter_results.size(); ++i) {
        CHECK_NEAR(results.parameter_results[i].summary_statistics.ess, seeded.esss[i], 1e-12);
    }
}

// Recompute_PreservesAutocorrelation.
void test_recompute_preserves_autocorrelation() {
    auto results = build_results(0.10);
    auto seeded = seed_diagnostics(results);

    results.recompute_parameter_results(0.05);

    for (std::size_t i = 0; i < results.parameter_results.size(); ++i) {
        const auto& acf_after = results.parameter_results[i].autocorrelation;
        CHECK_EQ(acf_after.size(), seeded.acfs[i].size());
        for (std::size_t r = 0; r < seeded.acfs[i].size(); ++r) {
            for (std::size_t c = 0; c < 2; ++c) {
                CHECK_NEAR(acf_after[r][c], seeded.acfs[i][r][c], 1e-12);
            }
        }
    }
}

// Recompute_NarrowsCIWhenAlphaIncreases: alpha=0.10 -> 90% CI; alpha=0.20 -> 80% CI (narrower
// band).
void test_recompute_narrows_ci_when_alpha_increases() {
    auto results = build_results(0.10);
    std::vector<double> lower90, upper90;
    for (const auto& pr : results.parameter_results) {
        lower90.push_back(pr.summary_statistics.lower_ci);
        upper90.push_back(pr.summary_statistics.upper_ci);
    }

    results.recompute_parameter_results(0.20);

    for (std::size_t i = 0; i < results.parameter_results.size(); ++i) {
        double lower80 = results.parameter_results[i].summary_statistics.lower_ci;
        double upper80 = results.parameter_results[i].summary_statistics.upper_ci;
        CHECK_TRUE(lower80 > lower90[i]);
        CHECK_TRUE(upper80 < upper90[i]);
    }
}

// Recompute_WidensCIWhenAlphaDecreases: alpha=0.10 -> 90% CI; alpha=0.05 -> 95% CI (wider
// band).
void test_recompute_widens_ci_when_alpha_decreases() {
    auto results = build_results(0.10);
    std::vector<double> lower90, upper90;
    for (const auto& pr : results.parameter_results) {
        lower90.push_back(pr.summary_statistics.lower_ci);
        upper90.push_back(pr.summary_statistics.upper_ci);
    }

    results.recompute_parameter_results(0.05);

    for (std::size_t i = 0; i < results.parameter_results.size(); ++i) {
        double lower95 = results.parameter_results[i].summary_statistics.lower_ci;
        double upper95 = results.parameter_results[i].summary_statistics.upper_ci;
        CHECK_TRUE(lower95 < lower90[i]);
        CHECK_TRUE(upper95 > upper90[i]);
    }
}

// Recompute_PreservesMeanAndMedian: Mean and Median are alpha-independent -- should be
// preserved (to within the recompute's own percentile/KDE re-evaluation noise, matching the
// C# test's 1e-10 tolerance).
void test_recompute_preserves_mean_and_median() {
    auto results = build_results(0.10);
    std::vector<double> means_before, medians_before;
    for (const auto& pr : results.parameter_results) {
        means_before.push_back(pr.summary_statistics.mean);
        medians_before.push_back(pr.summary_statistics.median);
    }

    results.recompute_parameter_results(0.05);

    for (std::size_t i = 0; i < results.parameter_results.size(); ++i) {
        CHECK_NEAR(results.parameter_results[i].summary_statistics.mean, means_before[i], 1e-10);
        CHECK_NEAR(results.parameter_results[i].summary_statistics.median, medians_before[i], 1e-10);
    }
}

// Recompute_OnEmptyResults_DoesNotThrow: the `parameter_results.empty() || output.empty()`
// guard.
void test_recompute_on_empty_results_does_not_throw() {
    mcmc::MCMCResults results;

    results.recompute_parameter_results(0.05);

    // Should be a no-op -- parameter_results stays empty, no exception.
    CHECK_TRUE(results.parameter_results.empty());
}

}  // namespace

int main() {
    test_all_invalid_weights_throws();
    test_mixed_finite_and_invalid_weights_normalizes();
    test_hmc_non_finite_gradient_does_not_crash();
    test_gelman_rubin_with_warmup();
    test_gelman_rubin_edge_cases();
    test_recompute_preserves_output_reference();
    test_recompute_preserves_map();
    test_recompute_preserves_rhat();
    test_recompute_preserves_ess();
    test_recompute_preserves_autocorrelation();
    test_recompute_narrows_ci_when_alpha_increases();
    test_recompute_widens_ci_when_alpha_decreases();
    test_recompute_preserves_mean_and_median();
    test_recompute_on_empty_results_does_not_throw();
    return chtest::summary("test_mcmc_extra");
}
