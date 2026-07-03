// C++-only MCMC tests that don't fit the fixture-driven `mcmc_sampler` kind's declarative
// {construct, assertions} shape -- see fixtures/README.md's "special_function"/C++-only
// convention and the P3.6 task brief.
//
// Both cases below are transcribed from Test_SNIS.cs's two invalid-weight tests
// (Test_SNIS_AllInvalidWeights_ThrowsInvalidOperationException,
// Test_SNIS_MixedFiniteAndInvalidWeights_NormalizesFiniteWeights). They are intentionally
// NOT fixture cases: their log-likelihood closures are inline C# lambdas with no declarative
// encoding in this port's closed named-function registries (`x < 0.5 ? 0 : -inf` has no
// counterpart in model_registry.hpp, which only builds full Bayesian models, not raw
// closures), and the mixed-weights test's own C# assertions are explicitly ORDER-INSENSITIVE
// (sum/found-zero/found-positive/no-NaN/no-Inf over the whole MarkovChains[0] list) precisely
// because -Infinity fitness ties make any particular sort order irreproducible across
// languages -- there is no literal chain digest to lock here, unlike RWMH/ARWMH/Gibbs.
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/sampling/mcmc/snis.hpp"
#include "check.hpp"

namespace mcmc = bestfit::numerics::sampling::mcmc;
using bestfit::numerics::distributions::Uniform;
using bestfit::numerics::distributions::UnivariateDistributionBase;

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

}  // namespace

int main() {
    test_all_invalid_weights_throws();
    test_mixed_finite_and_invalid_weights_normalizes();
    return bftest::summary("test_mcmc_extra");
}
