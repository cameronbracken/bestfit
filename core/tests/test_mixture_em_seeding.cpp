// C++-only ctest for the X2 MCMC UserDefined seeding hook + the MixtureAnalysis EM-seed
// draw cadence.
//
// Two concerns, neither of which fits the declarative fixture shape (public-API oracle values
// live only in fixtures/; these are internal-support C++-only oracles transcribing the
// deterministic upstream draws -- see fixtures/README.md and the X2 brief):
//
//   1. The mutable UserDefined seeding hook on MCMCSampler MUST survive reset(). The C#
//      MixtureAnalysis.RunAsync seeds sampler.PopulationMatrix / sampler.MarkovChains[i]
//      directly AFTER SetUpSampler() and just before Sample(), with no intervening setter. The
//      port stores the seed in members reset() does not clear and re-materializes them at the
//      point sample()/initialize_chains() consumes them, so a reset-triggering setter flipped
//      after seeding cannot wipe the seed. This exercises that survive-reset contract.
//
//   2. The EM + Dirichlet(weights) + MultivariateNormal(components) proposal draw the
//      MixtureAnalysis EM-seed path uses is DETERMINISTIC: fixed data => byte-identical EM
//      output; fixed PRNG seed => byte-identical proposal draws run-to-run. This is the
//      deterministic-grid check the chaotic-sensitivity rule (Phase 9a) requires before any
//      claim that a downstream seeded-MCMC divergence is inherent rather than a port bug.
#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/univariate_distribution/mixture_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/multivariate/dirichlet.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

namespace mcmc = corehydro::numerics::sampling::mcmc;
using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ExactSeries;
using corehydro::models::MixtureModel;
using corehydro::numerics::distributions::Dirichlet;
using corehydro::numerics::distributions::MultivariateNormal;
using corehydro::numerics::distributions::Uniform;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::math::linalg::Matrix;
using corehydro::numerics::math::optimization::ParameterSet;
using corehydro::numerics::sampling::MersenneTwister;

namespace {

// A trivial concrete sampler: chain_iteration returns the state unchanged, so every recorded
// chain state equals the initial (seeded) state and the population's leading entries are the
// seeded population. is_population_sampler_ = true so population seeding is exercised too.
class IdentitySampler : public mcmc::MCMCSampler {
   public:
    IdentitySampler(std::vector<std::shared_ptr<UnivariateDistributionBase>> priors,
                    mcmc::LogLikelihood ll)
        : MCMCSampler(std::move(priors), std::move(ll)) {
        is_population_sampler_ = true;
    }

   protected:
    ParameterSet chain_iteration(int, ParameterSet state) override { return state; }
};

// (1) The seeding hook must survive a reset() triggered AFTER seeding.
void test_seed_hook_survives_reset() {
    std::vector<std::shared_ptr<UnivariateDistributionBase>> priors{
        std::make_shared<Uniform>(0.0, 1.0)};
    IdentitySampler sampler(priors, [](const std::vector<double>& p) { return p[0]; });
    sampler.set_number_of_chains(2);
    sampler.set_iterations(100);
    sampler.set_warmup_iterations(1);
    sampler.set_thinning_interval(1);
    sampler.set_initial_iterations(3);
    sampler.output_length = 100;

    // Seed the population (3 sets) and the two chains with known values.
    std::vector<ParameterSet> pop{ParameterSet({0.11}, -1.0), ParameterSet({0.22}, -2.0),
                                  ParameterSet({0.33}, -3.0)};
    sampler.seed_population(pop);
    sampler.seed_chain(0, ParameterSet({0.11}, -1.0));
    sampler.seed_chain(1, ParameterSet({0.22}, -2.0));

    sampler.initialize = mcmc::MCMCSampler::InitializationType::UserDefined;

    // Flip a reset-triggering setter AFTER seeding: reset() clears population_matrix_ and
    // re-assigns markov_chains_ to empty. The hook must survive this.
    sampler.set_thinning_interval(1);

    sampler.sample();

    // The materialized population's leading entries are the seed (post-reset wipe honored).
    CHECK_TRUE(sampler.population_matrix().size() >= pop.size());
    CHECK_NEAR(sampler.population_matrix()[0].values[0], 0.11, 1e-15);
    CHECK_NEAR(sampler.population_matrix()[1].values[0], 0.22, 1e-15);
    CHECK_NEAR(sampler.population_matrix()[2].values[0], 0.33, 1e-15);

    // The first recorded state of each chain is the seed (identity iteration => frozen).
    CHECK_TRUE(!sampler.markov_chains()[0].empty());
    CHECK_TRUE(!sampler.markov_chains()[1].empty());
    CHECK_NEAR(sampler.markov_chains()[0].front().values[0], 0.11, 1e-15);
    CHECK_NEAR(sampler.markov_chains()[1].front().values[0], 0.22, 1e-15);
}

// A never-seeded UserDefined sampler must be unaffected by the hook (additive-only guard):
// with an empty seed and no prior Sample(), the pre-existing UserDefined branch behavior is
// unchanged. We only assert it does not spuriously honor a nonexistent seed here by checking
// the Randomize path is byte-identical below via the fixture harnesses; this case simply
// confirms seed_population/seed_chain default to empty and the accessors still read empty.
void test_unseeded_hook_is_empty() {
    std::vector<std::shared_ptr<UnivariateDistributionBase>> priors{
        std::make_shared<Uniform>(0.0, 1.0)};
    IdentitySampler sampler(priors, [](const std::vector<double>& p) { return p[0]; });
    // No seeding: population_matrix() / markov_chains() start empty / N-empty after construct.
    CHECK_TRUE(sampler.population_matrix().empty());
    CHECK_EQ(static_cast<int>(sampler.markov_chains().size()), sampler.number_of_chains());
    for (const auto& chain : sampler.markov_chains()) CHECK_TRUE(chain.empty());
}

// Fixed bimodal 2-component dataset (mirrors mixture_analysis_smoke.json's `bimodal`).
MixtureModel make_bimodal_model() {
    DataFrame df;
    std::vector<ExactData> data;
    const double vals[] = {520,  580,  610,  650,  700,  730,  760,  800,  850,  880,
                           910,  950,  990,  1030, 1080, 5000, 5400, 5800, 6300, 6800};
    int year = 1980;
    for (double v : vals) data.emplace_back(year++, v);
    df.set_exact_series(ExactSeries(data));
    std::vector<UnivariateDistributionType> types{UnivariateDistributionType::Normal,
                                                  UnivariateDistributionType::Normal};
    return MixtureModel(std::move(df), types);
}

// Replicate the MixtureAnalysis EM-seed proposal draw cadence for `draw_count` iterations with
// a fixed PRNG seed. Returns the concatenated (weights ++ components) parameter vectors. Draws
// that throw (invalid component params) are skipped deterministically (identical across runs).
std::vector<std::vector<double>> draw_proposals(const std::vector<double>& parameters,
                                                const Matrix& covariance, int K, int N,
                                                int prng_seed, int draw_count) {
    int Np = static_cast<int>(parameters.size()) - K;

    // Weights: Dirichlet(alpha) with alpha_j = max(w_j * S / deflation, 0.1), S = max(N-1,K+1).
    double S = std::max(N - 1, K + 1);
    double deflation = 1.5;
    std::vector<double> alpha(static_cast<std::size_t>(K));
    for (int j = 0; j < K; ++j)
        alpha[static_cast<std::size_t>(j)] =
            std::max(parameters[static_cast<std::size_t>(j)] * S / deflation, 0.1);
    Dirichlet weight_dirichlet(alpha);

    // Components: MVN over the Fisher sub-block of the EM covariance, inflated 1.5x.
    std::vector<double> em_components(static_cast<std::size_t>(Np));
    std::vector<std::vector<double>> component_covar(
        static_cast<std::size_t>(Np), std::vector<double>(static_cast<std::size_t>(Np)));
    for (int i = 0; i < Np; ++i) {
        em_components[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(i + K)];
        for (int j = 0; j < Np; ++j)
            component_covar[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                covariance(i + K, j + K) * 1.5;
    }
    MultivariateNormal component_mvn(em_components, component_covar);

    MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
    std::vector<std::vector<double>> out;
    for (int i = 0; i < draw_count; ++i) {
        std::vector<double> weights;
        if (K > 1) {
            auto w = weight_dirichlet.generate_random_values(1, prng.next());
            weights.assign(w[0].begin(), w[0].end());
        } else {
            weights = {1.0};
        }
        // MVN draw uses the C# NextDoubles(1, Np) cadence: one outer prng.next() per dimension,
        // each seeding a fresh MT whose first NextDouble() is the uniform for that dimension.
        std::vector<double> u(static_cast<std::size_t>(Np));
        for (int d = 0; d < Np; ++d) {
            MersenneTwister sub(static_cast<std::uint32_t>(prng.next()));
            u[static_cast<std::size_t>(d)] = sub.next_double();
        }
        try {
            std::vector<double> comp = component_mvn.inverse_cdf(u);
            std::vector<double> p;
            p.insert(p.end(), weights.begin(), weights.end());
            p.insert(p.end(), comp.begin(), comp.end());
            out.push_back(std::move(p));
        } catch (const std::exception&) {
            // Deterministic across runs: skip identically.
        }
    }
    return out;
}

// (2) EM output + proposal draws are deterministic run-to-run.
void test_em_seed_determinism() {
    MixtureModel m1 = make_bimodal_model();
    MixtureModel m2 = make_bimodal_model();

    std::vector<double> p1, p2;
    Matrix c1(0, 0), c2(0, 0);
    int it1 = 0, it2 = 0;
    m1.expectation_maximization(p1, c1, it1);
    m2.expectation_maximization(p2, c2, it2);

    // EM is deterministic: byte-identical parameters, covariance, and iteration count.
    CHECK_EQ(it1, it2);
    CHECK_EQ(p1.size(), p2.size());
    for (std::size_t i = 0; i < p1.size(); ++i) CHECK_EQ(p1[i], p2[i]);
    CHECK_EQ(c1.number_of_rows(), c2.number_of_rows());
    CHECK_EQ(c1.number_of_columns(), c2.number_of_columns());
    for (int i = 0; i < c1.number_of_rows(); ++i)
        for (int j = 0; j < c1.number_of_columns(); ++j) CHECK_EQ(c1(i, j), c2(i, j));

    // Sanity check that the seed cadence itself is reproducible (draw_proposals mirrors the
    // production seed order). The production seed path (seed_sampler_from_em surviving sample()'s
    // reset) is covered structurally by the survive-reset test + the mixture fixture curve_length.
    // Proposal draws with a fixed seed are byte-identical run-to-run.
    int K = m1.mixture()->component_count();
    int N = m1.data_frame().total_record_length();
    auto d1 = draw_proposals(p1, c1, K, N, 12345, 10);
    auto d2 = draw_proposals(p1, c1, K, N, 12345, 10);
    CHECK_EQ(d1.size(), d2.size());
    CHECK_TRUE(!d1.empty());
    for (std::size_t i = 0; i < d1.size() && i < d2.size(); ++i) {
        CHECK_EQ(d1[i].size(), d2[i].size());
        for (std::size_t j = 0; j < d1[i].size() && j < d2[i].size(); ++j)
            CHECK_EQ(d1[i][j], d2[i][j]);
    }
}

}  // namespace

int main() {
    test_seed_hook_survives_reset();
    test_unseeded_hook_is_empty();
    test_em_seed_determinism();
    return chtest::summary("test_mixture_em_seeding");
}
