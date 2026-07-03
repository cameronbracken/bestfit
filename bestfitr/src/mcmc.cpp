// cpp11 glue exposing the MCMC sampler surface (model registry + RWMH, extensible to later
// samplers) of the shared C++ core to R. Unlike the per-method dispatch style used elsewhere
// in this package (bf_dist_val_, bf_cop_val_, ...), `mcmc_sampler` fixtures are inherently
// STATEFUL -- one sampler construct + settings + a single sample() run backs every assertion
// in a case (see fixtures/README.md's mcmc_sampler schema) -- so this file exposes ONE
// function, `bf_mcmc_run_`, that builds the model, configures and runs the sampler once, and
// returns every value the test-fixtures.R dispatcher needs in one named list. This avoids a
// "seq machinery" batching mechanism entirely: there is only ever one run per case.
// Core headers are vendored under src/bestfit_core/include (see tools/sync_core.py).
#include <cpp11.hpp>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/sampling/mcmc/arwmh.hpp"
#include "bestfit/numerics/sampling/mcmc/gibbs.hpp"
#include "bestfit/numerics/sampling/mcmc/model_registry.hpp"
#include "bestfit/numerics/sampling/mcmc/rwmh.hpp"
#include "bestfit/numerics/sampling/mcmc/snis.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_results.hpp"

namespace mcmc = bestfit::numerics::sampling::mcmc;
namespace la = bestfit::numerics::math::linalg;
using namespace cpp11;

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema for why
// "identity" exists alongside the C# test's literal "zeros" (an all-zero proposal covariance
// is only safe when MAP initialization is expected to overwrite it before first use).
static la::Matrix parse_proposal_sigma(const std::string& s, int dimension) {
    if (s == "zeros") return la::Matrix(dimension);
    if (s == "identity") return la::Matrix::identity(dimension);
    stop("unknown proposal_sigma sentinel '%s'", s.c_str());
}

static mcmc::MCMCSampler::InitializationType parse_initialize(const std::string& s) {
    if (s == "MAP") return mcmc::MCMCSampler::InitializationType::MAP;
    if (s == "Randomize") return mcmc::MCMCSampler::InitializationType::Randomize;
    if (s == "UserDefined") return mcmc::MCMCSampler::InitializationType::UserDefined;
    stop("unknown initialize value '%s'", s.c_str());
}

// Builds the named model, configures `sampler_type` with `settings` (a named R list; every
// key optional, matching fixtures/README.md), samples once, and returns a named list with the
// full posterior surface `test-fixtures.R`'s mcmc_sampler dispatcher reads assertions from:
//   chains             -- list of NumberOfChains [n_draws x n_params] matrices (MarkovChains)
//   chain_fitness      -- list of NumberOfChains [n_draws] vectors
//   acceptance_rates   -- [n_chains] vector
//   map_values         -- [n_params] vector (MCMCResults.MAP.Values)
//   map_fitness        -- scalar (MCMCResults.MAP.Fitness)
//   mean_log_likelihood -- [iterations] vector
//   posterior_mean/sd/median/lower_ci/upper_ci -- [n_params] vectors
//   rhat/ess           -- [n_params] vectors
[[cpp11::register]]
list bf_mcmc_run_(std::string sampler_type, std::string model_name, std::string family,
                   doubles dataset, list settings) {
    std::vector<double> data(dataset.begin(), dataset.end());
    auto model = mcmc::build_model(model_name, family, data);
    int d = static_cast<int>(model.priors.size());

    SEXP ps = settings["proposal_sigma"];
    la::Matrix proposal_sigma =
        ps == R_NilValue ? la::Matrix(d) : parse_proposal_sigma(as_cpp<std::string>(ps), d);

    std::unique_ptr<mcmc::MCMCSampler> sampler;
    if (sampler_type == "RWMH") {
        sampler = std::make_unique<mcmc::RWMH>(model.priors, model.log_likelihood, proposal_sigma);
    } else if (sampler_type == "ARWMH") {
        auto arwmh = std::make_unique<mcmc::ARWMH>(model.priors, model.log_likelihood);
        SEXP sc = settings["scale"];
        if (sc != R_NilValue) arwmh->scale = as_cpp<double>(sc);
        SEXP be = settings["beta"];
        if (be != R_NilValue) arwmh->beta = as_cpp<double>(be);
        sampler = std::move(arwmh);
    } else if (sampler_type == "Gibbs") {
        if (!model.proposal) stop("Gibbs model has no proposal function");
        sampler = std::make_unique<mcmc::Gibbs>(model.priors, model.log_likelihood, model.proposal);
    } else if (sampler_type == "SNIS") {
        sampler = std::make_unique<mcmc::SNIS>(model.priors, model.log_likelihood);
    } else {
        stop("unknown mcmc_sampler target '%s'", sampler_type.c_str());
    }

    SEXP init = settings["initialize"];
    if (init != R_NilValue) sampler->initialize = parse_initialize(as_cpp<std::string>(init));

    auto set_int = [&](const char* key, const std::function<void(int)>& setter) {
        SEXP v = settings[key];
        if (v != R_NilValue) setter(static_cast<int>(as_cpp<double>(v)));
    };
    set_int("prng_seed", [&](int v) { sampler->set_prng_seed(v); });
    set_int("initial_iterations", [&](int v) { sampler->set_initial_iterations(v); });
    set_int("warmup_iterations", [&](int v) { sampler->set_warmup_iterations(v); });
    set_int("iterations", [&](int v) { sampler->set_iterations(v); });
    set_int("number_of_chains", [&](int v) { sampler->set_number_of_chains(v); });
    set_int("thinning_interval", [&](int v) { sampler->set_thinning_interval(v); });
    set_int("output_length", [&](int v) { sampler->output_length = v; });

    sampler->sample();
    mcmc::MCMCResults results(*sampler);

    int n_chains = sampler->number_of_chains();
    int p = sampler->number_of_parameters();

    writable::list chains(n_chains);
    writable::list chain_fitness(n_chains);
    for (int c = 0; c < n_chains; ++c) {
        const auto& chain = sampler->markov_chains()[static_cast<std::size_t>(c)];
        int n = static_cast<int>(chain.size());
        writable::doubles_matrix<by_column> m(n, p);
        writable::doubles fit(n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < p; ++j) m(i, j) = chain[static_cast<std::size_t>(i)].values[static_cast<std::size_t>(j)];
            fit[i] = chain[static_cast<std::size_t>(i)].fitness;
        }
        chains[c] = m;
        chain_fitness[c] = fit;
    }

    writable::doubles map_values(p);
    writable::doubles post_mean(p), post_sd(p), post_median(p), post_lo(p), post_hi(p), rhat(p), ess(p);
    for (int j = 0; j < p; ++j) {
        map_values[j] = results.map.values[static_cast<std::size_t>(j)];
        const auto& stats = results.parameter_results[static_cast<std::size_t>(j)].summary_statistics;
        post_mean[j] = stats.mean;
        post_sd[j] = stats.standard_deviation;
        post_median[j] = stats.median;
        post_lo[j] = stats.lower_ci;
        post_hi[j] = stats.upper_ci;
        rhat[j] = stats.rhat;
        ess[j] = stats.ess;
    }

    return writable::list({
        "chains"_nm = chains,
        "chain_fitness"_nm = chain_fitness,
        "acceptance_rates"_nm = writable::doubles(sampler->acceptance_rates()),
        "map_values"_nm = map_values,
        "map_fitness"_nm = writable::doubles({results.map.fitness}),
        "mean_log_likelihood"_nm = writable::doubles(sampler->mean_log_likelihood()),
        "posterior_mean"_nm = post_mean,
        "posterior_sd"_nm = post_sd,
        "posterior_median"_nm = post_median,
        "posterior_lower_ci"_nm = post_lo,
        "posterior_upper_ci"_nm = post_hi,
        "rhat"_nm = rhat,
        "ess"_nm = ess,
    });
}
