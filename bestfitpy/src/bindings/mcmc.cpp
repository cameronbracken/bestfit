// pybind11 glue exposing the MCMC sampler surface (model registry + RWMH, extensible to
// later samplers) of the shared C++ core to Python. Unlike the per-method dispatch style
// used elsewhere in this package (dist_val, cop_val, ...), `mcmc_sampler` fixtures are
// inherently STATEFUL -- one sampler construct + settings + a single sample() run backs
// every assertion in a case (see fixtures/README.md's mcmc_sampler schema) -- so this file
// exposes ONE function, `mcmc_run`, that builds the model, configures and runs the sampler
// once, and returns every value test_fixtures.py's dispatcher needs in one dict.
// Core headers are vendored under ../bestfit_core/include (see tools/sync_core.py).
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/sampling/mcmc/model_registry.hpp"
#include "bestfit/numerics/sampling/mcmc/rwmh.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "bindings.hpp"

namespace py = pybind11;
namespace mcmc = bestfit::numerics::sampling::mcmc;
namespace la = bestfit::numerics::math::linalg;

// `proposal_sigma` sentinel strings -- see fixtures/README.md's mcmc_sampler schema for why
// "identity" exists alongside the C# test's literal "zeros" (an all-zero proposal covariance
// is only safe when MAP initialization is expected to overwrite it before first use).
static la::Matrix parse_proposal_sigma(const std::string& s, int dimension) {
    if (s == "zeros") return la::Matrix(dimension);
    if (s == "identity") return la::Matrix::identity(dimension);
    throw py::value_error("unknown proposal_sigma sentinel: " + s);
}

static mcmc::MCMCSampler::InitializationType parse_initialize(const std::string& s) {
    if (s == "MAP") return mcmc::MCMCSampler::InitializationType::MAP;
    if (s == "Randomize") return mcmc::MCMCSampler::InitializationType::Randomize;
    if (s == "UserDefined") return mcmc::MCMCSampler::InitializationType::UserDefined;
    throw py::value_error("unknown initialize value: " + s);
}

void register_mcmc(py::module_& m) {
    // Builds the named model, configures `sampler_type` with `settings` (a dict; every key
    // optional, matching fixtures/README.md), samples once, and returns a dict with the full
    // posterior surface test_fixtures.py's mcmc_sampler dispatcher reads assertions from:
    //   chains              -- list of NumberOfChains [n_draws][n_params] nested lists
    //                          (MarkovChains)
    //   chain_fitness       -- list of NumberOfChains [n_draws] lists
    //   acceptance_rates    -- [n_chains] list
    //   map_values          -- [n_params] list (MCMCResults.MAP.Values)
    //   map_fitness         -- scalar (MCMCResults.MAP.Fitness)
    //   mean_log_likelihood -- [iterations] list
    //   posterior_mean/sd/median/lower_ci/upper_ci -- [n_params] lists
    //   rhat/ess            -- [n_params] lists
    m.def(
        "mcmc_run",
        [](const std::string& sampler_type, const std::string& model_name, const std::string& family,
           const std::vector<double>& dataset, const py::dict& settings) {
            auto model = mcmc::build_model(model_name, family, dataset);
            int d = static_cast<int>(model.priors.size());

            la::Matrix proposal_sigma(d);
            if (settings.contains("proposal_sigma"))
                proposal_sigma = parse_proposal_sigma(settings["proposal_sigma"].cast<std::string>(), d);

            std::unique_ptr<mcmc::MCMCSampler> sampler;
            if (sampler_type == "RWMH") {
                sampler = std::make_unique<mcmc::RWMH>(model.priors, model.log_likelihood, proposal_sigma);
            } else {
                throw py::value_error("unknown mcmc_sampler target: " + sampler_type);
            }

            if (settings.contains("initialize"))
                sampler->initialize = parse_initialize(settings["initialize"].cast<std::string>());

            auto set_int = [&](const char* key, const std::function<void(int)>& setter) {
                if (settings.contains(key)) setter(settings[key].cast<int>());
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

            std::vector<std::vector<std::vector<double>>> chains(static_cast<std::size_t>(n_chains));
            std::vector<std::vector<double>> chain_fitness(static_cast<std::size_t>(n_chains));
            for (int c = 0; c < n_chains; ++c) {
                const auto& chain = sampler->markov_chains()[static_cast<std::size_t>(c)];
                chains[static_cast<std::size_t>(c)].resize(chain.size());
                chain_fitness[static_cast<std::size_t>(c)].resize(chain.size());
                for (std::size_t i = 0; i < chain.size(); ++i) {
                    chains[static_cast<std::size_t>(c)][i] = chain[i].values;
                    chain_fitness[static_cast<std::size_t>(c)][i] = chain[i].fitness;
                }
            }

            std::vector<double> map_values(static_cast<std::size_t>(p));
            std::vector<double> post_mean(static_cast<std::size_t>(p)), post_sd(static_cast<std::size_t>(p)),
                post_median(static_cast<std::size_t>(p)), post_lo(static_cast<std::size_t>(p)),
                post_hi(static_cast<std::size_t>(p)), rhat(static_cast<std::size_t>(p)),
                ess(static_cast<std::size_t>(p));
            for (int j = 0; j < p; ++j) {
                map_values[static_cast<std::size_t>(j)] = results.map.values[static_cast<std::size_t>(j)];
                const auto& stats = results.parameter_results[static_cast<std::size_t>(j)].summary_statistics;
                post_mean[static_cast<std::size_t>(j)] = stats.mean;
                post_sd[static_cast<std::size_t>(j)] = stats.standard_deviation;
                post_median[static_cast<std::size_t>(j)] = stats.median;
                post_lo[static_cast<std::size_t>(j)] = stats.lower_ci;
                post_hi[static_cast<std::size_t>(j)] = stats.upper_ci;
                rhat[static_cast<std::size_t>(j)] = stats.rhat;
                ess[static_cast<std::size_t>(j)] = stats.ess;
            }

            py::dict out;
            out["chains"] = chains;
            out["chain_fitness"] = chain_fitness;
            out["acceptance_rates"] = sampler->acceptance_rates();
            out["map_values"] = map_values;
            out["map_fitness"] = results.map.fitness;
            out["mean_log_likelihood"] = sampler->mean_log_likelihood();
            out["posterior_mean"] = post_mean;
            out["posterior_sd"] = post_sd;
            out["posterior_median"] = post_median;
            out["posterior_lower_ci"] = post_lo;
            out["posterior_upper_ci"] = post_hi;
            out["rhat"] = rhat;
            out["ess"] = ess;
            return out;
        },
        py::arg("sampler_type"), py::arg("model_name"), py::arg("family"), py::arg("dataset"),
        py::arg("settings"));
}
