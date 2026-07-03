// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/BayesianAnalysis.cs @ fc28c0c --
// COMPUTATIONAL-CORE SUBSET (Phase 4, Task T9). BayesianAnalysis.cs is 2518 LOC, almost all
// WPF/XML/async/GUI; this port keeps only: the `SamplerType` enum (C# 234-254) and
// `PointEstimateType` enum (C# 867-876), the knob properties (C# ~490-885), the
// `SetDefaultSimulationOptions` (938)/`SetDefaultAdvancedSimulationOptions` (991) defaulting
// logic, `Validate` (1041), `SetUpSampler` (1170) -- building the 4-way sampler from the
// model's priors + posterior and applying every knob -- and the synchronous compute core of
// `RunAsync` (1234): `sampler.Sample()` then `Results = new MCMCResults(sampler, 1 -
// CredibleIntervalWidth)`.
//
// SCOPE DECISIONS (read before using this class):
//
//   1. SYNCHRONOUS ONLY. `estimate()` is the compute core of `RunAsync` with every
//      Task/async/SafeProgressReporter/CancellationTokenSource/`ParallelizeChains` concern
//      dropped (`mcmc_sampler.hpp` already omits `ParallelizeChains` for the same reason --
//      see that file's header). No `CancelSimulation`, no progress events.
//
//   2. KNOB SETTERS ARE PLAIN (no `ClearResults()` cascade). In C#, nearly every knob
//      property setter calls `ClearResults()` (which itself calls `SetUpSampler()`) AND,
//      for `Type`/`Model`, conditionally re-invokes `SetDefaultSimulationOptions()` /
//      `SetDefaultAdvancedSimulationOptions()` when the corresponding `UseXDefaults` flag is
//      set -- a property-changed-notification cascade that exists so a WPF-bound `Sampler`
//      is always kept in sync with whatever the user just typed into a settings panel. This
//      port has no notification system, so every knob setter here is a plain field write
//      (matching the brief: "Knob properties: plain getters/setters, mutation relaxed").
//      Consequence: `estimate()` below does NOT lazily reuse an existing `sampler_` (C#'s
//      `if (Sampler == null) SetUpSampler();`); it unconditionally calls `set_up_sampler()`
//      immediately before `sample()`, which is simpler than replicating the cascade and
//      strictly more correct (the sampler is always built from whatever knob values are
//      current at call time, never stale).
//
//   3. THE CTOR reproduces the C# ctor's NET EFFECT (`Model = model; Type = type;`, both of
//      which are property setters whose cascades -- because of point 2 -- would otherwise
//      run SetDefaultSimulationOptions/SetDefaultAdvancedSimulationOptions/SetUpSampler
//      multiple times before settling) directly: apply the simulation-option defaults for
//      the given (model, type) once each (when the corresponding `UseXDefaults` flag, true
//      by default, is set), then build the sampler once. This lands in the same final state
//      as the C# ctor without replaying its multi-step property cascade.
//
//   4. DEFERRED TO T10 (not implemented here; each is a separate, larger port): DIC / WAIC /
//      WAIC_pD / LOOIC / LOOIC_SE / ParetoK computation (`ComputeDIC`/`ComputeWAIC`/
//      `ComputePSISLOO`, C# ~1400-1870), `PointEstimateType`-driven point estimates,
//      credible intervals beyond what `MCMCResults`/`ParameterResults` already compute,
//      `GetPosteriorCovarianceMatrix`/`GetPosteriorCorrelationMatrix` (C# 2047/2071).
//
//   5. GATED (Diagnostics layer deferred past Phase 4, see `.claude/PLAN.md`):
//      `ComputeInfluenceDiagnostics` (1870), `ComputePriorInfluenceDiagnostics` (1927),
//      `ComputeLeverageDiagnostics` (1955) are provided as throwing stubs (same convention
//      as `maximum_a_posteriori.hpp`'s `compute_leverage_diagnostics()`), not omitted
//      entirely, so callers get a clear compile-time-visible member and a clear runtime
//      message.
//
//   6. SKIPPED (WPF/GUI/async/XML, no compute-layer analogue): `INotifyPropertyChanged` /
//      `PropertyChanged` / `RaisePropertyChange`, the `RunAsync` async/Task/
//      SafeProgressReporter/parallel/`CancelSimulation` surface (see point 1), `Clone`,
//      `ToXElement`/the XElement ctors/`DeserializeFromXElement`, `GenerateReport`,
//      `ElapsedTime`, `SetCustomMCMCResults`'s XML paths, `ParameterNames`/
//      `NumberOfEstimatedParameters`/`ModelPropertiesToIgnore` (pure passthroughs to
//      `Model.Parameters` with no compute-path caller), `LastError` (exception-based error
//      reporting is used directly instead -- see `estimate()`).
//
// PRIORS-FROM-MODEL WIRING: `set_up_sampler()` builds `priors` by cloning each
// `model_.parameters()[i].prior_distribution()` into a fresh
// `std::shared_ptr<UnivariateDistributionBase>` (mirroring C#'s `Model.Parameters.Select(x =>
// (IUnivariateDistribution)x.PriorDistribution.Clone())`), and passes
// `[this](const std::vector<double>& p) { return model_.log_likelihood(p); }` as the
// sampler's `LogLikelihood` -- the POSTERIOR (data + prior), exactly matching `x =>
// Model.LogLikelihood(x)` and `mcmc_sampler.hpp`'s documented `LogLikelihood` contract (the
// closure must already include the prior).
#pragma once
#include <algorithm>
#include <cmath>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bestfit/models/support/model_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/sampling/mcmc/arwmh.hpp"
#include "bestfit/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "bestfit/numerics/sampling/mcmc/demcz.hpp"
#include "bestfit/numerics/sampling/mcmc/demczs.hpp"
#include "bestfit/numerics/sampling/mcmc/nuts.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_diagnostics.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_results.hpp"

namespace bestfit::estimation {

// The MCMC sampler type to use for estimating model parameters (C# `SamplerType`, C# 234-254 --
// exactly the 4 values ported through Phase 3, same order).
enum class SamplerType { DEMCz, DEMCzs, ARWMH, NUTS };

// The point estimator used to summarize the posterior (C# `PointEstimateType`, C# 867-876).
// Ported here for API completeness / forward reference from `point_estimator()`; the
// point-estimate COMPUTATION that consumes it is deferred to T10 (see scope decision 4).
enum class PointEstimateType { PosteriorMean, PosteriorMode };

class BayesianAnalysis {
   public:
    using MCMCSampler = bestfit::numerics::sampling::mcmc::MCMCSampler;
    using MCMCResults = bestfit::numerics::sampling::mcmc::MCMCResults;

    // Constructs a new Bayesian analysis for `model` (held by reference; NOT owned -- mirrors
    // the C# ctor's `Model` property, which merely stores the passed-in reference; a C++
    // reference cannot be null, so the C# ArgumentNullException guard has no analogue). See
    // scope decision 3 for how this reproduces the C# ctor's net effect.
    explicit BayesianAnalysis(bestfit::models::ModelBase& model, SamplerType type = SamplerType::DEMCzs)
        : model_(model), type_(type) {
        if (use_simulation_defaults_) set_default_simulation_options();
        if (use_advanced_simulation_defaults_) set_default_advanced_simulation_options();
        set_up_sampler();
    }

    // --- Model / sampler access -----------------------------------------------------------

    bestfit::models::ModelBase& model() { return model_; }
    const bestfit::models::ModelBase& model() const { return model_; }

    // The MCMC sampler instance built by the most recent `set_up_sampler()` call (directly or
    // via `estimate()`). May be `nullptr` if the model has no parameters.
    MCMCSampler* sampler() { return sampler_.get(); }
    const MCMCSampler* sampler() const { return sampler_.get(); }

    // --- General settings (C# ~490-620) ---------------------------------------------------

    SamplerType type() const { return type_; }
    void set_type(SamplerType value) { type_ = value; }

    int number_of_chains() const { return number_of_chains_; }
    void set_number_of_chains(int value) { number_of_chains_ = value; }

    int thinning_interval() const { return thinning_interval_; }
    void set_thinning_interval(int value) { thinning_interval_ = value; }

    int warmup_iterations() const { return warmup_iterations_; }
    void set_warmup_iterations(int value) { warmup_iterations_ = value; }

    int iterations() const { return iterations_; }
    void set_iterations(int value) { iterations_ = value; }

    int prng_seed() const { return prng_seed_; }
    void set_prng_seed(int value) { prng_seed_ = value; }

    int initial_iterations() const { return initial_iterations_; }
    void set_initial_iterations(int value) { initial_iterations_ = value; }

    bool use_simulation_defaults() const { return use_simulation_defaults_; }
    void set_use_simulation_defaults(bool value) { use_simulation_defaults_ = value; }

    // --- Advanced settings (sampler-specific; C# ~648-830) ---------------------------------

    bool use_advanced_simulation_defaults() const { return use_advanced_simulation_defaults_; }
    void set_use_advanced_simulation_defaults(bool value) { use_advanced_simulation_defaults_ = value; }

    // DEMCz / DEMCzs.
    double jump() const { return jump_; }
    void set_jump(double value) { jump_ = value; }

    double jump_threshold() const { return jump_threshold_; }
    void set_jump_threshold(double value) { jump_threshold_ = value; }

    double snooker_threshold() const { return snooker_threshold_; }
    void set_snooker_threshold(double value) { snooker_threshold_ = value; }

    double noise() const { return noise_; }
    void set_noise(double value) { noise_ = value; }

    // ARWMH.
    double scale() const { return scale_; }
    void set_scale(double value) { scale_ = value; }

    double beta() const { return beta_; }
    void set_beta(double value) { beta_ = value; }

    // NUTS.
    int max_tree_depth() const { return max_tree_depth_; }
    void set_max_tree_depth(int value) { max_tree_depth_ = value; }

    // --- Output settings (C# ~832-885) ------------------------------------------------------

    double credible_interval_width() const { return credible_interval_width_; }
    void set_credible_interval_width(double value) { credible_interval_width_ = value; }

    int output_length() const { return output_length_; }
    void set_output_length(int value) { output_length_ = value; }

    PointEstimateType point_estimator() const { return point_estimator_; }
    void set_point_estimator(PointEstimateType value) { point_estimator_ = value; }

    // --- Status ------------------------------------------------------------------------

    bool is_estimated() const { return is_estimated_; }

    const std::optional<MCMCResults>& results() const { return results_; }

    // --- Methods (C# ~938-1233) --------------------------------------------------------

    // Sets default simulation options based on the current model and credible interval width
    // (C# `SetDefaultSimulationOptions`, line 938). `Model == null` never applies here (see
    // ctor comment), so this omits that ternary branch but keeps the `d <= 0` clamp for
    // parity in case a model reports zero parameters.
    void set_default_simulation_options() {
        int d = model_.number_of_parameters();
        if (d <= 0) d = 1;

        if (type_ == SamplerType::DEMCz || type_ == SamplerType::DEMCzs) {
            // ter Braak & Vrugt (2008): N=4 suffices for DEMCzs, but 2d gives better
            // diversity for differential-evolution proposals.
            number_of_chains_ = std::max(4, std::min(20, 2 * d));
            thinning_interval_ = std::max(1, std::min(100, 10 * d));
        } else if (type_ == SamplerType::ARWMH) {
            number_of_chains_ = 4;
            thinning_interval_ = std::max(1, std::min(100, 10 * d));
        } else if (type_ == SamplerType::NUTS) {
            number_of_chains_ = 4;
            thinning_interval_ = 1;
        } else {
            number_of_chains_ = 4;
            thinning_interval_ = 20;
        }

        double ci = credible_interval_width_;
        if (ci <= 0.0 || ci >= 1.0) ci = 0.9;

        double alpha = (1.0 - ci) / 2.0;
        iterations_ = bestfit::numerics::sampling::mcmc::minimum_sample_size(ci, 0.01, 1.0 - alpha);
        // 50% warmup is the consensus default (Stan, PyMC, Hoffman & Gelman 2014).
        warmup_iterations_ = iterations_ / 2;

        prng_seed_ = 12345;
        initial_iterations_ = std::min(1000, std::max(100, d * 100));
    }

    // Sets default advanced (sampler-specific) simulation options (C#
    // `SetDefaultAdvancedSimulationOptions`, line 991).
    void set_default_advanced_simulation_options() {
        if (type_ == SamplerType::DEMCz || type_ == SamplerType::DEMCzs) {
            int d = model_.number_of_parameters();
            if (d <= 0) d = 1;

            jump_ = 2.38 / std::sqrt(2.0 * d);
            jump_threshold_ = 0.1;
            noise_ = 1e-12;
        }

        if (type_ == SamplerType::DEMCzs) {
            snooker_threshold_ = 0.1;
        }

        if (type_ == SamplerType::ARWMH) {
            int d = model_.number_of_parameters();
            if (d <= 0) d = 1;

            scale_ = 2.38 * 2.38 / d;
            beta_ = 0.05;
        }

        if (type_ == SamplerType::NUTS) {
            max_tree_depth_ = 10;
        }
    }

    // Validates the current knob configuration (C# `Validate`, line 1041). Returns
    // (is_valid, messages) -- messages include both warnings (do not affect is_valid) and
    // errors (each sets is_valid = false), transcribed in the same order as the C# source.
    std::pair<bool, std::vector<std::string>> validate() const {
        std::vector<std::string> messages;
        bool is_valid = true;

        // Warnings.
        if (!is_estimated_) {
            messages.emplace_back("Warning: The model has not been estimated.");
        }
        if (warmup_iterations_ > static_cast<int>(0.5 * iterations_)) {
            messages.emplace_back(
                "Warning: The number of warmup iterations exceeds half of the total iterations.");
        }
        if (iterations_ < 1000) {
            messages.emplace_back(
                "Warning: The number of iterations is below 1,000, which may reduce the accuracy of "
                "the posterior distribution.");
        }
        if (output_length_ < 1000) {
            messages.emplace_back(
                "Warning: The output length is below 1,000, which may reduce the accuracy of the "
                "posterior distribution.");
        }

        // Errors.
        if (number_of_chains_ < 4 || number_of_chains_ > 20) {
            messages.emplace_back("Error: The number of Markov chains must be between 4 and 20.");
            is_valid = false;
        }
        if (thinning_interval_ < 1 || thinning_interval_ > 100) {
            messages.emplace_back("Error: The thinning interval must be between 1 and 100.");
            is_valid = false;
        }
        if (warmup_iterations_ < 50 || warmup_iterations_ > 100000) {
            messages.emplace_back("Error: The number of warmup iterations must be between 50 and 100,000.");
            is_valid = false;
        }
        if (iterations_ < 100 || iterations_ > 1000000) {
            messages.emplace_back("Error: The number of iterations must be between 100 and 1,000,000.");
            is_valid = false;
        }
        if (prng_seed_ < 0) {
            messages.emplace_back("Error: The PRNG seed cannot be negative.");
            is_valid = false;
        }
        if (initial_iterations_ < number_of_chains_ || initial_iterations_ > 1000) {
            messages.emplace_back(
                "Error: The initial iterations must be at least equal to the number of chains and no "
                "more than 1,000.");
            is_valid = false;
        }
        if (credible_interval_width_ <= 0.0 || credible_interval_width_ >= 1.0) {
            messages.emplace_back(
                "Error: The credible interval width must be greater than 0 and less than 1 (for "
                "example, 0.90 for a 90 percent interval).");
            is_valid = false;
        }
        if (output_length_ < 100 || output_length_ > 1000000) {
            messages.emplace_back("Error: The output length must be between 100 and 1,000,000.");
            is_valid = false;
        }

        // DEMCz / DEMCzs sampler validation.
        if (type_ == SamplerType::DEMCz || type_ == SamplerType::DEMCzs) {
            if (jump_ <= 0.0 || jump_ >= 2.0) {
                messages.emplace_back("Error: The jump parameter must be greater than 0 and less than 2.");
                is_valid = false;
            }
            if (jump_threshold_ < 0.0 || jump_threshold_ > 1.0) {
                messages.emplace_back("Error: The jump threshold must be between 0 and 1.");
                is_valid = false;
            }
            if (noise_ < 0.0 || noise_ > 0.1) {
                messages.emplace_back("Error: The noise parameter should be small and between 0 and 0.1.");
                is_valid = false;
            }
        }

        // DEMCzs sampler validation.
        if (type_ == SamplerType::DEMCzs) {
            if (snooker_threshold_ < 0.0 || snooker_threshold_ > 0.5) {
                messages.emplace_back("Error: The snooker threshold must be between 0 and 0.5.");
                is_valid = false;
            }
        }

        // ARWMH sampler validation.
        if (type_ == SamplerType::ARWMH) {
            if (scale_ <= 0.0) {
                messages.emplace_back("Error: The scale parameter must be greater than 0.");
                is_valid = false;
            }
            if (beta_ < 0.0 || beta_ > 1.0) {
                messages.emplace_back("Error: The beta parameter must be between 0 and 1.");
                is_valid = false;
            }
        }

        // NUTS sampler validation.
        if (type_ == SamplerType::NUTS) {
            if (max_tree_depth_ < 1 || max_tree_depth_ > 15) {
                messages.emplace_back("Error: The maximum tree depth must be between 1 and 15.");
                is_valid = false;
            }
        }

        return {is_valid, messages};
    }

    // Builds the MCMC sampler from the model's cloned priors + posterior log-likelihood, then
    // applies every knob (C# `SetUpSampler`, line 1170). Always rebuilds `sampler_` from
    // scratch (see scope decision 2) using the CURRENT values of every knob field.
    void set_up_sampler() {
        if (model_.number_of_parameters() == 0) {
            sampler_.reset();
            return;
        }

        // Clone each parameter's prior distribution (C# `Model.Parameters.Select(x =>
        // (IUnivariateDistribution)x.PriorDistribution.Clone())`).
        std::vector<std::shared_ptr<bestfit::numerics::distributions::UnivariateDistributionBase>> priors;
        priors.reserve(static_cast<std::size_t>(model_.number_of_parameters()));
        for (auto& parameter : model_.parameters()) {
            priors.push_back(std::shared_ptr<bestfit::numerics::distributions::UnivariateDistributionBase>(
                parameter.prior_distribution().clone()));
        }

        // The posterior (data + prior) log-likelihood closure (C# `x =>
        // Model.LogLikelihood(x)`).
        bestfit::numerics::sampling::mcmc::LogLikelihood posterior = [this](const std::vector<double>& p) {
            return model_.log_likelihood(p);
        };

        switch (type_) {
            case SamplerType::DEMCz: {
                auto sampler =
                    std::make_unique<bestfit::numerics::sampling::mcmc::DEMCz>(priors, posterior);
                sampler->jump = jump_;
                sampler->jump_threshold = jump_threshold_;
                sampler->set_noise(noise_);
                sampler_ = std::move(sampler);
                break;
            }
            case SamplerType::DEMCzs: {
                auto sampler =
                    std::make_unique<bestfit::numerics::sampling::mcmc::DEMCzs>(priors, posterior);
                sampler->jump = jump_;
                sampler->jump_threshold = jump_threshold_;
                sampler->snooker_threshold = snooker_threshold_;
                sampler->set_noise(noise_);
                sampler_ = std::move(sampler);
                break;
            }
            case SamplerType::ARWMH: {
                auto sampler =
                    std::make_unique<bestfit::numerics::sampling::mcmc::ARWMH>(priors, posterior);
                sampler->scale = scale_;
                sampler->beta = beta_;
                sampler_ = std::move(sampler);
                break;
            }
            case SamplerType::NUTS: {
                // `NUTS`'s ctor takes an optional mass vector + step size ahead of
                // `max_tree_depth` (see nuts.hpp); pass the C# defaults for both (no BestFit
                // knob customizes them) and this analysis's `max_tree_depth_` knob.
                sampler_ = std::make_unique<bestfit::numerics::sampling::mcmc::NUTS>(
                    priors, posterior, std::nullopt, 0.1, max_tree_depth_);
                break;
            }
        }

        // Apply the general knobs (C# assigns these in this exact order: NumberOfChains,
        // ThinningInterval, WarmupIterations, Iterations, PRNGSeed, InitialIterations,
        // OutputLength).
        sampler_->set_number_of_chains(number_of_chains_);
        sampler_->set_thinning_interval(thinning_interval_);
        sampler_->set_warmup_iterations(warmup_iterations_);
        sampler_->set_iterations(iterations_);
        sampler_->set_prng_seed(prng_seed_);
        sampler_->set_initial_iterations(initial_iterations_);
        sampler_->output_length = output_length_;
    }

    // Runs the MCMC simulation synchronously: validates, (re)builds the sampler, samples,
    // then post-processes into `MCMCResults` (the compute core of C# `RunAsync`, line 1234 --
    // async/progress/cancellation machinery dropped, see scope decision 1). Throws
    // `std::invalid_argument` if `validate()` reports invalid (C# throws
    // `InvalidOperationException` from the same check). Returns `false` (without throwing) if
    // the model has no parameters, mirroring C#'s early return when `SetUpSampler` leaves
    // `Sampler == null`.
    bool estimate() {
        auto [valid, messages] = validate();
        (void)messages;
        if (!valid) {
            throw std::invalid_argument(
                "Bayesian Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        set_up_sampler();
        if (!sampler_) return false;

        is_estimated_ = false;
        results_.reset();

        sampler_->sample();
        results_.emplace(*sampler_, 1.0 - credible_interval_width_);
        is_estimated_ = true;
        return true;
    }

    // --- Gated (Diagnostics layer deferred past Phase 4; see scope decision 5) -------------

    [[noreturn]] void compute_influence_diagnostics() const {
        throw std::logic_error(
            "BayesianAnalysis::compute_influence_diagnostics: Diagnostics layer is deferred past "
            "Phase 4 (see .claude/PLAN.md); not yet ported.");
    }

    [[noreturn]] void compute_prior_influence_diagnostics(int thin_every = 10) const {
        (void)thin_every;
        throw std::logic_error(
            "BayesianAnalysis::compute_prior_influence_diagnostics: Diagnostics layer is deferred "
            "past Phase 4 (see .claude/PLAN.md); not yet ported.");
    }

    [[noreturn]] void compute_leverage_diagnostics() const {
        throw std::logic_error(
            "BayesianAnalysis::compute_leverage_diagnostics: Diagnostics layer is deferred past "
            "Phase 4 (see .claude/PLAN.md); not yet ported.");
    }

   private:
    bestfit::models::ModelBase& model_;
    SamplerType type_ = SamplerType::DEMCzs;

    // General settings.
    int number_of_chains_ = 4;
    int thinning_interval_ = 20;
    int warmup_iterations_ = 1500;
    int iterations_ = 3000;
    int prng_seed_ = 12345;
    int initial_iterations_ = 300;
    bool use_simulation_defaults_ = true;

    // Advanced settings.
    bool use_advanced_simulation_defaults_ = true;

    // DEMCz / DEMCzs.
    double jump_ = 1.0;
    double jump_threshold_ = 0.1;
    double snooker_threshold_ = 0.1;
    double noise_ = 1e-12;

    // ARWMH.
    double scale_ = 2.38 * 2.38;
    double beta_ = 0.05;

    // NUTS.
    int max_tree_depth_ = 10;

    // Output settings.
    double credible_interval_width_ = 0.9;
    int output_length_ = 10000;
    PointEstimateType point_estimator_ = PointEstimateType::PosteriorMean;

    bool is_estimated_ = false;

    std::unique_ptr<MCMCSampler> sampler_;
    std::optional<MCMCResults> results_;
};

}  // namespace bestfit::estimation
