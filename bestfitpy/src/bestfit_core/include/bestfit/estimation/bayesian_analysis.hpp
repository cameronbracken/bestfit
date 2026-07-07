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
//   4. TASK T10 ADDITIONS (this port, extending the T9 slice above): `compute_dic()` /
//      `compute_waic()` / `compute_psis_loo()` (C# `ComputeDIC`/`ComputeWAIC`/
//      `ComputePSISLOO`, ~1384-1801) populate `DIC`/`WAIC`/`WAIC_pD`/`LOOIC`/`LOO_pD` (+
//      `LOOIC_SE`/`ParetoK` for fidelity with the C# method, though the brief only requires
//      the first five); `clear_results()` (C# `ClearResults`, 1364) resets them to NaN;
//      `point_estimate()` is a bestfit ADDITION centralizing the `PointEstimator ==
//      PosteriorMean ? Results.PosteriorMean : Results.MAP` ternary that ~15 consumer
//      Analysis classes (ARAnalysis.cs:444, UnivariateAnalysis.cs:620, etc.) each repeat
//      inline in C# -- BayesianAnalysis.cs itself never centralizes it, so there is no
//      single C# method this ports; `get_posterior_covariance_matrix()` /
//      `get_posterior_correlation_matrix()` (C# 2047/2071) reuse the already-ported
//      `bestfit::numerics::data::RunningCovarianceMatrix`. `estimate()` now calls the three
//      compute_* methods after building `Results`, mirroring `RunAsync`'s post-await block
//      (C# 1301-1304). See each method's doc comment below for numerics-fidelity notes.
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
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bestfit/diagnostics/influence_diagnostics.hpp"
#include "bestfit/diagnostics/leverage_diagnostics.hpp"
#include "bestfit/diagnostics/prior_influence_diagnostics.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/numerics/data/running_covariance_matrix.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/generalized_pareto.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
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
// The point-estimate COMPUTATION that consumes it is `point_estimate()` below (T10, see scope
// decision 4).
enum class PointEstimateType { PosteriorMean, PosteriorMode };

class BayesianAnalysis {
   public:
    using MCMCSampler = bestfit::numerics::sampling::mcmc::MCMCSampler;
    using MCMCResults = bestfit::numerics::sampling::mcmc::MCMCResults;
    using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;

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

    // Information criteria (Task T10; C# `DIC`/`WAIC`/`WAIC_pD`/`LOOIC`/`LOO_pD`/`LOOIC_SE`
    // properties, C# ~404-460). NaN before `estimate()` succeeds or after `clear_results()`.
    double dic() const { return dic_; }
    double waic() const { return waic_; }
    double waic_pd() const { return waic_pd_; }
    double looic() const { return looic_; }
    double loo_pd() const { return loo_pd_; }
    double looic_se() const { return looic_se_; }

    // Per-observation Pareto shape-parameter diagnostics from the most recent
    // `compute_psis_loo()` (C# `ParetoK`, C# 479). Empty before estimation / after
    // `clear_results()`.
    const std::vector<double>& pareto_k() const { return pareto_k_; }

    // Returns the posterior point estimate selected by `point_estimator()` (bestfit
    // ADDITION -- see header comment point 4: centralizes a ternary that ~15 C# consumer
    // Analysis classes each repeat inline). Throws `std::logic_error` if `estimate()` has
    // not yet produced `Results` (every C# call site guards on `IsEstimated`/`Results !=
    // null` before reading `PosteriorMean`/`MAP`; there is no null `ParameterSet` to fall
    // back to here).
    const ParameterSet& point_estimate() const {
        if (!results_) {
            throw std::logic_error(
                "BayesianAnalysis::point_estimate: estimate() has not produced Results yet.");
        }
        return point_estimator_ == PointEstimateType::PosteriorMean ? results_->posterior_mean
                                                                     : results_->map;
    }

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
        // Model.LogLikelihood(x)`). DOCUMENTED DEVIATION (M14): the MCMC sampler's
        // LogLikelihood type is const-ref (Phase 3, oracle-locked), so the model sees a mutable
        // COPY here -- MixtureModel's C# weight-normalization write-back into the SAMPLER'S
        // proposal/chain arrays is NOT ported. No fixture runs a mixture model through
        // BayesianAnalysis; every other model leaves the vector untouched, so the copy is
        // behaviorally identical for all wired paths. Flagged in docs for the follow-up.
        bestfit::numerics::sampling::mcmc::LogLikelihood posterior = [this](const std::vector<double>& p) {
            std::vector<double> point = p;
            return model_.log_likelihood(point);
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
        // Post-Results information criteria (C# `RunAsync`'s post-await block, C#
        // 1301-1304: `Results = capturedResults; ComputeDIC(); ComputeWAIC();
        // ComputePSISLOO();`, before `IsEstimated = true`). No progress/async/parallel
        // machinery to mirror here (see scope decision 1).
        compute_dic();
        compute_waic();
        compute_psis_loo();
        is_estimated_ = true;
        return true;
    }

    // Injects externally-produced MCMC results (C# `SetCustomMCMCResults(MCMCResults, bool)`,
    // C# 1819-1829). ADDITIVE A5 hook (the T9/T10 slice deliberately skipped the XML-restore
    // overloads -- see header comment point 6): stores `results`, optionally recomputes
    // DIC/WAIC/LOOIC (skipped when the caller already has them, e.g. XML restore), and flips
    // `is_estimated_` true -- exactly the C# body minus the dropped PropertyChanged raise. This
    // lets a caller (UnivariateAnalysis's structural tests, mirroring the C# reprocess tests)
    // flip the analysis into the estimated state without running a chain. The three-arg
    // `parameterNames` overload (C# 1838) is not ported (`_parameterNames` is not in this
    // slice -- header point 6).
    void set_custom_mcmc_results(MCMCResults results, bool skip_information_criteria = false) {
        results_ = std::move(results);
        if (!skip_information_criteria) {
            compute_dic();
            compute_waic();
            compute_psis_loo();
        }
        is_estimated_ = true;
    }

    // Clears the analysis results and information criteria, then rebuilds the sampler from
    // the current knob values (C# `ClearResults`, C# 1364). `ElapsedTime`/`_parameterNames`
    // are not ported (see header comment points 1/6 -- no ElapsedTime/ParameterNames surface
    // in this port).
    void clear_results() {
        results_.reset();
        dic_ = kNaN;
        waic_ = kNaN;
        waic_pd_ = kNaN;
        looic_ = kNaN;
        loo_pd_ = kNaN;
        looic_se_ = kNaN;
        pareto_k_.clear();
        elpd_loo_.clear();
        is_estimated_ = false;
        set_up_sampler();
    }

    // --- Information criteria (Task T10; C# ~1384-1801) -------------------------------------

    // Deviance Information Criterion (C# `ComputeDIC`, C# 1384-1404): DIC = 2*dicHat -
    // dicMu, where dicHat is the mean of -2*DataLogLikelihood over every posterior sample
    // and dicMu is -2*DataLogLikelihood at the posterior mean. C# accumulates dicHat via a
    // `Parallel.For` reduction (`Tools.ParallelAdd`); this port sums serially in sample
    // order 0..N-1 (see scope decision 1 -- no ParallelizeChains/parallel machinery
    // anywhere in this port), which is a valid floating-point reduction order but not
    // guaranteed bit-identical to whatever order the CLR's parallel reduction produces --
    // an execution-order difference, not a formula difference (see the T10 report for the
    // full fidelity discussion).
    void compute_dic() {
        if (!results_ || results_->output.empty()) {
            dic_ = kNaN;
            return;
        }

        std::size_t n = results_->output.size();
        double dic_hat = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            dic_hat += -2.0 * model_.data_log_likelihood(results_->output[j].values);
        }
        dic_hat /= static_cast<double>(n);

        double dic_mu = -2.0 * model_.data_log_likelihood(results_->posterior_mean.values);
        dic_ = 2.0 * dic_hat - dic_mu;
    }

    // Watanabe-Akaike Information Criterion (C# `ComputeWAIC`, C# 1430-1523): WAIC =
    // -2*lppd + 2*p_WAIC, using the pointwise log-likelihood matrix
    // logLik[observation][sample]. `lppd_i` is computed via the log-sum-exp trick (shift by
    // the per-observation max before exponentiating, matching C# exactly, including its
    // `NegativeInfinity`-not-`MinValue` choice for an all -inf row -- `kNegInf` here is
    // `-std::numeric_limits<double>::infinity()`, C#'s `double.NegativeInfinity`
    // equivalent). `p_WAIC_i` uses the unbiased sample-variance estimator (divisor S-1,
    // Vehtari/Gelman/Gabry 2017 Eq. 12), clamped to >= 0 for floating-point safety, exactly
    // matching the C# formula. Reference: Watanabe (2010), JMLR 11, 3571-3594.
    void compute_waic() {
        if (!results_ || results_->output.empty()) {
            waic_ = kNaN;
            waic_pd_ = kNaN;
            return;
        }

        std::size_t s_count = results_->output.size();
        std::vector<double> first_pointwise =
            model_.pointwise_data_log_likelihood(results_->output[0].values);
        std::size_t n = first_pointwise.size();
        if (n == 0) {
            waic_ = kNaN;
            waic_pd_ = kNaN;
            return;
        }

        // pointwise_log_lik[i][s]: log-likelihood of observation i under posterior sample s.
        std::vector<std::vector<double>> pointwise_log_lik(n, std::vector<double>(s_count));
        for (std::size_t i = 0; i < n; ++i) pointwise_log_lik[i][0] = first_pointwise[i];
        for (std::size_t s = 1; s < s_count; ++s) {
            std::vector<double> log_liks =
                model_.pointwise_data_log_likelihood(results_->output[s].values);
            for (std::size_t i = 0; i < n; ++i) pointwise_log_lik[i][s] = log_liks[i];
        }

        double total_lppd = 0.0;
        double total_p_waic = 0.0;
        const double kNegInf = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < n; ++i) {
            double max_log_lik = kNegInf;
            double sum_log_lik = 0.0;
            double sum_log_lik_sq = 0.0;
            for (std::size_t s = 0; s < s_count; ++s) {
                double ll = pointwise_log_lik[i][s];
                sum_log_lik += ll;
                sum_log_lik_sq += ll * ll;
                if (ll > max_log_lik) max_log_lik = ll;
            }

            double sum_exp = 0.0;
            for (std::size_t s = 0; s < s_count; ++s) {
                sum_exp += std::exp(pointwise_log_lik[i][s] - max_log_lik);
            }
            double lppd_i = max_log_lik + std::log(sum_exp) - std::log(static_cast<double>(s_count));

            double mean_log_lik = sum_log_lik / static_cast<double>(s_count);
            double p_waic_i = s_count > 1 ? (sum_log_lik_sq - static_cast<double>(s_count) * mean_log_lik *
                                                                    mean_log_lik) /
                                                 static_cast<double>(s_count - 1)
                                           : 0.0;
            if (p_waic_i < 0.0) p_waic_i = 0.0;

            total_lppd += lppd_i;
            total_p_waic += p_waic_i;
        }

        waic_pd_ = total_p_waic;
        waic_ = -2.0 * total_lppd + 2.0 * total_p_waic;
    }

    // Leave-One-Out Information Criterion via Pareto-Smoothed Importance Sampling (C#
    // `ComputePSISLOO`, C# 1541-1679, + `ParetoSmoothWeights`, C# 1687-1801). For each
    // observation: log importance weights are `-logLik`, Pareto-smoothed in the tail (the
    // largest M = clamp(min(S/5, 3*sqrt(S)), 3, S-1) weights) by fitting a Generalized
    // Pareto distribution via MLE (reusing the already-ported
    // `bestfit::numerics::distributions::GeneralizedPareto::mle()`, which returns
    // {xi_fixed, alpha, kappa} exactly like C#'s `GeneralizedPareto().MLE(tailWeights)` --
    // both use Hosking's parameterization, so the same sign flip `k = -kappa` converts to
    // the PSIS shape convention), falling back to method-of-moments if the MLE throws
    // (mirrored via try/catch, matching C#'s `catch (Exception)` fallback). elpd_loo_i and
    // lppd_i both use the log-sum-exp trick. LOOIC = -2*sum(elpd_loo); LOO_pD =
    // sum(lppd) - sum(elpd_loo); LOOIC_SE = 2*sqrt(n*Var[elpd_loo]) (SE of a sum scales by
    // sqrt(n) per C#'s comment). `ParetoK`/`LOOIC_SE` are ported alongside `LOOIC`/`LOO_pD`
    // for fidelity with the C# method (the brief only requires the latter two, but the
    // whole method sets all four together in C# and `ParetoK` is needed by the still-gated
    // `compute_influence_diagnostics`). Reference: Vehtari, Gelman & Gabry (2017),
    // Statistics and Computing 27(5), 1413-1432.
    void compute_psis_loo() {
        if (!results_ || results_->output.empty()) {
            looic_ = kNaN;
            loo_pd_ = kNaN;
            looic_se_ = kNaN;
            pareto_k_.clear();
            elpd_loo_.clear();
            return;
        }

        std::size_t s_count = results_->output.size();
        std::vector<double> first_pointwise =
            model_.pointwise_data_log_likelihood(results_->output[0].values);
        std::size_t n = first_pointwise.size();
        if (n == 0) {
            looic_ = kNaN;
            loo_pd_ = kNaN;
            looic_se_ = kNaN;
            pareto_k_.clear();
            elpd_loo_.clear();
            return;
        }

        std::vector<std::vector<double>> pointwise_log_lik(n, std::vector<double>(s_count));
        for (std::size_t i = 0; i < n; ++i) pointwise_log_lik[i][0] = first_pointwise[i];
        for (std::size_t s = 1; s < s_count; ++s) {
            std::vector<double> log_liks =
                model_.pointwise_data_log_likelihood(results_->output[s].values);
            for (std::size_t i = 0; i < n; ++i) pointwise_log_lik[i][s] = log_liks[i];
        }

        pareto_k_.assign(n, 0.0);
        std::vector<double> elpd_loo(n);
        std::vector<double> lppd(n);

        int s_int = static_cast<int>(s_count);
        int m = static_cast<int>(std::min(s_int / 5.0, 3.0 * std::sqrt(static_cast<double>(s_int))));
        m = std::max(m, 3);
        m = std::min(m, s_int - 1);

        for (std::size_t i = 0; i < n; ++i) {
            const std::vector<double>& log_liks = pointwise_log_lik[i];

            std::vector<double> log_weights(s_count);
            for (std::size_t s = 0; s < s_count; ++s) log_weights[s] = -log_liks[s];

            double max_log_weight = log_weights[0];
            for (std::size_t s = 1; s < s_count; ++s) {
                if (log_weights[s] > max_log_weight) max_log_weight = log_weights[s];
            }
            std::vector<double> shifted_log_weights(s_count);
            for (std::size_t s = 0; s < s_count; ++s) shifted_log_weights[s] = log_weights[s] - max_log_weight;

            double pareto_k_i = pareto_smooth_weights(shifted_log_weights, m);
            pareto_k_[i] = pareto_k_i;

            std::vector<double> weights(s_count);
            double sum_weights = 0.0;
            for (std::size_t s = 0; s < s_count; ++s) {
                weights[s] = std::exp(shifted_log_weights[s]);
                sum_weights += weights[s];
            }
            for (std::size_t s = 0; s < s_count; ++s) weights[s] /= sum_weights;

            double max_ll = log_liks[0];
            for (std::size_t s = 1; s < s_count; ++s) {
                if (log_liks[s] > max_ll) max_ll = log_liks[s];
            }

            double sum_weighted_exp = 0.0;
            for (std::size_t s = 0; s < s_count; ++s) sum_weighted_exp += weights[s] * std::exp(log_liks[s] - max_ll);
            elpd_loo[i] = sum_weighted_exp > 0.0 ? max_ll + std::log(sum_weighted_exp)
                                                 : -std::numeric_limits<double>::infinity();

            double sum_exp = 0.0;
            for (std::size_t s = 0; s < s_count; ++s) sum_exp += std::exp(log_liks[s] - max_ll);
            lppd[i] = max_ll + std::log(sum_exp) - std::log(static_cast<double>(s_count));
        }

        // Retain the pointwise elpd_loo the same way pareto_k_ is retained (D4 additive member),
        // so compute_influence_diagnostics can consume it without a second PSIS pass. The C#
        // ComputeInfluenceDiagnostics recomputes this via ComputePointwiseElpdLoo (identical
        // formula); retaining it here is a DRY equivalent. See the D4 report.
        elpd_loo_ = elpd_loo;

        double total_elpd_loo = std::accumulate(elpd_loo.begin(), elpd_loo.end(), 0.0);
        double total_lppd = std::accumulate(lppd.begin(), lppd.end(), 0.0);

        double mean_elpd_loo = total_elpd_loo / static_cast<double>(n);
        double variance = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double diff = elpd_loo[i] - mean_elpd_loo;
            variance += diff * diff;
        }
        variance /= static_cast<double>(n - 1);
        double se_elpd_loo = std::sqrt(static_cast<double>(n) * variance);

        looic_ = -2.0 * total_elpd_loo;
        loo_pd_ = total_lppd - total_elpd_loo;
        looic_se_ = 2.0 * se_elpd_loo;
    }

    // --- Posterior summaries (Task T10; C# 2047/2071) --------------------------------------

    // Posterior sample covariance matrix (C# `GetPosteriorCovarianceMatrix`, C# 2047-2058),
    // computed by pushing every posterior output sample through the already-ported
    // `bestfit::numerics::data::RunningCovarianceMatrix` (Welford's online algorithm) and
    // reading `sample_covariance()` (Bessel's correction, N-1 denominator), exactly
    // matching C#'s `RunningCovarianceMatrix` + `.SampleCovariance`. `std::nullopt` when
    // not yet estimated or fewer than 2 output samples, mirroring the C# null-return guard.
    std::optional<bestfit::numerics::math::linalg::Matrix> get_posterior_covariance_matrix() const {
        if (!is_estimated_ || !results_ || results_->output.size() < 2) return std::nullopt;

        bestfit::numerics::data::RunningCovarianceMatrix rcm(model_.number_of_parameters());
        for (const auto& ps : results_->output) rcm.push(ps.values);
        return rcm.sample_covariance();
    }

    // Posterior sample correlation matrix (C# `GetPosteriorCorrelationMatrix`, C#
    // 2071-2082): `Corr[i,j] = Cov[i,j] / (SD[i] * SD[j])`, read from
    // `RunningCovarianceMatrix::sample_correlation()`. Same null-guard as
    // `get_posterior_covariance_matrix()`.
    std::optional<bestfit::numerics::math::linalg::Matrix> get_posterior_correlation_matrix() const {
        if (!is_estimated_ || !results_ || results_->output.size() < 2) return std::nullopt;

        bestfit::numerics::data::RunningCovarianceMatrix rcm(model_.number_of_parameters());
        for (const auto& ps : results_->output) rcm.push(ps.values);
        return rcm.sample_correlation();
    }

    // --- Influence diagnostics (D4 un-stub -- the Diagnostics layer is now ported) ----------

    // Computes PSIS-LOO influence diagnostics (C# BayesianAnalysis 1870). Consumes the
    // already-populated `pareto_k_` and the retained `elpd_loo_` (both set by the last
    // `compute_psis_loo()` inside `estimate()`; the C# recomputes elpd_loo via
    // ComputePointwiseElpdLoo -- an identical formula -- but this port retains it, see the D4
    // report). Forwards the model's per-observation DataComponents for labels when available
    // (silent no-throw guard, mirroring the C# `catch` fallback to no metadata). Empty pareto_k_
    // yields an empty InfluenceDiagnostics, matching the C#. The C# InvalidOperationException
    // maps to std::invalid_argument (this file's convention).
    bestfit::diagnostics::InfluenceDiagnostics compute_influence_diagnostics() const {
        if (!is_estimated_ || !results_)
            throw std::invalid_argument(
                "Estimation must be completed before computing influence diagnostics.");

        if (pareto_k_.empty()) return bestfit::diagnostics::InfluenceDiagnostics();

        std::optional<std::vector<bestfit::models::DataComponent>> data_components;
        try {
            data_components =
                model_.pointwise_data_log_likelihood_components(results_->output[0].values);
        } catch (const std::exception&) {
            // C# Debug.WriteLine then falls back to no metadata; silent guard here.
        }
        return bestfit::diagnostics::InfluenceDiagnostics(pareto_k_, elpd_loo_,
                                                          std::move(data_components));
    }

    // Computes prior influence diagnostics (C# BayesianAnalysis 1927): delegates to the
    // PriorInfluenceDiagnostics(IModel, MCMCResults, thinEvery) constructor at the estimator's
    // MCMC results. The C# InvalidOperationException maps to std::invalid_argument.
    bestfit::diagnostics::PriorInfluenceDiagnostics compute_prior_influence_diagnostics(
        int thin_every = 10) const {
        if (!is_estimated_ || !results_)
            throw std::invalid_argument(
                "Estimation must be completed before computing prior influence diagnostics.");
        return bestfit::diagnostics::PriorInfluenceDiagnostics(model_, *results_, thin_every);
    }

    // Computes leverage diagnostics at the MAP estimate using the Hessian of the full
    // posterior (C# BayesianAnalysis 1955). Delegates to the LeverageDiagnostics(IModel,
    // double[]) constructor at the MCMC MAP values (Results.MAP.Values); no optimization step
    // is performed (D3 un-stub; the Diagnostics layer is now ported). The C#
    // InvalidOperationException maps to std::invalid_argument (this file's convention).
    bestfit::diagnostics::LeverageDiagnostics compute_leverage_diagnostics() const {
        if (!is_estimated_ || !results_)
            throw std::invalid_argument(
                "Estimation must be completed before computing leverage diagnostics.");
        return bestfit::diagnostics::LeverageDiagnostics(model_, results_->map.values);
    }

   private:
    // Applies Pareto smoothing to the tail of log importance weights (C#
    // `ParetoSmoothWeights`, C# 1687-1801). `log_weights` is modified in place: the M
    // largest entries (the tail) are replaced by the expected order statistics of the
    // fitted Generalized Pareto distribution, when the fitted shape k lands in (-0.5, 1).
    // Returns the estimated PSIS shape parameter k.
    static double pareto_smooth_weights(std::vector<double>& log_weights, int m) {
        int s_count = static_cast<int>(log_weights.size());

        std::vector<int> indices(static_cast<std::size_t>(s_count));
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&log_weights](int a, int b) { return log_weights[static_cast<std::size_t>(a)] >
                                                         log_weights[static_cast<std::size_t>(b)]; });

        std::vector<double> tail_log_weights(static_cast<std::size_t>(m));
        for (int j = 0; j < m; ++j) {
            tail_log_weights[static_cast<std::size_t>(j)] =
                log_weights[static_cast<std::size_t>(indices[static_cast<std::size_t>(j)])];
        }

        double cutoff = tail_log_weights[static_cast<std::size_t>(m - 1)];
        for (int j = 0; j < m; ++j) tail_log_weights[static_cast<std::size_t>(j)] -= cutoff;

        std::vector<double> tail_weights(static_cast<std::size_t>(m));
        for (int j = 0; j < m; ++j) {
            tail_weights[static_cast<std::size_t>(j)] = std::exp(tail_log_weights[static_cast<std::size_t>(j)]);
        }

        // Fit the GPD by MLE (Numerics/bestfit's Hosking parameterization: Kappa has the
        // OPPOSITE sign of the PSIS k convention -- flip on the way out), falling back to
        // method-of-moments if the MLE throws (mirrors C#'s try/catch fallback).
        double k;
        double sigma;
        try {
            bestfit::numerics::distributions::GeneralizedPareto gpd_mle;
            std::vector<double> mle_params = gpd_mle.mle(tail_weights);
            sigma = mle_params[1];
            k = -mle_params[2];
        } catch (const std::exception&) {
            double mean = 0.0;
            for (int j = 0; j < m; ++j) mean += tail_weights[static_cast<std::size_t>(j)];
            mean /= static_cast<double>(m);
            double variance = 0.0;
            for (int j = 0; j < m; ++j) {
                double diff = tail_weights[static_cast<std::size_t>(j)] - mean;
                variance += diff * diff;
            }
            variance /= static_cast<double>(m - 1);
            if (variance > 0.0 && mean > 0.0) {
                double cv2 = variance / (mean * mean);
                k = 0.5 * (cv2 - 1.0) / (cv2 + 1.0);
                sigma = mean * (1.0 - k);
                if (sigma <= 0.0) sigma = mean;
            } else {
                k = 0.0;
                sigma = mean > 0.0 ? mean : 1.0;
            }
        }

        k = std::max(-0.5, std::min(k, 1.5));
        if (sigma <= 0.0) sigma = std::numeric_limits<double>::denorm_min();

        if (k < 1.0 && k > -0.5) {
            for (int j = 0; j < m; ++j) {
                double p = (j + 0.5) / static_cast<double>(m);
                double quantile;
                if (std::fabs(k) < 1e-8) {
                    quantile = -sigma * std::log(1.0 - p);
                } else {
                    quantile = sigma / k * (std::pow(1.0 - p, -k) - 1.0);
                }
                quantile = std::max(0.0, quantile);

                double smoothed_log_weight =
                    quantile > 0.0 ? std::log(quantile) + cutoff : cutoff - 300.0 * std::log(10.0);
                log_weights[static_cast<std::size_t>(indices[static_cast<std::size_t>(j)])] = smoothed_log_weight;
            }
        }

        return k;
    }

    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

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

    // Information criteria (Task T10; C# fields backing the properties at C# ~404-479).
    // NaN/empty before a successful `estimate()`, matching C#'s field initializers -- C#'s
    // `double` properties get 0.0 by default, but every read path (`DIC`/`WAIC`/etc.) is
    // only ever consulted after `ClearResults()` (which sets them to NaN) or a completed
    // `RunAsync`, so NaN is the correct "not yet computed" sentinel here too.
    double dic_ = kNaN;
    double waic_ = kNaN;
    double waic_pd_ = kNaN;
    double looic_ = kNaN;
    double loo_pd_ = kNaN;
    double looic_se_ = kNaN;
    std::vector<double> pareto_k_;
    // Pointwise elpd_loo retained alongside pareto_k_ (D4 additive member; see compute_psis_loo).
    std::vector<double> elpd_loo_;
};

}  // namespace bestfit::estimation
