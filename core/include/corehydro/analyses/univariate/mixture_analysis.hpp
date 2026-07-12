// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/MixtureAnalysis.cs @ fc28c0c
//
// Bayesian MCMC frequency-curve analysis over a MixtureModel (a finite mixture of 1-3 component
// distributions, optionally zero-inflated). A near-mechanical sibling of UnivariateAnalysis
// (A5): it drives a BayesianAnalysis (Phase 4) over the model, then assembles the Bayesian
// frequency curve + credible intervals into an UncertaintyAnalysisResults (A2) + point-estimate
// goodness-of-fit. The mixture model is stationary (IsNonstationary => false), so there is NO
// chronology / nonstationary path here.
//
// The DROPPED surface mirrors the UnivariateAnalysis precedent (GUI/threading/serialization,
// no numerical content):
//   * both XML constructors (C# 75-116) + `ToXElement` (727) -- XElement (de)serialization.
//   * the `*_PropertyChanged` handlers (C# 232/250/276) -- INotifyPropertyChanged /
//     INotifyCollectionChanged binding cascades.
//   * `CancellationTokenSource` + `CancelAnalysis` (530), `SafeProgressReporter`/`IProgress`,
//     the `_reprocessGate` semaphore, and the `AnalysisStarting`/`AnalysisCompleted` events.
//     `RaisePropertyChange` calls throughout.
//
// The C# `async Task RunAsync` ports to a synchronous `run()`; the `...Async` helpers port to
// synchronous methods (every `Parallel.For` becomes a serial loop -- independent writes).
//
// DEVIATIONS (documented):
//   1. EXCEEDANCE <-> NON-EXCEEDANCE FLIP (C# 668/602).
//   2. OWNERSHIP. The analysis OWNS the model via `std::unique_ptr` (ctor null-guard <=> C#
//      ArgumentNullException); `BayesianAnalysis` holds a `ModelBase&` into it, so the analysis
//      is non-copyable / non-movable.
//   3. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`.
//   4. `get_distribution` / `get_point_estimate_distribution` return a NON-owning raw pointer;
//      cloned distributions are OWNED by the analysis in `distribution_cache_`.
//   5. RUN ERROR REPORTING. With events dropped, `run()` lets exceptions propagate.
//
//   6. THE MANUAL EM-SEEDING PATH IS NOW WIRED (X2). The C# `RunAsync` (lines 360-486) does NOT
//      let `BayesianAnalysis.estimate()` auto-initialize the sampler. Instead it calls
//      `BayesianAnalysis.SetUpSampler()`, grabs `BayesianAnalysis.Sampler!`, sets
//      `sampler.Initialize = MCMCSampler.InitializationType.UserDefined`, runs
//      `MixtureModel.ExpectationMaximization(...)` for an initial (parameters, covariance), draws
//      `InitialIterations` proposals from a Dirichlet (weights) + MultivariateNormal (component
//      parameters) prior-predictive, and SEEDS the sampler by adding to `sampler.PopulationMatrix`
//      and `sampler.MarkovChains[i]`; on ANY exception during this block the C# falls back to
//      `sampler.Initialize = Randomize`. `run()` below reproduces this exactly, using the X2
//      additive seeding hooks: `MCMCSampler::seed_population(...)` / `seed_chain(...)` (which
//      survive `reset()`) and `BayesianAnalysis::estimate(/*set_up=*/false)` (mirroring the C#
//      `RunAsync(..., false)` that reuses the already-seeded sampler rather than rebuilding it).
//      On any throw in the seed block the sampler falls back to `Randomize` -- byte-identical to
//      the C#'s init-failure branch (and to the pre-X2 port's fallback-only behavior).
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/analyses/support/i_univariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/univariate_distribution/mixture_model.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/probability_ordinates.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/multivariate/dirichlet.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mcmc/base/mcmc_sampler.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::analyses {

class MixtureAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using MixtureModel = corehydro::models::MixtureModel;
    using UnivariateDistributionBase = corehydro::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using ProbabilityOrdinates = corehydro::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using PointEstimateType = corehydro::estimation::PointEstimateType;

    // C# ctor `MixtureAnalysis(MixtureModel)` (C# 55). The C# `?? throw ArgumentNullException`
    // maps to the null-guard here (deviation 2).
    explicit MixtureAnalysis(std::unique_ptr<MixtureModel> mixture_distribution)
        : mixture_distribution_(require_non_null(std::move(mixture_distribution))),
          bayesian_analysis_(*mixture_distribution_),
          probability_ordinates_() {}

    ~MixtureAnalysis() override = default;

    // Non-copyable / non-movable (deviation 2).
    MixtureAnalysis(const MixtureAnalysis&) = delete;
    MixtureAnalysis& operator=(const MixtureAnalysis&) = delete;
    MixtureAnalysis(MixtureAnalysis&&) = delete;
    MixtureAnalysis& operator=(MixtureAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `MixtureDistribution` (C# 137): the model being estimated.
    MixtureModel& mixture_distribution() { return *mixture_distribution_; }
    const MixtureModel& mixture_distribution() const { return *mixture_distribution_; }

    // C# `BayesianAnalysis` (C# 187). IBayesianAnalysis override.
    corehydro::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const corehydro::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ProbabilityOrdinates` (C# 162). IProbabilityOrdinates override.
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `AnalysisResults` (C# 218): the frequency-analysis results (null until estimated).
    // IBayesianAnalysis override (const pointer <=> nullable).
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 305).
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        distribution_cache_.clear();
        set_is_estimated(false);
    }

    // C# `ClearFrequencyAnalysisResults` (C# 318).
    void clear_frequency_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (C# 325), synchronous. Validate guard -> prepare input data
    // (ProcessThresholdSeries + ProcessQuantilePriors) -> set up + EM-seed the sampler ->
    // BayesianAnalysis.estimate(set_up=false) -> IF estimated: create frequency results ->
    // IsEstimated mirrors the inner fit. The EM-seed of the sampler (C# 366-486) is wired via
    // the X2 additive seeding hooks (see the file header, deviation 6).
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        // Prepare input data (C# 363-364).
        mixture_distribution_->data_frame().process_threshold_series();
        mixture_distribution_->process_quantile_priors();

        // Set up the sampler with EM initialization (C# 366-368).
        bayesian_analysis_.set_up_sampler();
        auto* sampler = bayesian_analysis_.sampler();
        if (sampler != nullptr) {
            using MCMCSampler = corehydro::numerics::sampling::mcmc::MCMCSampler;
            sampler->initialize = MCMCSampler::InitializationType::UserDefined;

            // EM-seed the sampler's population + chains (C# 372-483; Task.Run -> serial). On
            // ANY throw during initialization, fall back to Randomize (C# 478-482) -- the
            // C#'s own init-failure branch. The exception is swallowed (Debug.WriteLine ->
            // silent no-throw per the standing rule).
            try {
                seed_sampler_from_em(*sampler);
            } catch (...) {
                sampler->clear_seed();
                sampler->initialize = MCMCSampler::InitializationType::Randomize;
            }
        }

        // Run the Bayesian analysis WITHOUT rebuilding the seeded sampler (C# 484
        // `RunAsync(..., false)`).
        bayesian_analysis_.estimate(/*set_up=*/false);

        // Post-process.
        if (bayesian_analysis_.is_estimated()) {
            create_frequency_analysis_results();
        }

        // Mirror the inner fit's success state (C# 499).
        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // C# `GetDistribution(int)` (C# 537): clones the wrapped Mixture and applies the posterior
    // output sample. Null when unestimated. The clone is owned by the analysis (deviation 4).
    UnivariateDistributionBase* get_distribution(int index) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        std::unique_ptr<UnivariateDistributionBase> result =
            mixture_distribution_->mixture()->clone();
        result->set_parameters(
            bayesian_analysis_.results()->output[static_cast<std::size_t>(index)].values);
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // C# `GetPointEstimateDistribution()` (C# 551).
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_analysis_.point_estimator());
    }

    // C# `GetPointEstimateDistribution(PointEstimateType)` (C# 555).
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        const std::vector<double>& parms =
            point_estimator == PointEstimateType::PosteriorMean
                ? bayesian_analysis_.results()->posterior_mean.values
                : bayesian_analysis_.results()->map.values;

        std::unique_ptr<UnivariateDistributionBase> result =
            mixture_distribution_->mixture()->clone();
        result->set_parameters(parms);
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 641), synchronous.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        // Set model parameters to the MAP (C# 652).
        mixture_distribution_->set_parameter_values(results.map.values);

        // Build B sampled distributions, one per posterior output sample (C# 655-663).
        int b = bayesian_analysis_.output_length();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> owned;
        owned.reserve(static_cast<std::size_t>(b));
        for (int idx = 0; idx < b; ++idx) {
            std::unique_ptr<UnivariateDistributionBase> d = mixture_distribution_->mixture()->clone();
            d->set_parameters(results.output[static_cast<std::size_t>(idx)].values);
            owned.push_back(std::move(d));
        }

        std::vector<const UnivariateDistributionBase*> sampled;
        sampled.reserve(owned.size());
        for (const auto& u : owned) sampled.push_back(u.get());

        // Exceedance -> non-exceedance FLIP (deviation 1; C# 668 `p => 1.0 - p`).
        std::vector<double> probabilities;
        probabilities.reserve(probability_ordinates_.count());
        for (double p : probability_ordinates_) probabilities.push_back(1.0 - p);

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        analysis_results_.emplace(*mixture_distribution_->mixture(), sampled, probabilities, alpha);

        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 579), synchronous.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        MixtureModel& model = *mixture_distribution_;

        // Set the point estimator (C# 590-597).
        if (bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean) {
            model.set_parameter_values(results.posterior_mean.values);
        } else {
            model.set_parameter_values(results.map.values);
        }

        // Mode curve on the non-exceedance grid (C# 600-602; deviation 1).
        analysis_results_->mode_curve.assign(probability_ordinates_.count(), 0.0);
        for (std::size_t i = 0; i < probability_ordinates_.count(); ++i) {
            analysis_results_->mode_curve[i] =
                model.mixture()->inverse_cdf(1.0 - probability_ordinates_[i]);
        }

        // Information criteria at the MAP estimate (C# 605-612).
        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double log_l = model.log_likelihood(map_values);
        int k = model.number_of_parameters();
        int n = model.data_frame().total_record_length();
        double aic = GoodnessOfFit::aic(k, log_l);
        double bic = GoodnessOfFit::bic(n, k, log_l);
        double dic = bayesian_analysis_.dic();

        // RMSE over Exact + Uncertain + Interval data (C# 615-621).
        const corehydro::models::DataFrame& df = model.data_frame();
        std::vector<double> values = df.exact_series().values_to_list();
        {
            std::vector<double> u = df.uncertain_series().values_to_list();
            values.insert(values.end(), u.begin(), u.end());
            std::vector<double> iv = df.interval_series().values_to_list();
            values.insert(values.end(), iv.begin(), iv.end());
        }
        std::vector<double> probs;
        probs.reserve(values.size());
        for (std::size_t i = 0; i < df.exact_series().count(); ++i)
            probs.push_back(df.exact_series()[i].plotting_position_complement());
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i)
            probs.push_back(df.uncertain_series()[i].plotting_position_complement());
        for (std::size_t i = 0; i < df.interval_series().count(); ++i)
            probs.push_back(df.interval_series()[i].plotting_position_complement());
        double rmse = GoodnessOfFit::rmse(values, probs, *model.mixture());

        analysis_results_->aic = aic;
        analysis_results_->bic = bic;
        analysis_results_->dic = dic;
        analysis_results_->rmse = rmse;
    }

    // C# `Validate` (C# 677).
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        corehydro::models::ValidationResult dist_valid = mixture_distribution_->validate();
        if (!dist_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              dist_valid.validation_messages.begin(),
                                              dist_valid.validation_messages.end());
        }

        auto prob_valid = probability_ordinates_.validate();
        if (!prob_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              prob_valid.messages.begin(), prob_valid.messages.end());
        }

        auto bayes_valid = bayesian_analysis_.validate();
        if (!bayes_valid.first) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              bayes_valid.second.begin(), bayes_valid.second.end());
        }

        return result;
    }

   private:
    // EM-seeds `sampler`'s population + first-N chains (C# 374-477). Runs
    // ExpectationMaximization for an initial (parameters, covariance), builds a Dirichlet
    // proposal for the weights + a MultivariateNormal proposal for the component parameters,
    // draws `InitialIterations` prior-predictive proposals (retry-up-to-20 on invalid MVN
    // draws; throw "Bad parameters" past 20), seeds them into the population, then seeds the
    // best `NumberOfChains` sets (by descending fitness) as the per-chain initial states. Uses
    // the SAME PRNG cadence as C#: one `prng.next()` for the Dirichlet draw, then the
    // `NextDoubles(1, Np)` cadence (one `prng.next()`-seeded fresh MT per dimension) for the
    // MVN draw. Any throw propagates to `run()`'s catch (-> Randomize fallback).
    void seed_sampler_from_em(corehydro::numerics::sampling::mcmc::MCMCSampler& sampler) {
        namespace dists = corehydro::numerics::distributions;
        using ParameterSet = corehydro::numerics::math::optimization::ParameterSet;
        using MersenneTwister = corehydro::numerics::sampling::MersenneTwister;

        // Initial parameters + covariance from Expectation-Maximization (C# 375).
        std::vector<double> parameters;
        corehydro::numerics::math::linalg::Matrix covariance(0, 0);
        int em_iterations = 0;
        mixture_distribution_->expectation_maximization(parameters, covariance, em_iterations);

        const int K = mixture_distribution_->mixture()->component_count();
        const int Np = static_cast<int>(parameters.size()) - K;
        MersenneTwister prng(static_cast<std::uint32_t>(sampler.prng_seed()));
        std::vector<ParameterSet> temp_population;

        // Weights proposal: Dirichlet centered on the EM weights (C# 388-401). Only for K > 1;
        // a single component always has weight 1.0.
        std::optional<dists::Dirichlet> weight_dirichlet;
        if (K > 1) {
            const int N = mixture_distribution_->data_frame().total_record_length();
            const double S = static_cast<double>(std::max(N - 1, K + 1));
            const double deflation = 1.5;
            std::vector<double> alpha(static_cast<std::size_t>(K));
            for (int j = 0; j < K; ++j)
                alpha[static_cast<std::size_t>(j)] =
                    std::max(parameters[static_cast<std::size_t>(j)] * S / deflation, 0.1);
            weight_dirichlet.emplace(std::move(alpha));
        }

        // Component proposal: MVN over the Fisher sub-block of the EM covariance, inflated 1.5x
        // (C# 405-413).
        std::vector<double> em_components(static_cast<std::size_t>(Np));
        std::vector<std::vector<double>> component_covar(
            static_cast<std::size_t>(Np), std::vector<double>(static_cast<std::size_t>(Np)));
        for (int i = 0; i < Np; ++i) {
            em_components[static_cast<std::size_t>(i)] = parameters[static_cast<std::size_t>(i + K)];
            for (int j = 0; j < Np; ++j)
                component_covar[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    covariance(i + K, j + K) * 1.5;
        }
        dists::MultivariateNormal component_mvn(em_components, component_covar);

        // Draw InitialIterations proposals (C# 415-467).
        std::vector<ParameterSet> seeded_population;
        seeded_population.reserve(static_cast<std::size_t>(sampler.initial_iterations()));
        for (int i = 0; i < sampler.initial_iterations(); ++i) {
            // Weights from the Dirichlet (always valid on the simplex), or {1.0} for K == 1.
            std::vector<double> weights;
            if (weight_dirichlet.has_value()) {
                std::vector<std::vector<double>> w_sample =
                    weight_dirichlet->generate_random_values(1, prng.next());
                weights.assign(w_sample[0].begin(), w_sample[0].end());
            } else {
                weights = {1.0};
            }

            bool failed = true;
            int failed_count = 0;
            std::vector<double> p;
            double lh = -std::numeric_limits<double>::infinity();
            while (failed) {
                try {
                    // MVN component draw via the C# NextDoubles(1, Np) cadence: one outer
                    // prng.next() per dimension, each seeding a fresh MT whose first
                    // next_double() is the uniform for that dimension.
                    std::vector<double> u(static_cast<std::size_t>(Np));
                    for (int d = 0; d < Np; ++d) {
                        MersenneTwister sub(static_cast<std::uint32_t>(prng.next()));
                        u[static_cast<std::size_t>(d)] = sub.next_double();
                    }
                    std::vector<double> comp = component_mvn.inverse_cdf(u);

                    // Concatenate weights ++ component parameters (C# 447-450).
                    p.clear();
                    p.reserve(static_cast<std::size_t>(K + Np));
                    p.insert(p.end(), weights.begin(), weights.end());
                    p.insert(p.end(), comp.begin(), comp.end());

                    // MixtureModel::log_likelihood normalizes the weight entries IN PLACE (C#
                    // `SetParameters(ref parameters)`, M14), so the seeded set stores the
                    // normalized weights -- exactly as the C# `new ParameterSet(p, lh)` does.
                    lh = mixture_distribution_->log_likelihood(p);
                    failed = false;
                } catch (const std::exception&) {
                    // Invalid component draw -- retry (C# 458-465).
                    ++failed_count;
                    if (failed_count > 20) break;
                }
            }

            if (failed) throw std::runtime_error("Bad parameters");  // C# 466.

            seeded_population.emplace_back(p, lh);
            temp_population.emplace_back(p, lh);
        }

        // Seed the sampler's population (C# 465 `PopulationMatrix.Add` collapsed to one call).
        sampler.seed_population(std::move(seeded_population));

        // Sort temp population by log-likelihood descending (C# 470).
        std::stable_sort(
            temp_population.begin(), temp_population.end(),
            [](const ParameterSet& a, const ParameterSet& b) { return a.fitness > b.fitness; });

        // Seed the first NumberOfChains chains with the best-performing sets (C# 473-476).
        for (int i = 0; i < sampler.number_of_chains(); ++i)
            sampler.seed_chain(i, temp_population[static_cast<std::size_t>(i)].clone());
    }

    // Null-guard helper for the ctor init list (C# `?? throw new ArgumentNullException`).
    static std::unique_ptr<MixtureModel> require_non_null(std::unique_ptr<MixtureModel> model) {
        if (model == nullptr) {
            throw std::invalid_argument("mixtureDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // Owned model (deviation 2). Declared BEFORE bayesian_analysis_ so it is constructed first.
    std::unique_ptr<MixtureModel> mixture_distribution_;
    corehydro::estimation::BayesianAnalysis bayesian_analysis_;
    ProbabilityOrdinates probability_ordinates_;

    // Result object (C# nullable -> optional; deviation 3).
    std::optional<UncertaintyAnalysisResults> analysis_results_;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (deviation 4).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
};

}  // namespace corehydro::analyses
