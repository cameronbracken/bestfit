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
//   6. THE MANUAL EM-SEEDING PATH IS NOT WIRED (primary port deviation). The C# `RunAsync`
//      (lines 360-486) does NOT let `BayesianAnalysis.estimate()` auto-initialize the sampler.
//      Instead it calls `BayesianAnalysis.SetUpSampler()`, grabs `BayesianAnalysis.Sampler!`,
//      sets `sampler.Initialize = MCMCSampler.InitializationType.UserDefined`, runs
//      `MixtureModel.ExpectationMaximization(...)` for an initial (parameters, covariance), draws
//      `InitialIterations` proposals from a Dirichlet (weights) + MultivariateNormal (component
//      parameters) prior-predictive, and SEEDS the sampler by MUTATING `sampler.PopulationMatrix`
//      (`.Add(...)`) and `sampler.MarkovChains[i]` (`.Add(...)`); on ANY exception during this
//      block the C# falls back to `sampler.Initialize = Randomize`. This manual seeding CANNOT be
//      ported against the current oracle-locked Phase-3 MCMC API without editing it, which the D1
//      task scope forbids (additive-only; D1 adds only new headers). Two hard blocks:
//        (a) `MCMCSampler` exposes `population_matrix()` / `markov_chains()` as CONST accessors
//            only (mcmc_sampler.hpp) -- there is no public mutator to `.Add(...)` a seeded
//            ParameterSet, and the backing fields are `protected`.
//        (b) The ported `BayesianAnalysis::estimate()` UNCONDITIONALLY rebuilds the sampler via
//            `set_up_sampler()` immediately before `sample()` (bayesian_analysis.hpp scope
//            decision 2) rather than reusing an externally-configured `Sampler`, so any external
//            seeding would be discarded even if it could be written.
//      Consequently `run()` here takes the C#'s OWN documented fallback branch: it drives the
//      standard `bayesian_analysis_.estimate()` path (random chain initialization), which is
//      exactly the state the C# lands in when its EM-seeding block throws. The EM machinery
//      itself (`MixtureModel::expectation_maximization`) is fully ported and independently tested
//      in the models layer; only its use as a SAMPLER SEED is unavailable at this API boundary.
//      Re-wiring the seed is tracked as a follow-up (needs a mutable seeding hook on the ported
//      MCMCSampler / BayesianAnalysis, which is out of D1's additive-only scope).
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/univariate_distribution/mixture_model.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/probability_ordinates.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace bestfit::analyses {

class MixtureAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using MixtureModel = bestfit::models::MixtureModel;
    using UnivariateDistributionBase = bestfit::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using ProbabilityOrdinates = bestfit::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using PointEstimateType = bestfit::estimation::PointEstimateType;

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
    bestfit::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const bestfit::estimation::BayesianAnalysis& bayesian_analysis() const {
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
    // (ProcessThresholdSeries + ProcessQuantilePriors) -> BayesianAnalysis.estimate() ->
    // IF estimated: create frequency results -> IsEstimated mirrors the inner fit.
    //
    // NOTE (deviation 6, see the file header): the C# manual EM-seeding of the sampler
    // (SetUpSampler / Sampler / Initialize = UserDefined + population/chain seeding) is NOT
    // wired here -- the ported MCMC API exposes no mutable seeding hook and estimate() rebuilds
    // the sampler unconditionally. This takes the C#'s own init-failure fallback: the standard
    // random-initialized estimate() path.
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

        // Run the Bayesian analysis. (The C# manual EM-seed of the sampler -- C# 366-486 -- is
        // not portable against the current MCMC API; see deviation 6. This is the C# init-
        // failure fallback branch: standard random-initialized estimation.)
        bayesian_analysis_.estimate();

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
        const bestfit::models::DataFrame& df = model.data_frame();
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
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        bestfit::models::ValidationResult dist_valid = mixture_distribution_->validate();
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
    // Null-guard helper for the ctor init list (C# `?? throw new ArgumentNullException`).
    static std::unique_ptr<MixtureModel> require_non_null(std::unique_ptr<MixtureModel> model) {
        if (model == nullptr) {
            throw std::invalid_argument("mixtureDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // Owned model (deviation 2). Declared BEFORE bayesian_analysis_ so it is constructed first.
    std::unique_ptr<MixtureModel> mixture_distribution_;
    bestfit::estimation::BayesianAnalysis bayesian_analysis_;
    ProbabilityOrdinates probability_ordinates_;

    // Result object (C# nullable -> optional; deviation 3).
    std::optional<UncertaintyAnalysisResults> analysis_results_;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (deviation 4).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
};

}  // namespace bestfit::analyses
