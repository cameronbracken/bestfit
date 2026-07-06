// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/PointProcessAnalysis.cs @ fc28c0c
//
// Bayesian MCMC frequency-curve analysis over a PointProcessModel (peaks-over-threshold). A
// mechanical sibling of UnivariateAnalysis (A5): it drives a BayesianAnalysis (Phase 4) over
// the model, then assembles the Bayesian frequency curve + credible intervals into an
// UncertaintyAnalysisResults (A2) + point-estimate goodness-of-fit. The point-process model is
// stationary (IsNonstationary => false), so there is NO chronology / nonstationary path here
// (that half of UnivariateAnalysis is simply absent, not deferred).
//
// The DROPPED surface mirrors the UnivariateAnalysis precedent (each is GUI/threading/
// serialization with no numerical content):
//   * both XML constructors (C# 70-111) + `ToXElement` (604) -- XElement (de)serialization.
//   * the `*_PropertyChanged` handlers (ProbabilityOrdinates_CollectionChanged 227,
//     Model_PropertyChanged 245, BayesianAnalysis_PropertyChanged 275) -- INotifyPropertyChanged
//     / INotifyCollectionChanged binding cascades; the reprocess decisions they encode are
//     exercised HERE by calling the public reprocess methods directly, exactly as the C# tests do.
//   * `CancellationTokenSource` + `CancelAnalysis` (408), `SafeProgressReporter`/`IProgress`,
//     the `_reprocessGate` semaphore, and the `AnalysisStarting`/`AnalysisCompleted` events --
//     run-lifecycle plumbing. `RaisePropertyChange` calls throughout.
//
// The C# `async Task RunAsync` ports to a synchronous `run()`; the `...Async` helpers port to
// synchronous methods (every `Parallel.For` becomes a serial loop -- independent writes, so the
// result is numerically identical).
//
// DEVIATIONS (documented; same as UnivariateAnalysis):
//   1. EXCEEDANCE <-> NON-EXCEEDANCE FLIP. `ProbabilityOrdinates` holds EXCEEDANCE probabilities;
//      the frequency curve is tabulated on NON-exceedance probabilities, so
//      `create_frequency_analysis_results()` builds `probabilities = ordinates.map(p -> 1 - p)`
//      and the mode curve evaluates `inverse_cdf(1 - ordinate)` (C# 545/481).
//   2. OWNERSHIP. The analysis OWNS the model via `std::unique_ptr` (the ctor null-guard maps to
//      the C# ArgumentNullException). `BayesianAnalysis` holds a `ModelBase&` into that owned
//      model, so the analysis is non-copyable / non-movable.
//   3. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`;
//      `analysis_results()` returns a `const ...*` (null <=> empty optional).
//   4. `get_distribution` / `get_point_estimate_distribution` return a NON-owning raw pointer per
//      the IUnivariateAnalysis contract; the cloned distributions are OWNED by the analysis in
//      `distribution_cache_` (kept alive until the analysis dies).
//   5. RUN ERROR REPORTING. With events dropped, `run()` lets exceptions propagate to the caller
//      (there is nothing to report them to); `bayesian.estimate()` already throws on an invalid
//      configuration.
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/univariate_distribution/point_process_model.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/probability_ordinates.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace bestfit::analyses {

class PointProcessAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using PointProcessModel = bestfit::models::PointProcessModel;
    using UnivariateDistributionBase = bestfit::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using ProbabilityOrdinates = bestfit::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using PointEstimateType = bestfit::estimation::PointEstimateType;

    // C# ctor `PointProcessAnalysis(PointProcessModel)` (C# 50). The C# `?? throw
    // ArgumentNullException` maps to the null-guard here (deviation 2).
    explicit PointProcessAnalysis(std::unique_ptr<PointProcessModel> point_process)
        : point_process_(require_non_null(std::move(point_process))),
          bayesian_analysis_(*point_process_),
          probability_ordinates_() {}

    ~PointProcessAnalysis() override = default;

    // Non-copyable / non-movable (deviation 2: bayesian_analysis_ holds a reference to the owned
    // model; a defaulted move would leave that reference dangling on the moved-from object).
    PointProcessAnalysis(const PointProcessAnalysis&) = delete;
    PointProcessAnalysis& operator=(const PointProcessAnalysis&) = delete;
    PointProcessAnalysis(PointProcessAnalysis&&) = delete;
    PointProcessAnalysis& operator=(PointProcessAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `PointProcess` (C# 132): the model being estimated.
    PointProcessModel& point_process() { return *point_process_; }
    const PointProcessModel& point_process() const { return *point_process_; }

    // C# `BayesianAnalysis` (C# 182). IBayesianAnalysis override.
    bestfit::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const bestfit::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ProbabilityOrdinates` (C# 157). IProbabilityOrdinates override.
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `AnalysisResults` (C# 213): the frequency-analysis results (null until estimated).
    // IBayesianAnalysis override (const pointer <=> nullable).
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 304): clears the Bayesian fit + result object and resets
    // IsEstimated. The RaisePropertyChange calls are dropped.
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        distribution_cache_.clear();
        set_is_estimated(false);
    }

    // C# `ClearFrequencyAnalysisResults` (C# 317): clears ONLY the frequency results.
    void clear_frequency_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (C# 324), synchronous. Validate guard -> prepare input data
    // (ProcessThresholdSeries + ProcessQuantilePriors) -> BayesianAnalysis.estimate() ->
    // IF estimated: create frequency results -> IsEstimated mirrors the inner fit.
    // (Cancellation/gate/events/progress dropped; deviation 5: exceptions propagate.)
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        // Prepare input data (C# 362-363).
        point_process_->data_frame().process_threshold_series();
        point_process_->process_quantile_priors();

        // Run the Bayesian analysis.
        bayesian_analysis_.estimate();

        // Post-process.
        if (bayesian_analysis_.is_estimated()) {
            create_frequency_analysis_results();
        }

        // Mirror the inner fit's success state (C# 377).
        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // C# `GetDistribution(int)` (C# 415): the distribution for a given posterior output index,
    // or null when unestimated. IUnivariateAnalysis override. The clone is owned by the analysis
    // (deviation 4).
    UnivariateDistributionBase* get_distribution(int index) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        std::unique_ptr<UnivariateDistributionBase> result = point_process_->distribution()->clone();
        result->set_parameters(
            bayesian_analysis_.results()->output[static_cast<std::size_t>(index)].values);
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // C# `GetPointEstimateDistribution()` (C# 430): uses the analysis's configured PointEstimator.
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_analysis_.point_estimator());
    }

    // C# `GetPointEstimateDistribution(PointEstimateType)` (C# 434): uses a caller-supplied
    // estimator without mutating the analysis's own PointEstimator. Null when unestimated.
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        const std::vector<double>& parms =
            point_estimator == PointEstimateType::PosteriorMean
                ? bayesian_analysis_.results()->posterior_mean.values
                : bayesian_analysis_.results()->map.values;

        std::unique_ptr<UnivariateDistributionBase> result = point_process_->distribution()->clone();
        result->set_parameters(parms);
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 519), synchronous. Sets the model params to
    // MAP, builds one sampled distribution per posterior output sample via the model's
    // seasonal-aware GetDistribution, assembles the UncertaintyAnalysisResults on the
    // NON-exceedance grid (deviation 1) at alpha = 1 - CredibleIntervalWidth, then updates the
    // point-estimate scalars.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        // Set model parameters to the MAP (C# 530).
        point_process_->set_parameter_values(results.map.values);

        // Build B sampled distributions, one per posterior output sample (C# 533-540).
        int b = bayesian_analysis_.output_length();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> owned;
        owned.reserve(static_cast<std::size_t>(b));
        for (int idx = 0; idx < b; ++idx) {
            owned.push_back(
                point_process_->get_distribution(results.output[static_cast<std::size_t>(idx)].values));
        }

        std::vector<const UnivariateDistributionBase*> sampled;
        sampled.reserve(owned.size());
        for (const auto& u : owned) sampled.push_back(u.get());

        // Exceedance -> non-exceedance FLIP (deviation 1; C# 545 `p => 1.0 - p`).
        std::vector<double> probabilities;
        probabilities.reserve(probability_ordinates_.count());
        for (double p : probability_ordinates_) probabilities.push_back(1.0 - p);

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        analysis_results_.emplace(*point_process_->distribution(), sampled, probabilities, alpha);

        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 458), synchronous. Sets the model params to the
    // selected point estimator, rebuilds the mode curve on the non-exceedance grid, and writes
    // the goodness-of-fit scalars (AIC/BIC at MAP, DIC from the Bayesian analysis, RMSE over the
    // Exact + Uncertain + Interval data using plotting-position complements).
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        PointProcessModel& model = *point_process_;

        // Set the point estimator (C# 469-476).
        if (bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean) {
            model.set_parameter_values(results.posterior_mean.values);
        } else {
            model.set_parameter_values(results.map.values);
        }

        // Mode curve on the non-exceedance grid (C# 479-481; deviation 1).
        analysis_results_->mode_curve.assign(probability_ordinates_.count(), 0.0);
        for (std::size_t i = 0; i < probability_ordinates_.count(); ++i) {
            analysis_results_->mode_curve[i] =
                model.distribution()->inverse_cdf(1.0 - probability_ordinates_[i]);
        }

        // Information criteria at the MAP estimate (C# 485-490).
        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double log_l = model.log_likelihood(map_values);
        int k = model.number_of_parameters();
        int n = model.data_frame().total_record_length();
        double aic = GoodnessOfFit::aic(k, log_l);
        double bic = GoodnessOfFit::bic(n, k, log_l);
        double dic = bayesian_analysis_.dic();

        // RMSE over Exact + Uncertain + Interval data (C# 493-499).
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
        double rmse = GoodnessOfFit::rmse(values, probs, *model.distribution());

        analysis_results_->aic = aic;
        analysis_results_->bic = bic;
        analysis_results_->dic = dic;
        analysis_results_->rmse = rmse;
    }

    // C# `Validate` (C# 554): aggregates the model, ordinate, and Bayesian-analysis validations.
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        bestfit::models::ValidationResult dist_valid = point_process_->validate();
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
    static std::unique_ptr<PointProcessModel> require_non_null(
        std::unique_ptr<PointProcessModel> model) {
        if (model == nullptr) {
            throw std::invalid_argument("pointProcess");  // C# ArgumentNullException
        }
        return model;
    }

    // Owned model (deviation 2). Declared BEFORE bayesian_analysis_ so it is constructed first.
    std::unique_ptr<PointProcessModel> point_process_;
    bestfit::estimation::BayesianAnalysis bayesian_analysis_;
    ProbabilityOrdinates probability_ordinates_;

    // Result object (C# nullable -> optional; deviation 3).
    std::optional<UncertaintyAnalysisResults> analysis_results_;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (deviation 4).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
};

}  // namespace bestfit::analyses
