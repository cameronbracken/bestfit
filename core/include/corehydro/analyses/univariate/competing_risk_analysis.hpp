// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/CompetingRiskAnalysis.cs @ fc28c0c
//
// Bayesian MCMC frequency-curve analysis over a CompetingRisksModel (multiple independent
// parent distributions whose observed maximum is the result of competing processes). A
// mechanical sibling of UnivariateAnalysis (A5): it drives a BayesianAnalysis (Phase 4) over
// the model, then assembles the Bayesian frequency curve + credible intervals into an
// UncertaintyAnalysisResults (A2) + point-estimate goodness-of-fit. The model is stationary
// (IsNonstationary => false), so there is NO chronology / nonstationary path here.
//
// The DROPPED surface mirrors the UnivariateAnalysis precedent (GUI/threading/serialization,
// no numerical content):
//   * both XML constructors (C# 54-106) + `ToXElement` (591) -- XElement (de)serialization.
//   * the `*_PropertyChanged` handlers (C# 221/239/264) -- INotifyPropertyChanged /
//     INotifyCollectionChanged binding cascades; the reprocess decisions they encode are
//     exercised HERE by calling the public reprocess methods directly.
//   * `CancellationTokenSource` + `CancelAnalysis` (396), `SafeProgressReporter`/`IProgress`,
//     the `_reprocessGate` semaphore, and the `AnalysisStarting`/`AnalysisCompleted` events.
//     `RaisePropertyChange` calls throughout.
//
// The C# `async Task RunAsync` ports to a synchronous `run()`; the `...Async` helpers port to
// synchronous methods (every `Parallel.For` becomes a serial loop -- independent writes).
//
// DEVIATIONS (documented; same as UnivariateAnalysis):
//   1. EXCEEDANCE <-> NON-EXCEEDANCE FLIP (C# 532/468).
//   2. OWNERSHIP. The analysis OWNS the model via `std::unique_ptr` (the ctor null-guard maps to
//      the C# ArgumentNullException); `BayesianAnalysis` holds a `ModelBase&` into it, so the
//      analysis is non-copyable / non-movable.
//   3. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`.
//   4. `get_distribution` / `get_point_estimate_distribution` return a NON-owning raw pointer;
//      the cloned distributions are OWNED by the analysis in `distribution_cache_`.
//   5. RUN ERROR REPORTING. With events dropped, `run()` lets exceptions propagate.
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/analyses/support/i_univariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/univariate_distribution/competing_risks_model.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/probability_ordinates.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace corehydro::analyses {

class CompetingRiskAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using CompetingRisksModel = corehydro::models::CompetingRisksModel;
    using UnivariateDistributionBase = corehydro::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using ProbabilityOrdinates = corehydro::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using PointEstimateType = corehydro::estimation::PointEstimateType;

    // C# ctor `CompetingRiskAnalysis(CompetingRisksModel)` (C# 54). The C# `?? throw
    // ArgumentNullException` maps to the null-guard here (deviation 2).
    explicit CompetingRiskAnalysis(std::unique_ptr<CompetingRisksModel> competing_risks_distribution)
        : competing_risks_distribution_(require_non_null(std::move(competing_risks_distribution))),
          bayesian_analysis_(*competing_risks_distribution_),
          probability_ordinates_() {}

    ~CompetingRiskAnalysis() override = default;

    // Non-copyable / non-movable (deviation 2).
    CompetingRiskAnalysis(const CompetingRiskAnalysis&) = delete;
    CompetingRiskAnalysis& operator=(const CompetingRiskAnalysis&) = delete;
    CompetingRiskAnalysis(CompetingRiskAnalysis&&) = delete;
    CompetingRiskAnalysis& operator=(CompetingRiskAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `CompetingRisksDistribution` (C# 127): the model being estimated.
    CompetingRisksModel& competing_risks_distribution() { return *competing_risks_distribution_; }
    const CompetingRisksModel& competing_risks_distribution() const {
        return *competing_risks_distribution_;
    }

    // C# `BayesianAnalysis` (C# 176). IBayesianAnalysis override.
    corehydro::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const corehydro::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ProbabilityOrdinates` (C# 152). IProbabilityOrdinates override.
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `AnalysisResults` (C# 207): the frequency-analysis results (null until estimated).
    // IBayesianAnalysis override (const pointer <=> nullable).
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 293).
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        distribution_cache_.clear();
        set_is_estimated(false);
    }

    // C# `ClearFrequencyAnalysisResults` (C# 306).
    void clear_frequency_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (C# 313), synchronous. Validate guard -> prepare input data
    // (ProcessThresholdSeries + ProcessQuantilePriors) -> BayesianAnalysis.estimate() ->
    // IF estimated: create frequency results -> IsEstimated mirrors the inner fit.
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        // Prepare input data (C# 350-351).
        competing_risks_distribution_->data_frame().process_threshold_series();
        competing_risks_distribution_->process_quantile_priors();

        // Run the Bayesian analysis.
        bayesian_analysis_.estimate();

        // Post-process.
        if (bayesian_analysis_.is_estimated()) {
            create_frequency_analysis_results();
        }

        // Mirror the inner fit's success state (C# 365).
        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // C# `GetDistribution(int)` (C# 403): clones the competing-risks distribution and applies the
    // posterior output sample. Null when unestimated. The clone is owned by the analysis
    // (deviation 4).
    UnivariateDistributionBase* get_distribution(int index) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        std::unique_ptr<UnivariateDistributionBase> result =
            competing_risks_distribution_->competing_risks()->clone();
        result->set_parameters(
            bayesian_analysis_.results()->output[static_cast<std::size_t>(index)].values);
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // C# `GetPointEstimateDistribution()` (C# 417).
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_analysis_.point_estimator());
    }

    // C# `GetPointEstimateDistribution(PointEstimateType)` (C# 421).
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        const std::vector<double>& parms =
            point_estimator == PointEstimateType::PosteriorMean
                ? bayesian_analysis_.results()->posterior_mean.values
                : bayesian_analysis_.results()->map.values;

        std::unique_ptr<UnivariateDistributionBase> result =
            competing_risks_distribution_->competing_risks()->clone();
        result->set_parameters(parms);
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 506), synchronous.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        // Set model parameters to the MAP (C# 517).
        competing_risks_distribution_->set_parameter_values(results.map.values);

        // Build B sampled distributions, one per posterior output sample (C# 520-527).
        int b = bayesian_analysis_.output_length();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> owned;
        owned.reserve(static_cast<std::size_t>(b));
        for (int idx = 0; idx < b; ++idx) {
            std::unique_ptr<UnivariateDistributionBase> d =
                competing_risks_distribution_->competing_risks()->clone();
            d->set_parameters(results.output[static_cast<std::size_t>(idx)].values);
            owned.push_back(std::move(d));
        }

        std::vector<const UnivariateDistributionBase*> sampled;
        sampled.reserve(owned.size());
        for (const auto& u : owned) sampled.push_back(u.get());

        // Exceedance -> non-exceedance FLIP (deviation 1; C# 532 `p => 1.0 - p`).
        std::vector<double> probabilities;
        probabilities.reserve(probability_ordinates_.count());
        for (double p : probability_ordinates_) probabilities.push_back(1.0 - p);

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        analysis_results_.emplace(*competing_risks_distribution_->competing_risks(), sampled,
                                  probabilities, alpha);

        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 445), synchronous.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        CompetingRisksModel& model = *competing_risks_distribution_;

        // Set the point estimator (C# 456-463).
        if (bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean) {
            model.set_parameter_values(results.posterior_mean.values);
        } else {
            model.set_parameter_values(results.map.values);
        }

        // Mode curve on the non-exceedance grid (C# 466-468; deviation 1).
        analysis_results_->mode_curve.assign(probability_ordinates_.count(), 0.0);
        for (std::size_t i = 0; i < probability_ordinates_.count(); ++i) {
            analysis_results_->mode_curve[i] =
                model.competing_risks()->inverse_cdf(1.0 - probability_ordinates_[i]);
        }

        // Information criteria at the MAP estimate (C# 472-477).
        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double log_l = model.log_likelihood(map_values);
        int k = model.number_of_parameters();
        int n = model.data_frame().total_record_length();
        double aic = GoodnessOfFit::aic(k, log_l);
        double bic = GoodnessOfFit::bic(n, k, log_l);
        double dic = bayesian_analysis_.dic();

        // RMSE over Exact + Uncertain + Interval data (C# 480-486).
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
        double rmse = GoodnessOfFit::rmse(values, probs, *model.competing_risks());

        analysis_results_->aic = aic;
        analysis_results_->bic = bic;
        analysis_results_->dic = dic;
        analysis_results_->rmse = rmse;
    }

    // C# `Validate` (C# 541).
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        corehydro::models::ValidationResult dist_valid = competing_risks_distribution_->validate();
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
    static std::unique_ptr<CompetingRisksModel> require_non_null(
        std::unique_ptr<CompetingRisksModel> model) {
        if (model == nullptr) {
            throw std::invalid_argument("competingRisksDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // Owned model (deviation 2). Declared BEFORE bayesian_analysis_ so it is constructed first.
    std::unique_ptr<CompetingRisksModel> competing_risks_distribution_;
    corehydro::estimation::BayesianAnalysis bayesian_analysis_;
    ProbabilityOrdinates probability_ordinates_;

    // Result object (C# nullable -> optional; deviation 3).
    std::optional<UncertaintyAnalysisResults> analysis_results_;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (deviation 4).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
};

}  // namespace corehydro::analyses
