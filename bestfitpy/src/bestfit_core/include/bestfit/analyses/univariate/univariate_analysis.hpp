// ported from: Analyses/Univariate/UnivariateAnalysis.cs @ fc28c0c
//
// The first concrete Analyses-layer port (A5): a Bayesian MCMC frequency-curve analysis over a
// univariate distribution model. It drives a BayesianAnalysis (Phase 4) over the model, then
// assembles the Bayesian frequency curve + credible intervals into an UncertaintyAnalysisResults
// (A2), plus a time-varying chronology curve for nonstationary models.
//
// LOAD-BEARING compute sequence ported; WPF/event/gate/XML plumbing DROPPED per the A4 precedent.
// Specifically SKIPPED (each is GUI/threading/serialization with no numerical content):
//   * both XML constructors (C# 82-125) + `ToXElement` (838) -- XElement (de)serialization.
//   * every `*_PropertyChanged` handler (ProbabilityOrdinates_CollectionChanged 275,
//     Model_PropertyChanged 304, BayesianAnalysis_PropertyChanged 376) and the
//     `_previousTimeIndex` / `TryGetMaxDataIndex` (414) tracking they use -- INotifyPropertyChanged
//     / INotifyCollectionChanged binding cascades. The reprocess-vs-clear decisions they encode
//     are exercised HERE by calling the (public) reprocess methods directly, exactly as the C#
//     tests do.
//   * the reprocess gate (`_reprocessGate.WaitAsync/Release`), `CancellationTokenSource` + cancel
//     (`CancelAnalysis`, 541), `SafeProgressReporter`, and the `AnalysisStarting`/
//     `AnalysisCompleted` events + `OnAnalysisStarting/Completed` -- run-lifecycle plumbing.
//   * `RaisePropertyChange` calls throughout -- no notification system in this port.
//
// The C# `async Task RunAsync` ports to a synchronous `run()`; the C# `...Async` helper methods
// (`CreateFrequencyAnalysisResultsAsync`, `UpdatePointEstimateResultsAsync`,
// `CreateChronologyResultsAsync`) port to synchronous methods (every `Parallel.For` becomes a
// serial loop -- the loop bodies are independent writes, so the result is numerically identical).
//
// DEVIATIONS (documented):
//   1. EXCEEDANCE <-> NON-EXCEEDANCE FLIP. `ProbabilityOrdinates` holds EXCEEDANCE probabilities;
//      the frequency curve is tabulated on NON-exceedance probabilities. So
//      `create_frequency_analysis_results()` builds `probabilities = ordinates.map(p -> 1 - p)`
//      before handing them to `UncertaintyAnalysisResults`, and `update_point_estimate_results()` /
//      the mode curve evaluate `inverse_cdf(1 - ordinate)` -- transcribed from C# 632/712.
//   2. OWNERSHIP. The C# `UnivariateDistribution` model is a GC reference type; here the analysis
//      OWNS the model via `std::unique_ptr` (the ctor's null-guard maps to the C#
//      ArgumentNullException). `BayesianAnalysis` holds a `ModelBase&` into that owned model, so
//      the analysis is non-copyable / non-movable (a moved reference member cannot rebind).
//   3. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`. The
//      `analysis_results()` interface accessor returns a `const ...*` (null <=> empty optional).
//   4. `get_distribution` / `get_point_estimate_distribution` return a NON-owning raw pointer per
//      the IUnivariateAnalysis contract; the cloned distributions they hand back are OWNED by the
//      analysis in `distribution_cache_` (kept alive until the analysis dies). The C# returns a GC
//      clone; this is the C++ lifetime-faithful equivalent.
//   5. CHRONOLOGY PARENT. C# sets `ChronologyAnalysisResults.ParentDistribution =
//      Distribution.Clone()` (a fresh owned clone); the A2 DTO's `parent_distribution` is a
//      non-owning pointer, so the analysis owns that clone in `chronology_parent_` and points the
//      DTO at it.
//   6. RUN ERROR REPORTING. C#'s `RunAsync` wraps the compute body in try/catch that reports
//      failures through `AnalysisCompleted`. With events dropped, `run()` lets exceptions
//      propagate to the caller instead (there is nothing to report them to); `bayesian.estimate()`
//      already throws on an invalid configuration.
#pragma once
#include <algorithm>
#include <array>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/probability_ordinates.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace bestfit::analyses {

class UnivariateAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using UnivariateDistributionModel = bestfit::models::UnivariateDistributionModel;
    using UnivariateDistributionBase = bestfit::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using ProbabilityOrdinates = bestfit::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using PointEstimateType = bestfit::estimation::PointEstimateType;

    // C# ctor `UnivariateAnalysis(UnivariateDistribution)` (C# 49): builds the BayesianAnalysis
    // over the model and a default ProbabilityOrdinates. The C# `?? throw ArgumentNullException`
    // maps to the null-guard here (see deviation 2).
    explicit UnivariateAnalysis(std::unique_ptr<UnivariateDistributionModel> univariate_distribution)
        : univariate_distribution_(require_non_null(std::move(univariate_distribution))),
          bayesian_analysis_(*univariate_distribution_),
          probability_ordinates_() {}

    ~UnivariateAnalysis() override = default;

    // Non-copyable / non-movable (deviation 2: bayesian_analysis_ holds a reference to the owned
    // model; a defaulted move would leave that reference dangling on the moved-from object).
    UnivariateAnalysis(const UnivariateAnalysis&) = delete;
    UnivariateAnalysis& operator=(const UnivariateAnalysis&) = delete;
    UnivariateAnalysis(UnivariateAnalysis&&) = delete;
    UnivariateAnalysis& operator=(UnivariateAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `UnivariateDistribution` (C# 146): the model being estimated.
    UnivariateDistributionModel& univariate_distribution() { return *univariate_distribution_; }
    const UnivariateDistributionModel& univariate_distribution() const {
        return *univariate_distribution_;
    }

    // C# `BayesianAnalysis` (C# 208). IBayesianAnalysis override.
    bestfit::estimation::BayesianAnalysis& bayesian_analysis() override {
        return bayesian_analysis_;
    }
    const bestfit::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ProbabilityOrdinates` (C# 184). IProbabilityOrdinates override.
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `AnalysisResults` (C# 239): the frequency-analysis results (null until estimated).
    // IBayesianAnalysis override (const pointer <=> nullable).
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // C# `ChronologyAnalysisResults` (C# 250): the nonstationary time-varying results.
    const UncertaintyAnalysisResults* chronology_analysis_results() const {
        return chronology_analysis_results_ ? &*chronology_analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 425): clears the Bayesian fit + both result objects and resets
    // IsEstimated. The RaisePropertyChange calls are dropped (deviation).
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        chronology_analysis_results_.reset();
        distribution_cache_.clear();
        chronology_parent_.reset();
        set_is_estimated(false);
    }

    // C# `ClearFrequencyAnalysisResults` (C# 445): clears ONLY the frequency results; the fit,
    // the chronology results, and IsEstimated survive.
    void clear_frequency_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (C# 452), synchronous. Validate guard -> prepare input data
    // (ProcessThresholdSeries + CreateFullTimeSeries when nonstationary + ProcessQuantilePriors)
    // -> BayesianAnalysis.estimate() -> IF estimated: create frequency + chronology results ->
    // IsEstimated mirrors the inner fit. (Cancellation/gate/events/progress dropped; see the file
    // header. Deviation 6: exceptions propagate instead of routing through AnalysisCompleted.)
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        // Prepare input data.
        univariate_distribution_->data_frame().process_threshold_series();
        if (univariate_distribution_->is_nonstationary()) {
            univariate_distribution_->data_frame().create_full_time_series();
        }
        univariate_distribution_->process_quantile_priors();

        // Run the Bayesian analysis.
        bayesian_analysis_.estimate();

        // Post-process.
        if (bayesian_analysis_.is_estimated()) {
            create_frequency_analysis_results();
            create_chronology_results();
        }

        // Mirror the inner fit's success state (C# 510).
        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // C# `GetDistribution(int)` (C# 548): the distribution for a given posterior output index,
    // or null when unestimated. IUnivariateAnalysis override. The clone is owned by the analysis
    // (deviation 4).
    UnivariateDistributionBase* get_distribution(int index) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        std::unique_ptr<UnivariateDistributionBase> result;
        const auto& output = bayesian_analysis_.results()->output;
        if (univariate_distribution_->is_nonstationary()) {
            UnivariateDistributionModel dist = univariate_distribution_->clone();
            dist.set_parameter_values(output[static_cast<std::size_t>(index)].values);
            result = dist.distribution().clone();
        } else {
            result = univariate_distribution_->distribution().clone();
            result->set_parameters(output[static_cast<std::size_t>(index)].values);
        }
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // C# `GetPointEstimateDistribution()` (C# 571): uses the analysis's configured PointEstimator.
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_analysis_.point_estimator());
    }

    // C# `GetPointEstimateDistribution(PointEstimateType)` (C# 575): uses a caller-supplied
    // estimator without mutating the analysis's own PointEstimator. Null when unestimated.
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return nullptr;

        const std::vector<double>& parms =
            point_estimator == PointEstimateType::PosteriorMean
                ? bayesian_analysis_.results()->posterior_mean.values
                : bayesian_analysis_.results()->map.values;

        std::unique_ptr<UnivariateDistributionBase> result;
        if (univariate_distribution_->is_nonstationary()) {
            UnivariateDistributionModel dist = univariate_distribution_->clone();
            dist.set_parameter_values(parms);
            result = dist.distribution().clone();
        } else {
            result = univariate_distribution_->distribution().clone();
            result->set_parameters(parms);
        }
        UnivariateDistributionBase* raw = result.get();
        distribution_cache_.push_back(std::move(result));
        return raw;
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 674), synchronous. Sets the model params to
    // MAP, builds one sampled distribution per posterior output sample, assembles the
    // UncertaintyAnalysisResults on the NON-exceedance grid (deviation 1) at
    // alpha = 1 - CredibleIntervalWidth, then updates the point-estimate scalars.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        // Set model parameters to the MAP (C# 685).
        univariate_distribution_->set_parameter_values(results.map.values);

        // Build B sampled distributions, one per posterior output sample (C# 688-707).
        int b = bayesian_analysis_.output_length();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> owned;
        owned.reserve(static_cast<std::size_t>(b));
        if (univariate_distribution_->is_nonstationary()) {
            for (int idx = 0; idx < b; ++idx) {
                UnivariateDistributionModel ud = univariate_distribution_->clone();
                ud.set_parameter_values(results.output[static_cast<std::size_t>(idx)].values);
                owned.push_back(ud.distribution().clone());
            }
        } else {
            for (int idx = 0; idx < b; ++idx) {
                std::unique_ptr<UnivariateDistributionBase> d =
                    univariate_distribution_->distribution().clone();
                d->set_parameters(results.output[static_cast<std::size_t>(idx)].values);
                owned.push_back(std::move(d));
            }
        }

        std::vector<const UnivariateDistributionBase*> sampled;
        sampled.reserve(owned.size());
        for (const auto& u : owned) sampled.push_back(u.get());

        // Exceedance -> non-exceedance FLIP (deviation 1; C# 712 `p => 1.0 - p`).
        std::vector<double> probabilities;
        probabilities.reserve(probability_ordinates_.count());
        for (double p : probability_ordinates_) probabilities.push_back(1.0 - p);

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        analysis_results_.emplace(univariate_distribution_->distribution(), sampled, probabilities,
                                  alpha);

        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 609), synchronous. Sets the model params to the
    // selected point estimator, rebuilds the mode curve on the non-exceedance grid, and writes
    // the goodness-of-fit scalars (AIC/BIC at MAP, DIC from the Bayesian analysis, RMSE over the
    // Exact + Uncertain + Interval data using plotting-position complements).
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        UnivariateDistributionModel& model = *univariate_distribution_;

        // Set the point estimator (C# 620-627).
        if (bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean) {
            model.set_parameter_values(results.posterior_mean.values);
        } else {
            model.set_parameter_values(results.map.values);
        }

        // Mode curve on the non-exceedance grid (C# 630-632; deviation 1).
        analysis_results_->mode_curve.assign(probability_ordinates_.count(), 0.0);
        for (std::size_t i = 0; i < probability_ordinates_.count(); ++i) {
            analysis_results_->mode_curve[i] =
                model.distribution().inverse_cdf(1.0 - probability_ordinates_[i]);
        }

        // Information criteria at the MAP estimate (C# 640-645).
        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double log_l = model.log_likelihood(map_values);
        int k = model.number_of_parameters();
        int n = model.data_frame().total_record_length();
        double aic = GoodnessOfFit::aic(k, log_l);
        double bic = GoodnessOfFit::bic(n, k, log_l);
        double dic = bayesian_analysis_.dic();

        // RMSE over Exact + Uncertain + Interval data (C# 648-654). The three-arg overload
        // (observed, plotting-position complements, model) matches the C# call site.
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
        double rmse = GoodnessOfFit::rmse(values, probs, model.distribution());

        analysis_results_->aic = aic;
        analysis_results_->bic = bic;
        analysis_results_->dic = dic;
        analysis_results_->rmse = rmse;
    }

    // C# `CreateChronologyResultsAsync` (C# 729), synchronous. NONSTATIONARY ONLY: builds the
    // time-varying return-level mode curve, then the posterior mean + credible band at each time
    // step via percentiles at a = (1 - CredibleIntervalWidth) / 2.
    void create_chronology_results() {
        chronology_analysis_results_.reset();
        chronology_parent_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !univariate_distribution_->is_nonstationary()) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        UnivariateDistributionModel& model = *univariate_distribution_;

        // Set the point estimator (C# 742-749).
        if (bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean) {
            model.set_parameter_values(results.posterior_mean.values);
        } else {
            model.set_parameter_values(results.map.values);
        }

        UncertaintyAnalysisResults chrono;
        // Own the parent clone (deviation 5).
        chronology_parent_ = model.distribution().clone();
        chrono.parent_distribution = chronology_parent_.get();
        chrono.mode_curve = model.get_nonstationary_return_level();

        int realz = bayesian_analysis_.output_length();
        std::size_t length = chrono.mode_curve.size();
        // ts[realz][length] (C# `new double[realz, length]`).
        std::vector<std::vector<double>> ts(static_cast<std::size_t>(realz),
                                            std::vector<double>(length));
        for (int idx = 0; idx < realz; ++idx) {
            UnivariateDistributionModel ud = model.clone();
            ud.set_parameter_values(results.output[static_cast<std::size_t>(idx)].values);
            ts[static_cast<std::size_t>(idx)] = ud.get_nonstationary_return_level();
        }

        std::vector<double> mean(length);
        std::vector<std::array<double, 2>> ci(length);
        double a = (1.0 - bayesian_analysis_.credible_interval_width()) / 2.0;
        for (std::size_t col = 0; col < length; ++col) {
            std::vector<double> data(static_cast<std::size_t>(realz));
            for (int row = 0; row < realz; ++row)
                data[static_cast<std::size_t>(row)] = ts[static_cast<std::size_t>(row)][col];
            std::sort(data.begin(), data.end());
            mean[col] = bestfit::numerics::data::mean(data);
            ci[col][0] = bestfit::numerics::data::percentile(data, a, true);
            ci[col][1] = bestfit::numerics::data::percentile(data, 1.0 - a, true);
        }

        chrono.mean_curve = std::move(mean);
        chrono.confidence_intervals = std::move(ci);
        chronology_analysis_results_ = std::move(chrono);
    }

    // C# `Validate` (C# 788): aggregates the model, ordinate, and Bayesian-analysis validations.
    // const per the A4 IAnalysis contract (the C# body only reads state).
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        // Validate the univariate distribution.
        bestfit::models::ValidationResult dist_valid = univariate_distribution_->validate();
        if (!dist_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              dist_valid.validation_messages.begin(),
                                              dist_valid.validation_messages.end());
        }

        // Validate the probability ordinates.
        auto prob_valid = probability_ordinates_.validate();
        if (!prob_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              prob_valid.messages.begin(), prob_valid.messages.end());
        }

        // Validate the Bayesian analysis.
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
    static std::unique_ptr<UnivariateDistributionModel> require_non_null(
        std::unique_ptr<UnivariateDistributionModel> model) {
        if (model == nullptr) {
            throw std::invalid_argument("univariateDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // Owned model (deviation 2). Declared BEFORE bayesian_analysis_ so it is constructed first
    // (BayesianAnalysis stores a reference into it).
    std::unique_ptr<UnivariateDistributionModel> univariate_distribution_;
    bestfit::estimation::BayesianAnalysis bayesian_analysis_;
    ProbabilityOrdinates probability_ordinates_;

    // Result objects (C# nullable -> optional; deviation 3).
    std::optional<UncertaintyAnalysisResults> analysis_results_;
    std::optional<UncertaintyAnalysisResults> chronology_analysis_results_;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (deviation 4) and the chronology parent clone (deviation 5).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
    std::unique_ptr<UnivariateDistributionBase> chronology_parent_;
};

}  // namespace bestfit::analyses
