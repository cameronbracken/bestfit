// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/RatingCurve/RatingCurveAnalysis.cs @ fc28c0c
//
// A Bayesian MCMC frequency analysis over a stage-discharge RatingCurve model (X3): drives a
// BayesianAnalysis (Phase 4) over the model, then assembles the predicted-discharge mode/mean
// curves + credible band across a stage grid into an UncertaintyAnalysisResults (A2). A standard
// AnalysisBase clone, structurally near-identical to the A5 UnivariateAnalysis template.
//
// LOAD-BEARING compute sequence ported; WPF/event/gate/XML plumbing DROPPED per the A5 precedent.
// Specifically SKIPPED (each is GUI/threading/serialization with no numerical content):
//   * the XML ctor (C# 69-111) + `ToXElement` (751) -- XElement (de)serialization.
//   * every `*_PropertyChanged` handler (`Model_PropertyChanged` 294, `BayesianAnalysis_-
//     PropertyChanged` 335) and the INotifyPropertyChanged/INotifyCollectionChanged cascades they
//     drive. The reprocess-vs-clear decisions they encode are exercised HERE by calling the (public)
//     grid setters + reprocess methods directly, exactly as the C# grid-reprocess tests do.
//   * the reprocess gate (`_reprocessGate.WaitAsync/Release`), `CancellationTokenSource` + cancel
//     (`CancelAnalysis`, 521), `SafeProgressReporter`, and the `AnalysisStarting`/
//     `AnalysisCompleted` events + `OnAnalysisStarting/Completed` -- run-lifecycle plumbing.
//   * `RaisePropertyChange` calls throughout -- no notification system in this port.
//
// The C# `async Task RunAsync` ports to a synchronous `run()`; the C# `...Async` helper methods
// (`CreateUncertaintyAnalysisResultsAsync`, `UpdatePointEstimateResultsAsync`) port to synchronous
// methods (every `Parallel.For` becomes a serial loop -- the loop bodies are independent writes /
// order-independent reductions, so the result is numerically identical).
//
// DEVIATIONS (documented):
//   1. OWNERSHIP. The C# `RatingCurve` model is a GC reference type; here the analysis OWNS the
//      model via `std::unique_ptr` (the ctor's null-guard maps to the C# ArgumentNullException).
//      `BayesianAnalysis` holds a `ModelBase&` into that owned model, so the analysis is
//      non-copyable / non-movable (a moved reference member cannot rebind). Mirrors A5 deviation 2.
//   2. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`. `analysis_results()`
//      returns a `const ...*` (null <=> empty optional). RatingCurveAnalysis is NOT an
//      IBayesianAnalysis here (the C# class extends only AnalysisBase), so this is a plain accessor.
//   3. CONFIDENCE-INTERVAL SHAPE. The C# `ConfidenceIntervals` is `double[n, 3]` and packs the stage
//      grid into column 0 (`[i,0] = bins[i]`), with the lower/upper percentiles in columns 1/2. The
//      ported UncertaintyAnalysisResults DTO (oracle-locked) stores `std::array<double, 2>` per row,
//      so the port drops the redundant stage column and stores `{lower, upper}`. The stage grid is
//      the deterministic Stratify grid of `min_stage_`/`max_stage_`/`stage_bins_`; any consumer
//      recomputes it identically (see `build_stage_bins`). No numerical content is lost.
//   4. SEEDED PREDICT RNG. `create_uncertainty_analysis_results` calls `RatingCurve::predict(params,
//      stage, seed)`, whose stochastic-noise draw substitutes the ported MersenneTwister for the C#
//      `System.Random` (a documented RatingCurve-model deviation, see rating_curve.hpp). Same-seed
//      reproducibility holds; the exact seeded VALUES are not C#-reproducible (flagged for X12).
//   5. RUN ERROR REPORTING. C#'s `RunAsync` wraps the compute body in try/catch that reports
//      failures through `AnalysisCompleted`. With events dropped, `run()` lets exceptions propagate
//      to the caller instead (there is nothing to report them to). Mirrors A5 deviation 6.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/rating_curve/rating_curve.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/data/time_series/time_series.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/sampling/stratification_options.hpp"
#include "corehydro/numerics/sampling/stratify.hpp"

namespace corehydro::analyses {

class RatingCurveAnalysis : public AnalysisBase {
   public:
    using RatingCurve = corehydro::models::RatingCurve;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using PointEstimateType = corehydro::estimation::PointEstimateType;

    // C# ctor `RatingCurveAnalysis(RatingCurve)` (C# 47): builds the BayesianAnalysis over the
    // model, then SetDefaultStageBins(). The C# `?? throw ArgumentNullException` maps to the
    // null-guard here (see deviation 1).
    explicit RatingCurveAnalysis(std::unique_ptr<RatingCurve> rating_curve)
        : rating_curve_(require_non_null(std::move(rating_curve))),
          bayesian_analysis_(*rating_curve_) {
        set_default_stage_bins();
    }

    ~RatingCurveAnalysis() override = default;

    // Non-copyable / non-movable (deviation 1).
    RatingCurveAnalysis(const RatingCurveAnalysis&) = delete;
    RatingCurveAnalysis& operator=(const RatingCurveAnalysis&) = delete;
    RatingCurveAnalysis(RatingCurveAnalysis&&) = delete;
    RatingCurveAnalysis& operator=(RatingCurveAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `RatingCurve` (C# 134): the model being estimated.
    RatingCurve& rating_curve() { return *rating_curve_; }
    const RatingCurve& rating_curve() const { return *rating_curve_; }

    // C# `BayesianAnalysis` (C# 167). Plain accessor (no IBayesianAnalysis interface here).
    corehydro::estimation::BayesianAnalysis& bayesian_analysis() { return bayesian_analysis_; }
    const corehydro::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `MinStage` (C# 195): the setter's change-guard + grid-reprocess side effect are kept;
    // RaisePropertyChange is dropped.
    double min_stage() const { return min_stage_; }
    void set_min_stage(double value) {
        if (min_stage_ != value) {
            min_stage_ = value;
            reprocess_or_clear_uncertainty_grid();
        }
    }

    // C# `MaxStage` (C# 216).
    double max_stage() const { return max_stage_; }
    void set_max_stage(double value) {
        if (max_stage_ != value) {
            max_stage_ = value;
            reprocess_or_clear_uncertainty_grid();
        }
    }

    // C# `StageBins` (C# 237).
    int stage_bins() const { return stage_bins_; }
    void set_stage_bins(int value) {
        if (stage_bins_ != value) {
            stage_bins_ = value;
            reprocess_or_clear_uncertainty_grid();
        }
    }

    // C# `UseDefaultStageBins` (C# 258): on true, recompute the data-derived default bins.
    bool use_default_stage_bins() const { return use_default_stage_bins_; }
    void set_use_default_stage_bins(bool value) {
        if (use_default_stage_bins_ != value) {
            use_default_stage_bins_ = value;
            if (use_default_stage_bins_) set_default_stage_bins();
        }
    }

    // C# `AnalysisResults` (C# 282): the uncertainty results (null until estimated).
    const UncertaintyAnalysisResults* analysis_results() const {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 389): clears the Bayesian fit + the result object and resets
    // IsEstimated. The RaisePropertyChange calls are dropped.
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        set_is_estimated(false);
    }

    // C# `ClearUncertaintyAnalysisResults` (C# 407): clears ONLY the uncertainty grid output;
    // the fit and IsEstimated survive.
    void clear_uncertainty_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (C# 440), synchronous. Validate guard -> clear -> BayesianAnalysis.estimate()
    // -> IF estimated: create uncertainty results -> IsEstimated mirrors the inner fit.
    // (Cancellation/gate/events/progress dropped; see the file header. Deviation 5: exceptions
    // propagate instead of routing through AnalysisCompleted.)
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        // Run the Bayesian analysis.
        bayesian_analysis_.estimate();

        // Post-process.
        if (bayesian_analysis_.is_estimated()) {
            create_uncertainty_analysis_results();
        }

        // Mirror the inner fit's success state (C# 490).
        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateUncertaintyAnalysisResultsAsync` (C# 609), synchronous. Builds the mode curve
    // (MAP prediction at each stage bin), the posterior mean curve, and the credible band from
    // the seeded per-sample predictions, then updates the point-estimate scalars.
    void create_uncertainty_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        const auto& output = results.output;
        if (output.empty()) return;

        std::vector<double> parameters = results.map.values;
        int prng_seed = bayesian_analysis_.prng_seed();
        int realz =
            std::min(bayesian_analysis_.output_length(), static_cast<int>(output.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        int n = stage_bins_;

        UncertaintyAnalysisResults r;
        r.mode_curve.assign(static_cast<std::size_t>(n), 0.0);
        r.mean_curve.assign(static_cast<std::size_t>(n), 0.0);
        r.confidence_intervals.assign(static_cast<std::size_t>(n), {0.0, 0.0});

        // Stratify stage bins (C# 648-650).
        std::vector<double> bins = build_stage_bins(min_stage_, max_stage_, n);

        // Seed stream for the per-sample predictions (C# 652-653). MersenneTwister::next() ==
        // the C# `NextIntegers` (Random.Next) fill; seeded-predict RNG is deviation 4.
        corehydro::numerics::sampling::MersenneTwister prng(static_cast<std::uint32_t>(prng_seed));
        std::vector<int> seeds(static_cast<std::size_t>(realz));
        for (int i = 0; i < realz; ++i) seeds[static_cast<std::size_t>(i)] = prng.next();

        for (int i = 0; i < n; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            r.mode_curve[ui] = rating_curve_->predict(parameters, bins[ui]);

            std::vector<double> q(static_cast<std::size_t>(realz));
            for (int idx = 0; idx < realz; ++idx) {
                q[static_cast<std::size_t>(idx)] = rating_curve_->predict(
                    output[static_cast<std::size_t>(idx)].values, bins[ui],
                    seeds[static_cast<std::size_t>(idx)]);
            }
            r.mean_curve[ui] = corehydro::numerics::data::mean(q);
            std::sort(q.begin(), q.end());
            // Deviation 3: column 0 = lower percentile, column 1 = upper (C# packs bins[i] into
            // its column 0 and shifts the percentiles to columns 1/2; the 2-col DTO drops it).
            r.confidence_intervals[ui][0] =
                corehydro::numerics::data::percentile(q, alpha / 2.0, true);
            r.confidence_intervals[ui][1] =
                corehydro::numerics::data::percentile(q, 1.0 - alpha / 2.0, true);
        }

        analysis_results_ = std::move(r);
        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 536), synchronous. Rebuilds the mode curve at the
    // selected point estimator, computes RMSE over the date-aligned observations, and writes the
    // AIC/BIC (at MAP) + DIC scalars. Publishes the point-estimate parameters to the model.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        // Pick the point estimator (C# 546-548).
        std::vector<double> parameters =
            bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean
                ? results.posterior_mean.values
                : results.map.values;

        // Mode curve at the stratified stage bins (C# 557-566).
        int n = stage_bins_;
        analysis_results_->mode_curve.assign(static_cast<std::size_t>(n), 0.0);
        std::vector<double> bins = build_stage_bins(min_stage_, max_stage_, n);
        for (int i = 0; i < n; ++i)
            analysis_results_->mode_curve[static_cast<std::size_t>(i)] =
                rating_curve_->predict(parameters, bins[static_cast<std::size_t>(i)]);

        // RMSE over the date-aligned observations (C# 571-579).
        const auto& aligned = rating_curve_->get_aligned_observations();
        std::vector<double> true_values;
        true_values.reserve(aligned.size());
        std::vector<double> model_values(aligned.size());
        for (std::size_t i = 0; i < aligned.size(); ++i) {
            true_values.push_back(aligned[i].discharge);
            model_values[i] = rating_curve_->predict(parameters, aligned[i].stage);
        }
        double rmse = GoodnessOfFit::rmse(true_values, model_values);

        // AIC/BIC at the MAP (full log-likelihood, data + prior) (C# 584-586).
        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double map_log_lh = rating_curve_->log_likelihood(map_values);
        analysis_results_->aic =
            GoodnessOfFit::aic(rating_curve_->number_of_parameters(), map_log_lh);
        analysis_results_->bic = GoodnessOfFit::bic(static_cast<int>(aligned.size()),
                                                    rating_curve_->number_of_parameters(),
                                                    map_log_lh);
        analysis_results_->dic = bayesian_analysis_.dic();
        analysis_results_->rmse = rmse;

        // Publish the point-estimate parameters to the shared model (C# 595).
        rating_curve_->set_parameter_values(parameters);
    }

    // C# `Validate` (C# 687): rating-curve model + stage-bin bounds + Bayesian-analysis
    // validations. const per the A4 IAnalysis contract (the C# body only reads state).
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        // Validate the rating curve model.
        corehydro::models::ValidationResult model_valid = rating_curve_->validate();
        if (!model_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              model_valid.validation_messages.begin(),
                                              model_valid.validation_messages.end());
        }

        // Validate stage bins (C# 701-720).
        if (std::isnan(min_stage_)) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Minimum stage value is invalid (NaN).");
        }
        if (std::isnan(max_stage_)) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Maximum stage value is invalid (NaN).");
        }
        if (min_stage_ >= max_stage_) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Minimum stage must be less than maximum stage.");
        }
        if (stage_bins_ < 10 || stage_bins_ > 1000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Number of stage bins must be between 10 and 1,000.");
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
    static std::unique_ptr<RatingCurve> require_non_null(std::unique_ptr<RatingCurve> model) {
        if (model == nullptr) {
            throw std::invalid_argument("ratingCurve");  // C# ArgumentNullException
        }
        return model;
    }

    // C# `SetDefaultStageBins` (C# 364): data-range +/- 10% padding, 100 bins; fallback 0..100.
    void set_default_stage_bins() {
        if (rating_curve_ && rating_curve_->has_stage_data() &&
            rating_curve_->stage_data().count() > 0) {
            double min = rating_curve_->stage_data().min_value();
            double max = stage_series_max(rating_curve_->stage_data());
            double range = max - min;
            min_stage_ = min - 0.1 * range;
            max_stage_ = max + 0.1 * range;
        } else {
            min_stage_ = 0;
            max_stage_ = 100;
        }
        stage_bins_ = 100;
    }

    // C# `ReprocessOrClearUncertaintyGrid` (C# 425): reprocess if the analysis is estimated and
    // the grid is valid; else clear the grid output. The fit is preserved either way.
    void reprocess_or_clear_uncertainty_grid() {
        if (!is_estimated()) return;

        bool valid = !std::isnan(min_stage_) && !std::isnan(max_stage_) &&
                     min_stage_ < max_stage_ && stage_bins_ >= 10 && stage_bins_ <= 1000;

        if (valid)
            create_uncertainty_analysis_results();
        else
            clear_uncertainty_analysis_results();
    }

    // C# `Stratify.XValues(new StratificationOptions(min, max, n - 1))` then LowerBounds + the
    // final UpperBound (n grid points).
    static std::vector<double> build_stage_bins(double min, double max, int n) {
        std::vector<double> bins;
        auto x_bins = corehydro::numerics::sampling::Stratify::XValues(
            corehydro::numerics::sampling::StratificationOptions(min, max, n - 1));
        bins.reserve(x_bins.size() + 1);
        for (const auto& b : x_bins) bins.push_back(b.lower_bound());
        if (!x_bins.empty()) bins.push_back(x_bins.back().upper_bound());
        return bins;
    }

    // Max stage value skipping NaN (C# TimeSeries.MaxValue; the ported adapter omits it).
    static double stage_series_max(const corehydro::numerics::data::TimeSeries& series) {
        double max = std::numeric_limits<double>::lowest();
        for (int i = 0; i < series.count(); ++i) {
            double v = series[i].value();
            if (!std::isnan(v) && v > max) max = v;
        }
        return max;
    }

    // Owned model (deviation 1). Declared BEFORE bayesian_analysis_ so it is constructed first
    // (BayesianAnalysis stores a reference into it).
    std::unique_ptr<RatingCurve> rating_curve_;
    corehydro::estimation::BayesianAnalysis bayesian_analysis_;

    // Stage-bin configuration (C# fields, 119-122).
    double min_stage_ = 0;
    double max_stage_ = 100;
    int stage_bins_ = 100;
    bool use_default_stage_bins_ = true;

    // Result object (C# nullable -> optional; deviation 2).
    std::optional<UncertaintyAnalysisResults> analysis_results_;
};

}  // namespace corehydro::analyses
