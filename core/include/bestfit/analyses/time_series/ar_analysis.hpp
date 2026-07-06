// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/TimeSeries/ARAnalysis.cs @ fc28c0c
//
// Bayesian MCMC estimation for an AutoRegressive AR(p) time-series model. A thin orchestrator
// (mechanical sibling of the Phase-8 UnivariateAnalysis and the D1 univariate-family analyses):
// it drives a BayesianAnalysis (Phase 4) over the model, then assembles the posterior forecast
// (mode/mean/credible-band curves + goodness-of-fit) into an UncertaintyAnalysisResults.
//
// UNLIKE the univariate-family analyses, this derives AnalysisBase ONLY -- the C# signature is
// literally `public class ARAnalysis : AnalysisBase` (ARAnalysis.cs:36); it does NOT implement
// IUnivariateAnalysis (no ProbabilityOrdinates / GetDistribution surface). It still exposes the
// same BayesianAnalysis + AnalysisResults members.
//
// The RESULT is built as a plain DTO (empty UncertaintyAnalysisResults ctor + direct field
// writes), NOT via the parent-distribution "compute ctor" the univariate analyses use: the
// forecast curves come from the time-series model's Predict() + a MersenneTwister-seeded posterior
// ensemble, not from a distribution's InverseCDF grid.
//
// DROPPED C# surface (mirrors the D1 severance list -- GUI/threading/serialization, no numerical
// content):
//   * both XML ctors (the XElement-deserializing ctor, ARAnalysis.cs:69) + `ToXElement`
//     (ARAnalysis.cs:621) -- XElement (de)serialization.
//   * the `Model_PropertyChanged` (ARAnalysis.cs:225) / `BayesianAnalysis_PropertyChanged`
//     (ARAnalysis.cs:258) handlers + `ReprocessOrClearForecast` (322) / `ReprocessIfEstimated`
//     re-derivation cadence -- INotifyPropertyChanged binding cascades. The reprocess decisions
//     they encode are exercised by calling the (public) reprocess methods directly.
//   * `CancellationTokenSource` + `CancelAnalysis` (ARAnalysis.cs:420), `SafeProgressReporter`/
//     `IProgress`, the `_reprocessGate` semaphore, and the `AnalysisStarting`/`AnalysisCompleted`
//     events. Every `RaisePropertyChange` call. `Debug.WriteLine`/swallowed-exception guards.
//   The C# `async Task RunAsync` ports to a synchronous `run()`; the `...Async` helpers port to
//   synchronous methods (every `Parallel.For` becomes a serial loop -- independent writes).
//
// DEVIATIONS (documented; C# governs):
//   1. OWNERSHIP. The C# AutoRegressive model is a GC reference type; here the analysis OWNS it via
//      `std::unique_ptr` (the ctor null-guard maps to the C# ArgumentNullException, ARAnalysis.cs:52).
//      `BayesianAnalysis` holds a `ModelBase&` into that owned model, so the analysis is
//      non-copyable / non-movable (a moved reference member cannot rebind).
//   2. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`; `analysis_results()`
//      returns a `const ...*` (null <=> empty optional).
//   3. CONFIDENCE-INTERVAL SHAPE. C# `ConfidenceIntervals` is `double[n, 3]`: column 0 is the time
//      index (== row index), columns 1/2 the lower/upper credible bounds. The ported
//      `UncertaintyAnalysisResults::confidence_intervals` is `std::vector<std::array<double, 2>>`,
//      so only [lower, upper] is stored; the redundant time-index column (implicitly the row index)
//      is dropped. No information is lost.
//   4. FORECAST RNG. `AutoRegressive::predict_components(..., seed)` draws its stochastic-forecast
//      noise from the ported MersenneTwister (the model already substitutes it for C#'s
//      System.Random -- see auto_regressive.hpp). C#'s `prng.NextIntegers(realz)` per-realization
//      seed draw is reproduced as `realz` sequential `MersenneTwister::next()` draws. Same-seed
//      reproducibility holds; the exact seeded forecast VALUES are not C#-reproducible (a P4-class
//      concern; there is no numeric oracle for this analysis).
//   5. RUN ERROR REPORTING. With events dropped, `run()` lets exceptions propagate (nothing to
//      route them to); `bayesian_analysis_.estimate()` already throws on an invalid configuration.
//   6. SIM-DEFAULTS GUARD IN run(). Per the D2 brief, `run()` applies the
//      `if (use_simulation_defaults) set_default_simulation_options()` /
//      `if (use_advanced_simulation_defaults) set_default_advanced_simulation_options()` guard
//      (ARAnalysis.cs:230-234, the property-change cascade this port drops) so the sampler is built
//      from current-model defaults, reproducing the effective C# state. Idempotent given the
//      BayesianAnalysis ctor already applied them.
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/time_series/auto_regressive.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"

namespace bestfit::analyses {

class ARAnalysis : public AnalysisBase {
   public:
    using AutoRegressive = bestfit::models::AutoRegressive;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using PointEstimateType = bestfit::estimation::PointEstimateType;

    // C# ctor `ARAnalysis(AutoRegressive)` (ARAnalysis.cs:50): builds the BayesianAnalysis over the
    // model. The C# `?? throw ArgumentNullException` maps to the null-guard here (deviation 1).
    explicit ARAnalysis(std::unique_ptr<AutoRegressive> auto_regressive)
        : auto_regressive_(require_non_null(std::move(auto_regressive))),
          bayesian_analysis_(*auto_regressive_) {}

    ~ARAnalysis() override = default;

    // Non-copyable / non-movable (deviation 1).
    ARAnalysis(const ARAnalysis&) = delete;
    ARAnalysis& operator=(const ARAnalysis&) = delete;
    ARAnalysis(ARAnalysis&&) = delete;
    ARAnalysis& operator=(ARAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `AutoRegressive` (ARAnalysis.cs:116).
    AutoRegressive& auto_regressive() { return *auto_regressive_; }
    const AutoRegressive& auto_regressive() const { return *auto_regressive_; }

    // C# `BayesianAnalysis` (ARAnalysis.cs:149).
    bestfit::estimation::BayesianAnalysis& bayesian_analysis() { return bayesian_analysis_; }
    const bestfit::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ForecastingTimeSteps` (ARAnalysis.cs:185): steps to forecast past the observed series.
    // The setter CLAMPS to [0, 100] (ARAnalysis.cs:190); the notify/reprocess side effects drop.
    int forecasting_time_steps() const { return forecasting_time_steps_; }
    void set_forecasting_time_steps(int value) {
        int clamped = std::max(0, std::min(100, value));
        if (forecasting_time_steps_ != clamped) forecasting_time_steps_ = clamped;
    }

    // C# `AnalysisResults` (ARAnalysis.cs:209): null until run() completes (deviation 2).
    const UncertaintyAnalysisResults* analysis_results() const {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (ARAnalysis.cs:287): clears the Bayesian fit + AnalysisResults and resets
    // IsEstimated. The RaisePropertyChange calls are dropped.
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        set_is_estimated(false);
    }

    // C# `ClearUncertaintyAnalysisResults` (ARAnalysis.cs:305): clears ONLY the forecast output;
    // the Bayesian fit + IsEstimated survive.
    void clear_uncertainty_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (ARAnalysis.cs:336), synchronous. Validate guard -> sim-defaults guard
    // (deviation 6) -> BayesianAnalysis.estimate() -> IF estimated: build the uncertainty results
    // -> IsEstimated mirrors the inner fit. (Cancellation/gate/events/progress dropped; deviation 5.)
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        // Sim-defaults guard (deviation 6; ARAnalysis.cs:230-234).
        if (bayesian_analysis_.use_simulation_defaults())
            bayesian_analysis_.set_default_simulation_options();
        if (bayesian_analysis_.use_advanced_simulation_defaults())
            bayesian_analysis_.set_default_advanced_simulation_options();

        bayesian_analysis_.estimate();

        if (bayesian_analysis_.is_estimated()) {
            create_uncertainty_analysis_results();
        }

        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateUncertaintyAnalysisResultsAsync` (ARAnalysis.cs:494), synchronous. Builds the mode
    // curve at MAP, then the posterior forecast ensemble (one seeded Predict per posterior sample)
    // and its mean + credible band, then updates the point-estimate scalars.
    void create_uncertainty_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        const auto& posterior = results.output;
        if (posterior.empty()) return;
        std::vector<double> map_values = results.map.values;

        int data_length = auto_regressive_->time_series().count();
        int forecast_steps_for_predict =
            (data_length - auto_regressive_->training_time_steps()) + forecasting_time_steps_;
        int n = data_length + forecasting_time_steps_;

        UncertaintyAnalysisResults uar;  // plain DTO (empty ctor)
        uar.mode_curve = auto_regressive_->predict_components(map_values, forecast_steps_for_predict).y;
        uar.mean_curve.assign(static_cast<std::size_t>(n), 0.0);
        uar.confidence_intervals.assign(static_cast<std::size_t>(n), {0.0, 0.0});

        // Per-realization seeds (deviation 4: NextIntegers(realz) -> realz sequential next()).
        bestfit::numerics::sampling::MersenneTwister prng(bayesian_analysis_.prng_seed());
        int realz = std::min(bayesian_analysis_.output_length(), static_cast<int>(posterior.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        std::vector<int> seeds(static_cast<std::size_t>(realz));
        for (int i = 0; i < realz; ++i) seeds[static_cast<std::size_t>(i)] = prng.next();

        // series[realz][n] (C# `new double[n, realz]`, column per realization; serial loop).
        std::vector<std::vector<double>> series(static_cast<std::size_t>(realz));
        for (int idx = 0; idx < realz; ++idx) {
            series[static_cast<std::size_t>(idx)] = auto_regressive_->predict_components(
                posterior[static_cast<std::size_t>(idx)].values, forecast_steps_for_predict,
                seeds[static_cast<std::size_t>(idx)]).y;
        }

        // Summary statistics per time step (deviation 3: [lower, upper] only).
        for (int t = 0; t < n; ++t) {
            std::vector<double> y(static_cast<std::size_t>(realz));
            for (int idx = 0; idx < realz; ++idx)
                y[static_cast<std::size_t>(idx)] = series[static_cast<std::size_t>(idx)][static_cast<std::size_t>(t)];
            std::sort(y.begin(), y.end());
            uar.mean_curve[static_cast<std::size_t>(t)] = bestfit::numerics::data::mean(y);
            uar.confidence_intervals[static_cast<std::size_t>(t)][0] =
                bestfit::numerics::data::percentile(y, alpha / 2.0, true);
            uar.confidence_intervals[static_cast<std::size_t>(t)][1] =
                bestfit::numerics::data::percentile(y, 1.0 - alpha / 2.0, true);
        }

        analysis_results_ = std::move(uar);
        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (ARAnalysis.cs:435), synchronous. Rebuilds the mode
    // curve at the selected point estimator, computes in-sample RMSE, and the AIC/BIC (at MAP) /
    // DIC scalars, then publishes the point-estimate parameters to the shared model (C# 480).
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() || !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        std::vector<double> parameters =
            bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean
                ? results.posterior_mean.values
                : results.map.values;

        int data_length = auto_regressive_->time_series().count();
        int forecast_steps_for_predict =
            (data_length - auto_regressive_->training_time_steps()) + forecasting_time_steps_;
        analysis_results_->mode_curve =
            auto_regressive_->predict_components(parameters, forecast_steps_for_predict).y;

        // In-sample RMSE (observed vs predicted over the observed range; C# 467-469).
        std::vector<double> true_values = auto_regressive_->time_series().values_to_array();
        std::vector<double> model_values(
            analysis_results_->mode_curve.begin(),
            analysis_results_->mode_curve.begin() + std::min<std::size_t>(
                                                        static_cast<std::size_t>(data_length),
                                                        analysis_results_->mode_curve.size()));
        double rmse = GoodnessOfFit::rmse(true_values, model_values);

        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double map_log_lh = auto_regressive_->log_likelihood(map_values);
        analysis_results_->aic =
            GoodnessOfFit::aic(auto_regressive_->number_of_parameters(), map_log_lh);
        analysis_results_->bic =
            GoodnessOfFit::bic(data_length, auto_regressive_->number_of_parameters(), map_log_lh);
        analysis_results_->dic = bayesian_analysis_.dic();
        analysis_results_->rmse = rmse;

        // Publish the point-estimate parameters to the shared model (C# 480).
        auto_regressive_->set_parameter_values(parameters);
    }

    // C# `Validate` (ARAnalysis.cs:567): model validation + the ForecastingTimeSteps [0,100] range
    // + Bayesian-analysis validation.
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        bestfit::models::ValidationResult model_valid = auto_regressive_->validate();
        if (!model_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              model_valid.validation_messages.begin(),
                                              model_valid.validation_messages.end());
        }

        if (forecasting_time_steps_ < 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Forecasting time steps must be non-negative.");
        }
        if (forecasting_time_steps_ > 100) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Forecasting time steps must not exceed 100.");
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
    static std::unique_ptr<AutoRegressive> require_non_null(std::unique_ptr<AutoRegressive> model) {
        if (model == nullptr) {
            throw std::invalid_argument("autoRegressive");  // C# ArgumentNullException
        }
        return model;
    }

    // Owned model (deviation 1). Declared BEFORE bayesian_analysis_ so it is constructed first.
    std::unique_ptr<AutoRegressive> auto_regressive_;
    bestfit::estimation::BayesianAnalysis bayesian_analysis_;
    int forecasting_time_steps_ = 0;

    // Result object (C# nullable -> optional; deviation 2).
    std::optional<UncertaintyAnalysisResults> analysis_results_;
};

}  // namespace bestfit::analyses
