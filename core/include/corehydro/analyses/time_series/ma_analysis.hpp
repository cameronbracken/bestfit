// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/TimeSeries/MAAnalysis.cs @ fc28c0c
//
// Bayesian MCMC estimation for a MovingAverage MA(q) time-series model. A mechanical sibling of
// ARAnalysis (see ar_analysis.hpp for the full architecture + deviation notes) wrapping the
// MovingAverage model instead of AutoRegressive. The MAAnalysis.cs body is byte-for-byte the
// ARAnalysis.cs body with `AutoRegressive` -> `MovingAverage` (the in-sample RMSE, forecast
// horizon, AIC/BIC/DIC block, ForecastingTimeSteps [0,100] clamp, and the create/update reprocess
// pair are identical), so the same dropped/deviation list applies verbatim:
//   * DROPPED: both XML ctors + ToXElement; the *_PropertyChanged handlers + Reprocess cadence;
//     CancellationTokenSource/CancelAnalysis; SafeProgressReporter; the _reprocessGate; the
//     AnalysisStarting/Completed events; every RaisePropertyChange; Debug.WriteLine guards.
//     `async Task RunAsync` -> synchronous `run()`; each `...Async` helper -> serial method.
//   * DEVIATIONS: (1) owned-model unique_ptr + non-copyable/non-movable (BayesianAnalysis holds a
//     ModelBase& into it); (2) `UncertaintyAnalysisResults?` -> std::optional; (3) confidence
//     intervals stored as [lower, upper] (the C# `double[n,3]` redundant time-index column is
//     dropped); (4) forecast RNG via the ported MersenneTwister (NextIntegers(realz) -> realz
//     sequential next() draws), exact seeded VALUES not C#-reproducible; (5) run() lets exceptions
//     propagate; (6) run() applies the sim-defaults guard (MAAnalysis.cs:231-235) per the D2 brief.
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/time_series/moving_average.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::analyses {

class MAAnalysis : public AnalysisBase {
   public:
    using MovingAverage = corehydro::models::MovingAverage;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using PointEstimateType = corehydro::estimation::PointEstimateType;

    // C# ctor `MAAnalysis(MovingAverage)` (MAAnalysis.cs:50).
    explicit MAAnalysis(std::unique_ptr<MovingAverage> moving_average)
        : moving_average_(require_non_null(std::move(moving_average))),
          bayesian_analysis_(*moving_average_) {}

    ~MAAnalysis() override = default;

    MAAnalysis(const MAAnalysis&) = delete;
    MAAnalysis& operator=(const MAAnalysis&) = delete;
    MAAnalysis(MAAnalysis&&) = delete;
    MAAnalysis& operator=(MAAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `MovingAverage` (MAAnalysis.cs:116).
    MovingAverage& moving_average() { return *moving_average_; }
    const MovingAverage& moving_average() const { return *moving_average_; }

    // C# `BayesianAnalysis` (MAAnalysis.cs:149).
    corehydro::estimation::BayesianAnalysis& bayesian_analysis() { return bayesian_analysis_; }
    const corehydro::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ForecastingTimeSteps` (MAAnalysis.cs:186); the setter clamps to [0, 100] (:191).
    int forecasting_time_steps() const { return forecasting_time_steps_; }
    void set_forecasting_time_steps(int value) {
        int clamped = std::max(0, std::min(100, value));
        if (forecasting_time_steps_ != clamped) forecasting_time_steps_ = clamped;
    }

    // C# `AnalysisResults` (MAAnalysis.cs:210): null until run() completes.
    const UncertaintyAnalysisResults* analysis_results() const {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (MAAnalysis.cs:288).
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        set_is_estimated(false);
    }

    // C# `ClearUncertaintyAnalysisResults` (MAAnalysis.cs:306).
    void clear_uncertainty_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (MAAnalysis.cs:334), synchronous.
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

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

    // C# `CreateUncertaintyAnalysisResultsAsync` (MAAnalysis.cs:492), synchronous.
    void create_uncertainty_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        const auto& posterior = results.output;
        if (posterior.empty()) return;
        std::vector<double> map_values = results.map.values;

        int data_length = moving_average_->time_series().count();
        int forecast_steps_for_predict =
            (data_length - moving_average_->training_time_steps()) + forecasting_time_steps_;
        int n = data_length + forecasting_time_steps_;

        UncertaintyAnalysisResults uar;
        uar.mode_curve = moving_average_->predict_components(map_values, forecast_steps_for_predict).y;
        uar.mean_curve.assign(static_cast<std::size_t>(n), 0.0);
        uar.confidence_intervals.assign(static_cast<std::size_t>(n), {0.0, 0.0});

        corehydro::numerics::sampling::MersenneTwister prng(bayesian_analysis_.prng_seed());
        int realz = std::min(bayesian_analysis_.output_length(), static_cast<int>(posterior.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        std::vector<int> seeds(static_cast<std::size_t>(realz));
        for (int i = 0; i < realz; ++i) seeds[static_cast<std::size_t>(i)] = prng.next();

        std::vector<std::vector<double>> series(static_cast<std::size_t>(realz));
        for (int idx = 0; idx < realz; ++idx) {
            series[static_cast<std::size_t>(idx)] = moving_average_->predict_components(
                posterior[static_cast<std::size_t>(idx)].values, forecast_steps_for_predict,
                seeds[static_cast<std::size_t>(idx)]).y;
        }

        for (int t = 0; t < n; ++t) {
            std::vector<double> y(static_cast<std::size_t>(realz));
            for (int idx = 0; idx < realz; ++idx)
                y[static_cast<std::size_t>(idx)] = series[static_cast<std::size_t>(idx)][static_cast<std::size_t>(t)];
            std::sort(y.begin(), y.end());
            uar.mean_curve[static_cast<std::size_t>(t)] = corehydro::numerics::data::mean(y);
            uar.confidence_intervals[static_cast<std::size_t>(t)][0] =
                corehydro::numerics::data::percentile(y, alpha / 2.0, true);
            uar.confidence_intervals[static_cast<std::size_t>(t)][1] =
                corehydro::numerics::data::percentile(y, 1.0 - alpha / 2.0, true);
        }

        analysis_results_ = std::move(uar);
        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (MAAnalysis.cs:433), synchronous.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() || !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        std::vector<double> parameters =
            bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean
                ? results.posterior_mean.values
                : results.map.values;

        int data_length = moving_average_->time_series().count();
        int forecast_steps_for_predict =
            (data_length - moving_average_->training_time_steps()) + forecasting_time_steps_;
        analysis_results_->mode_curve =
            moving_average_->predict_components(parameters, forecast_steps_for_predict).y;

        std::vector<double> true_values = moving_average_->time_series().values_to_array();
        std::vector<double> model_values(
            analysis_results_->mode_curve.begin(),
            analysis_results_->mode_curve.begin() + std::min<std::size_t>(
                                                        static_cast<std::size_t>(data_length),
                                                        analysis_results_->mode_curve.size()));
        double rmse = GoodnessOfFit::rmse(true_values, model_values);

        std::vector<double> map_values = results.map.values;
        double map_log_lh = moving_average_->log_likelihood(map_values);
        analysis_results_->aic =
            GoodnessOfFit::aic(moving_average_->number_of_parameters(), map_log_lh);
        analysis_results_->bic =
            GoodnessOfFit::bic(data_length, moving_average_->number_of_parameters(), map_log_lh);
        analysis_results_->dic = bayesian_analysis_.dic();
        analysis_results_->rmse = rmse;

        moving_average_->set_parameter_values(parameters);
    }

    // C# `Validate` (MAAnalysis.cs:565).
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        corehydro::models::ValidationResult model_valid = moving_average_->validate();
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
    static std::unique_ptr<MovingAverage> require_non_null(std::unique_ptr<MovingAverage> model) {
        if (model == nullptr) {
            throw std::invalid_argument("movingAverage");  // C# ArgumentNullException
        }
        return model;
    }

    std::unique_ptr<MovingAverage> moving_average_;
    corehydro::estimation::BayesianAnalysis bayesian_analysis_;
    int forecasting_time_steps_ = 0;
    std::optional<UncertaintyAnalysisResults> analysis_results_;
};

}  // namespace corehydro::analyses
