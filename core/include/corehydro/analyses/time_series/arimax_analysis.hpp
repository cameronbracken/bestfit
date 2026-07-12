// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/TimeSeries/ARIMAXAnalysis.cs @ fc28c0c
//
// Bayesian MCMC estimation for an ARIMAX (ARIMA + eXogenous covariates) time-series model. A
// sibling of ARAnalysis (see ar_analysis.hpp for the shared architecture + deviation notes)
// wrapping the ARIMAX model. The orchestration shape is identical, with three ARIMAX-specific
// differences the C# governs (verified against ARIMAXAnalysis.cs):
//   * Predict takes the parameter array explicitly and returns the `.Y` component
//     (ARIMAXAnalysis.cs:482/547) -- `predict_components(params, steps, seed).y` here.
//   * The in-sample RMSE compares over the TRAINING window only
//     (`Subset(0, TrainingTimeSteps-1)`, ARIMAXAnalysis.cs:486-487), and BIC uses
//     `TrainingTimeSteps` as the sample size (ARIMAXAnalysis.cs:493) -- ARAnalysis/MA/ARIMA use the
//     full observed range / dataLength.
//   * Validate adds a covariate-length check: with `CovariateExtension == None` and covariates
//     present, each covariate must cover `TimeSeries.Count + ForecastingTimeSteps` observations,
//     else the analysis is invalid (ARIMAXAnalysis.cs:614-632).
//
// DROPPED C# surface (mirrors ARAnalysis; GUI/threading/serialization, no numerical content):
//   * both XML ctors (incl. the MCMCResults/UncertaintyAnalysisResults restore overload,
//     ARIMAXAnalysis.cs:73) + ToXElement (:663); the *_PropertyChanged handlers (incl. the
//     `CovariateExtension` reprocess branch, :268) + Reprocess cadence; CancellationTokenSource/
//     CancelAnalysis; SafeProgressReporter; the _reprocessGate; the AnalysisStarting/Completed
//     events; every RaisePropertyChange; Debug.WriteLine guards. `async Task RunAsync` ->
//     synchronous `run()`; each `...Async` helper -> serial method.
//
// DEVIATIONS (documented; C# governs):
//   1. OWNERSHIP -- owned-model unique_ptr + non-copyable/non-movable (BayesianAnalysis holds a
//      ModelBase& into it); ctor null-guard maps to the C# ArgumentNullException.
//   2. `UncertaintyAnalysisResults?` -> std::optional; `analysis_results()` is a nullable pointer.
//   3. CONFIDENCE-INTERVAL SHAPE -- [lower, upper] only; the C# `double[n,3]` redundant time-index
//      column (== row index) is dropped.
//   4. FORECAST RNG -- via the ported MersenneTwister (NextIntegers(realz) -> realz sequential
//      next() draws); exact seeded forecast VALUES not C#-reproducible.
//   5. run() lets exceptions propagate (no AnalysisCompleted sink).
//   6. run() applies the sim-defaults guard (ARIMAXAnalysis.cs:244-248) per the D2 brief.
//   7. COVARIATE FORECAST-TAIL EXTENSION severed (Phase-7a documented omission carried forward):
//      the ported `ARIMAX::predict_components` THROWS when a forecast horizon needs covariate
//      values beyond the observed covariate length under CovariateExtension = BlockBootstrap/KNN
//      (deferred with the heavy TimeSeries container -- see arimax.hpp). With covariates present
//      and ForecastingTimeSteps > 0, run()'s forecast ensemble propagates that throw; the analysis
//      Validate() already surfaces the None-extension case as invalid. Fit-only (horizon 0) and the
//      no-covariate path are unaffected.
#pragma once
#include <algorithm>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/time_series/arimax.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::analyses {

class ARIMAXAnalysis : public AnalysisBase {
   public:
    using ARIMAX = corehydro::models::ARIMAX;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using PointEstimateType = corehydro::estimation::PointEstimateType;

    // C# ctor `ARIMAXAnalysis(ARIMAX)` (ARIMAXAnalysis.cs:52).
    explicit ARIMAXAnalysis(std::unique_ptr<ARIMAX> armax)
        : arimax_(require_non_null(std::move(armax))), bayesian_analysis_(*arimax_) {}

    ~ARIMAXAnalysis() override = default;

    ARIMAXAnalysis(const ARIMAXAnalysis&) = delete;
    ARIMAXAnalysis& operator=(const ARIMAXAnalysis&) = delete;
    ARIMAXAnalysis(ARIMAXAnalysis&&) = delete;
    ARIMAXAnalysis& operator=(ARIMAXAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `ARIMAX` (ARIMAXAnalysis.cs:129).
    ARIMAX& arimax() { return *arimax_; }
    const ARIMAX& arimax() const { return *arimax_; }

    // C# `BayesianAnalysis` (ARIMAXAnalysis.cs:162).
    corehydro::estimation::BayesianAnalysis& bayesian_analysis() { return bayesian_analysis_; }
    const corehydro::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ForecastingTimeSteps` (ARIMAXAnalysis.cs:198); the setter clamps to [0, 100] (:203).
    int forecasting_time_steps() const { return forecasting_time_steps_; }
    void set_forecasting_time_steps(int value) {
        int clamped = std::max(0, std::min(100, value));
        if (forecasting_time_steps_ != clamped) forecasting_time_steps_ = clamped;
    }

    // C# `AnalysisResults` (ARIMAXAnalysis.cs:222): null until run() completes.
    const UncertaintyAnalysisResults* analysis_results() const {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (ARIMAXAnalysis.cs:316).
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        set_is_estimated(false);
    }

    // C# `ClearUncertaintyAnalysisResults` (ARIMAXAnalysis.cs:334).
    void clear_uncertainty_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (ARIMAXAnalysis.cs:362), synchronous.
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

    // C# `CreateUncertaintyAnalysisResultsAsync` (ARIMAXAnalysis.cs:513), synchronous.
    void create_uncertainty_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        const auto& posterior = results.output;
        if (posterior.empty()) return;
        std::vector<double> map_values = results.map.values;

        int data_length = arimax_->time_series().count();
        int forecast_steps_for_predict =
            (data_length - arimax_->training_time_steps()) + forecasting_time_steps_;
        int n = data_length + forecasting_time_steps_;

        UncertaintyAnalysisResults uar;
        uar.mode_curve = arimax_->predict_components(map_values, forecast_steps_for_predict).y;
        uar.mean_curve.assign(static_cast<std::size_t>(n), 0.0);
        uar.confidence_intervals.assign(static_cast<std::size_t>(n), {0.0, 0.0});

        corehydro::numerics::sampling::MersenneTwister prng(bayesian_analysis_.prng_seed());
        int realz = std::min(bayesian_analysis_.output_length(), static_cast<int>(posterior.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        std::vector<int> seeds(static_cast<std::size_t>(realz));
        for (int i = 0; i < realz; ++i) seeds[static_cast<std::size_t>(i)] = prng.next();

        std::vector<std::vector<double>> series(static_cast<std::size_t>(realz));
        for (int idx = 0; idx < realz; ++idx) {
            series[static_cast<std::size_t>(idx)] = arimax_->predict_components(
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

    // C# `UpdatePointEstimateResultsAsync` (ARIMAXAnalysis.cs:461), synchronous. RMSE/BIC use the
    // TRAINING window (not the full observed range) -- see the file header.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() || !analysis_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        std::vector<double> parameters =
            bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean
                ? results.posterior_mean.values
                : results.map.values;

        int data_length = arimax_->time_series().count();
        int training = arimax_->training_time_steps();
        int forecast_steps_for_predict = (data_length - training) + forecasting_time_steps_;
        analysis_results_->mode_curve =
            arimax_->predict_components(parameters, forecast_steps_for_predict).y;

        // In-sample RMSE over the TRAINING window (C# Subset(0, TrainingTimeSteps-1)).
        std::vector<double> all_true = arimax_->time_series().values_to_array();
        std::size_t take = std::min<std::size_t>(
            static_cast<std::size_t>(std::max(0, training)),
            std::min(all_true.size(), analysis_results_->mode_curve.size()));
        std::vector<double> true_values(all_true.begin(), all_true.begin() + take);
        std::vector<double> model_values(analysis_results_->mode_curve.begin(),
                                         analysis_results_->mode_curve.begin() + take);
        double rmse = GoodnessOfFit::rmse(true_values, model_values);

        std::vector<double> map_values = results.map.values;
        double map_log_lh = arimax_->log_likelihood(map_values);
        analysis_results_->aic = GoodnessOfFit::aic(arimax_->number_of_parameters(), map_log_lh);
        analysis_results_->bic =
            GoodnessOfFit::bic(training, arimax_->number_of_parameters(), map_log_lh);
        analysis_results_->dic = bayesian_analysis_.dic();
        analysis_results_->rmse = rmse;

        arimax_->set_parameter_values(parameters);
    }

    // C# `Validate` (ARIMAXAnalysis.cs:583): model validation + ForecastingTimeSteps [0,100] +
    // the CovariateExtension = None covariate-length check + Bayesian-analysis validation.
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        corehydro::models::ValidationResult model_valid = arimax_->validate();
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

        // CovariateExtension = None requires covariates already long enough to cover the horizon
        // (ARIMAXAnalysis.cs:614-632).
        if (arimax_->has_covariates() && !arimax_->covariates().empty() &&
            arimax_->covariate_extension() == ARIMAX::CovariateExtensionMethod::None &&
            arimax_->has_time_series()) {
            int needed_length = arimax_->time_series().count() + forecasting_time_steps_;
            const auto& covs = arimax_->covariates();
            for (std::size_t i = 0; i < covs.size(); ++i) {
                if (covs[i].count() < needed_length) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Covariate " + std::to_string(i + 1) + " has " +
                        std::to_string(covs[i].count()) + " observations; with CovariateExtension = "
                        "None and ForecastingTimeSteps = " + std::to_string(forecasting_time_steps_) +
                        ", " + std::to_string(needed_length) +
                        " are required. Set CovariateExtension to BlockBootstrap or KNN, extend the "
                        "covariate, or set ForecastingTimeSteps = 0.");
                    break;
                }
            }
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
    static std::unique_ptr<ARIMAX> require_non_null(std::unique_ptr<ARIMAX> model) {
        if (model == nullptr) {
            throw std::invalid_argument("armax");  // C# ArgumentNullException
        }
        return model;
    }

    std::unique_ptr<ARIMAX> arimax_;
    corehydro::estimation::BayesianAnalysis bayesian_analysis_;
    int forecasting_time_steps_ = 0;
    std::optional<UncertaintyAnalysisResults> analysis_results_;
};

}  // namespace corehydro::analyses
