// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Bivariate/BivariateAnalysis.cs @ fc28c0c
//
// A Bayesian MCMC estimation over a BivariateDistribution (copula) model (X3): drives a
// BayesianAnalysis (Phase 4) over the model, then assembles the AND-joint-exceedance mode/mean
// curves + credible band across an XY ordinate grid into an UncertaintyAnalysisResults (A2). A
// standard AnalysisBase + IBayesianAnalysis clone, structurally near-identical to the A5
// UnivariateAnalysis template.
//
// LOAD-BEARING compute sequence ported; WPF/event/gate/XML plumbing DROPPED per the A5 precedent.
// Specifically SKIPPED (each is GUI/threading/serialization with no numerical content):
//   * the XML ctor (C# 47-89) + `ToXElement` (557) -- XElement (de)serialization.
//   * every `*_PropertyChanged` handler (`Model_PropertyChanged` 178, `BayesianAnalysis_-
//     PropertyChanged` 208) and the INotifyPropertyChanged/INotifyCollectionChanged cascades they
//     drive, plus the `BayesianAnalysis` property's `PropertyChanged +=/-=` subscribe/unsubscribe
//     plumbing (C# 129-140). The reprocess-vs-clear decisions those handlers encode are exercised
//     HERE by calling the (public) XYOrdinates setter + reprocess methods directly, as the C# tests
//     do.
//   * the reprocess gate (`_reprocessGate.WaitAsync/Release`), `CancellationTokenSource` + cancel
//     (`CancelAnalysis`, 362), `SafeProgressReporter`, and the `AnalysisStarting`/
//     `AnalysisCompleted` events + `OnAnalysisStarting/Completed` -- run-lifecycle plumbing.
//   * `RaisePropertyChange` calls throughout -- no notification system in this port.
//
// The C# `async Task RunAsync` ports to a synchronous `run()`; the C# `...Async` helper methods
// (`CreateFrequencyAnalysisResultsAsync`, `UpdatePointEstimateResultsAsync`) port to synchronous
// methods (every `Parallel.For` becomes a serial loop -- the loop bodies are independent writes /
// order-independent reductions, so the result is numerically identical).
//
// DEVIATIONS (documented):
//   1. OWNERSHIP. The C# `BivariateDistribution` model is a GC reference type; here the analysis
//      OWNS the model via `std::unique_ptr` (the ctor's null-guard maps to the C#
//      ArgumentNullException). `BayesianAnalysis` holds a `ModelBase&` into that owned model, so the
//      analysis is non-copyable / non-movable. The owned BivariateDistribution in turn holds
//      NON-owning pointers to its two marginals; the CALLER must keep the marginals alive for the
//      analysis's lifetime (the BivariateDistribution model contract). Mirrors A5 deviation 2.
//   2. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> `std::optional`; the
//      IBayesianAnalysis `analysis_results()` accessor returns a `const ...*` (null <=> empty).
//   3. XY ORDINATES. The C# `XYOrdinates` is an `UncertainOrderedPairedData` (a full paired-data /
//      XML container the estimators only read `[i].X` and `[i].Y.Mean` off). That container is NOT
//      ported; this analysis models the grid as a plain `std::vector<XYOrdinate>` of (x, y) pairs
//      (y == the C# `UncertainOrdinate.Y.Mean`, a Deterministic mean). Default is one (0, 0) pair,
//      mirroring the C# ctor's single `UncertainOrdinate(0, new Deterministic(0))`.
//   4. CONFIDENCE-INTERVAL SHAPE. C# `ConfidenceIntervals` is `double[n, 2]`; the ported DTO's
//      `std::array<double, 2>` maps it exactly ({lower, upper}) -- no drop (unlike the rating-curve
//      3-column case).
//   5. RUN ERROR REPORTING. With events dropped, `run()` lets exceptions propagate to the caller
//      instead of routing failures through the (removed) `AnalysisCompleted`. Mirrors A5 dev. 6.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_bayesian_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/bivariate_distribution/bivariate_distribution.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace bestfit::analyses {

class BivariateAnalysis : public AnalysisBase, public IBayesianAnalysis {
   public:
    using BivariateDistribution = bestfit::models::BivariateDistribution;
    using BivariateCopula = bestfit::numerics::distributions::copulas::BivariateCopula;
    using UnivariateDistributionBase = bestfit::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using PointEstimateType = bestfit::estimation::PointEstimateType;

    // One XY ordinate: the X value and the Y mean (deviation 3; the C# reads only `[i].X` and
    // `[i].Y.Mean` off each UncertainOrdinate).
    struct XYOrdinate {
        double x = 0.0;
        double y = 0.0;
    };

    // C# ctor `BivariateAnalysis(BivariateDistribution)` (C# 33): builds the BayesianAnalysis over
    // the model + a default single-(0,0) ordinate grid. The C# `?? throw ArgumentNullException`
    // maps to the null-guard here (see deviation 1).
    explicit BivariateAnalysis(std::unique_ptr<BivariateDistribution> bivariate_distribution)
        : bivariate_distribution_(require_non_null(std::move(bivariate_distribution))),
          bayesian_analysis_(*bivariate_distribution_),
          xy_ordinates_{{0.0, 0.0}} {}

    ~BivariateAnalysis() override = default;

    // Non-copyable / non-movable (deviation 1).
    BivariateAnalysis(const BivariateAnalysis&) = delete;
    BivariateAnalysis& operator=(const BivariateAnalysis&) = delete;
    BivariateAnalysis(BivariateAnalysis&&) = delete;
    BivariateAnalysis& operator=(BivariateAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `BivariateDistribution` (C# 102): the model being estimated.
    BivariateDistribution& bivariate_distribution() { return *bivariate_distribution_; }
    const BivariateDistribution& bivariate_distribution() const { return *bivariate_distribution_; }

    // C# `BayesianAnalysis` (C# 126). IBayesianAnalysis override (the C# PropertyChanged +=/-=
    // subscribe/unsubscribe plumbing is DROPPED, see the file header).
    bestfit::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const bestfit::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `XYOrdinates` (C# 152): the joint-exceedance evaluation grid. The setter reprocesses
    // derived results if estimated and the grid is valid; else clears them.
    std::vector<XYOrdinate>& xy_ordinates() { return xy_ordinates_; }
    const std::vector<XYOrdinate>& xy_ordinates() const { return xy_ordinates_; }
    void set_xy_ordinates(std::vector<XYOrdinate> value) {
        xy_ordinates_ = std::move(value);
        reprocess_or_clear_xy_ordinates();
    }

    // C# `AnalysisResults` (C# 166): the uncertainty results (null until estimated).
    // IBayesianAnalysis override (const pointer <=> nullable).
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 235): clears the Bayesian fit + the result object and resets
    // IsEstimated. The RaisePropertyChange calls are dropped.
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        set_is_estimated(false);
    }

    // C# `ClearFrequencyAnalysisResults` (C# 252): clears ONLY the joint-exceedance output; the
    // fit and IsEstimated survive.
    void clear_frequency_analysis_results() { analysis_results_.reset(); }

    // C# `ReprocessOrClearXYOrdinates` (C# 277): reprocess if estimated and the new grid is valid
    // (non-empty); else clear the grid output. The fit is preserved either way.
    void reprocess_or_clear_xy_ordinates() {
        if (!is_estimated()) return;

        bool valid = !xy_ordinates_.empty();
        if (valid)
            create_frequency_analysis_results();
        else
            clear_frequency_analysis_results();
    }

    // C# `RunAsync` (C# 293), synchronous. Validate guard -> clear -> SetSampleData ->
    // BayesianAnalysis.estimate() -> IF estimated: create frequency results -> IsEstimated
    // mirrors the inner fit. (Cancellation/gate/events/progress dropped; see the file header.
    // Deviation 5: exceptions propagate instead of routing through AnalysisCompleted.)
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error("Analysis is not valid.");
        }

        clear_results();

        // Prepare the joint sample from the marginals (C# 325).
        bivariate_distribution_->set_sample_data();

        // Run the Bayesian analysis.
        bayesian_analysis_.estimate();

        // Post-process.
        if (bayesian_analysis_.is_estimated()) {
            create_frequency_analysis_results();
        }

        // Mirror the inner fit's success state (C# 337).
        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // --- Reprocess helpers (public, as in C#) ----------------------------------------------

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 472), synchronous. Sets the model params to the
    // MAP, builds the AND-joint-exceedance mode curve, then the posterior mean + credible band per
    // XY ordinate from the posterior copula samples, then updates the point-estimate scalars.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        // Set model parameters to the MAP (C# 483).
        bivariate_distribution_->set_parameter_values(bayesian_analysis_.results()->map.values);

        const UnivariateDistributionBase* dist_x =
            bivariate_distribution_->marginal_x() ? bivariate_distribution_->marginal_x()->distribution()
                                                  : nullptr;
        const UnivariateDistributionBase* dist_y =
            bivariate_distribution_->marginal_y() ? bivariate_distribution_->marginal_y()->distribution()
                                                  : nullptr;
        const BivariateCopula* copula_template = &bivariate_distribution_->copula();
        if (dist_x == nullptr || dist_y == nullptr) return;

        int n = static_cast<int>(xy_ordinates_.size());
        UncertaintyAnalysisResults r;
        r.mode_curve.assign(static_cast<std::size_t>(n), 0.0);
        r.mean_curve.assign(static_cast<std::size_t>(n), 0.0);
        r.confidence_intervals.assign(static_cast<std::size_t>(n), {0.0, 0.0});

        int realz = bayesian_analysis_.output_length();
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        const auto& output = bayesian_analysis_.results()->output;

        for (int i = 0; i < n; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            double u = dist_x->cdf(xy_ordinates_[ui].x);
            double v = dist_y->cdf(xy_ordinates_[ui].y);
            r.mode_curve[ui] = copula_template->and_joint_exceedance_probability(u, v);

            std::vector<double> p(static_cast<std::size_t>(realz));
            for (int idx = 0; idx < realz; ++idx) {
                std::unique_ptr<BivariateCopula> copula = copula_template->clone();
                copula->set_copula_parameters(output[static_cast<std::size_t>(idx)].values);
                copula->marginal_distribution_x = dist_x->clone();
                copula->marginal_distribution_y = dist_y->clone();
                double ui_p = copula->marginal_distribution_x->cdf(xy_ordinates_[ui].x);
                double vi_p = copula->marginal_distribution_y->cdf(xy_ordinates_[ui].y);
                p[static_cast<std::size_t>(idx)] =
                    copula->and_joint_exceedance_probability(ui_p, vi_p);
            }

            r.mean_curve[ui] = bestfit::numerics::data::mean(p);
            std::sort(p.begin(), p.end());
            r.confidence_intervals[ui][0] =
                bestfit::numerics::data::percentile(p, alpha / 2.0, true);
            r.confidence_intervals[ui][1] =
                bestfit::numerics::data::percentile(p, 1.0 - alpha / 2.0, true);
        }

        analysis_results_ = std::move(r);
        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 371), synchronous. Sets the model params to the
    // selected point estimator, rebuilds the mode curve, computes an empirical-CDF RMSE over the
    // date-aligned non-outlier marginal observations, and writes AIC/BIC (at MAP) + DIC.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            !analysis_results_) {
            return;
        }

        // Set the point estimator (C# 380-383).
        if (bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean)
            bivariate_distribution_->set_parameter_values(
                bayesian_analysis_.results()->posterior_mean.values);
        else
            bivariate_distribution_->set_parameter_values(bayesian_analysis_.results()->map.values);

        auto* marginal_x = bivariate_distribution_->marginal_x();
        auto* marginal_y = bivariate_distribution_->marginal_y();
        const UnivariateDistributionBase* dist_x = marginal_x ? marginal_x->distribution() : nullptr;
        const UnivariateDistributionBase* dist_y = marginal_y ? marginal_y->distribution() : nullptr;
        const BivariateCopula* copula = &bivariate_distribution_->copula();
        if (marginal_x == nullptr || marginal_y == nullptr || dist_x == nullptr ||
            dist_y == nullptr) {
            return;
        }

        // Mode curve (C# 396-403).
        int n = static_cast<int>(xy_ordinates_.size());
        analysis_results_->mode_curve.assign(static_cast<std::size_t>(n), 0.0);
        for (int i = 0; i < n; ++i) {
            std::size_t ui = static_cast<std::size_t>(i);
            double u = dist_x->cdf(xy_ordinates_[ui].x);
            double v = dist_y->cdf(xy_ordinates_[ui].y);
            analysis_results_->mode_curve[ui] = copula->and_joint_exceedance_probability(u, v);
        }

        // Two-pointer index merge of the non-low-outlier exact series (C# 406-427).
        const bestfit::models::ExactSeries& series_x = marginal_x->data_frame().exact_series();
        const bestfit::models::ExactSeries& series_y = marginal_y->data_frame().exact_series();
        std::vector<const bestfit::models::ExactData*> data_x = non_low_outlier(series_x);
        std::vector<const bestfit::models::ExactData*> data_y = non_low_outlier(series_y);

        std::vector<double> fx;
        std::vector<double> fy;
        {
            std::size_t ii = 0, jj = 0;
            while (ii < data_x.size() && jj < data_y.size()) {
                int idx_x = data_x[ii]->index();
                int idx_y = data_y[jj]->index();
                if (idx_x == idx_y) {
                    fx.push_back(data_x[ii]->plotting_position_complement());
                    fy.push_back(data_y[jj]->plotting_position_complement());
                    ++ii;
                    ++jj;
                } else if (idx_x < idx_y) {
                    ++ii;
                } else {
                    ++jj;
                }
            }
        }

        int m = static_cast<int>(fx.size());
        if (m < 2) {
            // RMSE requires at least 2 matched observations (C# 429-436).
            analysis_results_->rmse = std::numeric_limits<double>::quiet_NaN();
        } else {
            double sse = 0.0;
            for (int i = 0; i < m; ++i) {
                double cdf = copula->cdf(fx[static_cast<std::size_t>(i)], fy[static_cast<std::size_t>(i)]);
                double ecdf = 0.0;
                for (int j = 0; j < m; ++j) {
                    if (fx[static_cast<std::size_t>(j)] <= fx[static_cast<std::size_t>(i)] &&
                        fy[static_cast<std::size_t>(j)] <= fy[static_cast<std::size_t>(i)])
                        ecdf += 1.0;
                }
                ecdf /= m;
                double d = cdf - ecdf;
                sse += d * d;
            }
            analysis_results_->rmse = std::sqrt(sse / (m - 1));
        }

        // AIC/BIC at the MAP (full log-likelihood) + DIC (C# 460-463).
        std::vector<double> map_values = bayesian_analysis_.results()->map.values;
        double map_log_lh = bivariate_distribution_->log_likelihood(map_values);
        analysis_results_->aic = GoodnessOfFit::aic(copula->number_of_copula_parameters(), map_log_lh);
        analysis_results_->bic = GoodnessOfFit::bic(m, copula->number_of_copula_parameters(), map_log_lh);
        analysis_results_->dic = bayesian_analysis_.dic();
    }

    // C# `Validate` (C# 532): aggregates the model + Bayesian-analysis validations. const per the
    // A4 IAnalysis contract (the C# body only reads state).
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        bestfit::models::ValidationResult dist_valid = bivariate_distribution_->validate();
        if (!dist_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              dist_valid.validation_messages.begin(),
                                              dist_valid.validation_messages.end());
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
    static std::unique_ptr<BivariateDistribution> require_non_null(
        std::unique_ptr<BivariateDistribution> model) {
        if (model == nullptr) {
            throw std::invalid_argument("bivariateDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // Collect pointers to the non-low-outlier exact ordinates (mirrors bivariate_distribution.hpp).
    static std::vector<const bestfit::models::ExactData*> non_low_outlier(
        const bestfit::models::ExactSeries& series) {
        std::vector<const bestfit::models::ExactData*> result;
        for (std::size_t i = 0; i < series.count(); ++i)
            if (!series[i].is_low_outlier()) result.push_back(&series[i]);
        return result;
    }

    // Owned model (deviation 1). Declared BEFORE bayesian_analysis_ so it is constructed first
    // (BayesianAnalysis stores a reference into it).
    std::unique_ptr<BivariateDistribution> bivariate_distribution_;
    bestfit::estimation::BayesianAnalysis bayesian_analysis_;

    // XY ordinate grid (deviation 3). Default: one (0, 0) pair.
    std::vector<XYOrdinate> xy_ordinates_;

    // Result object (C# nullable -> optional; deviation 2).
    std::optional<UncertaintyAnalysisResults> analysis_results_;
};

}  // namespace bestfit::analyses
