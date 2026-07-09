// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/SpatialExtremes/SpatialGEVAnalysis.cs @ fc28c0c
//
// A Bayesian MCMC estimation over the hierarchical SpatialGEV model (Phase 7a): drives a
// BayesianAnalysis (Phase 4) over the model, then turns the posterior draws into per-site
// GEV-parameter / quantile credible bands (SiteResults), a regional (site-averaged)
// UncertaintyAnalysisResults, ungauged-location predictions, and a leave-one-site-out cross
// -validation report. A standard AnalysisBase + IBayesianAnalysis clone, structurally following the
// X3 BivariateAnalysis / RatingCurveAnalysis templates.
//
// LOAD-BEARING compute sequence ported; WPF/event/gate/XML plumbing DROPPED per the A5/X3 precedent.
// Specifically SKIPPED (each is GUI/threading/serialization with no numerical content):
//   * the XML ctor (C# 110-145) + `ToXElement` (1723) -- XElement (de)serialization.
//   * every `*_PropertyChanged` / `*_CollectionChanged` handler (`Model_PropertyChanged` 328,
//     `BayesianAnalysis_PropertyChanged` 361, `ProbabilityOrdinates_CollectionChanged` 434,
//     `HandleOrdinatesChanged` 450) and the INotifyPropertyChanged/INotifyCollectionChanged cascades
//     they drive. Their reprocess-vs-clear decisions are surfaced HERE by the explicit-invalidation
//     mutators (`set_probability_ordinates` resets the derived optionals), matching how M4/X3
//     handled INPC. The property setters' RaisePropertyChange calls are dropped.
//   * the reprocess gate (`_reprocessGate.WaitAsync/Release`), `CancellationTokenSource` + cancel
//     (`CancelAnalysis`, 550), `SafeProgressReporter`, the `AnalysisStarting`/`AnalysisCompleted`
//     events + `OnAnalysisStarting/Completed`, and `ReprocessIfEstimated` -- run-lifecycle plumbing.
//   * `RaisePropertyChange` calls throughout -- no notification system in this port.
//
// The C# `async Task RunAsync` / `...Async` helpers port to synchronous methods; every `Parallel.For`
// becomes a serial loop (the loop bodies are independent writes / order-independent reductions, so
// the result is numerically identical).
//
// DEVIATIONS (documented):
//   1. OWNERSHIP. The C# `SpatialGEV` model is a GC reference type; here the analysis OWNS it via
//      `std::unique_ptr` (the ctor's null-guard maps to the C# ArgumentNullException). `BayesianAnalysis`
//      holds a `ModelBase&` into that owned model, so the analysis is non-copyable / non-movable.
//   2. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` / `SpatialGEVSiteResults[]?` /
//      `SpatialGEVCrossValidationResults?` / `double[,]?` -> `std::optional`; the accessors return a
//      `const ...*` (null <=> empty optional). `analysis_results()` is the IBayesianAnalysis override.
//   3. CONFIDENCE-INTERVAL SHAPE. The C# regional `ConfidenceIntervals` is `double[n, 3]` and packs
//      the probability into column 0 (`[p,0] = probs[p]`), lower/upper into columns 1/2. The ported
//      UncertaintyAnalysisResults DTO (oracle-locked) stores `std::array<double, 2>` per row, so the
//      port drops the redundant probability column and stores `{lower, upper}`. The probability grid
//      is `probability_ordinates()`; any consumer recomputes it identically. No numerical content lost.
//   4. ParallelMean -> serial mean. C# `Statistics.ParallelMean` is a threaded reduction; the ported
//      `numerics::data::mean` is its serial equivalent (identical result). `Array.Sort` -> `std::sort`.
//   5. UNCERTAINTY METHOD. `SpatialGEVUncertaintyMethod` leads with `BayesianPosterior` (the default,
//      C# 306) -- the task-note list omitted it; C# governs, all four members are ported and the
//      default is BayesianPosterior. `run()` does NOT branch on the method (neither does the C#
//      RunAsync); the alternative SE paths (Godambe sandwich, variance inflation, spatial bootstrap)
//      are the separately-invoked public methods the GUI drives.
//   6. SEEDED SPATIAL-BOOTSTRAP RNG. `run_spatial_bootstrap` / `create_spatial_blocks` substitute the
//      ported MersenneTwister for the C# one; same-seed reproducibility holds. That path and the
//      converged LOOCV/site-band numerics are compiled-but-not-runtime-driven here -- the seeded
//      end-to-end oracles land with the X12 emitter.
//   7. RUN ERROR REPORTING. With events dropped, `run()` lets exceptions propagate to the caller
//      instead of routing failures through the (removed) `AnalysisCompleted`.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bestfit/analyses/spatial_extremes/spatial_gev_cross_validation_results.hpp"
#include "bestfit/analyses/spatial_extremes/spatial_gev_site_results.hpp"
#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_bayesian_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/spatial_extremes/spatial_gev.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/probability_ordinates.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::analyses {

// C# `SpatialGEVUncertaintyMethod` (C# 19-45). Order + members governed by C# (deviation 5); the
// leading `BayesianPosterior` is the default.
enum class SpatialGEVUncertaintyMethod {
    BayesianPosterior,
    BayesianInflated,
    GodambeSandwich,
    SpatialBootstrap,
};

class SpatialGEVAnalysis : public AnalysisBase, public IBayesianAnalysis {
   public:
    using SpatialGEV = bestfit::models::spatial_extremes::SpatialGEV;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using ProbabilityOrdinates = bestfit::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using PointEstimateType = bestfit::estimation::PointEstimateType;
    using Grid = std::vector<std::vector<double>>;
    using GeneralizedExtremeValue = bestfit::numerics::distributions::GeneralizedExtremeValue;

    // (mean, lower, upper) triple returned by `get_site_quantiles` (C# named tuple, 770).
    struct SiteQuantiles {
        std::vector<double> mean;
        std::vector<double> lower;
        std::vector<double> upper;
    };

    // (probabilities, growth_factors, lower, upper) returned by `get_regional_growth_curve`
    // (C# named tuple, 1087).
    struct RegionalGrowthCurve {
        std::vector<double> probabilities;
        std::vector<double> growth_factors;
        std::vector<double> lower;
        std::vector<double> upper;
    };

    // C# ctor `SpatialGEVAnalysis(SpatialGEV)` (C# 90): builds the BayesianAnalysis over the model,
    // then SetDefaultProbabilityOrdinates(). The C# `?? throw ArgumentNullException` maps to the
    // null-guard here (deviation 1).
    explicit SpatialGEVAnalysis(std::unique_ptr<SpatialGEV> spatial_gev)
        : spatial_gev_(require_non_null(std::move(spatial_gev))),
          bayesian_analysis_(*spatial_gev_) {
        set_default_probability_ordinates();
    }

    ~SpatialGEVAnalysis() override = default;

    // Non-copyable / non-movable (deviation 1).
    SpatialGEVAnalysis(const SpatialGEVAnalysis&) = delete;
    SpatialGEVAnalysis& operator=(const SpatialGEVAnalysis&) = delete;
    SpatialGEVAnalysis(SpatialGEVAnalysis&&) = delete;
    SpatialGEVAnalysis& operator=(SpatialGEVAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `SpatialGEV` (C# 165): the model being estimated.
    SpatialGEV& spatial_gev() { return *spatial_gev_; }
    const SpatialGEV& spatial_gev() const { return *spatial_gev_; }

    // C# `BayesianAnalysis` (C# 198). IBayesianAnalysis override.
    bestfit::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const bestfit::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `ProbabilityOrdinates` (C# 232): the exceedance-probability grid. The setter's
    // reprocess-or-clear side effect ports as explicit invalidation (deviation, see file header);
    // RaisePropertyChange is dropped.
    ProbabilityOrdinates& probability_ordinates() { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }
    void set_probability_ordinates(ProbabilityOrdinates value) {
        probability_ordinates_ = std::move(value);
        handle_ordinates_changed();
    }

    // C# `AnalysisResults` (C# 263): the regional uncertainty results (null until estimated).
    // IBayesianAnalysis override.
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // C# `SiteResults` (C# 274): per-site GEV/quantile results (null until estimated).
    const std::vector<SpatialGEVSiteResults>* site_results() const {
        return site_results_ ? &*site_results_ : nullptr;
    }

    // C# `CrossValidationResults` (C# 285): LOOCV report (null until run_cross_validation()).
    const SpatialGEVCrossValidationResults* cross_validation_results() const {
        return cross_validation_results_ ? &*cross_validation_results_ : nullptr;
    }

    // C# `UncertaintyMethod` (C# 306): stored config; default BayesianPosterior (deviation 5).
    SpatialGEVUncertaintyMethod uncertainty_method() const { return uncertainty_method_; }
    void set_uncertainty_method(SpatialGEVUncertaintyMethod value) { uncertainty_method_ = value; }

    // C# `GodambeCovariance` (C# 311): the sandwich covariance after compute_godambe_covariance()
    // (null until computed).
    const Grid* godambe_covariance() const {
        return godambe_covariance_ ? &*godambe_covariance_ : nullptr;
    }

    // C# `VarianceInflationFactor` (C# 316): default 1.0.
    double variance_inflation_factor() const { return variance_inflation_factor_; }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `SetDefaultProbabilityOrdinates` (C# 390): a fresh default grid (25 ordinates).
    void set_default_probability_ordinates() { probability_ordinates_ = ProbabilityOrdinates(); }

    // C# `ClearResults` (C# 399): clears the Bayesian fit + all result objects + IsEstimated.
    void clear_results() {
        bayesian_analysis_.clear_results();
        analysis_results_.reset();
        site_results_.reset();
        cross_validation_results_.reset();
        set_is_estimated(false);
    }

    // C# `ClearUncertaintyAnalysisResults` (C# 417): clears ONLY the ordinate-keyed outputs
    // (AnalysisResults + SiteResults); the Bayesian fit and IsEstimated survive.
    void clear_uncertainty_analysis_results() {
        analysis_results_.reset();
        site_results_.reset();
    }

    // C# `RunAsync` (C# 467), synchronous. Validate guard -> clear -> BayesianAnalysis.estimate()
    // -> IF estimated: create site results + regional uncertainty results -> IsEstimated mirrors the
    // inner fit. (Cancellation/gate/events/progress dropped; deviation 7: exceptions propagate.)
    void run() override {
        if (!validate().is_valid) {
            throw std::runtime_error(
                "Analysis is not valid. Please check the configuration before running the "
                "analysis.");
        }

        clear_results();

        bayesian_analysis_.estimate();

        if (bayesian_analysis_.is_estimated()) {
            create_site_results();
            create_uncertainty_analysis_results();
        }

        set_is_estimated(bayesian_analysis_.is_estimated());
    }

    // --- Result builders (public, as in C#) ------------------------------------------------

    // C# `CreateSiteResultsAsync` (C# 559), synchronous. For each site: clone the model per posterior
    // draw, collect the site's (xi, alpha, kappa) trend values + per-probability quantiles, then form
    // the mean/lower/upper credible bands; the QuantileMode is the MAP quantile.
    void create_site_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results()) return;

        const auto& results = *bayesian_analysis_.results();
        const auto& output = results.output;
        int n_sites = spatial_gev_->sites();
        int realz = std::min(bayesian_analysis_.output_length(), static_cast<int>(output.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        const std::vector<double>& probs = probability_ordinates_.values();
        int n_probs = static_cast<int>(probs.size());

        std::vector<SpatialGEVSiteResults> site_results(static_cast<std::size_t>(n_sites));

        for (int j = 0; j < n_sites; ++j) {
            SpatialGEVSiteResults& sr = site_results[static_cast<std::size_t>(j)];
            sr.site_index = j;
            sr.coordinate = {spatial_gev_->coordinates()[static_cast<std::size_t>(j)][0],
                             spatial_gev_->coordinates()[static_cast<std::size_t>(j)][1]};

            std::vector<double> xi_vals(static_cast<std::size_t>(realz));
            std::vector<double> alpha_vals(static_cast<std::size_t>(realz));
            std::vector<double> kappa_vals(static_cast<std::size_t>(realz));
            // quantiles[p][idx]
            std::vector<std::vector<double>> quantiles(
                static_cast<std::size_t>(n_probs), std::vector<double>(static_cast<std::size_t>(realz)));

            for (int idx = 0; idx < realz; ++idx) {
                SpatialGEV temp = spatial_gev_->clone();
                temp.set_parameter_values(output[static_cast<std::size_t>(idx)].values);
                std::vector<double> gev_params = temp.get_gev_parameters(j);
                xi_vals[static_cast<std::size_t>(idx)] = gev_params[0];
                alpha_vals[static_cast<std::size_t>(idx)] = gev_params[1];
                kappa_vals[static_cast<std::size_t>(idx)] = gev_params[2];
                for (int p = 0; p < n_probs; ++p)
                    quantiles[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] =
                        temp.inverse_cdf(1.0 - probs[static_cast<std::size_t>(p)], j);
            }

            Band loc = band(xi_vals, alpha);
            sr.location_mean = loc.mean;
            sr.location_lower = loc.lower;
            sr.location_upper = loc.upper;
            Band scl = band(alpha_vals, alpha);
            sr.scale_mean = scl.mean;
            sr.scale_lower = scl.lower;
            sr.scale_upper = scl.upper;
            Band shp = band(kappa_vals, alpha);
            sr.shape_mean = shp.mean;
            sr.shape_lower = shp.lower;
            sr.shape_upper = shp.upper;

            sr.probabilities = probs;
            sr.quantile_mean.assign(static_cast<std::size_t>(n_probs), 0.0);
            sr.quantile_lower.assign(static_cast<std::size_t>(n_probs), 0.0);
            sr.quantile_upper.assign(static_cast<std::size_t>(n_probs), 0.0);
            sr.quantile_mode.assign(static_cast<std::size_t>(n_probs), 0.0);

            // Point estimate quantiles use the MAP (C# 636).
            spatial_gev_->set_parameter_values(results.map.values);
            for (int p = 0; p < n_probs; ++p) {
                Band q = band(quantiles[static_cast<std::size_t>(p)], alpha);
                sr.quantile_mean[static_cast<std::size_t>(p)] = q.mean;
                sr.quantile_lower[static_cast<std::size_t>(p)] = q.lower;
                sr.quantile_upper[static_cast<std::size_t>(p)] = q.upper;
                sr.quantile_mode[static_cast<std::size_t>(p)] =
                    spatial_gev_->inverse_cdf(1.0 - probs[static_cast<std::size_t>(p)], j);
            }
        }

        site_results_ = std::move(site_results);
    }

    // C# `CreateUncertaintyAnalysisResultsAsync` (C# 656), synchronous. Averages the per-site
    // quantile bands across sites into the regional mode/mean curve + credible band, then writes the
    // AIC/BIC (at MAP) + DIC scalars.
    void create_uncertainty_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() || !site_results_) {
            return;
        }

        const std::vector<SpatialGEVSiteResults>& site_results = *site_results_;
        const std::vector<double>& probs = probability_ordinates_.values();
        int n = static_cast<int>(probs.size());
        int sites = spatial_gev_->sites();

        UncertaintyAnalysisResults r;
        r.mode_curve.assign(static_cast<std::size_t>(n), 0.0);
        r.mean_curve.assign(static_cast<std::size_t>(n), 0.0);
        r.confidence_intervals.assign(static_cast<std::size_t>(n), {0.0, 0.0});

        for (int p = 0; p < n; ++p) {
            double sum_mode = 0, sum_mean = 0, sum_lower = 0, sum_upper = 0;
            for (int j = 0; j < sites; ++j) {
                const SpatialGEVSiteResults& sr = site_results[static_cast<std::size_t>(j)];
                sum_mode += sr.quantile_mode[static_cast<std::size_t>(p)];
                sum_mean += sr.quantile_mean[static_cast<std::size_t>(p)];
                sum_lower += sr.quantile_lower[static_cast<std::size_t>(p)];
                sum_upper += sr.quantile_upper[static_cast<std::size_t>(p)];
            }
            r.mode_curve[static_cast<std::size_t>(p)] = sum_mode / sites;
            r.mean_curve[static_cast<std::size_t>(p)] = sum_mean / sites;
            // Deviation 3: C# packs probs[p] into confidence_intervals[p,0]; the 2-col DTO drops it
            // and stores {lower, upper}.
            r.confidence_intervals[static_cast<std::size_t>(p)][0] = sum_lower / sites;
            r.confidence_intervals[static_cast<std::size_t>(p)][1] = sum_upper / sites;
        }

        // AIC/BIC at the MAP (full log-likelihood, data + prior) + DIC (C# 699-702).
        std::vector<double> map_values = bayesian_analysis_.results()->map.values;
        double map_log_lh = spatial_gev_->log_likelihood(map_values);
        r.aic = GoodnessOfFit::aic(spatial_gev_->number_of_parameters(), map_log_lh);
        r.bic = GoodnessOfFit::bic(sites * spatial_gev_->observations(),
                                   spatial_gev_->number_of_parameters(), map_log_lh);
        r.dic = bayesian_analysis_.dic();

        analysis_results_ = std::move(r);
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 711), synchronous. Republishes the selected point
    // estimator to the model, rebuilds the per-site QuantileMode curves + the regional mode curve.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() || !site_results_) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();
        std::vector<double> parameters =
            bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean
                ? results.posterior_mean.values
                : results.map.values;
        spatial_gev_->set_parameter_values(parameters);

        const std::vector<double>& probs = probability_ordinates_.values();
        int n = static_cast<int>(probs.size());
        int sites = spatial_gev_->sites();
        std::vector<SpatialGEVSiteResults>& site_results = *site_results_;

        // Update per-site mode curves (C# GetGEVParameters(j) call is dead -- dropped).
        for (int j = 0; j < sites; ++j) {
            SpatialGEVSiteResults& sr = site_results[static_cast<std::size_t>(j)];
            for (int p = 0; p < n; ++p)
                sr.quantile_mode[static_cast<std::size_t>(p)] =
                    spatial_gev_->inverse_cdf(1.0 - probs[static_cast<std::size_t>(p)], j);
        }

        // Update the regional mode curve (C# 745-754).
        if (analysis_results_ && !analysis_results_->mode_curve.empty()) {
            for (int p = 0; p < n; ++p) {
                double sum = 0;
                for (int j = 0; j < sites; ++j)
                    sum += site_results[static_cast<std::size_t>(j)]
                               .quantile_mode[static_cast<std::size_t>(p)];
                analysis_results_->mode_curve[static_cast<std::size_t>(p)] = sum / sites;
            }
        }
    }

    // C# `GetSiteQuantiles` (C# 770): the mean/lower/upper credible band over one site at the given
    // exceedance probabilities.
    SiteQuantiles get_site_quantiles(int site_index,
                                     const std::vector<double>& exceedance_probabilities) const {
        if (!is_estimated() || !bayesian_analysis_.results())
            throw std::runtime_error("Analysis must be run before computing quantiles.");
        if (site_index < 0 || site_index >= spatial_gev_->sites())
            throw std::out_of_range("siteIndex");

        const auto& output = bayesian_analysis_.results()->output;
        int realz = std::min(bayesian_analysis_.output_length(), static_cast<int>(output.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        int n_probs = static_cast<int>(exceedance_probabilities.size());

        std::vector<std::vector<double>> quantiles(
            static_cast<std::size_t>(n_probs), std::vector<double>(static_cast<std::size_t>(realz)));
        for (int idx = 0; idx < realz; ++idx) {
            SpatialGEV temp = spatial_gev_->clone();
            temp.set_parameter_values(output[static_cast<std::size_t>(idx)].values);
            for (int p = 0; p < n_probs; ++p)
                quantiles[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] =
                    temp.inverse_cdf(1.0 - exceedance_probabilities[static_cast<std::size_t>(p)],
                                     site_index);
        }

        SiteQuantiles out;
        out.mean.assign(static_cast<std::size_t>(n_probs), 0.0);
        out.lower.assign(static_cast<std::size_t>(n_probs), 0.0);
        out.upper.assign(static_cast<std::size_t>(n_probs), 0.0);
        for (int p = 0; p < n_probs; ++p) {
            Band q = band(quantiles[static_cast<std::size_t>(p)], alpha);
            out.mean[static_cast<std::size_t>(p)] = q.mean;
            out.lower[static_cast<std::size_t>(p)] = q.lower;
            out.upper[static_cast<std::size_t>(p)] = q.upper;
        }
        return out;
    }

    // C# `PredictAtUngaugedLocation` (C# 828): posterior prediction of GEV params + quantile bands at
    // an ungauged coordinate. Per draw: evaluate the trends (with covariates), apply link functions,
    // interpolate the spatial errors by inverse-distance weighting, and compute the GEV quantiles.
    SpatialGEVSiteResults predict_at_ungauged_location(
        const std::vector<double>& coordinates, const std::vector<double>& covariates,
        const std::vector<double>& exceedance_probabilities) const {
        if (!is_estimated() || !bayesian_analysis_.results())
            throw std::runtime_error(
                "Analysis must be run before predicting at ungauged locations.");
        if (coordinates.size() != 2)
            throw std::invalid_argument("Coordinates must be a 2-element array [X, Y].");

        const auto& output = bayesian_analysis_.results()->output;
        int realz = std::min(bayesian_analysis_.output_length(), static_cast<int>(output.size()));
        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        int n_probs = static_cast<int>(exceedance_probabilities.size());
        int sites = spatial_gev_->sites();

        SpatialGEVSiteResults result;
        result.site_index = -1;  // ungauged
        result.coordinate = coordinates;

        std::vector<double> xi_vals(static_cast<std::size_t>(realz));
        std::vector<double> alpha_vals(static_cast<std::size_t>(realz));
        std::vector<double> kappa_vals(static_cast<std::size_t>(realz));
        std::vector<std::vector<double>> quantiles(
            static_cast<std::size_t>(n_probs), std::vector<double>(static_cast<std::size_t>(realz)));

        // Inverse-distance weighting for spatial interpolation of the errors (C# 852-861).
        std::vector<double> distances(static_cast<std::size_t>(sites));
        double sum_inv_dist = 0;
        for (int j = 0; j < sites; ++j) {
            double dx = coordinates[0] - spatial_gev_->coordinates()[static_cast<std::size_t>(j)][0];
            double dy = coordinates[1] - spatial_gev_->coordinates()[static_cast<std::size_t>(j)][1];
            distances[static_cast<std::size_t>(j)] = std::max(std::sqrt(dx * dx + dy * dy), 1e-10);
            sum_inv_dist += 1.0 / distances[static_cast<std::size_t>(j)];
        }

        for (int idx = 0; idx < realz; ++idx) {
            SpatialGEV temp = spatial_gev_->clone();
            temp.set_parameter_values(output[static_cast<std::size_t>(idx)].values);

            double xi = temp.location().predict_with_covariates(covariates);
            double scl = temp.scale().predict_with_covariates(covariates);
            double kappa = temp.shape().predict_with_covariates(covariates);

            if (temp.use_log_link_for_location()) xi = std::exp(xi);
            if (temp.use_log_link_for_scale()) scl = std::exp(scl);

            if (temp.use_location_errors() && temp.location_errors() != nullptr) {
                double err_sum = 0;
                for (int j = 0; j < temp.sites(); ++j)
                    err_sum += temp.location_errors()->get_error(j) /
                               distances[static_cast<std::size_t>(j)];
                double interp_err = err_sum / sum_inv_dist;
                if (temp.use_log_link_for_location())
                    xi *= std::exp(interp_err);
                else
                    xi += interp_err;
            }
            if (temp.use_scale_errors() && temp.scale_errors() != nullptr) {
                double err_sum = 0;
                for (int j = 0; j < temp.sites(); ++j)
                    err_sum +=
                        temp.scale_errors()->get_error(j) / distances[static_cast<std::size_t>(j)];
                double interp_err = err_sum / sum_inv_dist;
                if (temp.use_log_link_for_scale())
                    scl *= std::exp(interp_err);
                else
                    scl = std::max(scl + interp_err,
                                   bestfit::numerics::kDoubleMachineEpsilon);
            }
            if (temp.use_shape_errors() && temp.shape_errors() != nullptr) {
                double err_sum = 0;
                for (int j = 0; j < temp.sites(); ++j)
                    err_sum +=
                        temp.shape_errors()->get_error(j) / distances[static_cast<std::size_t>(j)];
                kappa += err_sum / sum_inv_dist;
            }

            xi_vals[static_cast<std::size_t>(idx)] = xi;
            alpha_vals[static_cast<std::size_t>(idx)] = scl;
            kappa_vals[static_cast<std::size_t>(idx)] = kappa;

            GeneralizedExtremeValue gev(xi, scl, kappa);
            for (int p = 0; p < n_probs; ++p)
                quantiles[static_cast<std::size_t>(p)][static_cast<std::size_t>(idx)] =
                    gev.inverse_cdf(1.0 - exceedance_probabilities[static_cast<std::size_t>(p)]);
        }

        Band loc = band(xi_vals, alpha);
        result.location_mean = loc.mean;
        result.location_lower = loc.lower;
        result.location_upper = loc.upper;
        Band scl_b = band(alpha_vals, alpha);
        result.scale_mean = scl_b.mean;
        result.scale_lower = scl_b.lower;
        result.scale_upper = scl_b.upper;
        Band shp = band(kappa_vals, alpha);
        result.shape_mean = shp.mean;
        result.shape_lower = shp.lower;
        result.shape_upper = shp.upper;

        result.probabilities = exceedance_probabilities;
        result.quantile_mean.assign(static_cast<std::size_t>(n_probs), 0.0);
        result.quantile_lower.assign(static_cast<std::size_t>(n_probs), 0.0);
        result.quantile_upper.assign(static_cast<std::size_t>(n_probs), 0.0);
        result.quantile_mode.assign(static_cast<std::size_t>(n_probs), 0.0);
        for (int p = 0; p < n_probs; ++p) {
            std::vector<double>& q_vals = quantiles[static_cast<std::size_t>(p)];
            Band q = band(q_vals, alpha);
            result.quantile_mean[static_cast<std::size_t>(p)] = q.mean;
            result.quantile_lower[static_cast<std::size_t>(p)] = q.lower;
            result.quantile_upper[static_cast<std::size_t>(p)] = q.upper;
            // C# uses the default (unsorted) Percentile overload for the median (C# 954); q_vals is
            // already sorted by band(), so the copy-and-sort inside percentile is a no-op.
            result.quantile_mode[static_cast<std::size_t>(p)] =
                bestfit::numerics::data::percentile(q_vals, 0.5);
        }

        return result;
    }

    // C# `RunCrossValidationAsync` (C# 980), synchronous. Leave-one-site-out: zero the excluded
    // site's weight, re-fit, predict at the excluded site, and accumulate the error/RMSE/bias
    // diagnostics + aggregate MAE/RMSE/MeanBias. (Parallel/async -> serial; heavy path, compiled but
    // driven only by the X12 emitter.)
    void run_cross_validation() {
        if (!validate().is_valid) throw std::runtime_error("Model validation failed.");

        int sites = spatial_gev_->sites();
        SpatialGEVCrossValidationResults cv;
        cv.site_prediction_errors.assign(static_cast<std::size_t>(sites), 0.0);
        cv.site_rmse.assign(static_cast<std::size_t>(sites), 0.0);
        cv.site_bias.assign(static_cast<std::size_t>(sites), 0.0);
        cv.site_crps.assign(static_cast<std::size_t>(sites), 0.0);  // CRPS not computed (C# remark).

        std::vector<double> original_weights = spatial_gev_->site_weights();

        try {
            for (int j = 0; j < sites; ++j) {
                for (int k = 0; k < sites; ++k)
                    spatial_gev_->site_weights()[static_cast<std::size_t>(k)] =
                        k == j ? 0.0 : original_weights[static_cast<std::size_t>(k)];

                bayesian_analysis_.clear_results();
                bayesian_analysis_.estimate();
                if (!bayesian_analysis_.is_estimated()) continue;

                std::vector<double> coords = {
                    spatial_gev_->coordinates()[static_cast<std::size_t>(j)][0],
                    spatial_gev_->coordinates()[static_cast<std::size_t>(j)][1]};
                std::vector<double> probs = {0.5, 0.2, 0.1, 0.04, 0.02, 0.01};  // T=2,5,10,25,50,100
                SpatialGEVSiteResults prediction = predict_at_ungauged_location(coords, {}, probs);

                std::vector<double> site_data;
                for (int i = 0; i < spatial_gev_->observations(); ++i) {
                    double v = spatial_gev_->at_site_data()[static_cast<std::size_t>(i)]
                                                            [static_cast<std::size_t>(j)];
                    if (!std::isnan(v)) site_data.push_back(v);
                }

                if (!site_data.empty()) {
                    GeneralizedExtremeValue gev;
                    gev.estimate(site_data,
                                 bestfit::numerics::distributions::EstimationMethod::MaximumLikelihood);
                    double obs_q100 = gev.inverse_cdf(0.99);
                    double pred_q100 = prediction.quantile_mean[5];  // T=100

                    cv.site_prediction_errors[static_cast<std::size_t>(j)] = pred_q100 - obs_q100;
                    cv.site_bias[static_cast<std::size_t>(j)] = (pred_q100 - obs_q100) / obs_q100;

                    double sum_sq_err = 0;
                    for (std::size_t p = 0; p < probs.size(); ++p) {
                        double obs_q = gev.inverse_cdf(1.0 - probs[p]);
                        double pred_q = prediction.quantile_mean[p];
                        sum_sq_err += (pred_q - obs_q) * (pred_q - obs_q);
                    }
                    cv.site_rmse[static_cast<std::size_t>(j)] =
                        std::sqrt(sum_sq_err / static_cast<double>(probs.size()));
                }
            }

            spatial_gev_->set_site_weights(original_weights);

            std::vector<double> abs_err(cv.site_prediction_errors.size());
            std::vector<double> sq_err(cv.site_prediction_errors.size());
            for (std::size_t i = 0; i < cv.site_prediction_errors.size(); ++i) {
                abs_err[i] = std::fabs(cv.site_prediction_errors[i]);
                sq_err[i] = cv.site_prediction_errors[i] * cv.site_prediction_errors[i];
            }
            cv.mean_absolute_error = bestfit::numerics::data::mean(abs_err);
            cv.root_mean_square_error = std::sqrt(bestfit::numerics::data::mean(sq_err));
            cv.mean_bias = bestfit::numerics::data::mean(cv.site_bias);

            cross_validation_results_ = std::move(cv);

            // Re-run the full analysis (C# 1062-1063).
            bayesian_analysis_.clear_results();
            run();
        } catch (...) {
            spatial_gev_->set_site_weights(original_weights);
            throw;
        }
        spatial_gev_->set_site_weights(original_weights);
    }

    // C# `GetRegionalGrowthCurve` (C# 1087): index-flood growth factors (quantile / median), averaged
    // across sites.
    RegionalGrowthCurve get_regional_growth_curve(
        const std::vector<double>& exceedance_probabilities) const {
        if (!is_estimated() || !site_results_)
            throw std::runtime_error("Analysis must be run before computing growth curves.");

        int n_probs = static_cast<int>(exceedance_probabilities.size());
        int sites = spatial_gev_->sites();
        std::vector<double> growth_mean(static_cast<std::size_t>(n_probs), 0.0);
        std::vector<double> growth_lower(static_cast<std::size_t>(n_probs), 0.0);
        std::vector<double> growth_upper(static_cast<std::size_t>(n_probs), 0.0);
        const std::vector<SpatialGEVSiteResults>& site_results = *site_results_;

        for (int j = 0; j < sites; ++j) {
            const SpatialGEVSiteResults& sr = site_results[static_cast<std::size_t>(j)];
            // Index flood = median (T=2). Locate probability 0.5 in the site's grid.
            int median_idx = -1;
            for (std::size_t k = 0; k < sr.probabilities.size(); ++k)
                if (sr.probabilities[k] == 0.5) {
                    median_idx = static_cast<int>(k);
                    break;
                }
            double index_flood;
            if (median_idx >= 0 &&
                median_idx < static_cast<int>(sr.quantile_mean.size())) {
                index_flood = sr.quantile_mean[static_cast<std::size_t>(median_idx)];
            } else {
                index_flood = !sr.quantile_mean.empty() ? sr.quantile_mean[0] : 1.0;
            }
            if (index_flood <= 0)
                index_flood = !sr.quantile_mean.empty() ? sr.quantile_mean[0] : 1.0;

            for (int p = 0; p < n_probs; ++p) {
                SiteQuantiles q = get_site_quantiles(
                    j, {exceedance_probabilities[static_cast<std::size_t>(p)]});
                growth_mean[static_cast<std::size_t>(p)] += q.mean[0] / index_flood;
                growth_lower[static_cast<std::size_t>(p)] += q.lower[0] / index_flood;
                growth_upper[static_cast<std::size_t>(p)] += q.upper[0] / index_flood;
            }
        }

        for (int p = 0; p < n_probs; ++p) {
            growth_mean[static_cast<std::size_t>(p)] /= sites;
            growth_lower[static_cast<std::size_t>(p)] /= sites;
            growth_upper[static_cast<std::size_t>(p)] /= sites;
        }

        return {exceedance_probabilities, growth_mean, growth_lower, growth_upper};
    }

    // C# `ComputeGodambeCovariance` (C# 1160): the sandwich covariance H^-1 J H^-1 via central-
    // difference Hessian H and score outer-product variability matrix J (falls back to J if H is
    // singular). Stores + returns the result.
    Grid compute_godambe_covariance(const std::vector<double>& parameters_in = {}) {
        std::vector<double> parameters = parameters_in;
        if (parameters.empty()) {
            if (!bayesian_analysis_.results())
                throw std::runtime_error(
                    "Analysis must be run before computing Godambe covariance.");
            parameters = bayesian_analysis_.results()->map.values;
        }

        int n_params = static_cast<int>(parameters.size());
        double eps = 1e-5;

        Grid h(static_cast<std::size_t>(n_params),
               std::vector<double>(static_cast<std::size_t>(n_params), 0.0));
        std::vector<double> p0 = parameters;
        double f0 = spatial_gev_->data_log_likelihood(p0);

        for (int i = 0; i < n_params; ++i) {
            for (int j = i; j < n_params; ++j) {
                double hi = std::fabs(parameters[static_cast<std::size_t>(i)]) * eps + eps;
                double hj = std::fabs(parameters[static_cast<std::size_t>(j)]) * eps + eps;
                if (i == j) {
                    std::vector<double> p_plus = parameters, p_minus = parameters;
                    p_plus[static_cast<std::size_t>(i)] += hi;
                    p_minus[static_cast<std::size_t>(i)] -= hi;
                    double f_plus = spatial_gev_->data_log_likelihood(p_plus);
                    double f_minus = spatial_gev_->data_log_likelihood(p_minus);
                    h[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] =
                        (f_plus - 2 * f0 + f_minus) / (hi * hi);
                } else {
                    std::vector<double> p_pp = parameters, p_mm = parameters, p_pm = parameters,
                                        p_mp = parameters;
                    p_pp[static_cast<std::size_t>(i)] += hi;
                    p_pp[static_cast<std::size_t>(j)] += hj;
                    p_mm[static_cast<std::size_t>(i)] -= hi;
                    p_mm[static_cast<std::size_t>(j)] -= hj;
                    p_pm[static_cast<std::size_t>(i)] += hi;
                    p_pm[static_cast<std::size_t>(j)] -= hj;
                    p_mp[static_cast<std::size_t>(i)] -= hi;
                    p_mp[static_cast<std::size_t>(j)] += hj;
                    double f_pp = spatial_gev_->data_log_likelihood(p_pp);
                    double f_mm = spatial_gev_->data_log_likelihood(p_mm);
                    double f_pm = spatial_gev_->data_log_likelihood(p_pm);
                    double f_mp = spatial_gev_->data_log_likelihood(p_mp);
                    double val = (f_pp - f_pm - f_mp + f_mm) / (4 * hi * hj);
                    h[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = val;
                    h[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = val;
                }
            }
        }

        Grid jmat(static_cast<std::size_t>(n_params),
                  std::vector<double>(static_cast<std::size_t>(n_params), 0.0));
        std::vector<double> pointwise_ll = spatial_gev_->pointwise_data_log_likelihood(parameters);

        for (std::size_t obs = 0; obs < pointwise_ll.size(); ++obs) {
            std::vector<double> score(static_cast<std::size_t>(n_params), 0.0);
            for (int k = 0; k < n_params; ++k) {
                std::vector<double> p_plus = parameters, p_minus = parameters;
                double hh = std::fabs(parameters[static_cast<std::size_t>(k)]) * eps + eps;
                p_plus[static_cast<std::size_t>(k)] += hh;
                p_minus[static_cast<std::size_t>(k)] -= hh;
                std::vector<double> ll_plus = spatial_gev_->pointwise_data_log_likelihood(p_plus);
                std::vector<double> ll_minus = spatial_gev_->pointwise_data_log_likelihood(p_minus);
                score[static_cast<std::size_t>(k)] = (ll_plus[obs] - ll_minus[obs]) / (2 * hh);
            }
            for (int i = 0; i < n_params; ++i)
                for (int j = 0; j < n_params; ++j)
                    jmat[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] +=
                        score[static_cast<std::size_t>(i)] * score[static_cast<std::size_t>(j)];
        }

        std::optional<Grid> h_inv = invert_matrix(h);
        if (!h_inv) {
            godambe_covariance_ = jmat;  // Return J if H is singular (C# 1262-1264).
            return jmat;
        }
        Grid temp = multiply_matrices(*h_inv, jmat);
        godambe_covariance_ = multiply_matrices(temp, *h_inv);
        return *godambe_covariance_;
    }

    // C# `InflatePosteriorCovariance` (C# 1295): compute the variance inflation factor and widen each
    // site's credible intervals by sqrt(VIF). Returns the factor.
    double inflate_posterior_covariance() {
        variance_inflation_factor_ = spatial_gev_->compute_variance_inflation_factor();
        if (!site_results_ || !is_estimated()) return variance_inflation_factor_;

        double sqrt_vif = std::sqrt(variance_inflation_factor_);
        for (SpatialGEVSiteResults& site : *site_results_) {
            double loc_mid = site.location_mean;
            double loc_half = (site.location_upper - site.location_lower) / 2;
            site.location_lower = loc_mid - loc_half * sqrt_vif;
            site.location_upper = loc_mid + loc_half * sqrt_vif;

            double scl_mid = site.scale_mean;
            double scl_half = (site.scale_upper - site.scale_lower) / 2;
            site.scale_lower = std::max(scl_mid - scl_half * sqrt_vif, 0.0);
            site.scale_upper = scl_mid + scl_half * sqrt_vif;

            double shp_mid = site.shape_mean;
            double shp_half = (site.shape_upper - site.shape_lower) / 2;
            site.shape_lower = shp_mid - shp_half * sqrt_vif;
            site.shape_upper = shp_mid + shp_half * sqrt_vif;

            for (std::size_t p = 0; p < site.quantile_mean.size(); ++p) {
                double q_mid = site.quantile_mean[p];
                double q_half = (site.quantile_upper[p] - site.quantile_lower[p]) / 2;
                site.quantile_lower[p] = q_mid - q_half * sqrt_vif;
                site.quantile_upper[p] = q_mid + q_half * sqrt_vif;
            }
        }
        return variance_inflation_factor_;
    }

    // C# `RunSpatialBootstrapAsync` (C# 1372), synchronous. Spatial block bootstrap: resample
    // spatial blocks, re-fit a short MCMC per replicate, and overwrite each site's credible bounds
    // with the bootstrap percentiles. (Seeded MersenneTwister; deviation 6, compiled but driven only
    // by the X12 emitter.)
    void run_spatial_bootstrap(int n_bootstrap = 200, int block_size = 0) {
        if (!is_estimated() || !bayesian_analysis_.results())
            throw std::runtime_error("Analysis must be run before bootstrap.");
        if (block_size <= 0)
            block_size = std::max(2, static_cast<int>(std::sqrt(spatial_gev_->sites())));

        bestfit::numerics::sampling::MersenneTwister prng(
            static_cast<std::uint32_t>(bayesian_analysis_.prng_seed()));
        const std::vector<double>& probs = probability_ordinates_.values();
        int n_probs = static_cast<int>(probs.size());
        int n_sites = spatial_gev_->sites();

        std::vector<int> block_assignments = create_spatial_blocks(block_size);
        int n_blocks = *std::max_element(block_assignments.begin(), block_assignments.end()) + 1;

        // boot_quantiles[site][prob][b]; boot_{loc,scl,shp}[site][b].
        std::vector<std::vector<std::vector<double>>> boot_quantiles(
            static_cast<std::size_t>(n_sites),
            std::vector<std::vector<double>>(
                static_cast<std::size_t>(n_probs),
                std::vector<double>(static_cast<std::size_t>(n_bootstrap), 0.0)));
        std::vector<std::vector<double>> boot_location(
            static_cast<std::size_t>(n_sites),
            std::vector<double>(static_cast<std::size_t>(n_bootstrap), 0.0));
        std::vector<std::vector<double>> boot_scale(
            static_cast<std::size_t>(n_sites),
            std::vector<double>(static_cast<std::size_t>(n_bootstrap), 0.0));
        std::vector<std::vector<double>> boot_shape(
            static_cast<std::size_t>(n_sites),
            std::vector<double>(static_cast<std::size_t>(n_bootstrap), 0.0));

        for (int b = 0; b < n_bootstrap; ++b) {
            // Resample blocks (the resampled-site / boot-data matrix the C# builds here is dead --
            // it never sets the resampled data back into boot_model, so the replicate is a plain
            // clone re-fit; that no-op construction is dropped, preserving identical behavior).
            std::vector<int> resampled_blocks;
            while (static_cast<int>(resampled_blocks.size()) < n_blocks)
                resampled_blocks.push_back(prng.next(n_blocks));

            SpatialGEV boot_model = spatial_gev_->clone();
            bestfit::estimation::BayesianAnalysis boot_bayes(boot_model);
            boot_bayes.set_iterations(500);
            boot_bayes.set_warmup_iterations(250);
            boot_bayes.set_thinning_interval(5);

            bool ok = false;
            try {
                boot_bayes.estimate();
                ok = boot_bayes.is_estimated() && boot_bayes.results().has_value();
            } catch (...) {
                ok = false;
            }

            if (ok) {
                boot_model.set_parameter_values(boot_bayes.results()->map.values);
                for (int j = 0; j < n_sites; ++j) {
                    std::vector<double> gev_params = boot_model.get_gev_parameters(j);
                    boot_location[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)] =
                        gev_params[0];
                    boot_scale[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)] =
                        gev_params[1];
                    boot_shape[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)] =
                        gev_params[2];
                    for (int p = 0; p < n_probs; ++p)
                        boot_quantiles[static_cast<std::size_t>(j)][static_cast<std::size_t>(p)]
                                      [static_cast<std::size_t>(b)] =
                            boot_model.inverse_cdf(1.0 - probs[static_cast<std::size_t>(p)], j);
                }
            } else {
                double nan = std::numeric_limits<double>::quiet_NaN();
                for (int j = 0; j < n_sites; ++j) {
                    boot_location[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)] = nan;
                    boot_scale[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)] = nan;
                    boot_shape[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)] = nan;
                    for (int p = 0; p < n_probs; ++p)
                        boot_quantiles[static_cast<std::size_t>(j)][static_cast<std::size_t>(p)]
                                      [static_cast<std::size_t>(b)] = nan;
                }
            }
        }

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        if (!site_results_) return;
        std::vector<SpatialGEVSiteResults>& site_results = *site_results_;

        for (int j = 0; j < n_sites; ++j) {
            std::vector<double> valid_loc, valid_scl, valid_shp;
            for (int b = 0; b < n_bootstrap; ++b) {
                double lv = boot_location[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)];
                if (!std::isnan(lv)) {
                    valid_loc.push_back(lv);
                    valid_scl.push_back(
                        boot_scale[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)]);
                    valid_shp.push_back(
                        boot_shape[static_cast<std::size_t>(j)][static_cast<std::size_t>(b)]);
                }
            }

            if (valid_loc.size() >= 10) {
                SpatialGEVSiteResults& sr = site_results[static_cast<std::size_t>(j)];
                std::sort(valid_loc.begin(), valid_loc.end());
                std::sort(valid_scl.begin(), valid_scl.end());
                std::sort(valid_shp.begin(), valid_shp.end());
                sr.location_lower = bestfit::numerics::data::percentile(valid_loc, alpha / 2, true);
                sr.location_upper =
                    bestfit::numerics::data::percentile(valid_loc, 1 - alpha / 2, true);
                sr.scale_lower = bestfit::numerics::data::percentile(valid_scl, alpha / 2, true);
                sr.scale_upper = bestfit::numerics::data::percentile(valid_scl, 1 - alpha / 2, true);
                sr.shape_lower = bestfit::numerics::data::percentile(valid_shp, alpha / 2, true);
                sr.shape_upper = bestfit::numerics::data::percentile(valid_shp, 1 - alpha / 2, true);

                for (int p = 0; p < n_probs; ++p) {
                    std::vector<double> valid_q;
                    for (int b = 0; b < n_bootstrap; ++b) {
                        double qv =
                            boot_quantiles[static_cast<std::size_t>(j)][static_cast<std::size_t>(p)]
                                          [static_cast<std::size_t>(b)];
                        if (!std::isnan(qv)) valid_q.push_back(qv);
                    }
                    if (valid_q.size() >= 10) {
                        std::sort(valid_q.begin(), valid_q.end());
                        sr.quantile_lower[static_cast<std::size_t>(p)] =
                            bestfit::numerics::data::percentile(valid_q, alpha / 2, true);
                        sr.quantile_upper[static_cast<std::size_t>(p)] =
                            bestfit::numerics::data::percentile(valid_q, 1 - alpha / 2, true);
                    }
                }
            }
        }
    }

    // C# `Validate` (C# 1685): model + Bayesian-analysis validations. The C# `ProbabilityOrdinates ==
    // null` check is unrepresentable (value member, never null) and is dropped. const per the A4
    // IAnalysis contract.
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        bestfit::models::ValidationResult model_valid = spatial_gev_->validate();
        if (!model_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              model_valid.validation_messages.begin(),
                                              model_valid.validation_messages.end());
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
    // A mean/lower/upper credible band computed from a sample (C# Array.Sort + ParallelMean +
    // Percentile(., ., true) triple; deviation 4). Takes the sample by value (sorted in place).
    struct Band {
        double mean;
        double lower;
        double upper;
    };
    static Band band(std::vector<double> vals, double alpha) {
        std::sort(vals.begin(), vals.end());
        return {bestfit::numerics::data::mean(vals),
                bestfit::numerics::data::percentile(vals, alpha / 2.0, true),
                bestfit::numerics::data::percentile(vals, 1.0 - alpha / 2.0, true)};
    }

    // Null-guard helper for the ctor init list (C# `?? throw new ArgumentNullException`).
    static std::unique_ptr<SpatialGEV> require_non_null(std::unique_ptr<SpatialGEV> model) {
        if (model == nullptr) throw std::invalid_argument("spatialGEV");  // C# ArgumentNullException
        return model;
    }

    // C# `HandleOrdinatesChanged` (C# 450), explicit-invalidation form (INPC dropped): if estimated
    // with a valid grid, reprocess; if estimated with an invalid grid, clear the ordinate-keyed
    // outputs; on a fresh (unestimated) analysis, do nothing.
    void handle_ordinates_changed() {
        if (probability_ordinates_.validate().is_valid) {
            if (is_estimated()) {
                create_site_results();
                create_uncertainty_analysis_results();
            }
        } else if (is_estimated()) {
            clear_uncertainty_analysis_results();
        }
    }

    // C# `CreateSpatialBlocks` (C# 1543): greedy nearest-neighbour spatial clustering (fixed seed
    // 12345 MersenneTwister for the block seeds; deviation 6).
    std::vector<int> create_spatial_blocks(int block_size) const {
        int n_sites = spatial_gev_->sites();
        int n_blocks = std::max(1, (n_sites + block_size - 1) / block_size);
        std::vector<int> assignments(static_cast<std::size_t>(n_sites), 0);

        std::vector<int> unassigned(static_cast<std::size_t>(n_sites));
        for (int i = 0; i < n_sites; ++i) unassigned[static_cast<std::size_t>(i)] = i;
        bestfit::numerics::sampling::MersenneTwister prng(12345);

        for (int block = 0; block < n_blocks && !unassigned.empty(); ++block) {
            int start_idx = prng.next(static_cast<int>(unassigned.size()));
            int start_site = unassigned[static_cast<std::size_t>(start_idx)];
            unassigned.erase(unassigned.begin() + start_idx);
            assignments[static_cast<std::size_t>(start_site)] = block;

            int site_count = 1;
            while (site_count < block_size && !unassigned.empty()) {
                double min_dist = std::numeric_limits<double>::max();
                int nearest_idx = -1;
                for (std::size_t idx = 0; idx < unassigned.size(); ++idx) {
                    int site = unassigned[idx];
                    double dist = bestfit::numerics::distance(
                        spatial_gev_->coordinates()[static_cast<std::size_t>(site)][0],
                        spatial_gev_->coordinates()[static_cast<std::size_t>(site)][1],
                        spatial_gev_->coordinates()[static_cast<std::size_t>(start_site)][0],
                        spatial_gev_->coordinates()[static_cast<std::size_t>(start_site)][1]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        nearest_idx = static_cast<int>(idx);
                    }
                }
                if (nearest_idx >= 0) {
                    assignments[static_cast<std::size_t>(unassigned[static_cast<std::size_t>(
                        nearest_idx)])] = block;
                    unassigned.erase(unassigned.begin() + nearest_idx);
                    ++site_count;
                }
            }
        }
        return assignments;
    }

    // C# `InvertMatrix` (C# 1598): Gauss-Jordan with partial pivoting; std::nullopt if singular.
    static std::optional<Grid> invert_matrix(const Grid& a) {
        int n = static_cast<int>(a.size());
        Grid aug(static_cast<std::size_t>(n),
                 std::vector<double>(static_cast<std::size_t>(2 * n), 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j)
                aug[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    a[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            aug[static_cast<std::size_t>(i)][static_cast<std::size_t>(i + n)] = 1;
        }

        for (int col = 0; col < n; ++col) {
            int max_row = col;
            for (int row = col + 1; row < n; ++row)
                if (std::fabs(aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]) >
                    std::fabs(aug[static_cast<std::size_t>(max_row)][static_cast<std::size_t>(col)]))
                    max_row = row;
            std::swap(aug[static_cast<std::size_t>(col)], aug[static_cast<std::size_t>(max_row)]);
            if (std::fabs(aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)]) < 1e-12)
                return std::nullopt;
            double scale = aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
            for (int j = 0; j < 2 * n; ++j)
                aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)] /= scale;
            for (int row = 0; row < n; ++row) {
                if (row != col) {
                    double factor =
                        aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
                    for (int j = 0; j < 2 * n; ++j)
                        aug[static_cast<std::size_t>(row)][static_cast<std::size_t>(j)] -=
                            factor * aug[static_cast<std::size_t>(col)][static_cast<std::size_t>(j)];
                }
            }
        }

        Grid inv(static_cast<std::size_t>(n),
                 std::vector<double>(static_cast<std::size_t>(n), 0.0));
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                inv[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                    aug[static_cast<std::size_t>(i)][static_cast<std::size_t>(j + n)];
        return inv;
    }

    // C# `MultiplyMatrices` (C# 1663).
    static Grid multiply_matrices(const Grid& a, const Grid& b) {
        int m = static_cast<int>(a.size());
        int n = b.empty() ? 0 : static_cast<int>(b[0].size());
        int k = a.empty() ? 0 : static_cast<int>(a[0].size());
        Grid c(static_cast<std::size_t>(m), std::vector<double>(static_cast<std::size_t>(n), 0.0));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) {
                double sum = 0;
                for (int l = 0; l < k; ++l)
                    sum += a[static_cast<std::size_t>(i)][static_cast<std::size_t>(l)] *
                           b[static_cast<std::size_t>(l)][static_cast<std::size_t>(j)];
                c[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = sum;
            }
        return c;
    }

    // Owned model (deviation 1). Declared BEFORE bayesian_analysis_ so it is constructed first
    // (BayesianAnalysis stores a reference into it).
    std::unique_ptr<SpatialGEV> spatial_gev_;
    bestfit::estimation::BayesianAnalysis bayesian_analysis_;

    ProbabilityOrdinates probability_ordinates_;
    SpatialGEVUncertaintyMethod uncertainty_method_ = SpatialGEVUncertaintyMethod::BayesianPosterior;

    // Result objects (C# nullable -> optional; deviation 2).
    std::optional<UncertaintyAnalysisResults> analysis_results_;
    std::optional<std::vector<SpatialGEVSiteResults>> site_results_;
    std::optional<SpatialGEVCrossValidationResults> cross_validation_results_;
    std::optional<Grid> godambe_covariance_;
    double variance_inflation_factor_ = 1.0;
};

}  // namespace bestfit::analyses
