// ported from: Analyses/Bivariate/CoincidentFrequencyAnalysis.cs @ fc28c0c
//
// Coincident frequency analysis (X6). Consumes a fitted BivariateAnalysis (X3) -- two marginals
// X, Y plus a copula with its posterior copula chain -- plus an M x N response grid Z = f(X, Y),
// and constructs an UncertaintyAnalysisResults BY HAND via the conditional-frequency law of total
// probability over Y:
//
//   F_Z(z) = sum_j [ C(u_j, v_{j+1}) - C(u_j, v_j) ],   AEP(z) = 1 - F_Z(z),
//
// where u_j = F_X(x*_j) inverts the response column j at level z (interpolating in
// (z, zeta = Phi^-1(F_X(x))) probability-paper space) and v are the Y bin edges in copula space.
// This analysis runs NO MCMC of its own (it only reads the upstream copula posterior); the
// per-realisation loop is a plain serial average + percentile band.
//
// DEVIATIONS from the C# source, all deliberate:
//
//   1. WPF / ASYNC / THREADING DROPPED (mirrors the AnalysisBase + BivariateAnalysis precedent):
//      `RunAsync` async shell / `SafeProgressReporter` / `CancellationToken` / `Parallel.For`
//      (serial here -- the per-realisation writes are independent and the mean/percentile
//      reductions are order-independent, so the serial result is numerically identical) /
//      `AnalysisStarting`/`AnalysisCompleted` events / `_reprocessGate` / `OnAnalysis*` /
//      `Debug.WriteLine` / all `RaisePropertyChange`/INotifyPropertyChanged cascades. The C#
//      async `RunAsync` + `CreateFrequencyAnalysisResultsAsync` collapse into a synchronous
//      `run()` + `create_frequency_analysis_results()`.
//
//   2. XML DROPPED (project-wide non-port): the `XElement` ctor (C# 89-146), `ToXElement`
//      (1156), `JoinDoubles`/`ParseDoubleArray`, and the UI-restore setters
//      `SetAnalysisResults`/`RestoreAnalysisResults`/`SetZOutputValues` (C# 943-981). None have a
//      compute-layer caller here.
//
//   3. PRESENTATION SETTINGS. C# CFA owns a full `BayesianAnalysis` built via the parameterless
//      `BayesianAnalysis()` ctor purely for its presentation settings (CredibleIntervalWidth,
//      OutputLength, PointEstimator) -- CFA runs no chain. The ported `estimation::BayesianAnalysis`
//      REQUIRES a `ModelBase&` (a C++ reference cannot be null) and drags in the entire MCMC-knob
//      surface, so a model-less instance is not expressible. This port therefore owns a small
//      `BayesianSettings` value holding EXACTLY the three settings CFA reads. It is a distinct
//      object from the upstream BivariateAnalysis's `BayesianAnalysis` (the C#
//      `BayesianAnalysis_OwnedNotProxiedFromUpstream` invariant), reuses `estimation::PointEstimateType`
//      for the estimator field, and defaults to CredibleIntervalWidth = 0.90.
//
//   4. OWNERSHIP. The upstream `BivariateAnalysis` is a GC reference in C#; here CFA holds a
//      NON-OWNING `const BivariateAnalysis*` (nullable, mirroring the settable C# property + the
//      default ctor's null state). The ctor null-guard maps the C# ArgumentNullException. CFA only
//      READS the upstream (marginals/copula/posterior), so const access is faithful. The caller
//      must keep the BivariateAnalysis (and its marginals) alive for CFA's lifetime.
//
//   5. RESPONSE SHAPE. C# `double[,]` -> `std::vector<std::vector<double>>` (M rows x N cols),
//      always rectangular in the ported/consumer paths. The null-array ctor guards (X/Y/response)
//      collapse -- a `std::vector` cannot be null -- so only the BivariateAnalysis-pointer null
//      guard is testable (the vector-not-nullable pattern, mirroring BivariateAnalysis dev. 3).
//
//   6. INVALIDATION-ON-MUTATE. The C# input setters + `ClearResults` (326) each `RaisePropertyChange`;
//      with no notification system, every input setter simply calls `clear_results()` (nulls both
//      outputs + resets IsEstimated). The `BayesianAnalysis_PropertyChanged` reprocess/point-estimate
//      cascade (C# 301-323) is dropped with the notification system (matching X3/X4).
//
//   7. WHITE-BOX STATICS. The deterministic surface-integration helpers `build_z_output_bins`,
//      `compute_y_bin_edges`, `build_x_zetas`, `build_v_edges`, and `compute_fz_at_bin` are `private
//      static` in C#; they are exposed as PUBLIC static here so the C++-only ctest can pin the
//      hand-derived conditional-frequency oracle directly (deterministic math; no MCMC needed).
//      `find_u_star_in_column` / `clamp_unit` stay private (exercised through `compute_fz_at_bin`).
//
//   8. PRESENTATION HELPERS SKIPPED: `GetPointEstimateDistribution` / `GetEmpiricalDistribution`
//      (C# 862-940) build `EmpiricalDistribution` objects for GUI plotting; no ported/consumer path
//      needs them (they would require porting the EmpiricalDistribution SortOrder ctor). Skipped;
//      the exact per-draw AEP curve is available via `compute_aep_curve_for_draw` if a follow-up
//      needs it.
//
//   9. RUN ERROR REPORTING. With events dropped, `run()` lets a validation failure propagate as a
//      `std::runtime_error` (the C# `InvalidOperationException`); other exceptions propagate to the
//      caller instead of routing through the removed `AnalysisCompleted` (mirrors X3 dev. 5).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/analyses/bivariate/bivariate_analysis.hpp"
#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/data/statistics.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/copulas/base/bivariate_copula.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"

namespace corehydro::analyses::bivariate {

class CoincidentFrequencyAnalysis : public AnalysisBase {
   public:
    using BivariateAnalysis = corehydro::analyses::BivariateAnalysis;
    using BivariateCopula = corehydro::numerics::distributions::copulas::BivariateCopula;
    using UnivariateDistributionBase = corehydro::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using MCMCResults = corehydro::numerics::sampling::mcmc::MCMCResults;
    using PointEstimateType = corehydro::estimation::PointEstimateType;
    using Normal = corehydro::numerics::distributions::Normal;
    using Matrix = std::vector<std::vector<double>>;

    // Presentation settings owned by CFA (deviation 3). Holds only the three fields CFA reads.
    struct BayesianSettings {
        double credible_interval_width = 0.90;
        int output_length = 10000;
        PointEstimateType point_estimator = PointEstimateType::PosteriorMean;
    };

    // C# NumberOfBins default (C# 156).
    static constexpr int kDefaultNumberOfBins = 50;

    // C# parameterless ctor (C# 54-61): empty inputs.
    CoincidentFrequencyAnalysis() = default;

    // C# ctor `CoincidentFrequencyAnalysis(BivariateAnalysis, double[] x, double[] y, double[,] z)`
    // (C# 72-80). The BivariateAnalysis null-guard maps the C# ArgumentNullException.
    CoincidentFrequencyAnalysis(const BivariateAnalysis* bivariate_analysis, std::vector<double> x_values,
                                std::vector<double> y_values, Matrix bivariate_response)
        : bivariate_analysis_(require_non_null(bivariate_analysis)),
          x_values_(std::move(x_values)),
          y_values_(std::move(y_values)),
          bivariate_response_(std::move(bivariate_response)) {}

    ~CoincidentFrequencyAnalysis() override = default;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `BivariateAnalysis` (C# 170). const reference to the upstream fitted analysis.
    const BivariateAnalysis& bivariate_analysis() const { return *bivariate_analysis_; }
    void set_bivariate_analysis(const BivariateAnalysis* value) {
        bivariate_analysis_ = value;
        clear_results();
    }

    // C# `BayesianAnalysis` (C# 163): the owned presentation settings (deviation 3).
    BayesianSettings& bayesian_analysis() { return bayesian_settings_; }
    const BayesianSettings& bayesian_analysis() const { return bayesian_settings_; }

    // C# `XValues` (C# 192): primary ordinates. Setter clears results.
    const std::vector<double>& x_values() const { return x_values_; }
    void set_x_values(std::vector<double> value) {
        x_values_ = std::move(value);
        clear_results();
    }

    // C# `YValues` (C# 207): secondary ordinates. Setter clears results.
    const std::vector<double>& y_values() const { return y_values_; }
    void set_y_values(std::vector<double> value) {
        y_values_ = std::move(value);
        clear_results();
    }

    // C# `BivariateResponse` (C# 223): the M x N response surface. Setter clears results.
    const Matrix& bivariate_response() const { return bivariate_response_; }
    void set_bivariate_response(Matrix value) {
        bivariate_response_ = std::move(value);
        clear_results();
    }

    // C# `NumberOfBins` (C# 240): setter clears results only on an actual change.
    int number_of_bins() const { return number_of_bins_; }
    void set_number_of_bins(int value) {
        if (number_of_bins_ != value) {
            number_of_bins_ = value;
            clear_results();
        }
    }

    // C# `MarginalXChain` / `MarginalYChain` (C# 259-265): optional marginal posterior chains
    // (non-owning; nullptr => point-estimate marginal fall-back). Plain get/set.
    const MCMCResults* marginal_x_chain() const { return marginal_x_chain_; }
    void set_marginal_x_chain(const MCMCResults* value) { marginal_x_chain_ = value; }
    const MCMCResults* marginal_y_chain() const { return marginal_y_chain_; }
    void set_marginal_y_chain(const MCMCResults* value) { marginal_y_chain_ = value; }

    // C# `AnalysisResults` (C# 271): null until estimated.
    const UncertaintyAnalysisResults* analysis_results() const {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // C# `ZOutputValues` (C# 278): Z output bins; null until estimated.
    const std::vector<double>* z_output_values() const {
        return z_output_values_ ? &*z_output_values_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 329): nulls both outputs and resets IsEstimated.
    void clear_results() {
        analysis_results_.reset();
        z_output_values_.reset();
        set_is_estimated(false);
    }

    // C# `RunAsync` (C# 339), synchronous. Validate guard -> clear -> IsEstimated = true ->
    // build results (deviation 9: a validation failure throws instead of routing through the
    // removed AnalysisCompleted).
    void run() override {
        auto v = validate();
        if (!v.is_valid) {
            std::string joined;
            for (std::size_t i = 0; i < v.validation_messages.size(); ++i) {
                if (i != 0) joined += "; ";
                joined += v.validation_messages[i];
            }
            throw std::runtime_error(joined);
        }
        clear_results();
        // C# sets _isEstimated = true BEFORE building results (C# 378).
        set_is_estimated(true);
        create_frequency_analysis_results();
    }

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 417), synchronous. Builds ZOutputValues + the
    // Y bin edges, the point-estimate ModeCurve, then the posterior MeanCurve + credible band from
    // the upstream copula chain (Parallel.For -> serial). realz <= 0 => point-estimate-only path.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        z_output_values_.reset();

        if (bivariate_analysis_ == nullptr) return;

        int K = number_of_bins_;
        double alpha = 1.0 - bayesian_settings_.credible_interval_width;

        // Step 1 -- Z output bins.
        z_output_values_ = build_z_output_bins(bivariate_response_, K);
        // Step 2 -- Y bin edges.
        std::vector<double> y_edges = compute_y_bin_edges(y_values_);

        // Determine realisation count: copula chain is required; marginal chains optional.
        const std::optional<MCMCResults>& copula_opt = bivariate_analysis_->bayesian_analysis().results();
        std::int64_t copula_realz = copula_opt ? static_cast<std::int64_t>(copula_opt->output.size()) : 0;
        std::int64_t mx_realz = marginal_x_chain_ ? static_cast<std::int64_t>(marginal_x_chain_->output.size())
                                                  : kIntMax;
        std::int64_t my_realz = marginal_y_chain_ ? static_cast<std::int64_t>(marginal_y_chain_->output.size())
                                                  : kIntMax;
        std::int64_t realz = std::min(copula_realz, std::min(mx_realz, my_realz));

        UncertaintyAnalysisResults r;
        r.mode_curve = compute_point_estimate_aep_curve(y_edges);
        r.mean_curve.assign(static_cast<std::size_t>(K), 0.0);
        r.confidence_intervals.assign(static_cast<std::size_t>(K), {0.0, 0.0});

        if (realz <= 0) {
            // Point-estimate-only path (C# 462-473): mean curve == mode curve, CIs are NaN bands.
            r.mean_curve = r.mode_curve;
            for (int k = 0; k < K; ++k) {
                r.confidence_intervals[static_cast<std::size_t>(k)][0] = kNaN;
                r.confidence_intervals[static_cast<std::size_t>(k)][1] = kNaN;
            }
            analysis_results_ = std::move(r);
            return;
        }

        int R = static_cast<int>(realz);
        // aep_store[k][r] = AEP at output bin k under realisation r (C# 477-478).
        std::vector<std::vector<double>> aep_store(static_cast<std::size_t>(K),
                                                   std::vector<double>(static_cast<std::size_t>(R)));
        for (int rr = 0; rr < R; ++rr) {
            const std::vector<double>* mx = (marginal_x_chain_ &&
                                             rr < static_cast<int>(marginal_x_chain_->output.size()))
                                                ? &marginal_x_chain_->output[static_cast<std::size_t>(rr)].values
                                                : nullptr;
            const std::vector<double>* my = (marginal_y_chain_ &&
                                             rr < static_cast<int>(marginal_y_chain_->output.size()))
                                                ? &marginal_y_chain_->output[static_cast<std::size_t>(rr)].values
                                                : nullptr;
            std::vector<double> aep = compute_aep_curve_for_draw(
                &copula_opt->output[static_cast<std::size_t>(rr)].values, mx, my, y_edges);
            for (int k = 0; k < K; ++k) aep_store[static_cast<std::size_t>(k)][static_cast<std::size_t>(rr)] = aep[static_cast<std::size_t>(k)];
        }

        // Aggregate (C# 517-524).
        for (int k = 0; k < K; ++k) {
            std::vector<double>& aeps = aep_store[static_cast<std::size_t>(k)];
            r.mean_curve[static_cast<std::size_t>(k)] = corehydro::numerics::data::mean(aeps);
            std::sort(aeps.begin(), aeps.end());
            r.confidence_intervals[static_cast<std::size_t>(k)][0] =
                corehydro::numerics::data::percentile(aeps, alpha / 2.0, true);
            r.confidence_intervals[static_cast<std::size_t>(k)][1] =
                corehydro::numerics::data::percentile(aeps, 1.0 - alpha / 2.0, true);
        }

        analysis_results_ = std::move(r);
    }

    // --- Static surface-integration helpers (public for white-box tests; deviation 7) --------

    // C# `BuildZOutputBins` (C# 558-582): K evenly-spaced Z values spanning the response min..max.
    static std::vector<double> build_z_output_bins(const Matrix& response, int num_bins) {
        double z_min = std::numeric_limits<double>::max();
        double z_max = std::numeric_limits<double>::lowest();
        for (const auto& row : response) {
            for (double value : row) {
                if (value < z_min) z_min = value;
                if (value > z_max) z_max = value;
            }
        }

        std::vector<double> bins(static_cast<std::size_t>(num_bins));
        if (num_bins == 1 || z_max <= z_min) {
            for (int k = 0; k < num_bins; ++k) bins[static_cast<std::size_t>(k)] = z_min;
            return bins;
        }
        double step = (z_max - z_min) / (num_bins - 1);
        for (int k = 0; k < num_bins; ++k) bins[static_cast<std::size_t>(k)] = z_min + k * step;
        return bins;
    }

    // C# `ComputeYBinEdges` (C# 588-597): N+1 edges, +/-inf ends, interior midpoints.
    static std::vector<double> compute_y_bin_edges(const std::vector<double>& y_values) {
        std::size_t n = y_values.size();
        std::vector<double> edges(n + 1);
        edges[0] = -std::numeric_limits<double>::infinity();
        edges[n] = std::numeric_limits<double>::infinity();
        for (std::size_t j = 1; j < n; ++j) edges[j] = (y_values[j - 1] + y_values[j]) / 2.0;
        return edges;
    }

    // C# `BuildXZetas` (C# 732-742): zeta_i = Phi^-1(F_X(x_i)), the probability-paper ordinate.
    static std::vector<double> build_x_zetas(const std::vector<double>& x_values,
                                             const UnivariateDistributionBase& marginal_x) {
        std::vector<double> zetas(x_values.size());
        for (std::size_t i = 0; i < x_values.size(); ++i) {
            double u = clamp_unit(marginal_x.cdf(x_values[i]), kUEpsilon);
            zetas[i] = Normal::standard_z(u);
        }
        return zetas;
    }

    // C# `BuildVEdges` (C# 754-766): v_0 = 0, v_N = 1, interior v_j = F_Y(midpoint).
    static std::vector<double> build_v_edges(const std::vector<double>& y_values,
                                             const std::vector<double>& y_edges,
                                             const UnivariateDistributionBase& marginal_y) {
        std::size_t n = y_values.size();
        std::vector<double> v(n + 1);
        v[0] = 0.0;
        v[n] = 1.0;
        for (std::size_t j = 1; j < n; ++j) v[j] = clamp_unit(marginal_y.cdf(y_edges[j]), kUEpsilon);
        return v;
    }

    // C# `ComputeFZAtBin` (C# 635-661): F_Z(z) = sum_j [ C(u_j, v_{j+1}) - C(u_j, v_j) ] with the
    // copula boundary identities C(u,0)=0 (j=0) and C(u,1)=u (j=N-1) applied analytically.
    static double compute_fz_at_bin(double z, const Matrix& response, const std::vector<double>& x_zetas,
                                    const std::vector<double>& v_edges, const BivariateCopula& copula) {
        int M = static_cast<int>(response.size());
        int N = M > 0 ? static_cast<int>(response[0].size()) : 0;

        double fz = 0.0;
        for (int j = 0; j < N; ++j) {
            double u = clamp_unit(find_u_star_in_column(z, j, response, M, x_zetas), kUEpsilon);

            double upper_term = (j == N - 1) ? u : copula.cdf(u, v_edges[static_cast<std::size_t>(j + 1)]);
            double lower_term = (j == 0) ? 0.0 : copula.cdf(u, v_edges[static_cast<std::size_t>(j)]);
            double contribution = upper_term - lower_term;

            if (contribution > 0.0) fz += contribution;
        }

        if (fz < 0.0) fz = 0.0;
        if (fz > 1.0) fz = 1.0;
        return fz;
    }

    // --- Validate (C# 984-1151) -------------------------------------------------------------

    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        // BivariateAnalysis presence + estimated gate (each is an early return, per C#).
        if (bivariate_analysis_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Bivariate analysis is required.");
            return result;
        }
        const auto& bd = bivariate_analysis_->bivariate_distribution();
        const UnivariateDistributionBase* dist_x = bd.marginal_x() ? bd.marginal_x()->distribution() : nullptr;
        const UnivariateDistributionBase* dist_y = bd.marginal_y() ? bd.marginal_y()->distribution() : nullptr;
        if (dist_x == nullptr || dist_y == nullptr) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Bivariate analysis must have valid marginals and copula.");
            return result;
        }
        if (!bivariate_analysis_->is_estimated()) {
            result.is_valid = false;
            result.validation_messages.emplace_back(
                "Bivariate analysis has not been estimated yet. Run the bivariate analysis (or "
                "include it in the batch) before running the coincident frequency analysis.");
            return result;
        }

        // X / Y ordinate validation.
        if (x_values_.size() < 2) {
            result.is_valid = false;
            result.validation_messages.emplace_back("At least 2 X (primary) values are required.");
        } else {
            for (std::size_t i = 1; i < x_values_.size(); ++i) {
                if (!(x_values_[i] > x_values_[i - 1])) {
                    result.is_valid = false;
                    result.validation_messages.emplace_back("X (primary) values must be strictly ascending.");
                    break;
                }
            }
        }

        if (y_values_.size() < 2) {
            result.is_valid = false;
            result.validation_messages.emplace_back("At least 2 Y (secondary) values are required.");
        } else {
            for (std::size_t j = 1; j < y_values_.size(); ++j) {
                if (!(y_values_[j] > y_values_[j - 1])) {
                    result.is_valid = false;
                    result.validation_messages.emplace_back("Y (secondary) values must be strictly ascending.");
                    break;
                }
            }
        }

        // BivariateResponse validation.
        int rows = static_cast<int>(bivariate_response_.size());
        int cols = rows > 0 ? static_cast<int>(bivariate_response_[0].size()) : 0;
        if (rows != static_cast<int>(x_values_.size())) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Bivariate response must have " +
                                                    std::to_string(x_values_.size()) +
                                                    " rows (one per X value).");
        }
        if (cols != static_cast<int>(y_values_.size())) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Bivariate response must have " +
                                                    std::to_string(y_values_.size()) +
                                                    " columns (one per Y value).");
        }

        // No NaN / infinity.
        bool saw_bad_value = false;
        for (int i = 0; i < rows && !saw_bad_value; ++i) {
            for (int j = 0; j < static_cast<int>(bivariate_response_[static_cast<std::size_t>(i)].size()) && !saw_bad_value; ++j) {
                double value = bivariate_response_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
                if (std::isnan(value) || std::isinf(value)) saw_bad_value = true;
            }
        }
        if (saw_bad_value) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Bivariate response contains NaN or infinity values.");
        }

        // Monotonicity checks (only when the dimensions match, matching C# 1092).
        if (rows == static_cast<int>(x_values_.size()) && cols == static_cast<int>(y_values_.size())) {
            bool column_monotonic = true;
            for (int j = 0; j < cols && column_monotonic; ++j) {
                for (int i = 1; i < rows; ++i) {
                    if (!(bivariate_response_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] >
                          bivariate_response_[static_cast<std::size_t>(i - 1)][static_cast<std::size_t>(j)])) {
                        column_monotonic = false;
                        break;
                    }
                }
            }
            if (!column_monotonic) {
                result.is_valid = false;
                result.validation_messages.emplace_back(
                    "Bivariate response must be strictly increasing along X (each column).");
            }

            bool row_monotonic = true;
            for (int i = 0; i < rows && row_monotonic; ++i) {
                for (int j = 1; j < cols; ++j) {
                    if (!(bivariate_response_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] >
                          bivariate_response_[static_cast<std::size_t>(i)][static_cast<std::size_t>(j - 1)])) {
                        row_monotonic = false;
                        break;
                    }
                }
            }
            if (!row_monotonic) {
                result.is_valid = false;
                result.validation_messages.emplace_back(
                    "Bivariate response must be strictly increasing along Y (each row).");
            }
        }

        // NumberOfBins range (errors) + slow-run warning.
        if (number_of_bins_ < 5) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Error: NumberOfBins must be at least 5.");
        } else if (number_of_bins_ > 1000) {
            result.is_valid = false;
            result.validation_messages.emplace_back("Error: NumberOfBins must be at most 1000.");
        } else if (number_of_bins_ > 100) {
            result.validation_messages.emplace_back("Warning: NumberOfBins > 100 may result in slow run times.");
        }

        return result;
    }

    // --- Per-draw AEP curve (public: available to a follow-up that needs a single realisation) ---

    // C# `ComputeAEPCurveForDraw` (C# 791-819): clone copula + marginals, set params when provided,
    // pre-compute xZetas + vEdges once, then AEP[k] = 1 - F_Z(ZOutputValues[k]).
    std::vector<double> compute_aep_curve_for_draw(const std::vector<double>* copula_parameters,
                                                   const std::vector<double>* marginal_x_parameters,
                                                   const std::vector<double>* marginal_y_parameters,
                                                   const std::vector<double>& y_edges) const {
        int K = static_cast<int>(z_output_values_->size());
        const auto& bd = bivariate_analysis_->bivariate_distribution();

        std::unique_ptr<BivariateCopula> copula = bd.copula().clone();
        if (copula_parameters != nullptr) copula->set_copula_parameters(*copula_parameters);

        const UnivariateDistributionBase* dist_x = bd.marginal_x() ? bd.marginal_x()->distribution() : nullptr;
        const UnivariateDistributionBase* dist_y = bd.marginal_y() ? bd.marginal_y()->distribution() : nullptr;
        std::unique_ptr<UnivariateDistributionBase> mx = dist_x->clone();
        if (marginal_x_parameters != nullptr) mx->set_parameters(*marginal_x_parameters);
        std::unique_ptr<UnivariateDistributionBase> my = dist_y->clone();
        if (marginal_y_parameters != nullptr) my->set_parameters(*marginal_y_parameters);

        std::vector<double> x_zetas = build_x_zetas(x_values_, *mx);
        std::vector<double> v_edges = build_v_edges(y_values_, y_edges, *my);

        std::vector<double> aep(static_cast<std::size_t>(K));
        for (int k = 0; k < K; ++k) {
            double fz = compute_fz_at_bin((*z_output_values_)[static_cast<std::size_t>(k)], bivariate_response_,
                                          x_zetas, v_edges, *copula);
            aep[static_cast<std::size_t>(k)] = 1.0 - fz;
        }
        return aep;
    }

   private:
    static const BivariateAnalysis* require_non_null(const BivariateAnalysis* value) {
        if (value == nullptr) {
            throw std::invalid_argument("bivariateAnalysis");  // C# ArgumentNullException
        }
        return value;
    }

    // C# `ComputePointEstimateAEPCurve` (C# 832-853): point-estimate copula/marginal params per the
    // PointEstimator branch, then the shared per-draw AEP curve.
    std::vector<double> compute_point_estimate_aep_curve(const std::vector<double>& y_edges) const {
        const std::vector<double>* copula_parameters = nullptr;
        const std::vector<double>* marginal_x_parameters = nullptr;
        const std::vector<double>* marginal_y_parameters = nullptr;

        const std::optional<MCMCResults>& copula_opt = bivariate_analysis_->bayesian_analysis().results();
        if (bayesian_settings_.point_estimator == PointEstimateType::PosteriorMean) {
            if (copula_opt) copula_parameters = &copula_opt->posterior_mean.values;
            if (marginal_x_chain_) marginal_x_parameters = &marginal_x_chain_->posterior_mean.values;
            if (marginal_y_chain_) marginal_y_parameters = &marginal_y_chain_->posterior_mean.values;
        } else {
            if (copula_opt) copula_parameters = &copula_opt->map.values;
            if (marginal_x_chain_) marginal_x_parameters = &marginal_x_chain_->map.values;
            if (marginal_y_chain_) marginal_y_parameters = &marginal_y_chain_->map.values;
        }

        return compute_aep_curve_for_draw(copula_parameters, marginal_x_parameters, marginal_y_parameters,
                                          y_edges);
    }

    // C# `FindUStarInColumn` (C# 680-722): invert the response column at level z in (z, zeta) space
    // with end-segment extrapolation, then map back to u via Phi.
    static double find_u_star_in_column(double z, int col_idx, const Matrix& response, int M,
                                        const std::vector<double>& x_zetas) {
        std::size_t c = static_cast<std::size_t>(col_idx);
        double z_first = response[0][c];
        double z_last = response[static_cast<std::size_t>(M - 1)][c];
        double zeta_star;

        if (z <= z_first) {
            double dz = response[1][c] - z_first;
            zeta_star = (dz > 0.0) ? x_zetas[0] + (x_zetas[1] - x_zetas[0]) * (z - z_first) / dz : x_zetas[0];
        } else if (z >= z_last) {
            double dz = z_last - response[static_cast<std::size_t>(M - 2)][c];
            zeta_star = (dz > 0.0) ? x_zetas[static_cast<std::size_t>(M - 1)] +
                                         (x_zetas[static_cast<std::size_t>(M - 1)] -
                                          x_zetas[static_cast<std::size_t>(M - 2)]) *
                                             (z - z_last) / dz
                                   : x_zetas[static_cast<std::size_t>(M - 1)];
        } else {
            int i = 0;
            for (int k = 0; k < M - 1; ++k) {
                if (z >= response[static_cast<std::size_t>(k)][c] && z <= response[static_cast<std::size_t>(k + 1)][c]) {
                    i = k;
                    break;
                }
            }
            double z_low = response[static_cast<std::size_t>(i)][c];
            double z_high = response[static_cast<std::size_t>(i + 1)][c];
            zeta_star = (z_high > z_low)
                            ? x_zetas[static_cast<std::size_t>(i)] +
                                  (z - z_low) *
                                      (x_zetas[static_cast<std::size_t>(i + 1)] - x_zetas[static_cast<std::size_t>(i)]) /
                                      (z_high - z_low)
                            : x_zetas[static_cast<std::size_t>(i)];
        }

        return Normal::standard_cdf(zeta_star);
    }

    // C# `ClampUnit` (C# 772-777): clamp to (eps, 1 - eps).
    static double clamp_unit(double x, double eps) {
        if (x < eps) return eps;
        if (x > 1.0 - eps) return 1.0 - eps;
        return x;
    }

    static constexpr double kUEpsilon = 1e-12;
    static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    static constexpr std::int64_t kIntMax = static_cast<std::int64_t>(std::numeric_limits<int>::max());

    const BivariateAnalysis* bivariate_analysis_ = nullptr;
    std::vector<double> x_values_;
    std::vector<double> y_values_;
    Matrix bivariate_response_;
    int number_of_bins_ = kDefaultNumberOfBins;
    BayesianSettings bayesian_settings_;

    const MCMCResults* marginal_x_chain_ = nullptr;
    const MCMCResults* marginal_y_chain_ = nullptr;

    std::optional<UncertaintyAnalysisResults> analysis_results_;
    std::optional<std::vector<double>> z_output_values_;
};

}  // namespace corehydro::analyses::bivariate
