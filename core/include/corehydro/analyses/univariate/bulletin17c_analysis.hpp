// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/Bulletin17CAnalysis.cs @ c2e6192
//
// The Bulletin 17C flood-frequency analysis (GMM point estimate + uncertainty quantification).
// This header is delivered across THREE additive slices sharing one class:
//   * A7 (this file): the class skeleton, run(), the UncertaintyMethod dispatcher, and the DEFAULT
//     uncertainty path -- MultivariateNormal (GMM sandwich covariance -> Latin-Hypercube MVN
//     parameter ensemble -> UncertaintyAnalysisResults).
//   * A8 (adds into this same header): the parametric bootstrap + jackknife acceleration.
//   * A9 (adds into this same header): the Cohn-style delta-method CI machinery.
// A8/A9 extend this class ADDITIVELY; A7 declares only what the MVN slice needs.
//
// SHIPPED-DEFAULT DEVIATION (C# GOVERNS the enum, this port GOVERNS the shipped default):
//   The C# ctor (line 118) sets `UncertaintyMethod = LinkedMultivariateNormal`, while the field
//   initializer (line 228) is `MultivariateNormal`. LinkedMultivariateNormal now SHIPS (X8: its
//   ~13 link-builder helpers + InfluenceStatistics + the delta-method MVN sampler), but this port
//   keeps `MultivariateNormal` as the shipped C++ default. That is behavior-preserving: the C#
//   RunUncertaintyQuantification path (line 666-671) SILENTLY FALLS BACK to plain MultivariateNormal
//   whenever LinkedMVN returns null, and the LinkedMVN arm itself falls back to plain MVN inline.
//
// X8 PROVENANCE (LinkedMultivariateNormal, carried verbatim): the shipped C# path samples from
//   **MultivariateNormal** (NOT MultivariateStudentT), and the influence-function center shift
//   `etaHat[1]/[2] += shift` (C# 1044 / 1053) is **COMMENTED OUT** in the source. So
//   ComputeInfluenceStatistics is genuinely computed and then **DISCARDED**. This port mirrors that
//   EXACTLY: ComputeInfluenceStatistics + its two helpers are ported (real, reachable code), but no
//   MVT is applied and no center shift the C# source does not apply.
//
// UNCERTAINTY-METHOD DISPATCH (C# switch, lines 657-664):
//   * MultivariateNormal       -> SHIPPED here (get_parameter_sets_from_multivariate_normal).
//   * Bootstrap                -> ships in A8 (dispatch arm throws until then; clearly marked).
//   * LinkedMultivariateNormal -> SHIPPED (X8: get_parameter_sets_from_linked_multivariate_normal).
//   * BiasCorrectedBootstrap   -> SHIPPED (X9, reworked in T20:
//                                 get_parameter_sets_from_pivotal_bootstrap). After X9, the B17C
//                                 UncertaintyMethod dispatcher has NO throwing arm left.
//
// BAYESIANANALYSIS PLUMBING (C#-vs-port deviation, documented):
//   The C# ctor builds `new BayesianAnalysis()` whose `Model` is null -- this analysis uses GMM,
//   not MCMC, and the BayesianAnalysis is only a config + MCMCResults holder (output length, PRNG
//   seed, credible-interval width, point estimator, SetCustomMCMCResults, Results). The ported
//   BayesianAnalysis (Phase 4) holds a non-null `ModelBase&`, so this port backs it with a private
//   zero-parameter `PlumbingModel` stub. With zero parameters BayesianAnalysis::set_up_sampler()
//   early-returns (Sampler stays null), reproducing the C# null-model behavior EXACTLY -- the
//   plumbing BayesianAnalysis never builds a sampler and never runs a chain here.
//
// DROPPED / DEFERRED C# surface (each is documented; none is numerical for the shipped scope):
//   * The XML deserialization ctor (147-219), ToXElement (2382-2401), and all XElement plumbing.
//   * INotifyPropertyChanged: every RaisePropertyChange, the Model_PropertyChanged /
//     ProbabilityOrdinates_CollectionChanged / BayesianAnalysis_PropertyChanged handlers, and the
//     property setters' change cascades -- WPF binding, no compute content.
//   * CancellationTokenSource / CancelAnalysis / _reprocessGate / SafeProgressReporter / the
//     Stopwatch ElapsedTime/GMMElapsedTime/UncertaintyElapsedTime timing -- run-lifecycle plumbing.
//     `async Task RunAsync` -> synchronous `run()`; the `await Task.Run(...)` wrappers vanish.
//   * `Debug.WriteLine` + swallowed-exception guards -> silent no-throw guards (commented).
//   * SHIPPED in A8 (this header): GetParameterSetsFromParametricBootstrap (the Bootstrap dispatch
//     arm), AccelerationConstants (ported but UNCALLED -- the C# only references it from the
//     deferred BCa/pivot path), the BootstrapResults member, and the BootstrapDiagnostics DTO
//     (its own support header). See those methods for the per-line C# provenance.
//   * SHIPPED in A9 (this header): the Cohn-style delta-method CI machinery -- the public
//     ComputeCohnStyleConfidenceIntervals plus its private helpers (BuildQuadratureGrid,
//     CohnCholesky, BuildGridFromCholesky, ClampForCovariance, ClampForQuantile,
//     EvaluateQuantileSafe, WeightedCovariance, CohnAdjustedStudentTCI, EnforceMonotonicity) and the
//     CohnConfidenceIntervalResult DTO (its own support header). Deterministic (nested Gaussian
//     quadrature over the GMM fit -- no MCMC/bootstrap seed dependence).
//   * SHIPPED in X8 (this header): GetParameterSetsFromLinkedMultivariateNormal + its link-builder
//     helpers (CreatePositiveParameterLink / CreatePearson{Location,Scale}Link / CreateLocationLink /
//     CreateGammaShapeLink / OrientGammaWedsForLink / CleanWeds / SmoothStep / CreateGammaTailDelta /
//     StandardizedMagnitude / SafeStandardError / Relative{StandardError,UncertaintyScore} /
//     LogScaleFromRelativeStandardError) and ComputeInfluenceStatistics / InfluenceStatistics /
//     ComputeDegreesOfFreedomFromKurtosis / ComputeSkewnessFromInfluence (COMPUTED then DISCARDED --
//     see the X8 PROVENANCE note above).
//   * SHIPPED in X9, reworked in T20 (this header): GetParameterSetsFromPivotalBootstrap (the
//     BiasCorrectedBootstrap dispatch arm) -- the three-phase pivotal bootstrap (bootstrap fits
//     collecting theta*_b + Sigma*_b, guarded Yeo-Johnson/Log link fitting, and the seeded pivot
//     draws), plus its CreatePivotYeoJohnsonLink helper. DEFERRED / dropped: the ~617-line GMM
//     report generator, C# 3168-3785 -- GenerateGMMReport + its ReportAppend* / covariance-table
//     StringBuilder helpers (pure plain-text formatting, no compute content). AFormulaOverride is a
//     deferred A9-adjacent member (not declared here).
//   * T19 UPDATE (this header; upstream-sync, BestFit v2.0.0 @ c2e6192, commits 1b424e3 "Discard
//     failed bootstrap replicates instead of substituting parent parameters" + 71b7d4b "Extend
//     bootstrap diagnostics, report, and persistence for discard semantics" + 7efa9d0 "Improving
//     the bootstrap for B17C" -- the parametric-bootstrap arm ONLY):
//       - `get_parameter_sets_from_parametric_bootstrap` now matches the NET v2.0.0 C# exactly:
//         `maxRetries` 5 -> 10; the Mahalanobis rejection threshold is the ADAPTIVE
//         `ChiSquared(p).InverseCDF(1 - 1/(5B))` (was the fixed 0.9999); and each replicate warm-
//         starts via Task 18's `clone_with_data_frame()` + `set_parameter_values(thetaHat)` when
//         low outliers / uncertain / interval / threshold series are present on the parent
//         DataFrame (the C# `cloneWithDataFrame` local, mirrored EXACTLY) -- otherwise it clones
//         via the plain `clone()` + `set_data_frame()` path, unchanged from A8.
//       - IMPORTANT GROUND-TRUTH NOTE (the shipped v2.0.0 C# does NOT match every planning-doc
//         summary of 1b424e3's *commit message*): 1b424e3 introduced true discard-without-
//         substitution semantics (a null parameter-set slot, filtered downstream) plus a relaxed
//         "reject only hard Failure / non-finite" acceptance gate and a >50%-discarded abort
//         guard, but 7efa9d0 ("Improving the bootstrap for B17C", the LAST commit on this method
//         before the v2.0.0 tag) REVERTED both of those for GetParameterSetsFromParametricBootstrap
//         specifically -- the shipped acceptance gate is still the strict `!= Success`, and a
//         replicate that exhausts every retry still falls back to the parent vector `thetaHat`
//         (never discarded), so every bootstrap replicate slot is always populated and there is no
//         abort-on-high-discard-rate path in this arm. This port reproduces the ACTUAL shipped
//         behavior (verified by re-reading `Bulletin17CAnalysis.cs` at the `c2e6192` pin
//         byte-for-byte, not the commit message), since that is what the dotnet oracle gate
//         replays against. The per-replicate retry loop is factored into the new private
//         `collect_bootstrap_replicate_fit` helper (BootstrapReplicateFit), which T20's
//         pivot-bootstrap Phase 1 rewrite now shares (the SAME two 7efa9d0 changes above apply
//         there too -- see `get_parameter_sets_from_pivotal_bootstrap`).
//       - `BootstrapDiagnostics` gained `AttemptedReplicates` / `RetainedReplicates` (legacy
//         fallback to `TotalReplicates` / `ValidReplicates`) / `TransformFailures` / the five GMM
//         `OptimizationStatus` counters / `OptimizerFallbacks` (see its own header); NONE of these
//         are populated by this arm (the shipped C# never calls their increment/record methods
//         from GetParameterSetsFromParametricBootstrap), so a diagnostics object built by this
//         method always reads them through their legacy-fallback defaults.
//       - `UncertaintyDiagnosticMessage` (C# 409, a plain read-only string property + private
//         setter): ported as a class member/accessor for API completeness, but NOT populated by
//         this arm either (the shipped GetParameterSetsFromParametricBootstrap never calls
//         `SetUncertaintyDiagnosticMessage`); the other uncertainty paths that DO call it keep
//         their existing "silent no-throw guard" ports, unwired here.
//   * T20 UPDATE (this header; upstream-sync, BestFit v2.0.0 @ c2e6192 -- the PIVOT arm, renamed
//     upstream from GetParameterSetsFromPivotBootstrap to `GetParameterSetsFromPivotalBootstrap`):
//     the pivot arm is where the discard semantics 7efa9d0 reverted for the parametric arm
//     ACTUALLY shipped. `get_parameter_sets_from_pivotal_bootstrap` now reproduces the shipped
//     method exactly -- maxRetries 10, the parent covariance through `TryGetCovariance` +
//     `MakeSymmetricPositiveDefinite` (abort + diagnostic message on failure), the adaptive
//     `1 - 1/(5B)` Mahalanobis threshold, the conditional `cloneWithDataFrame` warm start, the
//     per-replicate covariance validation (TryGetCovariance -> regularize -> a throwaway Cholesky)
//     inside the retry loop, the guarded `CreatePivotYeoJohnsonLink` link builder, an abort (not a
//     parametric fallback) on a parent-Cholesky failure, a CLIPPED rather than rejected pivot, an
//     explicit non-finite check on the inverse-linked draw, and a Phase-3 failure that DISCARDS
//     the slot while incrementing `TransformFailures`, with `RetainedReplicates` set from the
//     survivors. `IncrementTransformFailure` + the `RetainedReplicates` setter are the ONLY two
//     new BootstrapDiagnostics members the shipped analysis populates ANYWHERE (IncrementAttempted
//     / RecordGMMStatus / AddOptimizerFallbacks are never called in the file), and
//     `IncrementPivotRejection` is now called nowhere (the z-limit became a clip). The single
//     numerically LIVE change: the pivot arm's `{ PenaltyIsRandom = false }` GMM initializer was
//     deleted upstream (0 occurrences at c2e6192), so the GMM default `true` applies and H is back
//     in the sandwich meat of every Sigma*_b. NOT a change (the planning text said otherwise; the
//     SHIPPED SOURCE GOVERNS): the links are still fitted over ALL B replicates -- there is no
//     compaction to accepted replicates -- and nothing is dropped for exceeding the z-limit.
//
// `EvaluateLogQuantileSafe` (C# 856-880) IS ported below per the A7 brief, though the shipped MVN
// path does not itself call it (the C# MVN sampler validates via ValidateParameters); it is the
// safe-quantile guard the LinkedMVN / Cohn paths use and lands here for structural fidelity.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/analyses/support/bootstrap_diagnostics.hpp"
#include "corehydro/analyses/support/cohn_confidence_interval_result.hpp"
#include "corehydro/analyses/support/i_univariate_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/estimation/generalized_method_of_moments.hpp"
#include "corehydro/models/link_functions/asinh_link.hpp"
#include "corehydro/models/link_functions/log_asinh_link.hpp"
#include "corehydro/models/support/model_base.hpp"
#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/probability_ordinates.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/chi_squared.hpp"
#include "corehydro/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/pearson_type_iii.hpp"
#include "corehydro/numerics/distributions/student_t.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "corehydro/numerics/functions/identity_link.hpp"
#include "corehydro/numerics/functions/link_controller.hpp"
#include "corehydro/numerics/functions/log_link.hpp"
#include "corehydro/numerics/functions/yeo_johnson_link.hpp"
#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/eigenvalue_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/matrix_regularization.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"

namespace corehydro::analyses {

// UncertaintyMethod (C# 33-57): the uncertainty-quantification method. All four members are
// declared so the enum contract is complete and A8's `Bootstrap` is already named. Order matches
// the C# enum (MultivariateNormal = 0). See the header comment for which arms ship / defer.
enum class UncertaintyMethod {
    MultivariateNormal,        // SHIPPED (A7)
    LinkedMultivariateNormal,  // SHIPPED (X8: get_parameter_sets_from_linked_multivariate_normal)
    Bootstrap,                 // SHIPPED (A8: get_parameter_sets_from_parametric_bootstrap)
    BiasCorrectedBootstrap,    // SHIPPED (X9/T20: get_parameter_sets_from_pivotal_bootstrap)
};

namespace detail {

// Zero-parameter ModelBase stub backing the plumbing BayesianAnalysis (see the file header:
// C# `new BayesianAnalysis()` has Model = null; this port needs a non-null ModelBase& but never
// samples through it). With zero parameters, BayesianAnalysis::set_up_sampler() early-returns.
class Bulletin17CPlumbingModel : public corehydro::models::ModelBase {
   public:
    double data_log_likelihood(std::vector<double>&) const override { return 0.0; }
    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>&) const override {
        return {};
    }
    std::vector<corehydro::models::DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>&) const override {
        return {};
    }
    void set_default_parameters() override {}
    corehydro::models::ValidationResult validate() const override { return {}; }
};

}  // namespace detail

// Grants the C++-only ctest access to the private acceleration_constants() method (C# private and
// uncalled on the shipped path; see that method's note). Keeps the public API unchanged.
struct Bulletin17CAnalysisTestAccess;

class Bulletin17CAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using Bulletin17CDistribution = corehydro::models::Bulletin17CDistribution;
    using UnivariateDistributionModel = corehydro::models::UnivariateDistributionModel;
    using UnivariateDistributionBase = corehydro::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using MultivariateNormal = corehydro::numerics::distributions::MultivariateNormal;
    using ProbabilityOrdinates = corehydro::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using GeneralizedMethodOfMoments = corehydro::estimation::GeneralizedMethodOfMoments;
    using BayesianAnalysis = corehydro::estimation::BayesianAnalysis;
    using PointEstimateType = corehydro::estimation::PointEstimateType;
    using ParameterSet = corehydro::numerics::math::optimization::ParameterSet;
    using MCMCResults = corehydro::numerics::sampling::mcmc::MCMCResults;
    using LinkController = corehydro::numerics::functions::LinkController;
    using OptimizationStatus = corehydro::numerics::math::optimization::OptimizationStatus;
    using DistributionType = corehydro::numerics::distributions::UnivariateDistributionType;
    using ILinkFunction = corehydro::numerics::functions::ILinkFunction;
    using ASinHLink = corehydro::models::link_functions::ASinHLink;
    using LogASinHLink = corehydro::models::link_functions::LogASinHLink;

    // C# ctor `Bulletin17CAnalysis(Bulletin17CDistribution)` (C# 113): stores the model, builds a
    // plumbing BayesianAnalysis (PointEstimator = PosteriorMode), defaults the UncertaintyMethod
    // (see the SHIPPED-DEFAULT DEVIATION note: MultivariateNormal, not the C# LinkedMVN), and a
    // default ProbabilityOrdinates. The C# `?? throw ArgumentNullException` maps to the null-guard.
    explicit Bulletin17CAnalysis(std::unique_ptr<Bulletin17CDistribution> bulletin17c_distribution)
        : bulletin17c_distribution_(require_non_null(std::move(bulletin17c_distribution))),
          plumbing_model_(),
          bayesian_analysis_(plumbing_model_),
          probability_ordinates_(),
          uncertainty_method_(UncertaintyMethod::MultivariateNormal) {
        bayesian_analysis_.set_point_estimator(PointEstimateType::PosteriorMode);
    }

    ~Bulletin17CAnalysis() override = default;

    // Non-copyable / non-movable: bayesian_analysis_ holds a reference to the owned plumbing_model_,
    // and gmm_ holds a pointer into the owned distribution -- a defaulted move would dangle either.
    Bulletin17CAnalysis(const Bulletin17CAnalysis&) = delete;
    Bulletin17CAnalysis& operator=(const Bulletin17CAnalysis&) = delete;
    Bulletin17CAnalysis(Bulletin17CAnalysis&&) = delete;
    Bulletin17CAnalysis& operator=(Bulletin17CAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `Bulletin17CDistribution` (C# 244): the model being estimated (owned; deviation).
    Bulletin17CDistribution& bulletin17c_distribution() { return *bulletin17c_distribution_; }
    const Bulletin17CDistribution& bulletin17c_distribution() const {
        return *bulletin17c_distribution_;
    }

    // C# `BayesianAnalysis` (C# 289). IBayesianAnalysis override.
    BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const BayesianAnalysis& bayesian_analysis() const { return bayesian_analysis_; }

    // C# `ProbabilityOrdinates` (C# 264). IProbabilityOrdinates override.
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `UncertaintyMethod` (C# 320). The C# setter calls ClearResults() + RaisePropertyChange;
    // the RaisePropertyChange is dropped, ClearResults() is kept (a method change invalidates any
    // prior fit).
    UncertaintyMethod uncertainty_method() const { return uncertainty_method_; }
    void set_uncertainty_method(UncertaintyMethod value) {
        if (uncertainty_method_ != value) {
            uncertainty_method_ = value;
            clear_results();
        }
    }

    // C# `AnalysisResults` (C# 335): the frequency-analysis results (null until estimated).
    // IBayesianAnalysis override (const pointer <=> nullable optional).
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // C# `GMM` (C# 346): the GMM estimator, populated after a successful run() (null until then).
    const GeneralizedMethodOfMoments* gmm() const { return gmm_ ? &*gmm_ : nullptr; }
    GeneralizedMethodOfMoments* gmm() { return gmm_ ? &*gmm_ : nullptr; }

    // C# `BootstrapResults` (C# 391): the bootstrap diagnostics from the most recent bootstrap
    // uncertainty analysis; null when the uncertainty method is not a bootstrap method (A8).
    const BootstrapDiagnostics* bootstrap_results() const {
        return bootstrap_results_ ? &*bootstrap_results_ : nullptr;
    }

    // C# `UncertaintyDiagnosticMessage` (T19, C# 409): a user-facing description of why the most
    // recent uncertainty quantification was degraded or aborted; empty when no issue occurred. NOT
    // populated by A8's get_parameter_sets_from_parametric_bootstrap() -- see the file header's T19
    // note; the accessor exists for the other uncertainty paths' existing (unwired) guards.
    const std::string& uncertainty_diagnostic_message() const {
        return uncertainty_diagnostic_message_;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 488): resets the fit + results to the un-estimated state; the model is
    // preserved. RaisePropertyChange / ElapsedTime dropped; BootstrapResults + (T19)
    // UncertaintyDiagnosticMessage reset (C# 500-501).
    void clear_results() {
        set_is_estimated(false);
        bayesian_analysis_.clear_results();
        gmm_.reset();
        analysis_results_.reset();
        bootstrap_results_.reset();
        uncertainty_diagnostic_message_.clear();
        distribution_cache_.clear();
    }

    // C# `ClearFrequencyAnalysisResults` (C# 510): clears ONLY the frequency results; the GMM fit,
    // the BayesianAnalysis results, and IsEstimated survive.
    void clear_frequency_analysis_results() { analysis_results_.reset(); }

    // C# `RunAsync` (C# 518), synchronous. Compute sequence (guards element-by-element, C#
    // 533-581): clear -> preprocess thresholds -> location/scale/shape LinkController -> initial
    // params -> penalty function -> build+run GMM -> bail on OptimizationStatus::Failure -> mark
    // estimated -> uncertainty quantification -> frequency results. Cancellation/gate/events/
    // progress/timing dropped (see the file header).
    void run() override {
        clear_results();

        // Preprocess data (C# 538-541). C# dereferences DataFrame unconditionally here; matched.
        bulletin17c_distribution_->data_frame().process_threshold_series();
        bulletin17c_distribution_->set_link_controller(LinkController::for_location_scale_shape());
        bulletin17c_distribution_->set_initial_parameters();
        bulletin17c_distribution_->set_penalty_function();

        // Build + run the GMM estimator (C# 542-551).
        gmm_.emplace(*bulletin17c_distribution_);
        gmm_->estimate();

        // Bail on solver failure (C# 553-558): a silent no-throw guard (the C# Debug.WriteLine is
        // dropped); IsEstimated stays false (cleared above).
        if (gmm_->status() == OptimizationStatus::Failure) return;

        // Mark estimated BEFORE building results (C# 563 sets _isEstimated = true first).
        set_is_estimated(true);

        // Heavy uncertainty quantification (runs once), then the fast frequency-result assembly
        // reading the persisted parameter sets (C# 568 / 581).
        run_uncertainty_quantification();
        create_frequency_analysis_results();
    }

    // C# `GetDistribution(int)` (C# 2280): the distribution for a stored posterior output index,
    // or null when unestimated. IUnivariateAnalysis override (clone owned in distribution_cache_).
    UnivariateDistributionBase* get_distribution(int index) override {
        if (!is_estimated() || !bayesian_analysis_.results()) return nullptr;
        std::unique_ptr<UnivariateDistributionBase> dist = bulletin17c_distribution_->distribution()->clone();
        dist->set_parameters(bayesian_analysis_.results()->output[static_cast<std::size_t>(index)].values);
        UnivariateDistributionBase* raw = dist.get();
        distribution_cache_.push_back(std::move(dist));
        return raw;
    }

    // C# `GetPointEstimateDistribution()` (C# 2291): uses the analysis's configured PointEstimator.
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_analysis_.point_estimator());
    }

    // C# `GetPointEstimateDistribution(PointEstimateType)` (C# 2295): a caller-supplied estimator
    // without mutating the analysis's own PointEstimator. Null when unestimated.
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        if (!is_estimated() || !bayesian_analysis_.results()) return nullptr;
        const std::vector<double>& parms =
            point_estimator == PointEstimateType::PosteriorMean
                ? bayesian_analysis_.results()->posterior_mean.values
                : bayesian_analysis_.results()->map.values;
        std::unique_ptr<UnivariateDistributionBase> dist = bulletin17c_distribution_->distribution()->clone();
        dist->set_parameters(parms);
        UnivariateDistributionBase* raw = dist.get();
        distribution_cache_.push_back(std::move(dist));
        return raw;
    }

    // C# `Validate` (C# 2357): aggregates the model + probability-ordinate validations (NOT the
    // BayesianAnalysis, matching the C# body). const per the A4 IAnalysis contract.
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        corehydro::models::ValidationResult dist_valid = bulletin17c_distribution_->validate();
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

        return result;
    }

    // C# `ComputeCohnStyleConfidenceIntervals` (C# 2513-2643): the flagship Cohn-style delta-method
    // confidence intervals. A DETERMINISTIC nested Gaussian-quadrature delta method over the GMM
    // fit (no MCMC/bootstrap seed dependence). Builds the outer 2^p Cholesky quadrature grid at
    // theta_hat, recomputes the covariance (with the degeneracy sanity check) and an inner grid per
    // outer node, then per probability level forms the 2x2 Cov(Q_hat, SE(Q_hat)) and applies Cohn's
    // adjusted Student-t CI. Returns nullopt when the GMM is null / not estimated (C# `return null`).
    std::optional<CohnConfidenceIntervalResult> compute_cohn_style_confidence_intervals() {
        using corehydro::numerics::math::linalg::Matrix;
        using corehydro::numerics::math::linalg::MatrixRegularization;

        // C# `if (_gmm == null || !_gmm.IsEstimated) return null;` (C# 2515-2516).
        if (!gmm_ || !gmm_->is_estimated()) return std::nullopt;

        int n_prob = static_cast<int>(probability_ordinates_.count());       // C# 2518
        int p = bulletin17c_distribution_->number_of_parameters();           // C# 2519
        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;  // C# 2521

        // Outer covariance from the GMM sandwich estimator at theta_hat (C# 2525-2526).
        Matrix sigma_hat = gmm_->get_covariance(clamp_for_covariance(theta_hat));
        sigma_hat = MatrixRegularization::make_symmetric_positive_definite(sigma_hat);

        // Outer 2^p Cholesky quadrature grid (C# 2529-2531).
        const int n_nodes_per_dim = 2;
        auto [outer_grid, outer_weights] =
            build_quadrature_grid(theta_hat, sigma_hat, p, n_nodes_per_dim);
        int n_outer = static_cast<int>(outer_grid.size());

        // Per outer node: recompute the covariance + build the inner grid (C# 2540-2577).
        std::vector<std::vector<std::vector<double>>> inner_grids(static_cast<std::size_t>(n_outer));
        std::vector<std::vector<double>> inner_weights(static_cast<std::size_t>(n_outer));
        for (int i = 0; i < n_outer; ++i) {
            std::size_t iu = static_cast<std::size_t>(i);
            Matrix sigma_at_i = sigma_hat;
            // C# try/catch + Debug.WriteLine -> silent no-throw guard falling back to sigma_hat.
            try {
                Matrix candidate = gmm_->get_covariance(clamp_for_covariance(outer_grid[iu]));
                candidate = MatrixRegularization::make_symmetric_positive_definite(candidate);

                // Degeneracy sanity check (C# 2554-2567): fall back to baseline when any diagonal is
                // NaN/Inf/<=0 or differs from the baseline diagonal by more than 10x or less than 0.1x.
                bool degenerate = false;
                for (int d = 0; d < p; ++d) {
                    double base_var = sigma_hat(d, d);
                    double grid_var = candidate(d, d);
                    if (std::isnan(grid_var) || std::isinf(grid_var) || grid_var <= 0.0 ||
                        (base_var > 0.0 &&
                         (grid_var > 10.0 * base_var || grid_var < 0.1 * base_var))) {
                        degenerate = true;
                        break;
                    }
                }
                sigma_at_i = degenerate ? sigma_hat : candidate;
            } catch (...) {
                sigma_at_i = sigma_hat;
            }
            auto [ig, iw] = build_quadrature_grid(outer_grid[iu], sigma_at_i, p, n_nodes_per_dim);
            inner_grids[iu] = std::move(ig);
            inner_weights[iu] = std::move(iw);
        }

        // Result arrays (C# 2580-2585).
        std::vector<double> point_estimates(static_cast<std::size_t>(n_prob));
        std::vector<double> lower_ci(static_cast<std::size_t>(n_prob));
        std::vector<double> upper_ci(static_cast<std::size_t>(n_prob));
        std::vector<double> beta1_array(static_cast<std::size_t>(n_prob));
        std::vector<double> nu_array(static_cast<std::size_t>(n_prob));
        std::vector<double> var_q_array(static_cast<std::size_t>(n_prob));

        // Per probability level: the 2x2 Cov(Q_hat_p, SE(Q_hat_p)) (C# 2588-2627).
        for (int k = 0; k < n_prob; ++k) {
            std::size_t ku = static_cast<std::size_t>(k);
            double non_exceed_prob = 1.0 - probability_ordinates_[ku];  // C# 2590

            point_estimates[ku] = evaluate_quantile_safe(theta_hat, non_exceed_prob);  // C# 2593

            std::vector<double> q_outer(static_cast<std::size_t>(n_outer));
            std::vector<double> se_outer(static_cast<std::size_t>(n_outer));
            for (int i = 0; i < n_outer; ++i) {
                std::size_t iu = static_cast<std::size_t>(i);
                q_outer[iu] = evaluate_quantile_safe(outer_grid[iu], non_exceed_prob);

                // Inner-level variance of Q_p (C# 2603-2610).
                int n_inner = static_cast<int>(inner_grids[iu].size());
                std::vector<double> q_inner(static_cast<std::size_t>(n_inner));
                for (int j = 0; j < n_inner; ++j)
                    q_inner[static_cast<std::size_t>(j)] =
                        evaluate_quantile_safe(inner_grids[iu][static_cast<std::size_t>(j)],
                                               non_exceed_prob);
                double var_inner = weighted_covariance(q_inner, q_inner, inner_weights[iu]);
                se_outer[iu] = std::sqrt(std::max(0.0, var_inner));
            }

            // 2x2 covariance (C# 2614-2616).
            double var_q = weighted_covariance(q_outer, q_outer, outer_weights);
            double cov_q_se = weighted_covariance(q_outer, se_outer, outer_weights);
            double var_se = weighted_covariance(se_outer, se_outer, outer_weights);

            // Cohn adjusted Student-t CI (C# 2619-2620).
            auto [low, high, beta1, nu] = cohn_adjusted_student_t_ci(
                point_estimates[ku], var_q, cov_q_se, var_se,
                bayesian_analysis_.credible_interval_width());

            lower_ci[ku] = std::pow(10.0, low);    // C# 2622
            upper_ci[ku] = std::pow(10.0, high);   // C# 2623
            beta1_array[ku] = beta1;
            nu_array[ku] = nu;
            var_q_array[ku] = var_q;
        }

        // Monotonicity enforcement (C# 2630).
        enforce_monotonicity(lower_ci, upper_ci, n_prob);

        // Fill the DTO (C# 2632-2642). ExceedanceProbabilities = ProbabilityOrdinates.ToArray()
        // (the ordinates ARE the exceedance probabilities in this codebase -- mirrored verbatim).
        CohnConfidenceIntervalResult result;
        result.exceedance_probabilities =
            std::vector<double>(probability_ordinates_.begin(), probability_ordinates_.end());
        result.point_estimates = std::move(point_estimates);
        result.lower_ci = std::move(lower_ci);
        result.upper_ci = std::move(upper_ci);
        result.confidence_level = bayesian_analysis_.credible_interval_width();
        result.beta1 = std::move(beta1_array);
        result.nu = std::move(nu_array);
        result.quantile_variance = std::move(var_q_array);
        return result;
    }

   private:
    // Guard-rail constants for the covariance/quantile parameter clamps (C#-governed literals,
    // named per the coding-style magic-number rule). EMA REGMOMS/VAR_MOM/QP3 provenance:
    //   kSigmaFloor  -- sigma > 0 floor for covariance/quantile evaluation.
    //   kSkewClampAbs -- REGMOMS |gamma| <= 1.5 outer clamp (covariance only).
    //   kSkewFloorAbs -- VAR_MOM |gamma| >= 0.0632 floor (covariance only; alpha = 4/gamma^2).
    static constexpr double kSigmaFloor = 1e-10;
    static constexpr double kSkewClampAbs = 1.5;
    static constexpr double kSkewFloorAbs = 0.063;

    // Null-guard helper for the ctor init list (C# `?? throw new ArgumentNullException`).
    static std::unique_ptr<Bulletin17CDistribution> require_non_null(
        std::unique_ptr<Bulletin17CDistribution> model) {
        if (model == nullptr) {
            throw std::invalid_argument("bullet17CDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // T19: mirrors C#'s `:N0` numeric format specifier (thousands-grouped, no decimal digits,
    // invariant/en-US comma separator) for the two exact-count UncertaintyDiagnosticMessage
    // strings below -- the only place this port needs C#-style number formatting.
    static std::string format_n0(long long value) {
        bool negative = value < 0;
        unsigned long long magnitude =
            negative ? static_cast<unsigned long long>(-value) : static_cast<unsigned long long>(value);
        std::string digits = std::to_string(magnitude);
        std::string grouped;
        int since_group = 0;
        for (auto it = digits.rbegin(); it != digits.rend(); ++it) {
            if (since_group != 0 && since_group % 3 == 0) grouped.push_back(',');
            grouped.push_back(*it);
            ++since_group;
        }
        std::reverse(grouped.begin(), grouped.end());
        return negative ? "-" + grouped : grouped;
    }

    // C# `RunUncertaintyQuantificationAsync` (C# 724-798 @ c2e6192), synchronous. Sets the parent
    // distribution to thetaHat, dispatches on UncertaintyMethod to build the parameter-set
    // ensemble, and stores it into the BayesianAnalysis results plumbing. The async/Task.Run/
    // progress/cancellation machinery is dropped.
    void run_uncertainty_quantification() {
        if (!gmm_ || gmm_->status() == OptimizationStatus::Failure) return;

        // Set parent distribution parameters to the GMM point estimate (C# 733).
        bulletin17c_distribution_->set_parameter_values(gmm_->best_parameter_set().values);

        // Dispatch (C# switch, 736-743). See the file header for the ship/defer contract.
        std::optional<std::vector<ParameterSet>> raw_sets;
        switch (uncertainty_method_) {
            case UncertaintyMethod::MultivariateNormal:
                raw_sets = get_parameter_sets_from_multivariate_normal();
                break;
            case UncertaintyMethod::Bootstrap:
                // C# `UncertaintyMethod.Bootstrap => GetParameterSetsFromParametricBootstrap(...)`.
                // The parametric bootstrap re-fits the GMM on B resampled frames.
                raw_sets = get_parameter_sets_from_parametric_bootstrap();
                break;
            case UncertaintyMethod::LinkedMultivariateNormal:
                // C# `UncertaintyMethod.LinkedMultivariateNormal => ...`. On a null return (e.g.
                // high rejection rate) the C# silently falls back to plain MVN (C# 746-750);
                // ported inline here.
                raw_sets = get_parameter_sets_from_linked_multivariate_normal();
                if (!raw_sets) raw_sets = get_parameter_sets_from_multivariate_normal();
                break;
            case UncertaintyMethod::BiasCorrectedBootstrap:
                // C# `UncertaintyMethod.BiasCorrectedBootstrap => GetParameterSetsFromPivotalBootstrap(...)`
                // (C# 741; T20: RENAMED from GetParameterSetsFromPivotBootstrap). The pivotal
                // bootstrap fits B replicates, builds guarded Yeo-Johnson/Log links, and draws
                // pivoted parameter sets. No caller fallback at all: T20 replaced the old
                // Cholesky-failure fallback to the parametric path with a null return + diagnostic
                // message, handled by the shared guards below.
                raw_sets = get_parameter_sets_from_pivotal_bootstrap();
                break;
        }

        // T19 (C# 755-767): a null rawSets means either a specific diagnostic was already set
        // deeper in the sampler (e.g. an abort) or the shared covariance path failed silently; the
        // default message is set ONLY when nothing more specific was already recorded. Previously
        // a silent no-throw guard with no message -- this is a genuine v2.0.0 behavioral delta (the
        // C# has zero UncertaintyDiagnosticMessage occurrences at the fc28c0c pin).
        if (!raw_sets) {
            if (uncertainty_diagnostic_message_.empty()) {
                set_uncertainty_diagnostic_message(
                    "Uncertainty quantification failed — the covariance matrix from GMM "
                    "estimation is not positive-definite. The point estimate is still valid but "
                    "confidence intervals cannot be computed. Consider using a different "
                    "distribution or the Bootstrap/Bias-Corrected Bootstrap uncertainty method.");
            }
            return;
        }

        // T19 (C# 769-777): filter unset, dimensionally invalid, or non-finite entries. MVN
        // samplers may discard invalid draws; bootstrap collectors replace failed candidates and
        // therefore must satisfy the exact-count invariant below.
        int expected_parameter_count = bulletin17c_distribution_->number_of_parameters();
        std::vector<ParameterSet> valid_sets;
        valid_sets.reserve(raw_sets->size());
        for (const auto& ps : *raw_sets) {
            if (!ps.values.empty() &&
                static_cast<int>(ps.values.size()) == expected_parameter_count &&
                std::all_of(ps.values.begin(), ps.values.end(),
                           [](double v) { return std::isfinite(v); }))
                valid_sets.push_back(ps);
        }

        // T19 (C# 778-786): bootstrap methods must deliver EXACTLY OutputLength valid sets, or the
        // ensemble is rejected wholesale (no partial publish). This is what makes a discard in the
        // pivot arm's Phase 3 (Task 20) -- or any future discard in this arm -- user-visible.
        bool is_bootstrap_method = uncertainty_method_ == UncertaintyMethod::Bootstrap ||
                                   uncertainty_method_ == UncertaintyMethod::BiasCorrectedBootstrap;
        auto output_length = static_cast<std::size_t>(bayesian_analysis_.output_length());
        if (is_bootstrap_method && (raw_sets->size() != output_length || valid_sets.size() != output_length)) {
            set_uncertainty_diagnostic_message(
                "Bootstrap uncertainty requires exactly " +
                format_n0(bayesian_analysis_.output_length()) + " valid parameter sets, but received " +
                format_n0(static_cast<long long>(valid_sets.size())) +
                ". No partial uncertainty result was published.");
            return;
        }

        // T19 (C# 787-793).
        if (valid_sets.size() < 2) {
            set_uncertainty_diagnostic_message(
                "Only " + format_n0(static_cast<long long>(valid_sets.size())) +
                " valid parameter sets out of " + format_n0(static_cast<long long>(raw_sets->size())) +
                " sampled realizations. Uncertainty analysis was skipped.");
            return;
        }

        // Sanitize non-finite Fitness/Values (C# 796) -- MVN sets Fitness = NaN by construction.
        sanitize_parameter_sets(valid_sets);

        // Sanitize the MAP (best) parameter set's Fitness (C# 800-802).
        ParameterSet best = gmm_->best_parameter_set();
        if (!std::isfinite(best.fitness)) best = ParameterSet(best.values, 0.0);

        // Store into the BayesianAnalysis results plumbing (C# 804-807).
        MCMCResults mcmc(std::move(best), std::move(valid_sets),
                         1.0 - bayesian_analysis_.credible_interval_width());
        bayesian_analysis_.set_custom_mcmc_results(std::move(mcmc), /*skip_information_criteria=*/true);
    }

    // C# `GetParameterSetsFromMultivariateNormal` (C# 763-848): draws B parameter sets from
    // MVN(thetaHat, sigmaHat) via Latin Hypercube (seeded by the BayesianAnalysis PRNG seed),
    // validating each draw and retrying / falling back exactly as the C#. Returns nullopt when the
    // GMM sandwich covariance is not positive-definite (C# `return null`). The C# Parallel.For ->
    // a serial loop (independent per-index writes; identical results). Progress/cancellation
    // dropped.
    std::optional<std::vector<ParameterSet>> get_parameter_sets_from_multivariate_normal() {
        int b = bayesian_analysis_.output_length();
        std::vector<ParameterSet> results(static_cast<std::size_t>(b));

        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;
        corehydro::numerics::math::linalg::Matrix sigma_hat = gmm_->get_covariance(theta_hat);

        // Validate positive-definiteness by constructing the MVN (throws otherwise) (C# 773-782).
        std::optional<MultivariateNormal> mvn;
        try {
            mvn.emplace(theta_hat, sigma_hat.to_array());
        } catch (...) {
            // C# Debug.WriteLine + return null -> silent nullopt guard.
            return std::nullopt;
        }

        const int seed = bayesian_analysis_.prng_seed();
        const int p = static_cast<int>(theta_hat.size());
        std::vector<std::vector<double>> draws = mvn->latin_hypercube_random_values(b, seed);

        // A single reusable validator clone (C# clones per-thread; serial here so one clone
        // suffices -- set_parameters mutates only this throwaway).
        std::unique_ptr<UnivariateDistributionBase> validator =
            bulletin17c_distribution_->distribution()->clone();
        auto accepts = [&validator](const std::vector<double>& theta) -> bool {
            try {
                validator->set_parameters(theta);  // C# ValidateParameters(theta, true)
                return validator->parameters_valid();
            } catch (...) {
                return false;  // C# catch -> rejected draw
            }
        };

        for (int idx = 0; idx < b; ++idx) {
            std::optional<std::vector<double>> accepted;

            // First try: the LHS draw (best space coverage) (C# 805-811).
            if (accepts(draws[static_cast<std::size_t>(idx)])) accepted = draws[static_cast<std::size_t>(idx)];

            // Retry with fresh MVN draws seeded per-index (C# 814-833).
            if (!accepted) {
                corehydro::numerics::sampling::MersenneTwister prng(
                    static_cast<std::uint32_t>(seed + b + idx));
                for (int retry = 0; retry < 10 && !accepted; ++retry) {
                    std::vector<double> u =
                        corehydro::numerics::utilities::next_doubles(prng, 1, p)[0];
                    std::vector<double> candidate = mvn->inverse_cdf(u);
                    if (accepts(candidate)) accepted = std::move(candidate);
                }
                // Fall back to the parent vector after 10 rejected retries (C# 832).
                if (!accepted) accepted = theta_hat;
            }

            results[static_cast<std::size_t>(idx)] =
                ParameterSet(std::move(*accepted), std::numeric_limits<double>::quiet_NaN());
        }

        return results;
    }

    // T19: one accepted bootstrap-replicate fit, returned by collect_bootstrap_replicate_fit().
    // T20: `covariance` carries the regularized, Cholesky-factorable replicate covariance
    // Sigma*_b -- populated ONLY when the caller sets `collect_covariance` (the pivot arm; the
    // parametric arm never asks for it, exactly as the C# only computes it in the pivot method).
    struct BootstrapReplicateFit {
        std::vector<double> params;
        int function_evaluations = 0;
        std::optional<corehydro::numerics::math::linalg::Matrix> covariance;
    };

    // T19/T20: the shared per-replicate bootstrap-fit attempt loop, factored out of
    // GetParameterSetsFromParametricBootstrap (C# 1990-2045) and reused verbatim by
    // GetParameterSetsFromPivotalBootstrap's Phase 1 (C# 2227-2288) -- at the c2e6192 pin the two
    // C# loops are byte-identical except for the pivot arm's extra covariance step (guarded here
    // by `collect_covariance`), so one helper reproduces both.
    //
    // Resamples a bootstrap data frame from `parent_distribution` via the parent's
    // BootstrapDataFrame, warm-starts the B17C clone through Task 18's `clone_with_data_frame()` +
    // `set_parameter_values(theta_hat)` when `clone_with_data_frame_flag` is set (mirroring the C#
    // `cloneWithDataFrame` local EXACTLY -- see the callers), otherwise clones via the plain
    // `clone()` + `set_data_frame()` path; sets a randomized penalty function; re-fits via GMM
    // (the GMM's own `penalty_is_random` default `true` is left alone -- T20: the pivot arm's
    // `new GeneralizedMethodOfMoments(...) { PenaltyIsRandom = false }` initializer was REMOVED
    // upstream and has ZERO occurrences in the file at c2e6192, so BOTH arms now run the default);
    // and retries up to `max_retries` times on any exception (a non-Success GMM status, a
    // Mahalanobis distance from `theta_hat` beyond `mahal_threshold`, or -- when
    // `collect_covariance` is set -- a covariance that TryGetCovariance rejects, that is not
    // Cholesky-factorable after regularization), incrementing `diag`'s Mahalanobis-rejection /
    // retry counters exactly as the C#. Returns the accepted parameter vector + its GMM
    // function-evaluation count (+ the regularized covariance when requested) on success; nullopt
    // when every attempt failed -- the CALLER decides fallback-vs-discard semantics for its own
    // arm (both shipped arms fall back to the parent fit; see the callers below).
    std::optional<BootstrapReplicateFit> collect_bootstrap_replicate_fit(
        corehydro::numerics::sampling::MersenneTwister& prng,
        const UnivariateDistributionBase& parent_distribution,
        const std::vector<double>& theta_hat,
        const corehydro::numerics::math::linalg::Matrix& sigma_inv, double mahal_threshold,
        bool clone_with_data_frame_flag, bool collect_covariance, int max_retries,
        BootstrapDiagnostics& diag) {
        int p = static_cast<int>(theta_hat.size());
        std::optional<std::vector<double>> accepted_params;
        std::optional<corehydro::numerics::math::linalg::Matrix> accepted_sigma;
        int accepted_fn_evals = 0;

        for (int attempt = 0; attempt < max_retries && !accepted_params; ++attempt) {
            try {
                // Resample a bootstrap frame from the parent (clone used only for sampling).
                std::unique_ptr<UnivariateDistributionBase> sampling_dist =
                    parent_distribution.clone();
                corehydro::models::DataFrame boot_data_frame =
                    bulletin17c_distribution_->data_frame().BootstrapDataFrame(*sampling_dist,
                                                                               prng);

                std::unique_ptr<Bulletin17CDistribution> boot_b17c;
                if (clone_with_data_frame_flag) {
                    // Warm start at the parent fit: preserves parameter bounds/priors/penalty
                    // configuration instead of re-deriving defaults from the boot data.
                    boot_b17c = bulletin17c_distribution_->clone_with_data_frame(boot_data_frame);
                    boot_b17c->set_parameter_values(theta_hat);
                } else {
                    std::unique_ptr<corehydro::models::IGMMModel> cloned =
                        bulletin17c_distribution_->clone();
                    boot_b17c.reset(static_cast<Bulletin17CDistribution*>(cloned.release()));
                    boot_b17c->set_data_frame(std::move(boot_data_frame));
                }
                boot_b17c->set_random_penalty_function(theta_hat, &prng);

                // T20: NO PenaltyIsRandom override -- the C# `{ PenaltyIsRandom = false }`
                // initializer was deleted upstream, so the GMM default (true) governs both arms.
                GeneralizedMethodOfMoments boot_gmm(*boot_b17c);
                boot_gmm.estimate();
                if (boot_gmm.status() != OptimizationStatus::Success)
                    throw std::runtime_error("bootstrap GMM solver failed");

                // Mahalanobis distance of the bootstrap fit from theta_hat.
                const std::vector<double>& boot_params = boot_gmm.best_parameter_set().values;
                double mahal_dist = 0.0;
                for (int j = 0; j < p; ++j) {
                    double tmp = 0.0;
                    for (int k = 0; k < p; ++k)
                        tmp += sigma_inv(j, k) * (boot_params[static_cast<std::size_t>(k)] -
                                                  theta_hat[static_cast<std::size_t>(k)]);
                    mahal_dist += (boot_params[static_cast<std::size_t>(j)] -
                                   theta_hat[static_cast<std::size_t>(j)]) *
                                  tmp;
                }
                if (mahal_dist > mahal_threshold) {
                    diag.increment_mahalanobis_rejection();
                    throw std::runtime_error("bootstrap replicate rejected: Mahalanobis");
                }

                // T20 (C# 2274-2277, pivot arm only): the replicate covariance must be finite,
                // regularizable, and Cholesky-factorable -- a failure here retries the WHOLE
                // replicate, which is why this step lives inside the attempt loop.
                if (collect_covariance) {
                    corehydro::numerics::math::linalg::Matrix boot_covariance(p);
                    if (!boot_gmm.try_get_covariance(boot_params, true, boot_covariance))
                        throw std::runtime_error(
                            "the pivot bootstrap GMM covariance could not be computed");
                    boot_covariance = corehydro::numerics::math::linalg::MatrixRegularization::
                        make_symmetric_positive_definite(boot_covariance);
                    // C# `_ = new CholeskyDecomposition(bootCovariance);` -- constructed purely to
                    // throw on a non-factorable matrix; the factor itself is discarded.
                    (void)corehydro::numerics::math::linalg::CholeskyDecomposition(boot_covariance);
                    accepted_sigma = std::move(boot_covariance);
                }

                accepted_params = boot_params;
                accepted_fn_evals = boot_gmm.total_function_evaluations();
            } catch (...) {
                // C# Debug.WriteLine dropped (silent guard); count a retry when more remain.
                if (attempt < max_retries - 1) diag.add_retries(1);
            }
        }

        if (!accepted_params) return std::nullopt;
        return BootstrapReplicateFit{std::move(*accepted_params), accepted_fn_evals,
                                     std::move(accepted_sigma)};
    }

    // C# `GetParameterSetsFromParametricBootstrap` (C# 1941-2073 @ c2e6192): the parametric-
    // bootstrap UQ path. Draws B resampled data frames from the fitted parent distribution,
    // re-fits the GMM on each via collect_bootstrap_replicate_fit() (T19), and keeps the accepted
    // parameter vectors. A degenerate fit (GMM failure, or a Mahalanobis distance from thetaHat
    // beyond the ADAPTIVE `chi2(p, 1 - 1/(5B))` threshold, T19: was the fixed 0.9999) is retried
    // up to maxRetries (T19: 10, was 5) times before falling back to the parent vector -- the
    // shipped v2.0.0 C# still falls back rather than discarding; see the file header's T19 note
    // for why. The up-front seed cascade (masterPRNG(PRNGSeed) -> B integer seeds -> a
    // per-replicate MersenneTwister) is IDENTICAL to the ported Bootstrap and is load-bearing for
    // MersenneTwister parity with the emitter -- do NOT reorder. The C# Parallel.For -> a serial
    // loop; per-index writes are independent, so the serial counts/results match.
    // Progress/cancellation/stopwatch lines dropped.
    std::optional<std::vector<ParameterSet>> get_parameter_sets_from_parametric_bootstrap() {
        const int max_retries = 10;                                   // T19: 5 -> 10
        int b = bayesian_analysis_.output_length();
        int p = bulletin17c_distribution_->number_of_parameters();
        std::vector<ParameterSet> results(static_cast<std::size_t>(b));

        // Up-front seed cascade: master PRNG seeds B per-replicate integers, keyed to the
        // ORIGINAL replicate index (load-bearing for MT parity -- do NOT reorder).
        corehydro::numerics::sampling::MersenneTwister master_prng(
            static_cast<std::uint32_t>(bayesian_analysis_.prng_seed()));
        std::vector<int> seeds = corehydro::numerics::utilities::next_integers(master_prng, b);

        // Parent distribution cloned + set to thetaHat as the data-generating model.
        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;
        std::unique_ptr<UnivariateDistributionBase> parent_distribution =
            bulletin17c_distribution_->distribution()->clone();
        parent_distribution->set_parameters(theta_hat);

        // Sandwich covariance + inverse for the Mahalanobis rejection. On a singular inverse the
        // C# regularizes via MatrixRegularization instead of aborting; ported as a try/catch
        // falling back to the regularized inverse (the C# Debug.WriteLine is a dropped silent
        // guard).
        corehydro::numerics::math::linalg::Matrix sigma_hat = gmm_->get_covariance(theta_hat);
        corehydro::numerics::math::linalg::Matrix sigma_inv(p);
        try {
            sigma_inv = sigma_hat.inverse();
        } catch (...) {
            sigma_inv = corehydro::numerics::math::linalg::MatrixRegularization::
                            make_symmetric_positive_definite(sigma_hat)
                                .inverse();
        }
        // T19: adaptive Mahalanobis threshold ChiSq(p) at 1 - 1/(5B) (was the fixed 0.9999) --
        // rejects extreme draws with p-value < 1/(5B).
        double mahal_pvalue = 1.0 - 1.0 / (b * 5.0);
        double mahal_threshold =
            corehydro::numerics::distributions::ChiSquared(p).inverse_cdf(mahal_pvalue);

        // Diagnostics: TotalReplicates = B.
        BootstrapDiagnostics diag;
        diag.set_total_replicates(b);

        // T19: warm-start every replicate via clone_with_data_frame() (Task 18) when low
        // outliers / uncertain / interval / threshold series are present on the parent DataFrame
        // -- mirrors the C# `cloneWithDataFrame` local EXACTLY (C# "Determine if we should clone
        // with the parent data frame or not").
        bool clone_with_data_frame_flag =
            bulletin17c_distribution_->data_frame().number_of_low_outliers() > 0 ||
            bulletin17c_distribution_->data_frame().uncertain_series().count() > 0 ||
            bulletin17c_distribution_->data_frame().interval_series().count() > 0 ||
            bulletin17c_distribution_->data_frame().threshold_series().count() > 0;

        // Serial replicate loop (C# Parallel.For).
        for (int idx = 0; idx < b; ++idx) {
            corehydro::numerics::sampling::MersenneTwister prng(
                static_cast<std::uint32_t>(seeds[static_cast<std::size_t>(idx)]));
            std::optional<BootstrapReplicateFit> fit = collect_bootstrap_replicate_fit(
                prng, *parent_distribution, theta_hat, sigma_inv, mahal_threshold,
                clone_with_data_frame_flag, /*collect_covariance=*/false, max_retries, diag);

            // Fall back to the parent vector after all retries fail -- the shipped v2.0.0 C#
            // does NOT discard (see the file header's T19 note).
            std::vector<double> accepted_params;
            if (fit) {
                diag.add_function_evaluations(fit->function_evaluations);
                accepted_params = std::move(fit->params);
            } else {
                diag.increment_failed();
                accepted_params = theta_hat;
            }

            results[static_cast<std::size_t>(idx)] =
                ParameterSet(std::move(accepted_params), std::numeric_limits<double>::quiet_NaN());
        }

        // Persist the diagnostics on the analysis and return the sets.
        bootstrap_results_ = std::move(diag);
        return results;
    }

    // ================= X9/T20: BiasCorrectedBootstrap (pivotal bootstrap) uncertainty path =================

   public:
    // C# `internal static ILinkFunction CreatePivotYeoJohnsonLink(IReadOnlyList<double> samples,
    // string parameterName)` (C# 2081-2117; NEW at c2e6192). Fits a Numerics YeoJohnsonLink to one
    // parameter's bootstrap samples, then VALIDATES it before handing it to the pivot transform:
    //   * at least two samples, every sample finite (C# 2087-2095);
    //   * |lambda| < 4.999 -- a fit railed at the YeoJohnson::fit_lambda [-5, 5] optimizer boundary
    //     is rejected as degenerate (C# 2098-2099);
    //   * link / d_link / inverse_link finite at EVERY sample (C# 2101-2108).
    // Any failure -- including the exceptions the ported YeoJohnsonLink values-constructor throws
    // for a degenerate sample (T2's NaN fit_lambda hardening) -- falls back to an IdentityLink
    // (C# 2112-2116; the C# Debug.WriteLine is a dropped silent guard). `parameter_name` is kept
    // for signature parity even though the log line is dropped. Public here because the C#
    // `internal` + InternalsVisibleTo makes it directly test-visible (see the two transcribed
    // Bulletin17CAnalysisTests cases in core/tests/test_bulletin17c_analysis.cpp).
    //
    // NOTE (T17): the YeoJohnsonLink used here is the NUMERICS one -- upstream deleted the
    // duplicate RMC.BestFit.Models.LinkFunctions.YeoJohnsonLink, so there is only one left.
    static std::unique_ptr<corehydro::numerics::functions::ILinkFunction>
    create_pivot_yeo_johnson_link(const std::vector<double>& samples,
                                  const std::string& parameter_name) {
        (void)parameter_name;  // C# only uses it in the dropped Debug.WriteLine.
        try {
            if (samples.size() < 2)
                throw std::invalid_argument(
                    "At least two samples are required to fit a Yeo-Johnson pivot link.");
            for (std::size_t i = 0; i < samples.size(); ++i)
                if (!std::isfinite(samples[i]))
                    throw std::out_of_range("Yeo-Johnson pivot link samples must be finite.");

            auto link = std::make_unique<corehydro::numerics::functions::YeoJohnsonLink>(samples);
            if (std::abs(link->lambda()) >= 4.999)
                throw std::runtime_error(
                    "Yeo-Johnson pivot link lambda fit railed at the optimizer boundary.");

            for (std::size_t i = 0; i < samples.size(); ++i) {
                double eta = link->link(samples[i]);
                double derivative = link->d_link(samples[i]);
                double round_trip = link->inverse_link(eta);
                if (!std::isfinite(eta) || !std::isfinite(derivative) || !std::isfinite(round_trip))
                    throw std::runtime_error(
                        "Yeo-Johnson pivot link produced non-finite behavior.");
            }

            return link;
        } catch (...) {
            return std::make_unique<corehydro::numerics::functions::IdentityLink>();
        }
    }

   private:
    // C# `GetParameterSetsFromPivotalBootstrap` (C# 2158-2459 @ c2e6192; RENAMED from
    // GetParameterSetsFromPivotBootstrap): the bias-corrected / pivotal-bootstrap UQ path. THREE
    // phases:
    //   Phase 1 (C# 2209-2312): B parametric-bootstrap GMM re-fits through the SHARED
    //     collect_bootstrap_replicate_fit() helper (T19) -- at c2e6192 this loop is byte-identical
    //     to the parametric arm's except that it ALSO demands a usable covariance Sigma*_b
    //     (TryGetCovariance -> MakeSymmetricPositiveDefinite -> a throwaway CholeskyDecomposition
    //     that throws on a non-factorable matrix, C# 2274-2277) inside the same try, so a bad
    //     covariance retries the whole replicate. A replicate that exhausts every retry falls back
    //     to (thetaHat, sigmaHat) and increments Failed (C# 2291-2296).
    //   Phase 2 (C# 2314-2363): fit link functions from ALL B bootstrap samples -- location
    //     (idx 0) -> CreatePivotYeoJohnsonLink; scale (idx 1) -> plain LogLink; shape (idx 2, if
    //     p >= 3) -> CreatePivotYeoJohnsonLink. Transform the parent to link-space, delta-method
    //     V_eta = G Sigma G', regularize, and take its Cholesky lower factor LHat.
    //   Phase 3 (C# 2365-2458): per draw, z = LStar^-1 (etaHat - etaStar) + StandardZ*smoothStd
    //     jitter, CLIPPED to +/- zLimit (6.0), etaDraw = etaHat + LHat*z, InverseLink, a
    //     finiteness check, then validate. On any failure the slot is LEFT UNSET and
    //     TransformFailures is incremented; RetainedReplicates counts the populated slots.
    //
    // T20 DELTAS from the previous (fc28c0c-era) port, ALL verified against the shipped file:
    //   1. maxRetries 5 -> 10 (C# 2160).
    //   2. The parent covariance goes through TryGetCovariance(thetaHat, true, out sigmaHat); a
    //      failure sets the uncertainty diagnostic message and returns null (C# 2174-2180), and
    //      the accepted sigmaHat is then REGULARIZED (C# 2181) before every downstream use
    //      (sigmaInv, the Phase-1 fallback, and V_eta_hat).
    //   3. The Mahalanobis threshold is the ADAPTIVE ChiSquared(p) at 1 - 1/(5B) (C# 2191-2192;
    //      was the fixed 0.999 -- numerically identical only at B = 200).
    //   4. The conditional cloneWithDataFrame warm start (C# 2204-2207), shared with the
    //      parametric arm through the collector.
    //   5. `new GeneralizedMethodOfMoments(...) { PenaltyIsRandom = false }` is GONE from the file
    //      (0 occurrences at c2e6192), so the GMM default `true` now applies -- H is back in the
    //      sandwich meat of Sigma*_b. This is numerically live and moves the pivot draws.
    //   6. The Yeo-Johnson links are built through the guarded CreatePivotYeoJohnsonLink (C# 2322,
    //      2332) rather than `new YeoJohnsonLink(samples)` directly, and the unused scaleSamples
    //      local is gone.
    //   7. A parent-Cholesky failure no longer falls back to the parametric bootstrap: it sets the
    //      diagnostic message and returns null (C# 2353-2360).
    //   8. Phase 3 CLIPS the pivot to +/- zLimit instead of rejecting the draw (C# 2409), so
    //      IncrementPivotRejection is no longer called anywhere and the PRNG always consumes
    //      exactly p StandardZ draws per replicate (the old early `break` made the draw count
    //      data-dependent).
    //   9. Phase 3 adds an explicit non-finite check on the inverse-linked theta (C# 2424-2425).
    //  10. A Phase-3 failure DISCARDS the slot (leaves default(ParameterSet), C# 2437-2438) and
    //      increments TransformFailures (C# 2434) instead of substituting thetaHat; RetainedReplicates
    //      is then set from the populated slots (C# 2447-2450). These two are the ONLY new
    //      BootstrapDiagnostics members the shipped analysis populates anywhere. A discard here is
    //      made user-visible by the T19 exact-count guard in run_uncertainty_quantification().
    //
    // NOT a delta (the brief's planning text said otherwise; the SHIPPED SOURCE GOVERNS): there is
    // NO compaction to accepted replicates before link fitting. Phase 1 always populates every
    // bootTheta[idx] (falling back to thetaHat), and Phase 2 fits the links over ALL B samples via
    // `bootTheta.Select(t => t[0])` (C# 2321, 2331). Likewise the z-limit is a CLIP, not a
    // rejection -- nothing is dropped for exceeding it.
    //
    // The C# Parallel.For -> a serial loop; the MersenneTwister draw cadence (seeds[idx] Phase 1,
    // seeds[idx]+B Phase 3, one StandardZ(NextDouble()) draw per parameter) is load-bearing for the
    // emitter's MT parity -- DO NOT reorder. Async/progress/cancellation/stopwatch lines dropped.
    std::optional<std::vector<ParameterSet>> get_parameter_sets_from_pivotal_bootstrap() {
        using corehydro::numerics::math::linalg::Matrix;
        using corehydro::numerics::math::linalg::MatrixRegularization;

        const int max_retries = 10;              // C# 2160 (T20: was 5)
        const double smooth_std_scale = 0.01;    // C# 2161
        const double z_limit = 6.0;              // C# 2162

        int b = bayesian_analysis_.output_length();                      // C# 2164
        int p = bulletin17c_distribution_->number_of_parameters();       // C# 2165
        std::vector<ParameterSet> results(static_cast<std::size_t>(b));  // C# 2166

        // Up-front seed cascade (C# 2167-2168): master PRNG seeds B per-replicate integers.
        corehydro::numerics::sampling::MersenneTwister master_prng(
            static_cast<std::uint32_t>(bayesian_analysis_.prng_seed()));
        std::vector<int> seeds = corehydro::numerics::utilities::next_integers(master_prng, b);

        // Parent distribution cloned + set to thetaHat as the data-generating model (C# 2170-2172).
        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;
        std::unique_ptr<UnivariateDistributionBase> parent_distribution =
            bulletin17c_distribution_->distribution()->clone();
        parent_distribution->set_parameters(theta_hat);

        // T20 (C# 2174-2181): the parent sandwich covariance must be finite and non-degenerate, or
        // the whole method aborts with a specific diagnostic; the accepted matrix is regularized.
        Matrix sigma_hat(p);
        if (!gmm_->try_get_covariance(theta_hat, true, sigma_hat)) {
            set_uncertainty_diagnostic_message(
                "Pivot bootstrap could not compute a finite covariance matrix for the parent GMM "
                "fit. Use the plain Bootstrap uncertainty method for this analysis.");
            return std::nullopt;
        }
        sigma_hat = MatrixRegularization::make_symmetric_positive_definite(sigma_hat);

        // Sandwich inverse for the Mahalanobis rejection (C# 2184-2190). On a singular inverse the
        // C# regularizes; port as a try/catch falling back to the regularized inverse
        // (Debug.WriteLine is a dropped silent guard).
        Matrix sigma_inv(p);
        try {
            sigma_inv = sigma_hat.inverse();
        } catch (...) {
            sigma_inv = MatrixRegularization::make_symmetric_positive_definite(sigma_hat).inverse();
        }
        // T20: adaptive Mahalanobis threshold ChiSq(p) at 1 - 1/(5B) (C# 2191-2192; was 0.999).
        double mahal_pvalue = 1.0 - 1.0 / (b * 5.0);
        double mahal_threshold =
            corehydro::numerics::distributions::ChiSquared(p).inverse_cdf(mahal_pvalue);

        // Diagnostics (C# 2197): TotalReplicates = B.
        BootstrapDiagnostics diag;
        diag.set_total_replicates(b);

        // T20: the same cloneWithDataFrame warm-start condition the parametric arm uses
        // (C# 2204-2207) -- mirrored EXACTLY.
        bool clone_with_data_frame_flag =
            bulletin17c_distribution_->data_frame().number_of_low_outliers() > 0 ||
            bulletin17c_distribution_->data_frame().uncertain_series().count() > 0 ||
            bulletin17c_distribution_->data_frame().interval_series().count() > 0 ||
            bulletin17c_distribution_->data_frame().threshold_series().count() > 0;

        // --- Phase 1: collect bootstrap fits theta*_b AND Sigma*_b (C# 2209-2312) ---
        std::vector<std::vector<double>> boot_theta(static_cast<std::size_t>(b));
        std::vector<Matrix> boot_sigma(static_cast<std::size_t>(b), Matrix(p));
        for (int idx = 0; idx < b; ++idx) {
            std::size_t iu = static_cast<std::size_t>(idx);
            corehydro::numerics::sampling::MersenneTwister prng(
                static_cast<std::uint32_t>(seeds[iu]));  // C# 2223
            std::optional<BootstrapReplicateFit> fit = collect_bootstrap_replicate_fit(
                prng, *parent_distribution, theta_hat, sigma_inv, mahal_threshold,
                clone_with_data_frame_flag, /*collect_covariance=*/true, max_retries, diag);

            if (fit && fit->covariance) {
                diag.add_function_evaluations(fit->function_evaluations);  // C# 2281
                boot_theta[iu] = std::move(fit->params);                   // C# 2298
                boot_sigma[iu] = std::move(*fit->covariance);              // C# 2299
            } else {
                // Fall back to the parent fit if all retries failed (C# 2291-2296).
                boot_theta[iu] = theta_hat;
                boot_sigma[iu] = sigma_hat;
                diag.increment_failed();
            }
        }

        // --- Phase 2: fit link functions from the bootstrap parameter samples (C# 2314-2363) ---
        // NO compaction: the links are fitted over ALL B replicates (C# 2321/2331).
        std::vector<double> location_samples(static_cast<std::size_t>(b));
        for (int idx = 0; idx < b; ++idx)
            location_samples[static_cast<std::size_t>(idx)] =
                boot_theta[static_cast<std::size_t>(idx)][0];

        std::vector<std::unique_ptr<ILinkFunction>> links(static_cast<std::size_t>(p));
        // Location (idx 0): guarded Yeo-Johnson fitted to the bootstrap location estimates
        // (C# 2321-2322).
        links[0] = create_pivot_yeo_johnson_link(location_samples, "location");
        // Scale (idx 1): plain Log link -- scale is strictly positive (C# 2325; T20: the unused
        // `scaleSamples` local and its commented-out YeoJohnson alternative are gone upstream).
        links[1] = std::make_unique<corehydro::numerics::functions::LogLink>();
        // Shape (idx 2, if present): guarded Yeo-Johnson fitted to the shape estimates
        // (C# 2328-2333).
        if (p >= 3) {
            std::vector<double> shape_samples(static_cast<std::size_t>(b));
            for (int idx = 0; idx < b; ++idx)
                shape_samples[static_cast<std::size_t>(idx)] =
                    boot_theta[static_cast<std::size_t>(idx)][2];
            links[2] = create_pivot_yeo_johnson_link(shape_samples, "shape");
        }
        LinkController link_controller(std::move(links));  // C# 2336-2340

        // Transform the parent to link-space and take the parent Cholesky factor (C# 2343-2360).
        std::vector<double> eta_hat = link_controller.link(theta_hat);
        Matrix g_hat = link_controller.link_jacobian(theta_hat);
        Matrix v_eta_hat = g_hat * sigma_hat * g_hat.transpose();
        v_eta_hat = MatrixRegularization::make_symmetric_positive_definite(v_eta_hat);
        Matrix l_hat(p);
        try {
            l_hat = corehydro::numerics::math::linalg::CholeskyDecomposition(v_eta_hat).l();
        } catch (...) {
            // T20 (C# 2353-2359): a parent-Cholesky failure now ABORTS with a diagnostic message
            // instead of falling back to the parametric bootstrap.
            set_uncertainty_diagnostic_message(
                "Pivot bootstrap could not factor the parent link-space covariance matrix. Use "
                "the plain Bootstrap uncertainty method for this analysis.");
            return std::nullopt;
        }

        // --- Phase 3: generate the pivot draws (C# 2365-2458) ---
        double smooth_std = smooth_std_scale / std::sqrt(static_cast<double>(p));  // C# 2369

        // A single reusable validator clone (serial loop; C# clones per Parallel.For thread).
        std::unique_ptr<UnivariateDistributionBase> validator = parent_distribution->clone();

        for (int idx = 0; idx < b; ++idx) {
            std::size_t iu = static_cast<std::size_t>(idx);
            corehydro::numerics::sampling::MersenneTwister prng(
                static_cast<std::uint32_t>(seeds[iu] + b));  // C# 2378
            std::optional<std::vector<double>> accepted_theta;

            try {
                // Transform the bootstrap fit to link-space (C# 2386-2389).
                std::vector<double> eta_star = link_controller.link(boot_theta[iu]);
                Matrix g_star = link_controller.link_jacobian(boot_theta[iu]);
                Matrix v_eta_star = g_star * boot_sigma[iu] * g_star.transpose();
                v_eta_star = MatrixRegularization::make_symmetric_positive_definite(v_eta_star);

                Matrix l_star =
                    corehydro::numerics::math::linalg::CholeskyDecomposition(v_eta_star).l();  // C# 2391-2392
                Matrix l_star_inv = l_star.inverse();  // C# 2393

                // Pivot: z = L*^-1 (etaHat - etaStar) (C# 2396-2402).
                Matrix diff_matrix(p, 1);
                for (int j = 0; j < p; ++j)
                    diff_matrix(j, 0) = eta_hat[static_cast<std::size_t>(j)] -
                                        eta_star[static_cast<std::size_t>(j)];
                Matrix z_matrix = l_star_inv * diff_matrix;

                // Add the smoothing jitter and CLIP to the z-limit (C# 2405-2410; T20: the old
                // reject-and-break is gone, so exactly p StandardZ draws are always consumed).
                std::vector<double> z(static_cast<std::size_t>(p));
                for (int j = 0; j < p; ++j) {
                    std::size_t ju = static_cast<std::size_t>(j);
                    z[ju] = z_matrix(j, 0) +
                            corehydro::numerics::distributions::Normal::standard_z(
                                prng.next_double()) *
                                smooth_std;
                    z[ju] = std::max(-z_limit, std::min(z_limit, z[ju]));
                }

                // Map back: etaDraw = etaHat + LHat * z (C# 2413-2420).
                Matrix z_col(p, 1);
                for (int j = 0; j < p; ++j) z_col(j, 0) = z[static_cast<std::size_t>(j)];
                Matrix lz_matrix = l_hat * z_col;
                std::vector<double> eta_draw(static_cast<std::size_t>(p));
                for (int j = 0; j < p; ++j)
                    eta_draw[static_cast<std::size_t>(j)] =
                        eta_hat[static_cast<std::size_t>(j)] + lz_matrix(j, 0);

                // Inverse transform back to real-space, check finiteness, then validate
                // (C# 2423-2429). The C# `ValidateParameters(theta, true)` throws on an invalid
                // vector -> the catch below.
                std::vector<double> theta = link_controller.inverse_link(eta_draw);
                if (!std::all_of(theta.begin(), theta.end(),
                                 [](double value) { return std::isfinite(value); }))
                    throw std::runtime_error("the pivot transform produced non-finite parameters");
                validator->set_parameters(theta);
                if (!validator->parameters_valid())
                    throw std::runtime_error("pivot draw failed parameter validation");
                accepted_theta = std::move(theta);
            } catch (...) {
                // T20 (C# 2431-2435): the C# Debug.WriteLine is a dropped silent guard; the slot is
                // DISCARDED (no thetaHat substitution) and TransformFailures is incremented.
                diag.increment_transform_failure();
            }

            // C# 2437-2438: only a successful draw populates the slot; a failure leaves
            // default(ParameterSet) (an empty `values`, this port's "unset" -- see parameter_set.hpp).
            if (accepted_theta)
                results[iu] = ParameterSet(std::move(*accepted_theta),
                                           std::numeric_limits<double>::quiet_NaN());
        }

        // T20 (C# 2447-2450): RetainedReplicates counts the slots that survived Phase 3.
        long long retained = 0;
        for (const ParameterSet& parameter_set : results) {
            if (static_cast<int>(parameter_set.values.size()) == p &&
                std::all_of(parameter_set.values.begin(), parameter_set.values.end(),
                            [](double value) { return std::isfinite(value); }))
                ++retained;
        }
        diag.set_retained_replicates(static_cast<int>(retained));

        // Persist the diagnostics on the analysis and return the sets (C# 2451-2453).
        bootstrap_results_ = std::move(diag);
        return results;
    }
    // ================= X8: LinkedMultivariateNormal uncertainty path =================
    //
    // PROVENANCE (carried verbatim from the X8 brief): the shipped C# path samples from
    // **MultivariateNormal** (NOT MVT), and the influence-function center shift
    // `etaHat[1]/[2] += shift` (C# 1044 / 1053) is **COMMENTED OUT** in the source. So
    // ComputeInfluenceStatistics is genuinely computed and then **DISCARDED** -- it is real,
    // reachable code that is ported for fidelity, but its result never affects the draws. This
    // port mirrors that EXACTLY: no MVT, no center shift.

    // C# `GetParameterSetsFromLinkedMultivariateNormal` (C# 923-1162): fits per-family sinh-arcsinh
    // link functions from the GMM point estimate + sandwich covariance, transforms to link space,
    // delta-methods the covariance, and Latin-Hypercube samples MVN(etaHat, VetaHat) -- mapping
    // each draw back through the inverse links with a bounded retry. Returns nullopt when the
    // link-space covariance is not positive-definite, when the rejection rate exceeds 50%, or when
    // fewer than 2 draws survive (C# `return null`, each of which the caller falls back to plain
    // MVN for). The C# Parallel.For -> a serial loop; progress/cancellation/Debug.WriteLine dropped.
    std::optional<std::vector<ParameterSet>> get_parameter_sets_from_linked_multivariate_normal() {
        using corehydro::numerics::math::linalg::Matrix;
        using corehydro::numerics::math::linalg::MatrixRegularization;

        int b = bayesian_analysis_.output_length();                      // C# 925
        std::vector<ParameterSet> results(static_cast<std::size_t>(b));   // C# 926

        // Scope guard restoring the identity (empty) LinkController on EVERY exit path (C# 1157-
        // 1161 finally). Runs on the normal return, the early nullopt returns, and any throw.
        struct LinkControllerRestore {
            Bulletin17CDistribution* dist;
            ~LinkControllerRestore() { dist->set_link_controller(LinkController()); }
        } restore_guard{bulletin17c_distribution_.get()};

        // Step 1: thetaHat + sandwich covariance from GMM (C# 929-930). BEFORE any link install.
        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;
        Matrix sigma_hat = gmm_->get_covariance(theta_hat);

        // Step 2: WEDS in natural parameter space (C# 939) -- computed before the temporary links.
        std::vector<double> weds = bulletin17c_distribution_->weighted_error_direction_score(theta_hat);

        // Step 3: per-family link selection (C# 951-1010).
        int p = bulletin17c_distribution_->number_of_parameters();
        std::vector<std::unique_ptr<ILinkFunction>> links(static_cast<std::size_t>(p));
        DistributionType dist_type = bulletin17c_distribution_->distribution_type();

        if (dist_type == DistributionType::GammaDistribution) {
            // Gamma: [scale > 0, shape > 0] -- both positive-support LogASinH (C# 961-962).
            links[0] = std::make_unique<LogASinHLink>(
                create_positive_parameter_link(theta_hat[0], safe_standard_error(sigma_hat, 0)));
            links[1] = std::make_unique<LogASinHLink>(
                create_positive_parameter_link(theta_hat[1], safe_standard_error(sigma_hat, 1)));
        } else if (dist_type == DistributionType::PearsonTypeIII ||
                   dist_type == DistributionType::LogPearsonTypeIII) {
            // P3/LP3: [location(real), scale(>0), shape(real)] (C# 968-991).
            double gamma_hat = theta_hat[2];
            double mu_se = safe_standard_error(sigma_hat, 0);
            double scale_se = safe_standard_error(sigma_hat, 1);
            double gamma_se = safe_standard_error(sigma_hat, 2);
            links[0] = std::make_unique<ASinHLink>(
                create_pearson_location_link(theta_hat[0], mu_se, gamma_hat, clean_weds(weds, 0)));
            links[1] =
                std::make_unique<LogASinHLink>(create_pearson_scale_link(theta_hat[1], scale_se));
            double gamma_direction_score = orient_gamma_weds_for_link(gamma_hat, clean_weds(weds, 2));
            links[2] = std::make_unique<ASinHLink>(
                create_gamma_shape_link(gamma_hat, gamma_se, gamma_direction_score));
        } else {
            // Normal / LogNormal / Exponential: [location(real), scale(>0)] (C# 999-1009).
            double mu_se = safe_standard_error(sigma_hat, 0);
            links[0] = std::make_unique<ASinHLink>(
                create_location_link(theta_hat[0], mu_se, clean_weds(weds, 0)));
            double scale_se = safe_standard_error(sigma_hat, 1);
            links[1] = std::make_unique<LogASinHLink>(
                create_positive_parameter_link(theta_hat[1], scale_se));
        }

        // Step 4: install the temporary LinkController (C# 1013).
        bulletin17c_distribution_->set_link_controller(LinkController(std::move(links)));

        // Step 5-7: transform, diagonal link Jacobian G, delta-method V_eta = G Sigma G' (C# 1016-
        // 1023), then regularize to a symmetric positive-definite matrix.
        std::vector<double> eta_hat = bulletin17c_distribution_->link_controller().link(theta_hat);
        Matrix g_hat = bulletin17c_distribution_->link_controller().link_jacobian(theta_hat);
        Matrix v_eta_hat = g_hat * sigma_hat * g_hat.transpose();
        v_eta_hat = MatrixRegularization::make_symmetric_positive_definite(v_eta_hat);

        // T20 (adjacent v2.0.0 diff in this region): the old Step 7b -- the
        // `ComputeInfluenceStatistics(thetaHat, 0.999)` call plus the two commented-out center-shift
        // lines `etaHat[1]/[2] += shift` -- was DELETED WHOLESALE from this method at c2e6192, so
        // the call is gone here too. ComputeInfluenceStatistics itself is still DEFINED upstream
        // (C# 1714) with zero call sites, so its port below stays as-is. No numeric effect either
        // way: the C# shift lines were already commented out, so the statistics were computed and
        // discarded (the X8 PROVENANCE note above).

        // Step 8: MVN(etaHat, V_eta) -- NOT MVT (C# 1058-1067). A non-positive-definite covariance
        // throws in the ctor -> nullopt (the guard restores the identity links; caller falls back).
        std::optional<MultivariateNormal> mvn;
        try {
            mvn.emplace(eta_hat, v_eta_hat.to_array());
        } catch (...) {
            return std::nullopt;  // C# Debug.WriteLine + return null
        }

        const int seed = bayesian_analysis_.prng_seed();
        std::vector<std::vector<double>> eta_draws = mvn->latin_hypercube_random_values(b, seed);  // C# 1068

        // A single reusable validator clone (serial loop; C# clones per Parallel.For thread).
        std::unique_ptr<UnivariateDistributionBase> validator =
            bulletin17c_distribution_->distribution()->clone();
        auto accepts = [&validator](const std::vector<double>& theta) -> bool {
            try {
                validator->set_parameters(theta);  // C# ValidateParameters(theta, true)
                return validator->parameters_valid();
            } catch (...) {
                return false;  // C# catch -> rejected draw
            }
        };

        int rejection_count = 0;

        // Step 9: map each draw back theta = InverseLink(eta) (C# 1082-1130 serial).
        for (int idx = 0; idx < b; ++idx) {
            std::optional<std::vector<double>> accepted_theta;
            std::size_t iu = static_cast<std::size_t>(idx);

            // First try: the LHS draw (best space coverage) (C# 1090-1097).
            {
                std::vector<double> theta =
                    bulletin17c_distribution_->link_controller().inverse_link(eta_draws[iu]);
                if (accepts(theta)) accepted_theta = std::move(theta);
            }

            // Retry with fresh eta-space draws seeded per-index (C# 1099-1118).
            if (!accepted_theta) {
                int pp = static_cast<int>(eta_hat.size());
                corehydro::numerics::sampling::MersenneTwister prng(
                    static_cast<std::uint32_t>(seed + b + idx));
                for (int retry = 0; retry < 10 && !accepted_theta; ++retry) {
                    std::vector<double> u =
                        corehydro::numerics::utilities::next_doubles(prng, 1, pp)[0];
                    std::vector<double> eta = mvn->inverse_cdf(u);
                    std::vector<double> theta =
                        bulletin17c_distribution_->link_controller().inverse_link(eta);
                    if (accepts(theta)) accepted_theta = std::move(theta);
                }
                if (!accepted_theta) ++rejection_count;  // C# Interlocked.Increment(rejectionCount)
            }

            // Rejected draws stay as the default/empty ParameterSet -- filtered below; NO parent
            // thetaHat fallback (high rejection triggers the whole-method fallback) (C# 1120-1125).
            if (accepted_theta)
                results[iu] =
                    ParameterSet(std::move(*accepted_theta), std::numeric_limits<double>::quiet_NaN());
        }

        // Rejection-rate guard: too-aggressive links -> null -> caller falls back (C# 1132-1141).
        double rejection_rate = static_cast<double>(rejection_count) / static_cast<double>(b);
        if (rejection_rate > 0.50) return std::nullopt;

        // Filter rejected (empty) draws (C# 1143-1149).
        std::vector<ParameterSet> valid_results;
        valid_results.reserve(results.size());
        for (auto& ps : results)
            if (!ps.values.empty()) valid_results.push_back(std::move(ps));
        if (valid_results.size() < 2) return std::nullopt;

        return valid_results;
    }

    // ---- LinkedMVN link-builder helpers (C# 1175-1573, all pure/static) ----

    // C# `CleanWeds(double[] weds, int index)` (C# 1175-1181): finite WEDS component clamped to
    // [-1, 1], or 0 when unavailable. The C# null-array check maps to the empty/range check.
    static double clean_weds(const std::vector<double>& weds, int index) {
        if (index < 0 || index >= static_cast<int>(weds.size()) ||
            !std::isfinite(weds[static_cast<std::size_t>(index)]))
            return 0.0;
        return std::clamp(weds[static_cast<std::size_t>(index)], -1.0, 1.0);
    }

    // C# `CreateLocationLink` (C# 1197-1206): unbounded ASinH location link; signed WEDS drives
    // adaptive asymmetry (censoring direction), tail thickness fixed (Delta = 1).
    static ASinHLink create_location_link(double center, double standard_error, double weds_score) {
        ASinHLink link(center, standard_error);
        link.set_use_adaptive_epsilon(true);
        link.set_parent_indicator(std::clamp(weds_score, -1.0, 1.0));
        link.set_epsilon_max(0.50);
        link.set_epsilon_slope(1.0);
        return link;
    }

    // C# `CreatePearsonLocationLink` (C# 1223-1236): LP3/P3 location ASinH link; parent indicator
    // is the 0.5*gammaHat + WEDS blend, conservative fixed cap.
    static ASinHLink create_pearson_location_link(double center, double standard_error,
                                                  double gamma_hat, double weds_score) {
        ASinHLink link(center, standard_error);
        link.set_use_adaptive_epsilon(true);
        link.set_parent_indicator((0.5 * gamma_hat) + weds_score);
        link.set_epsilon_max(0.5);
        link.set_epsilon_slope(1.0);
        return link;
    }

    // C# `OrientGammaWedsForLink` (C# 1254-1276): WEDS is magnitude, gammaHat supplies direction.
    static double orient_gamma_weds_for_link(double gamma_hat, double raw_gamma_weds) {
        constexpr double kGammaAsymmetryFull = 0.50;
        constexpr double kMinPositiveDirection = 0.10;

        if (!std::isfinite(gamma_hat)) return 0.0;

        double weds_magnitude = std::abs(std::clamp(raw_gamma_weds, -1.0, 1.0));

        if (gamma_hat > 0.0) return std::max(kMinPositiveDirection, weds_magnitude);
        if (gamma_hat == 0.0) return 0.0;

        // Negative fitted gamma: smooth gate near zero (C# 1274-1275).
        double asymmetry_gate = smooth_step(0.0, kGammaAsymmetryFull, std::abs(gamma_hat));
        return -weds_magnitude * asymmetry_gate;
    }

    // C# `SmoothStep` (C# 1289-1299): cubic smoothstep in [0, 1].
    static double smooth_step(double edge0, double edge1, double x) {
        if (!std::isfinite(x)) return 0.0;
        if (edge1 <= edge0) return x >= edge1 ? 1.0 : 0.0;
        double t = std::clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
        return t * t * (3.0 - (2.0 * t));
    }

    // C# `CreateGammaTailDelta` (C# 1315-1332): ASinH tail-shape delta in [0.80, 1.00] for the
    // LP3/P3 gamma link; smaller delta gives heavier symmetric tails near a weakly identified zero.
    static double create_gamma_tail_delta(double gamma_hat, double gamma_se) {
        constexpr double kMinDelta = 0.80;
        constexpr double kMaxTailReduction = 0.20;
        constexpr double kGammaSEScale = 0.25;
        constexpr double kSignalForTailStart = 0.75;
        constexpr double kSignalForTailFade = 2.0;

        if (!std::isfinite(gamma_hat) || !std::isfinite(gamma_se) || gamma_se <= 0.0) return 1.0;

        double gamma_signal = standardized_magnitude(gamma_hat, gamma_se);
        double weak_direction_gate =
            1.0 - smooth_step(kSignalForTailStart, kSignalForTailFade, gamma_signal);
        double uncertainty_gate = gamma_se / (gamma_se + kGammaSEScale);
        double delta = 1.0 - (kMaxTailReduction * weak_direction_gate * uncertainty_gate);
        return std::clamp(delta, kMinDelta, 1.0);
    }

    // C# `StandardizedMagnitude` (C# 1346-1355): |estimate| / SE signal-to-noise ratio.
    static double standardized_magnitude(double estimate, double standard_error) {
        if (!std::isfinite(estimate)) return 0.0;
        if (!std::isfinite(standard_error) || standard_error <= 1e-12)
            return std::abs(estimate) > 0.0 ? std::numeric_limits<double>::infinity() : 0.0;
        return std::abs(estimate) / standard_error;
    }

    // C# `SafeStandardError` (C# 1368-1378): robust SE from a covariance diagonal; 1e-12 floor on
    // out-of-range / non-finite / non-positive variance. The C# null-matrix check is dropped (the
    // ported Matrix is a value type, never null).
    static double safe_standard_error(const corehydro::numerics::math::linalg::Matrix& covariance,
                                      int index) {
        if (index < 0 || index >= covariance.number_of_rows() ||
            index >= covariance.number_of_columns())
            return 1e-12;
        double variance = covariance(index, index);
        if (!std::isfinite(variance) || variance <= 0.0) return 1e-12;
        return std::sqrt(variance);
    }

    // C# `CreatePositiveParameterLink` (C# 1396-1417): positive-support LogASinH driven by the
    // relative standard error SE/estimate (unit-invariant).
    static LogASinHLink create_positive_parameter_link(double center, double standard_error) {
        constexpr double kScaleCVReference = 0.25;
        constexpr double kEpsilonMax = 0.75;
        constexpr double kEpsilonSlope = 1.25;
        constexpr double kMaxTailReduction = 0.18;
        constexpr double kMinDelta = 0.82;

        double relative_se = relative_standard_error(center, standard_error);
        double log_scale = log_scale_from_relative_standard_error(relative_se);
        double uncertainty = relative_uncertainty_score(relative_se, kScaleCVReference);
        double delta = 1.0 - (kMaxTailReduction * uncertainty);

        LogASinHLink link(center, log_scale);
        link.set_delta(std::clamp(delta, kMinDelta, 1.0));
        link.set_use_adaptive_epsilon(true);
        link.set_parent_indicator(uncertainty);
        link.set_epsilon_max(kEpsilonMax);
        link.set_epsilon_slope(kEpsilonSlope);
        return link;
    }

    // C# `CreatePearsonScaleLink` (C# 1431-1434): LP3/P3 scale uses the positive-parameter rule.
    static LogASinHLink create_pearson_scale_link(double center, double standard_error) {
        return create_positive_parameter_link(center, standard_error);
    }

    // C# `RelativeStandardError` (C# 1447-1457): dimensionless |SE / estimate| for positive params.
    static double relative_standard_error(double center, double standard_error) {
        if (!std::isfinite(center) || center <= 0.0 || !std::isfinite(standard_error) ||
            standard_error <= 0.0)
            return 0.0;
        double relative_se = std::abs(standard_error / center);
        if (!std::isfinite(relative_se)) return 0.0;
        return relative_se;
    }

    // C# `RelativeUncertaintyScore` (C# 1470-1479): saturating map relSE/(relSE + reference).
    static double relative_uncertainty_score(double relative_se, double reference_relative_se) {
        if (!std::isfinite(relative_se) || relative_se <= 0.0) return 0.0;
        if (!std::isfinite(reference_relative_se) || reference_relative_se <= 0.0) return 1.0;
        return std::clamp(relative_se / (relative_se + reference_relative_se), 0.0, 1.0);
    }

    // C# `LogScaleFromRelativeStandardError` (C# 1493-1508): sqrt(log(1 + CV^2)) log-scale.
    static double log_scale_from_relative_standard_error(double relative_se) {
        constexpr double kMinLogScale = 1e-12;
        constexpr double kMaxRelativeSE = 10.0;

        if (!std::isfinite(relative_se) || relative_se <= 0.0) return kMinLogScale;
        double bounded_relative_se = std::min(relative_se, kMaxRelativeSE);
        double log_scale = std::sqrt(std::log(1.0 + (bounded_relative_se * bounded_relative_se)));
        if (!std::isfinite(log_scale) || log_scale <= 0.0) return kMinLogScale;
        return std::max(log_scale, kMinLogScale);
    }

    // C# `CreateGammaShapeLink` (C# 1525-1566): LP3/P3 gamma ASinH link; oriented WEDS supplies
    // asymmetry after a deadband + asymmetric positive/negative epsilon caps, tail-shape via delta.
    static ASinHLink create_gamma_shape_link(double gamma_hat, double gamma_se,
                                             double gamma_direction_score) {
        constexpr double kWedsDeadband = 0.10;
        constexpr double kMaxPositiveEpsilon = 1.0;
        constexpr double kMaxNegativeEpsilon = 0.5;
        constexpr double kMinPositiveEpsilon = 0.50;
        constexpr double kMinNegativeEpsilon = 0.25;

        double bounded_weds = std::clamp(gamma_direction_score, -1.0, 1.0);
        double abs_weds = std::abs(bounded_weds);
        double epsilon_max = 0.0;
        double parent_indicator = 0.0;

        if (abs_weds >= kWedsDeadband) {
            parent_indicator = bounded_weds;
            if (bounded_weds < 0.0) {
                epsilon_max =
                    kMinNegativeEpsilon + ((kMaxNegativeEpsilon - kMinNegativeEpsilon) * abs_weds);
                epsilon_max = std::min(epsilon_max, kMaxNegativeEpsilon);
            } else {
                epsilon_max =
                    kMinPositiveEpsilon + ((kMaxPositiveEpsilon - kMinPositiveEpsilon) * abs_weds);
                if (gamma_hat > 0.75) epsilon_max = std::max(epsilon_max, 0.75);
            }
            epsilon_max = std::clamp(epsilon_max, 0.0, kMaxPositiveEpsilon);
        }

        ASinHLink link(gamma_hat, gamma_se);
        link.set_delta(create_gamma_tail_delta(gamma_hat, gamma_se));
        link.set_use_adaptive_epsilon(true);
        link.set_parent_indicator(parent_indicator);
        link.set_epsilon_max(epsilon_max);
        link.set_epsilon_slope(1.5);
        return link;
    }

    // C# `InfluenceStatistics` struct (C# 1582-1598): quantile degrees of freedom + per-parameter
    // skewness of the influence functions. COMPUTED then DISCARDED on the shipped path (see the
    // section provenance note): the MVT nu and the center-shift skewness are never applied.
    struct InfluenceStatistics {
        double nu_quantile = 0.0;
        std::vector<double> parameter_skewness;
    };

    // C# `ComputeInfluenceStatistics` (C# 1628-1740): Cohn et al. (2001) influence-function
    // statistics from the per-observation GMM moment conditions psi_i = Bread^-1 D'W m_i. Ported
    // for fidelity; the result is DISCARDED (no MVT, no center shift). C# Debug.WriteLine +
    // default-return guards (PointwiseMomentConditions unavailable / singular bread) preserved.
    InfluenceStatistics compute_influence_statistics(const std::vector<double>& theta_hat,
                                                     double target_non_exceedance_probability) {
        using corehydro::numerics::math::linalg::Matrix;
        int n_params = static_cast<int>(theta_hat.size());
        InfluenceStatistics result;
        result.nu_quantile = 30.0;                                            // C# 1633
        result.parameter_skewness.assign(static_cast<std::size_t>(n_params), 0.0);

        // 1. Quantile gradient g_p = dX_p/dtheta (C# 1638).
        std::vector<double> gp =
            bulletin17c_distribution_->quantile_gradient(target_non_exceedance_probability, theta_hat);

        // 2. Per-observation moment conditions [n x q] (C# 1641-1649).
        const auto& pointwise_mc = gmm_->pointwise_moment_conditions();
        if (!pointwise_mc) return result;  // C# 1642-1646
        corehydro::numerics::math::linalg::Matrix2D mi = pointwise_mc(theta_hat);
        int n = static_cast<int>(mi.size());
        int q = (n > 0) ? static_cast<int>(mi[0].size()) : 0;

        // 3. Bread = D'WD + H (C# 1652-1667).
        Matrix d = gmm_->get_jacobian(theta_hat);
        Matrix dt = d.transpose();
        const Matrix& w = gmm_->w().value();  // C# `_gmm.W!`
        Matrix h = gmm_->get_penalty_hessian(theta_hat);
        Matrix bread = dt * w * d + h;
        Matrix bread_inv(n_params);
        try {
            bread_inv = bread.inverse();
        } catch (...) {
            return result;  // C# 1663-1667 singular-bread guard
        }
        Matrix dt_w = dt * w;

        // 4. psi_i vectors and quantile influence scores s_i = g_p' psi_i (C# 1671-1703). NOTE:
        // s_i is computed exactly as the C# but the shipped path derives nu from per-parameter
        // kurtosis (5a) rather than from s -- so s is dead, mirrored for structural fidelity.
        std::vector<std::vector<double>> psi_all(static_cast<std::size_t>(n));
        std::vector<double> s(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) {
            std::size_t iu = static_cast<std::size_t>(i);
            std::vector<double> dt_wm(static_cast<std::size_t>(n_params));
            for (int j = 0; j < n_params; ++j) {
                double sum = 0.0;
                for (int k = 0; k < q; ++k)
                    sum += dt_w(j, k) * mi[iu][static_cast<std::size_t>(k)];
                dt_wm[static_cast<std::size_t>(j)] = sum;
            }
            std::vector<double> psi(static_cast<std::size_t>(n_params));
            for (int j = 0; j < n_params; ++j) {
                double sum = 0.0;
                for (int k = 0; k < n_params; ++k)
                    sum += bread_inv(j, k) * dt_wm[static_cast<std::size_t>(k)];
                psi[static_cast<std::size_t>(j)] = sum;
            }
            psi_all[iu] = psi;
            double si = 0.0;
            for (int j = 0; j < n_params; ++j)
                si += gp[static_cast<std::size_t>(j)] * psi[static_cast<std::size_t>(j)];
            s[iu] = si;
        }
        (void)s;  // computed per the C#, unused on the shipped path.

        // 5a. Per-parameter nu from kurtosis, geometric mean for the MVT nu (C# 1705-1725).
        double log_nu_sum = 0.0;
        int nu_count = 0;
        for (int j = 0; j < n_params; ++j) {
            std::vector<double> psi_j(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) psi_j[static_cast<std::size_t>(i)] = psi_all[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            double nu_j = compute_degrees_of_freedom_from_kurtosis(psi_j, n);
            log_nu_sum += std::log(nu_j);
            ++nu_count;
        }
        result.nu_quantile = nu_count > 0 ? std::exp(log_nu_sum / nu_count) : 1000.0;

        // 5b. Per-parameter skewness for the (discarded) center shift (C# 1727-1737).
        for (int j = 0; j < n_params; ++j) {
            std::vector<double> psi_j(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i) psi_j[static_cast<std::size_t>(i)] = psi_all[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            result.parameter_skewness[static_cast<std::size_t>(j)] =
                compute_skewness_from_influence(psi_j, n);
        }

        return result;
    }

    // C# `ComputeDegreesOfFreedomFromKurtosis` (C# 1761-1794): match empirical kurtosis to the
    // Student-t kurtosis nu = 2(2k-3)/(k-3); sub-Gaussian returns 1000, else clamped to [5, 1000].
    static double compute_degrees_of_freedom_from_kurtosis(const std::vector<double>& scores, int n) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += scores[static_cast<std::size_t>(i)];
        mean /= n;

        double m2 = 0.0, m4 = 0.0;
        for (int i = 0; i < n; ++i) {
            double dd = scores[static_cast<std::size_t>(i)] - mean;
            double d2 = dd * dd;
            m2 += d2;
            m4 += d2 * d2;
        }
        m2 /= n;
        m4 /= n;

        if (m2 <= 0.0) return 1000.0;
        double kappa = m4 / (m2 * m2);
        if (kappa <= 3.0 + 1e-6) return 1000.0;
        double nu = 2.0 * (2.0 * kappa - 3.0) / (kappa - 3.0);
        return std::clamp(nu, 5.0, 1000.0);
    }

    // C# `ComputeSkewnessFromInfluence` (C# 1806-1831): Fisher skewness M3 / M2^{3/2}, or 0 when
    // the variance is near zero.
    static double compute_skewness_from_influence(const std::vector<double>& scores, int n) {
        double mean = 0.0;
        for (int i = 0; i < n; ++i) mean += scores[static_cast<std::size_t>(i)];
        mean /= n;

        double m2 = 0.0, m3 = 0.0;
        for (int i = 0; i < n; ++i) {
            double dd = scores[static_cast<std::size_t>(i)] - mean;
            double d2 = dd * dd;
            m2 += d2;
            m3 += d2 * dd;
        }
        m2 /= n;
        m3 /= n;

        if (m2 <= 1e-30) return 0.0;
        double sigma = std::sqrt(m2);
        return m3 / (sigma * sigma * sigma);
    }

    // C# `AccelerationConstants(double[] thetaHats)` (C# 2408-2464): the BCa jackknife acceleration
    // constant a_i = I3_i / (6 * I2_i^1.5), where I2/I3 accumulate the squared/cubed leave-one-out
    // deltas of each parameter. Ported per the A8 brief for structural fidelity; the C# NEVER calls
    // it on any shipped path (it is BCa building-block machinery the C# never wires -- the X9 pivot
    // bootstrap does not use it), so it lands here UNCALLED -- mirroring the A7
    // evaluate_log_quantile_safe precedent. The
    // jackknife is fully deterministic (no PRNG). The C# Parallel.For -> a serial loop and
    // Tools.ParallelAdd -> plain `+=`; a GMM failure is the C#'s swallowed exception (silent guard).
    std::vector<double> acceleration_constants(const std::vector<double>& theta_hats) {
        bulletin17c_distribution_->data_frame().create_full_time_series();  // C# 2417
        // C# also reads N = SampleSize (C# 2418) but never uses it downstream; omitted.
        const auto& full = bulletin17c_distribution_->data_frame().full_time_series();
        int start_index = full.front()->index();                        // C# 2419
        int end_index = full.back()->index();                           // C# 2420
        int p = bulletin17c_distribution_->number_of_parameters();      // C# 2421
        std::vector<double> i2(static_cast<std::size_t>(p), 0.0);       // C# 2422
        std::vector<double> i3(static_cast<std::size_t>(p), 0.0);       // C# 2423
        std::vector<double> a(static_cast<std::size_t>(p), 0.0);        // C# 2424
        DistributionType type = bulletin17c_distribution_->distribution_type();  // C# `dist.Type`

        // Serial jackknife over the full inclusive index range (C# Parallel.For 2427-2455).
        for (int idx = start_index; idx <= end_index; ++idx) {
            corehydro::models::DataFrame jack_data =
                bulletin17c_distribution_->data_frame().JackKnife(idx);  // C# 2430
            try {
                Bulletin17CDistribution model(std::move(jack_data), type);  // C# 2437
                GeneralizedMethodOfMoments gmm(model);                       // C# 2438
                gmm.estimate();                                             // C# 2439
                if (gmm.status() == OptimizationStatus::Success) {          // C# 2440
                    const std::vector<double>& theta_jack = gmm.best_parameter_set().values;
                    for (int i = 0; i < p; ++i) {
                        double d = theta_hats[static_cast<std::size_t>(i)] -
                                   theta_jack[static_cast<std::size_t>(i)];
                        i2[static_cast<std::size_t>(i)] += std::pow(d, 2);  // C# 2445
                        i3[static_cast<std::size_t>(i)] += std::pow(d, 3);  // C# 2446
                    }
                }
            } catch (...) {
                // C# swallows: the GMM solver can fail to find a solution (silent guard).
            }
        }

        // Acceleration constant (C# 2458-2461).
        for (int i = 0; i < p; ++i)
            a[static_cast<std::size_t>(i)] =
                i3[static_cast<std::size_t>(i)] /
                (std::pow(i2[static_cast<std::size_t>(i)], 1.5) * 6.0);
        return a;
    }

    // C# `EvaluateLogQuantileSafe` (C# 856-880). Ported per the A7 brief for structural fidelity;
    // the shipped MVN path does not call it (the C# MVN sampler validates via ValidateParameters),
    // but the shipped LinkedMVN path (X8) uses it. Evaluates the log-space quantile for
    // LP3/LogNormal (via the P3/Normal base) or the real-space quantile otherwise, falling back to
    // the point estimate on any failure.
    double evaluate_log_quantile_safe(const std::vector<double>& parameters,
                                      double non_exceedance_probability) const {
        try {
            DistributionType dist_type = bulletin17c_distribution_->distribution_type();
            std::unique_ptr<UnivariateDistributionBase> dist;
            if (dist_type == DistributionType::LogPearsonTypeIII)
                dist = Bulletin17CDistribution::create_distribution(DistributionType::PearsonTypeIII);
            else if (dist_type == DistributionType::LogNormal)
                dist = Bulletin17CDistribution::create_distribution(DistributionType::Normal);
            else
                dist = bulletin17c_distribution_->distribution()->clone();

            dist->set_parameters(parameters);
            if (!dist->parameters_valid())
                throw std::invalid_argument("invalid parameters");  // C# ValidateParameters throw
            return dist->inverse_cdf(non_exceedance_probability);
        } catch (...) {
            // Fallback: the point estimate (C# 874-879; Debug.WriteLine dropped).
            return bulletin17c_distribution_->distribution()->inverse_cdf(non_exceedance_probability);
        }
    }

    // C# `SanitizeParameterSets` (C# 3140-3164): replaces non-finite Fitness/Values with 0 so the
    // stored parameter sets are finite.
    static void sanitize_parameter_sets(std::vector<ParameterSet>& parameter_sets) {
        for (auto& ps : parameter_sets) {
            if (!std::isfinite(ps.fitness)) ps.fitness = 0.0;
            for (double& v : ps.values)
                if (!std::isfinite(v)) v = 0.0;
        }
    }

    // ================= A9: Cohn-style delta-method CI helpers =================

    // C# `BuildQuadratureGrid` (C# 2672-2767): Cohn (2013) GRIDMAKE. Regularizes the covariance,
    // builds standardized per-dimension nodes/weights (Gamma quadrature for the scale dim sigma;
    // Gauss-Hermite +/-1 weight 0.5 for the others), factors S via CohnCholesky (falling back to
    // standard Cholesky, then diagonal sqrt on failure), and tensor-products through the factor.
    // n_nodes_per_dim is C#-signature-only (the body hard-codes 2 nodes/dim). Non-static to mirror
    // the C# private instance method (it reads no instance state).
    std::pair<std::vector<std::vector<double>>, std::vector<double>> build_quadrature_grid(
        const std::vector<double>& mean, const corehydro::numerics::math::linalg::Matrix& covariance,
        int dimension, int n_nodes_per_dim) const {
        using corehydro::numerics::math::linalg::CholeskyDecomposition;
        using corehydro::numerics::math::linalg::Matrix;
        using corehydro::numerics::math::linalg::MatrixRegularization;
        (void)n_nodes_per_dim;  // C# takes it but hard-codes 2 nodes/dim below.

        // Regularize to ensure positive-definiteness (C# 2676).
        Matrix s = MatrixRegularization::make_symmetric_positive_definite(covariance);

        // Scale (sigma) dimension uses Gamma quadrature; index 1 for B17C dists (C# 2680).
        int scale_idx = (dimension >= 2) ? 1 : -1;

        std::vector<std::vector<double>> per_dim_nodes(static_cast<std::size_t>(dimension));
        std::vector<std::vector<double>> per_dim_weights(static_cast<std::size_t>(dimension));
        for (int d = 0; d < dimension; ++d) {
            std::size_t du = static_cast<std::size_t>(d);
            if (d == scale_idx && mean[du] > 0.0) {
                // 2-point generalized Gauss-Laguerre (Gamma) quadrature for sigma (C# 2708-2732).
                // Standardized nodes have mean 0, variance 1 exactly. alpha = sigma_hat^2 / Var(sigma_hat).
                double var_sigma = std::max(s(d, d), 1e-30);
                double alpha = mean[du] * mean[du] / var_sigma;
                if (alpha > 50.0) {
                    // For very large alpha, Gamma ~ Normal (C# 2712-2716).
                    per_dim_nodes[du] = {-1.0, 1.0};
                    per_dim_weights[du] = {0.5, 0.5};
                } else {
                    double sqrt_alpha = std::sqrt(alpha);
                    double sqrt_alpha_p1 = std::sqrt(alpha + 1.0);
                    double z1 = (1.0 - sqrt_alpha_p1) / sqrt_alpha;
                    double z2 = (1.0 + sqrt_alpha_p1) / sqrt_alpha;
                    double w1 = (alpha + 1.0 + sqrt_alpha_p1) / (2.0 * (alpha + 1.0));
                    double w2 = (alpha + 1.0 - sqrt_alpha_p1) / (2.0 * (alpha + 1.0));
                    per_dim_nodes[du] = {z1, z2};
                    per_dim_weights[du] = {w1, w2};
                }
            } else {
                // Normal (Gauss-Hermite) quadrature for unconstrained params (C# 2736-2738).
                per_dim_nodes[du] = {-1.0, 1.0};
                per_dim_weights[du] = {0.5, 0.5};
            }
        }

        // Modified Cholesky (CHOL33) with the standard-Cholesky / diagonal-sqrt fallbacks (C# 2744-2765).
        Matrix v(dimension, dimension);
        try {
            v = cohn_cholesky(s, dimension, scale_idx);
        } catch (...) {
            // C# Debug.WriteLine dropped -> silent guard.
            try {
                CholeskyDecomposition chol(s);
                v = chol.l();
            } catch (...) {
                v = Matrix(dimension, dimension);
                for (int i = 0; i < dimension; ++i) v(i, i) = std::sqrt(std::max(0.0, s(i, i)));
            }
        }

        return build_grid_from_cholesky(mean, v, dimension, per_dim_nodes, per_dim_weights);
    }

    // C# `CohnCholesky` (C# 2796-2824): EMA's CHOL33 modified Cholesky. The shipped C# body returns
    // the STANDARD lower-triangular Cholesky in BOTH branches (it concludes any valid Cholesky
    // reproduces S = L*L^T; the EMA distinction lives in the Gamma nodes, not the factor ordering).
    // The branch structure is mirrored verbatim. Static.
    static corehydro::numerics::math::linalg::Matrix cohn_cholesky(
        const corehydro::numerics::math::linalg::Matrix& s, int dimension, int pivot_idx) {
        using corehydro::numerics::math::linalg::CholeskyDecomposition;
        if (dimension != 3 || pivot_idx != 1) {
            CholeskyDecomposition chol(s);  // non-3D: standard Cholesky
            return chol.l();
        }
        CholeskyDecomposition chol2(s);  // 3D pivot case: also standard Cholesky (see the note)
        return chol2.l();
    }

    // C# `BuildGridFromCholesky` (C# 2836-2882): tensor product of the per-dim standardized nodes
    // mapped through the lower-triangular factor L (theta = mean + L*z), product weights. Static.
    static std::pair<std::vector<std::vector<double>>, std::vector<double>> build_grid_from_cholesky(
        const std::vector<double>& mean, const corehydro::numerics::math::linalg::Matrix& l,
        int dimension, const std::vector<std::vector<double>>& per_dim_nodes,
        const std::vector<std::vector<double>>& per_dim_weights) {
        int total_points = 1;
        for (int d = 0; d < dimension; ++d)
            total_points *= static_cast<int>(per_dim_nodes[static_cast<std::size_t>(d)].size());

        std::vector<std::vector<double>> grid(static_cast<std::size_t>(total_points));
        std::vector<double> weights(static_cast<std::size_t>(total_points));

        std::vector<int> indices(static_cast<std::size_t>(dimension), 0);
        for (int pt = 0; pt < total_points; ++pt) {
            std::size_t ptu = static_cast<std::size_t>(pt);
            std::vector<double> z(static_cast<std::size_t>(dimension));
            double w = 1.0;
            for (int d = 0; d < dimension; ++d) {
                std::size_t du = static_cast<std::size_t>(d);
                std::size_t idx = static_cast<std::size_t>(indices[du]);
                z[du] = per_dim_nodes[du][idx];
                w *= per_dim_weights[du][idx];
            }
            // Apply the Cholesky: theta = mean + L*z (L lower-triangular).
            std::vector<double> theta(static_cast<std::size_t>(dimension));
            for (int i = 0; i < dimension; ++i) {
                double sum = 0.0;
                for (int j = 0; j <= i; ++j) sum += l(i, j) * z[static_cast<std::size_t>(j)];
                theta[static_cast<std::size_t>(i)] = mean[static_cast<std::size_t>(i)] + sum;
            }
            grid[ptu] = std::move(theta);
            weights[ptu] = w;

            // Increment the multi-index (odometer pattern) (C# 2873-2878).
            for (int d = 0; d < dimension; ++d) {
                std::size_t du = static_cast<std::size_t>(d);
                indices[du]++;
                if (indices[du] < static_cast<int>(per_dim_nodes[du].size())) break;
                indices[du] = 0;
            }
        }

        return {std::move(grid), std::move(weights)};
    }

    // C# `ClampForCovariance` (C# 2903-2922): clone; sigma > kSigmaFloor; |gamma| clamped to
    // kSkewClampAbs and floored (sign-preserving) at kSkewFloorAbs. Static.
    static std::vector<double> clamp_for_covariance(const std::vector<double>& parameters) {
        std::vector<double> clamped = parameters;  // clone
        if (clamped.size() >= 2) clamped[1] = std::max(clamped[1], kSigmaFloor);
        if (clamped.size() >= 3) {
            clamped[2] = std::clamp(clamped[2], -kSkewClampAbs, kSkewClampAbs);
            if (std::abs(clamped[2]) < kSkewFloorAbs)
                clamped[2] = std::copysign(kSkewFloorAbs, clamped[2]);
        }
        return clamped;
    }

    // C# `ClampForQuantile` (C# 2935-2948): clone; sigma > kSigmaFloor; gamma left UNCLAMPED
    // (matching EMA's QP3, which has no skew clamp). Static.
    static std::vector<double> clamp_for_quantile(const std::vector<double>& parameters) {
        std::vector<double> clamped = parameters;  // clone
        if (clamped.size() >= 2) clamped[1] = std::max(clamped[1], kSigmaFloor);
        // gamma intentionally unclamped for quantile evaluation.
        return clamped;
    }

    // C# `EvaluateQuantileSafe` (C# 2956-2980): quantile at the given parameters via a PearsonTypeIII
    // clamped for quantile evaluation; NaN/Inf or any throw falls back to
    // log10(Distribution.InverseCDF(p)). DISTINCT from evaluate_log_quantile_safe (A7). The C#
    // `new PearsonTypeIII(); SetParameters(clamped)` validates the length and throws for a non-3
    // vector; our set_parameters does NOT length-check, so an explicit size guard reproduces the
    // C# throw+fallback (P3 has exactly 3 parameters).
    double evaluate_quantile_safe(const std::vector<double>& parameters,
                                  double non_exceedance_probability) const {
        try {
            std::vector<double> clamped = clamp_for_quantile(parameters);
            if (clamped.size() != 3)
                throw std::invalid_argument("PearsonTypeIII requires 3 parameters");
            corehydro::numerics::distributions::PearsonTypeIII dist;
            dist.set_parameters(clamped);
            double q = dist.inverse_cdf(non_exceedance_probability);
            if (std::isnan(q) || std::isinf(q))
                return std::log10(
                    bulletin17c_distribution_->distribution()->inverse_cdf(non_exceedance_probability));
            return q;
        } catch (...) {
            // C# Debug.WriteLine dropped -> silent guard; fall back to the point-estimate quantile.
            return std::log10(
                bulletin17c_distribution_->distribution()->inverse_cdf(non_exceedance_probability));
        }
    }

    // C# `WeightedCovariance` (C# 2990-3017): EMA's COVW -- weighted means then
    // Cov = sum w_i (x_i - xbar)(y_i - ybar) / sum w_i; wSum <= 0 -> 0. Static.
    static double weighted_covariance(const std::vector<double>& x, const std::vector<double>& y,
                                      const std::vector<double>& weights) {
        std::size_t n = x.size();
        double w_sum = 0.0;
        for (std::size_t i = 0; i < n; ++i) w_sum += weights[i];
        if (w_sum <= 0.0) return 0.0;

        double x_bar = 0.0, y_bar = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            x_bar += weights[i] * x[i];
            y_bar += weights[i] * y[i];
        }
        x_bar /= w_sum;
        y_bar /= w_sum;

        double cov = 0.0;
        for (std::size_t i = 0; i < n; ++i) cov += weights[i] * (x[i] - x_bar) * (y[i] - y_bar);
        return cov / w_sum;
    }

    // C# `CohnAdjustedStudentTCI` (C# 3051-3089): the CI_EMA_M3B adjusted Student-t CI. Returns
    // (lower, upper, beta1, nu). nuMin = 5, cMin = 0.5; varQ <= 0 -> (qHat, qHat, 0, nuMin). Static.
    static std::tuple<double, double, double, double> cohn_adjusted_student_t_ci(
        double q_hat, double var_q, double cov_q_se, double var_se, double confidence_level) {
        constexpr double kNuMin = 5.0;
        constexpr double kCMin = 0.5;

        if (var_q <= 0.0) return {q_hat, q_hat, 0.0, kNuMin};

        double beta1 = cov_q_se / var_q;                        // regression coefficient of SE on Q
        double var_se_given_q = var_se - cov_q_se * cov_q_se / var_q;  // conditional variance of SE
        double nu = (var_se_given_q <= 0.0) ? 1000.0 : 0.5 * var_q / var_se_given_q;
        nu = std::max(nu, kNuMin);

        double p_high = (1.0 + confidence_level) / 2.0;
        corehydro::numerics::distributions::StudentT t_dist(nu);
        double t = t_dist.inverse_cdf(p_high);
        double se_q = std::sqrt(var_q);

        // beta1-corrected CI bounds (C# 3085-3086).
        double ci_high = q_hat + se_q * t / std::max(kCMin, 1.0 - beta1 * t);
        double ci_low = q_hat + se_q * (-t) / std::max(kCMin, 1.0 - beta1 * (-t));
        return {ci_low, ci_high, beta1, nu};
    }

    // C# `EnforceMonotonicity` (C# 3126-3133): backward sweep i = nProb-2..0 setting each bound to
    // max(self, next). C#-DOC NOTE: the source carries TWO contradictory xml-doc blocks above this
    // single body; the BODY governs -- ordinates are AEPs in ASCENDING order, quantiles DECREASE
    // with index, so both CI bounds must be non-increasing with index. The stale "descending
    // ordinates" doc block is ignored (see the A9 report). Static.
    static void enforce_monotonicity(std::vector<double>& lower_ci, std::vector<double>& upper_ci,
                                     int n_prob) {
        for (int i = n_prob - 2; i >= 0; --i) {
            std::size_t iu = static_cast<std::size_t>(i);
            lower_ci[iu] = std::max(lower_ci[iu], lower_ci[iu + 1]);
            upper_ci[iu] = std::max(upper_ci[iu], upper_ci[iu + 1]);
        }
    }

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 722-757), synchronous fast path. Reads the
    // stored parameter sets, clones the parent per set, and assembles the UncertaintyAnalysisResults
    // on the NON-exceedance grid (the exceedance<->non-exceedance flip, C# 751), then updates the
    // point-estimate scalars. Safe to call repeatedly without resampling.
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        if (!bayesian_analysis_.is_estimated() || !bayesian_analysis_.results() ||
            bulletin17c_distribution_->distribution() == nullptr) {
            return;
        }

        const auto& results = *bayesian_analysis_.results();

        // Set parent distribution parameters from the stored MAP (the GMM fit) (C# 735).
        bulletin17c_distribution_->set_parameter_values(results.map.values);

        // Clone the parent per stored parameter set (C# 739-746).
        std::size_t b = results.output.size();
        std::vector<std::unique_ptr<UnivariateDistributionBase>> owned;
        owned.reserve(b);
        for (std::size_t idx = 0; idx < b; ++idx) {
            std::unique_ptr<UnivariateDistributionBase> d =
                bulletin17c_distribution_->distribution()->clone();
            d->set_parameters(results.output[idx].values);
            owned.push_back(std::move(d));
        }
        std::vector<const UnivariateDistributionBase*> sampled;
        sampled.reserve(owned.size());
        for (const auto& u : owned) sampled.push_back(u.get());

        // Exceedance -> non-exceedance FLIP (C# 751, `p => 1.0 - p`).
        std::vector<double> probabilities;
        probabilities.reserve(probability_ordinates_.count());
        for (double p : probability_ordinates_) probabilities.push_back(1.0 - p);

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        // C# passes recordParameterSets: false (the default here).
        analysis_results_.emplace(*bulletin17c_distribution_->distribution(), sampled, probabilities,
                                  alpha);

        update_point_estimate_results();
    }

    // C# `UpdatePointEstimateResultsAsync` (C# 2313-2354), synchronous. Sets the model to the
    // selected point estimator, rebuilds the mode curve on the non-exceedance grid, and writes
    // AIC/BIC (at MAP), RMSE (over Exact + Interval data with plotting-position complements), and
    // the effective record length (ERL) from the covariance eigen-decomposition.
    void update_point_estimate_results() {
        if (!bayesian_analysis_.results() || !analysis_results_ || !gmm_) return;

        const auto& results = *bayesian_analysis_.results();

        // Set point estimator parameters (C# 2319-2323).
        const std::vector<double>& parms =
            bayesian_analysis_.point_estimator() == PointEstimateType::PosteriorMean
                ? results.posterior_mean.values
                : results.map.values;
        bulletin17c_distribution_->set_parameter_values(parms);

        // Mode curve on the non-exceedance grid (C# 2326-2328).
        std::size_t count = probability_ordinates_.count();
        analysis_results_->mode_curve.assign(count, 0.0);
        for (std::size_t i = 0; i < count; ++i)
            analysis_results_->mode_curve[i] =
                bulletin17c_distribution_->distribution()->inverse_cdf(1.0 - probability_ordinates_[i]);

        // Goodness-of-fit metrics (C# 2330-2345). nt drops the low outliers (C# 2331-2333).
        const corehydro::models::DataFrame& df = bulletin17c_distribution_->data_frame();
        int nt = static_cast<int>(df.exact_series().count()) - df.number_of_low_outliers() +
                 static_cast<int>(df.interval_series().count());

        // logLH at the MAP through a UnivariateDistribution over the frame + distribution (C# 2335).
        UnivariateDistributionModel univariate_dist(df.clone(),
                                                    bulletin17c_distribution_->distribution()->clone());
        std::vector<double> map_values = results.map.values;  // log_likelihood takes a mutable ref
        double log_lh = univariate_dist.log_likelihood(map_values);
        int k = bulletin17c_distribution_->number_of_parameters();
        analysis_results_->aic = GoodnessOfFit::aic(k, log_lh);
        analysis_results_->bic = GoodnessOfFit::bic(nt, k, log_lh);

        // RMSE over Exact + Interval data (C# 2340-2345).
        std::vector<double> values = df.exact_series().values_to_list();
        {
            std::vector<double> iv = df.interval_series().values_to_list();
            values.insert(values.end(), iv.begin(), iv.end());
        }
        std::vector<double> pp;
        pp.reserve(values.size());
        for (std::size_t i = 0; i < df.exact_series().count(); ++i)
            pp.push_back(df.exact_series()[i].plotting_position_complement());
        for (std::size_t i = 0; i < df.interval_series().count(); ++i)
            pp.push_back(df.interval_series()[i].plotting_position_complement());
        analysis_results_->rmse =
            GoodnessOfFit::rmse(values, pp, *bulletin17c_distribution_->distribution());

        // Effective record length from the covariance eigen-decomposition (C# 2348-2351).
        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;
        corehydro::numerics::math::linalg::Matrix sigma_hat = gmm_->get_covariance(theta_hat);
        corehydro::numerics::math::linalg::EigenValueDecomposition eig(sigma_hat);
        analysis_results_->erl = eig.effective_sample_size();
    }

    // Owned model (deviation). Declared FIRST so it is constructed before gmm_ points into it.
    std::unique_ptr<Bulletin17CDistribution> bulletin17c_distribution_;
    // Zero-parameter plumbing model backing bayesian_analysis_ (see the file header). Declared
    // BEFORE bayesian_analysis_ so it outlives the reference the BayesianAnalysis stores.
    detail::Bulletin17CPlumbingModel plumbing_model_;
    BayesianAnalysis bayesian_analysis_;
    ProbabilityOrdinates probability_ordinates_;
    UncertaintyMethod uncertainty_method_;

    // The GMM estimator (C# nullable `_gmm`), populated by run() (null until then).
    std::optional<GeneralizedMethodOfMoments> gmm_;
    // The frequency-analysis results (C# nullable AnalysisResults) -> optional.
    std::optional<UncertaintyAnalysisResults> analysis_results_;
    // C# nullable `BootstrapResults` (A8): populated by the parametric-bootstrap UQ path.
    std::optional<BootstrapDiagnostics> bootstrap_results_;
    // C# `UncertaintyDiagnosticMessage` (T19); empty string default matches the C# field
    // initializer.
    std::string uncertainty_diagnostic_message_;

    // C# `SetUncertaintyDiagnosticMessage(string)` (T19, C# 421): records the reason the most
    // recent uncertainty quantification was degraded or aborted. RaisePropertyChange /
    // Debug.WriteLine dropped (WPF binding / trace-only). Not currently called by
    // get_parameter_sets_from_parametric_bootstrap (see the file header's T19 note); kept for the
    // other uncertainty paths' existing (unwired) guards and for Task 20.
    void set_uncertainty_diagnostic_message(const std::string& message) {
        uncertainty_diagnostic_message_ = message;
    }

    // Test-only access to the private acceleration_constants() (see the forward-declared struct).
    friend struct Bulletin17CAnalysisTestAccess;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (kept alive until the analysis dies; the interface returns non-owning pointers).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
};

}  // namespace corehydro::analyses
