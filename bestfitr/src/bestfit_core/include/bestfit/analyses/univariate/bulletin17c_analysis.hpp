// ported from: upstream/RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/Bulletin17CAnalysis.cs @ fc28c0c
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
//   initializer (line 228) is `MultivariateNormal`. LinkedMultivariateNormal is DEFERRED to
//   Phase 9 (its ~13 link-builder helpers + InfluenceStatistics are a large severable slice), so
//   its dispatch arm THROWS here. The shipped C++ default is therefore `MultivariateNormal`.
//   This is behavior-preserving for the shipped scope: the C# RunUncertaintyQuantification path
//   (line 666-671) SILENTLY FALLS BACK to plain MultivariateNormal whenever LinkedMVN returns
//   null, so a MultivariateNormal default reproduces the C# net behavior for every case the
//   shipped code can actually compute.
//
// UNCERTAINTY-METHOD DISPATCH (C# switch, lines 657-664):
//   * MultivariateNormal       -> SHIPPED here (get_parameter_sets_from_multivariate_normal).
//   * Bootstrap                -> ships in A8 (dispatch arm throws until then; clearly marked).
//   * LinkedMultivariateNormal -> DEFERRED to Phase 9 (dispatch arm throws "deferred to Phase 9").
//   * BiasCorrectedBootstrap   -> DEFERRED to Phase 9 (dispatch arm throws "deferred to Phase 9").
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
//   * DEFERRED to Phase 9: GetParameterSetsFromLinkedMultivariateNormal + its link-builder helpers
//     (CreatePositiveParameterLink / CreatePearson{Location,Scale}Link / CreateLocationLink /
//     CreateGammaShapeLink / OrientGammaWedsForLink / CleanWeds / SafeStandardError /
//     ComputeInfluenceStatistics / InfluenceStatistics), GetParameterSetsFromPivotBootstrap
//     (BiasCorrected), the Cohn CI machinery + CohnConfidenceIntervalResult DTO (A9), and the
//     ~617-line GMM report generator (incl. ReportAppendBootstrapDiagnostics). AFormulaOverride is
//     a deferred A9 member (not declared here).
//
// `EvaluateLogQuantileSafe` (C# 856-880) IS ported below per the A7 brief, though the shipped MVN
// path does not itself call it (the C# MVN sampler validates via ValidateParameters); it is the
// safe-quantile guard the LinkedMVN / Cohn paths use and lands here for structural fidelity.
#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/bootstrap_diagnostics.hpp"
#include "bestfit/analyses/support/i_univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/estimation/generalized_method_of_moments.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/data/goodness_of_fit.hpp"
#include "bestfit/numerics/data/probability_ordinates.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/chi_squared.hpp"
#include "bestfit/numerics/distributions/multivariate/multivariate_normal.hpp"
#include "bestfit/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"
#include "bestfit/numerics/functions/link_controller.hpp"
#include "bestfit/numerics/math/linalg/eigenvalue_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/matrix_regularization.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"
#include "bestfit/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/utilities/extension_methods.hpp"

namespace bestfit::analyses {

// UncertaintyMethod (C# 33-57): the uncertainty-quantification method. All four members are
// declared so the enum contract is complete and A8's `Bootstrap` is already named. Order matches
// the C# enum (MultivariateNormal = 0). See the header comment for which arms ship / defer.
enum class UncertaintyMethod {
    MultivariateNormal,        // SHIPPED (A7)
    LinkedMultivariateNormal,  // DEFERRED to Phase 9 (dispatch throws)
    Bootstrap,                 // ships in A8 (dispatch throws until then)
    BiasCorrectedBootstrap,    // DEFERRED to Phase 9 (dispatch throws)
};

namespace detail {

// Zero-parameter ModelBase stub backing the plumbing BayesianAnalysis (see the file header:
// C# `new BayesianAnalysis()` has Model = null; this port needs a non-null ModelBase& but never
// samples through it). With zero parameters, BayesianAnalysis::set_up_sampler() early-returns.
class Bulletin17CPlumbingModel : public bestfit::models::ModelBase {
   public:
    double data_log_likelihood(std::vector<double>&) const override { return 0.0; }
    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>&) const override {
        return {};
    }
    std::vector<bestfit::models::DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>&) const override {
        return {};
    }
    void set_default_parameters() override {}
    bestfit::models::ValidationResult validate() const override { return {}; }
};

}  // namespace detail

// Grants the C++-only ctest access to the private acceleration_constants() method (C# private and
// uncalled on the shipped path; see that method's note). Keeps the public API unchanged.
struct Bulletin17CAnalysisTestAccess;

class Bulletin17CAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using Bulletin17CDistribution = bestfit::models::Bulletin17CDistribution;
    using UnivariateDistributionModel = bestfit::models::UnivariateDistributionModel;
    using UnivariateDistributionBase = bestfit::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = bestfit::numerics::distributions::UncertaintyAnalysisResults;
    using MultivariateNormal = bestfit::numerics::distributions::MultivariateNormal;
    using ProbabilityOrdinates = bestfit::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = bestfit::numerics::data::GoodnessOfFit;
    using GeneralizedMethodOfMoments = bestfit::estimation::GeneralizedMethodOfMoments;
    using BayesianAnalysis = bestfit::estimation::BayesianAnalysis;
    using PointEstimateType = bestfit::estimation::PointEstimateType;
    using ParameterSet = bestfit::numerics::math::optimization::ParameterSet;
    using MCMCResults = bestfit::numerics::sampling::mcmc::MCMCResults;
    using LinkController = bestfit::numerics::functions::LinkController;
    using OptimizationStatus = bestfit::numerics::math::optimization::OptimizationStatus;
    using DistributionType = bestfit::numerics::distributions::UnivariateDistributionType;

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

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 488): resets the fit + results to the un-estimated state; the model is
    // preserved. RaisePropertyChange / ElapsedTime dropped; BootstrapResults reset (C# 500).
    void clear_results() {
        set_is_estimated(false);
        bayesian_analysis_.clear_results();
        gmm_.reset();
        analysis_results_.reset();
        bootstrap_results_.reset();
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
    bestfit::models::ValidationResult validate() const override {
        bestfit::models::ValidationResult result;

        bestfit::models::ValidationResult dist_valid = bulletin17c_distribution_->validate();
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

   private:
    // Null-guard helper for the ctor init list (C# `?? throw new ArgumentNullException`).
    static std::unique_ptr<Bulletin17CDistribution> require_non_null(
        std::unique_ptr<Bulletin17CDistribution> model) {
        if (model == nullptr) {
            throw std::invalid_argument("bullet17CDistribution");  // C# ArgumentNullException
        }
        return model;
    }

    // C# `RunUncertaintyQuantificationAsync` (C# 645), synchronous. Sets the parent distribution to
    // thetaHat, dispatches on UncertaintyMethod to build the parameter-set ensemble, and stores it
    // into the BayesianAnalysis results plumbing (C# 703-705). The async/Task.Run/progress/
    // cancellation machinery is dropped.
    void run_uncertainty_quantification() {
        if (!gmm_ || gmm_->status() == OptimizationStatus::Failure) return;

        // Set parent distribution parameters to the GMM point estimate (C# 654).
        bulletin17c_distribution_->set_parameter_values(gmm_->best_parameter_set().values);

        // Dispatch (C# switch, 657-664). See the file header for the ship/defer contract.
        std::optional<std::vector<ParameterSet>> raw_sets;
        switch (uncertainty_method_) {
            case UncertaintyMethod::MultivariateNormal:
                raw_sets = get_parameter_sets_from_multivariate_normal();
                break;
            case UncertaintyMethod::Bootstrap:
                // C# `UncertaintyMethod.Bootstrap => GetParameterSetsFromParametricBootstrap(...)`
                // (C# 661). The parametric bootstrap re-fits the GMM on B resampled frames.
                raw_sets = get_parameter_sets_from_parametric_bootstrap();
                break;
            case UncertaintyMethod::LinkedMultivariateNormal:
                throw std::runtime_error(
                    "Bulletin17CAnalysis: LinkedMultivariateNormal is deferred to Phase 9.");
            case UncertaintyMethod::BiasCorrectedBootstrap:
                throw std::runtime_error(
                    "Bulletin17CAnalysis: BiasCorrectedBootstrap is deferred to Phase 9.");
        }

        // Empty-result guard (C# 676-682): a non-positive-definite covariance leaves the point
        // estimate valid but yields no CI. Silent no-throw guard (C# Debug.WriteLine dropped).
        if (!raw_sets) return;

        // Filter unset entries (C# 687): the MVN sampler substitutes thetaHat on failure, so every
        // slot is populated, but the filter is ported faithfully.
        std::vector<ParameterSet> valid_sets;
        valid_sets.reserve(raw_sets->size());
        for (const auto& ps : *raw_sets)
            if (!ps.values.empty()) valid_sets.push_back(ps);
        if (valid_sets.size() < 2) return;  // C# 688-692

        // Sanitize non-finite Fitness/Values (C# 696) -- MVN sets Fitness = NaN by construction.
        sanitize_parameter_sets(valid_sets);

        // Sanitize the MAP (best) parameter set's Fitness (C# 699-701).
        ParameterSet best = gmm_->best_parameter_set();
        if (!std::isfinite(best.fitness)) best = ParameterSet(best.values, 0.0);

        // Store into the BayesianAnalysis results plumbing (C# 703-705).
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
        bestfit::numerics::math::linalg::Matrix sigma_hat = gmm_->get_covariance(theta_hat);

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
                bestfit::numerics::sampling::MersenneTwister prng(
                    static_cast<std::uint32_t>(seed + b + idx));
                for (int retry = 0; retry < 10 && !accepted; ++retry) {
                    std::vector<double> u =
                        bestfit::numerics::utilities::next_doubles(prng, 1, p)[0];
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

    // C# `GetParameterSetsFromParametricBootstrap` (C# 1855-1967): the parametric-bootstrap UQ
    // path. Draws B resampled data frames from the fitted parent distribution, re-fits the GMM on
    // each, and keeps the accepted parameter vectors. A degenerate fit (GMM failure, or a
    // Mahalanobis distance from thetaHat beyond the chi2(p, 0.9999) threshold) is retried up to
    // maxRetries times before falling back to the parent vector. The up-front seed cascade
    // (masterPRNG(PRNGSeed) -> B integer seeds -> a per-replicate MersenneTwister) is IDENTICAL to
    // the ported Bootstrap and is load-bearing for MersenneTwister parity with the A11 emitter --
    // do NOT reorder. The C# Parallel.For -> a serial loop; per-index writes are independent, so
    // the serial counts/results match. Progress/cancellation/stopwatch lines dropped.
    std::optional<std::vector<ParameterSet>> get_parameter_sets_from_parametric_bootstrap() {
        const int max_retries = 5;                                   // C# 1857
        int b = bayesian_analysis_.output_length();                  // C# 1858
        int p = bulletin17c_distribution_->number_of_parameters();   // C# 1859
        std::vector<ParameterSet> results(static_cast<std::size_t>(b));  // C# 1860

        // Up-front seed cascade (C# 1861-1862): master PRNG seeds B per-replicate integers.
        bestfit::numerics::sampling::MersenneTwister master_prng(
            static_cast<std::uint32_t>(bayesian_analysis_.prng_seed()));
        std::vector<int> seeds = bestfit::numerics::utilities::next_integers(master_prng, b);

        // Parent distribution cloned + set to thetaHat as the data-generating model (C# 1863-1865).
        const std::vector<double> theta_hat = gmm_->best_parameter_set().values;
        std::unique_ptr<UnivariateDistributionBase> parent_distribution =
            bulletin17c_distribution_->distribution()->clone();
        parent_distribution->set_parameters(theta_hat);

        // Sandwich covariance + inverse for the Mahalanobis rejection (C# 1867-1876). On a singular
        // inverse the C# regularizes; port as a try/catch falling back to the regularized inverse
        // (the C# Debug.WriteLine is a dropped silent guard).
        bestfit::numerics::math::linalg::Matrix sigma_hat = gmm_->get_covariance(theta_hat);
        bestfit::numerics::math::linalg::Matrix sigma_inv(p);
        try {
            sigma_inv = sigma_hat.inverse();
        } catch (...) {
            sigma_inv = bestfit::numerics::math::linalg::MatrixRegularization::
                            make_symmetric_positive_definite(sigma_hat)
                                .inverse();
        }
        double mahal_threshold =
            bestfit::numerics::distributions::ChiSquared(p).inverse_cdf(0.9999);  // C# 1876

        // Diagnostics (C# 1879): TotalReplicates = B.
        BootstrapDiagnostics diag;
        diag.set_total_replicates(b);

        // Serial replicate loop (C# Parallel.For 1892-1955).
        for (int idx = 0; idx < b; ++idx) {
            bestfit::numerics::sampling::MersenneTwister prng(
                static_cast<std::uint32_t>(seeds[static_cast<std::size_t>(idx)]));  // C# 1896
            std::optional<std::vector<double>> accepted_params;

            for (int attempt = 0; attempt < max_retries && !accepted_params; ++attempt) {  // C# 1899
                try {
                    // Resample a bootstrap frame from the parent (clone used only for sampling),
                    // clone the B17C model onto it, and set a randomized penalty (C# 1905-1909).
                    std::unique_ptr<UnivariateDistributionBase> sampling_dist =
                        parent_distribution->clone();
                    bestfit::models::DataFrame boot_data_frame =
                        bulletin17c_distribution_->data_frame().BootstrapDataFrame(*sampling_dist,
                                                                                   prng);
                    std::unique_ptr<bestfit::models::IGMMModel> boot_model =
                        bulletin17c_distribution_->clone();
                    auto* boot_b17c = static_cast<Bulletin17CDistribution*>(boot_model.get());
                    boot_b17c->set_data_frame(std::move(boot_data_frame));
                    boot_b17c->set_random_penalty_function(theta_hat, &prng);

                    GeneralizedMethodOfMoments boot_gmm(*boot_b17c);  // C# 1910
                    boot_gmm.estimate();                              // C# 1911
                    if (boot_gmm.status() != OptimizationStatus::Success)
                        throw std::runtime_error("bootstrap GMM solver failed");  // C# 1912-1913

                    // Mahalanobis distance of the bootstrap fit from thetaHat (C# 1916-1924).
                    const std::vector<double>& boot_params = boot_gmm.best_parameter_set().values;
                    double mahal_dist = 0.0;
                    for (int j = 0; j < p; ++j) {
                        double tmp = 0.0;
                        for (int k = 0; k < p; ++k)
                            tmp += sigma_inv(j, k) *
                                   (boot_params[static_cast<std::size_t>(k)] -
                                    theta_hat[static_cast<std::size_t>(k)]);
                        mahal_dist += (boot_params[static_cast<std::size_t>(j)] -
                                       theta_hat[static_cast<std::size_t>(j)]) *
                                      tmp;
                    }
                    if (mahal_dist > mahal_threshold) {  // C# 1925-1929
                        diag.increment_mahalanobis_rejection();
                        throw std::runtime_error("bootstrap replicate rejected: Mahalanobis");
                    }

                    accepted_params = boot_params;                                   // C# 1931
                    diag.add_function_evaluations(boot_gmm.total_function_evaluations());  // C# 1932
                } catch (...) {
                    // C# Debug.WriteLine dropped (silent guard); count a retry when more remain.
                    if (attempt < max_retries - 1) diag.add_retries(1);  // C# 1937
                }
            }

            // Fall back to the parent vector after all retries fail (C# 1944-1948).
            if (!accepted_params) {
                diag.increment_failed();
                accepted_params = theta_hat;
            }

            results[static_cast<std::size_t>(idx)] = ParameterSet(
                std::move(*accepted_params), std::numeric_limits<double>::quiet_NaN());  // C# 1950
        }

        // Persist the diagnostics on the analysis and return the sets (C# 1959-1961).
        bootstrap_results_ = std::move(diag);
        return results;
    }

    // C# `AccelerationConstants(double[] thetaHats)` (C# 2408-2464): the BCa jackknife acceleration
    // constant a_i = I3_i / (6 * I2_i^1.5), where I2/I3 accumulate the squared/cubed leave-one-out
    // deltas of each parameter. Ported per the A8 brief for structural fidelity; the C# NEVER calls
    // it on any shipped path (it is BCa/pivot-bootstrap building-block machinery deferred to Phase
    // 9), so it lands here UNCALLED -- mirroring the A7 evaluate_log_quantile_safe precedent. The
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
            bestfit::models::DataFrame jack_data =
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
    // but the LinkedMVN / Cohn paths (Phase 9 / A9) use it. Evaluates the log-space quantile for
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
        const bestfit::models::DataFrame& df = bulletin17c_distribution_->data_frame();
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
        bestfit::numerics::math::linalg::Matrix sigma_hat = gmm_->get_covariance(theta_hat);
        bestfit::numerics::math::linalg::EigenValueDecomposition eig(sigma_hat);
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

    // Test-only access to the private acceleration_constants() (see the forward-declared struct).
    friend struct Bulletin17CAnalysisTestAccess;

    // Owns the distributions handed out by get_distribution / get_point_estimate_distribution
    // (kept alive until the analysis dies; the interface returns non-owning pointers).
    std::vector<std::unique_ptr<UnivariateDistributionBase>> distribution_cache_;
};

}  // namespace bestfit::analyses
