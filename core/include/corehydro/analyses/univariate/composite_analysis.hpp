// ported from: RMC-BestFit/src/RMC.BestFit/Analyses/Univariate/CompositeAnalysis.cs @ fc28c0c
//
// Composite univariate analysis: combines several already-fitted child IUnivariateAnalysis into a
// single composite frequency curve via CompetingRisks, Mixture, or Model-Averaging. The composite
// has NO MCMC of its own -- it aggregates over the children's posterior samples -- so given fixed
// child posteriors the whole pipeline is deterministic.
//
// LOAD-BEARING compute sequence ported; WPF/event/gate/XML plumbing DROPPED per the A5 precedent.
// Specifically SKIPPED (each is GUI/threading/serialization with no numerical content):
//   * both XML constructors (C# 148-222) + `ToXElement` (1094) -- XElement (de)serialization.
//   * every `*_PropertyChanged` / `*_CollectionChanged` handler (C# 396-498) and all
//     `RaisePropertyChange` calls -- INotifyPropertyChanged / INotifyCollectionChanged binding
//     cascades. The reprocess/clear decisions they encode are exercised by driving the setters +
//     reprocess methods directly, exactly as the C# tests do.
//   * `_reprocessGate.WaitAsync/Release`, `CancellationTokenSource` + cancel (C# 683-691),
//     `SafeProgressReporter`/`ReportProgress`/`IndicateTaskStart-Ended`, and the AnalysisStarting/
//     Completed events + On* raisers -- run-lifecycle plumbing.
//   * `_isEstimatingWeights` (C# 249): it only gates the INPC ClearResults cascade in the Weight
//     setter; with INPC dropped it is unnecessary and is not ported.
//   * `UpdatePointEstimateResultsAsync` (C# 530) / `RestoreAnalysisResults` (C# 555) /
//     `ClearFrequencyAnalysisResults` (C# 515): the first two are INPC/point-estimator-refresh and
//     deserialization-restore helpers; the mode curve they rebuild is already supplied by the
//     ported UncertaintyAnalysisResults compute-ctor's process_mode_curve, so nothing re-derives it.
//
// The C# async `RunAsync` ports to a synchronous `run()`; `CreateFrequencyAnalysisResultsAsync`
// ports to a synchronous `create_frequency_analysis_results()` (every `Parallel.For` becomes a
// serial loop -- the loop bodies are independent writes, so the result is numerically identical).
//
// DEVIATIONS (documented):
//   1. NO BootstrapAnalysis. The C# tail (C# 892-893) builds the results via
//      `new BootstrapAnalysis(mode, ...).Estimate(probs, alpha, results, false)`; the C# certifies
//      that ctor == that Estimate() path to 1e-8, and the ported UncertaintyAnalysisResults
//      compute-ctor realizes exactly that pre-computed-distributions path. So `run()` calls the
//      compute-ctor DIRECTLY. BootstrapAnalysis is a separate, orchestrator-unused deliverable and
//      is NOT a dependency here.
//   2. OWNERSHIP. Child analyses are NON-owning (the caller keeps them alive; C# GC references).
//      The composite OWNS a placeholder UnivariateDistributionModel purely so its settings-holder
//      BayesianAnalysis (used only for PointEstimator + CredibleIntervalWidth) can bind a model
//      reference -- the ported BayesianAnalysis requires a live model where the C# allowed a null
//      Model. The composite also OWNS the aggregated `mode` distribution (the UAR parent, held by
//      non-owning pointer) and the sampled distributions for the lifetime of the results.
//      BayesianAnalysis holds a ModelBase& into the placeholder model, so the composite is
//      non-copyable / non-movable.
//   3. RESULT NULLABILITY. C# `UncertaintyAnalysisResults?` -> std::optional; the interface
//      accessor returns a `const ...*` (null <=> empty optional).
//   4. Validate() returns corehydro::models::ValidationResult (the (bool, List<string>) tuple),
//      matching the AnalysisBase contract; WeightedUnivariateAnalysis::validate()'s single-string
//      message is appended verbatim.
//   5. RUN ERROR REPORTING. Events dropped, so `run()` lets exceptions propagate instead of routing
//      through AnalysisCompleted.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/analyses/support/i_univariate_analysis.hpp"
#include "corehydro/analyses/support/weighted_univariate_analysis.hpp"
#include "corehydro/analyses/univariate/bulletin17c_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/data/probability_ordinates.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/distributions/uncertainty_analysis/uncertainty_analysis_results.hpp"

namespace corehydro::analyses {

// Enumeration of composite distribution types (C# CompositeType, exact names/order).
enum class CompositeType { CompetingRisks, Mixture, ModelAverage };

// Enumeration of model averaging methods (C# AverageMethod, exact names/order).
enum class AverageMethod { AIC, BIC, DIC, WAIC, LOOIC, Equal, RMSE };

class CompositeAnalysis : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using UnivariateDistributionBase = corehydro::numerics::distributions::UnivariateDistributionBase;
    using UncertaintyAnalysisResults = corehydro::numerics::distributions::UncertaintyAnalysisResults;
    using CompetingRisks = corehydro::numerics::distributions::CompetingRisks;
    using Mixture = corehydro::numerics::distributions::Mixture;
    using ProbabilityOrdinates = corehydro::numerics::data::ProbabilityOrdinates;
    using GoodnessOfFit = corehydro::numerics::data::GoodnessOfFit;
    using DependencyType = corehydro::numerics::data::probability::DependencyType;
    using PointEstimateType = corehydro::estimation::PointEstimateType;
    using Transform = corehydro::numerics::data::Transform;

    // C# parameterless ctor (C# 119): empty children, default ordinates, a settings-holder
    // BayesianAnalysis over the placeholder model (deviation 2).
    CompositeAnalysis()
        : placeholder_model_(std::make_unique<corehydro::models::UnivariateDistributionModel>()),
          bayesian_analysis_(*placeholder_model_) {}

    // C# ctor from a collection of weighted analyses (C# 131).
    explicit CompositeAnalysis(std::vector<WeightedUnivariateAnalysis> analyses)
        : placeholder_model_(std::make_unique<corehydro::models::UnivariateDistributionModel>()),
          bayesian_analysis_(*placeholder_model_),
          analyses_(std::move(analyses)) {}

    ~CompositeAnalysis() override = default;

    // Non-copyable / non-movable (deviation 2).
    CompositeAnalysis(const CompositeAnalysis&) = delete;
    CompositeAnalysis& operator=(const CompositeAnalysis&) = delete;
    CompositeAnalysis(CompositeAnalysis&&) = delete;
    CompositeAnalysis& operator=(CompositeAnalysis&&) = delete;

    // --- Members (C# properties) -----------------------------------------------------------

    // C# `Analyses` (C# 254): the weighted child analyses making up the composite.
    std::vector<WeightedUnivariateAnalysis>& analyses() { return analyses_; }
    const std::vector<WeightedUnivariateAnalysis>& analyses() const { return analyses_; }

    // C# `CompositeDistributionType` (C# 275): setter re-estimates ModelAverage weights and clears.
    CompositeType composite_distribution_type() const { return composite_distribution_type_; }
    void set_composite_distribution_type(CompositeType value) {
        if (composite_distribution_type_ != value) {
            composite_distribution_type_ = value;
            if (composite_distribution_type_ == CompositeType::ModelAverage) estimate_model_weights();
            clear_results();
        }
    }

    // C# `ModelAverageMethod` (C# 295): setter re-estimates weights (when ModelAverage) and clears.
    AverageMethod model_average_method() const { return model_average_method_; }
    void set_model_average_method(AverageMethod value) {
        if (model_average_method_ != value) {
            model_average_method_ = value;
            if (composite_distribution_type_ == CompositeType::ModelAverage) estimate_model_weights();
            clear_results();
        }
    }

    // C# `Dependency` (C# 318): the dependency between competing distributions.
    DependencyType dependency() const { return dependency_; }
    void set_dependency(DependencyType value) {
        if (dependency_ != value) {
            dependency_ = value;
            clear_results();
        }
    }

    // C# `IsMaximum` (C# 336): whether the distributions compete to be the maximum or minimum.
    bool is_maximum() const { return is_maximum_; }
    void set_is_maximum(bool value) {
        if (is_maximum_ != value) {
            is_maximum_ = value;
            clear_results();
        }
    }

    // C# `ProbabilityOrdinates` (C# 351). IProbabilityOrdinates override.
    ProbabilityOrdinates& probability_ordinates() override { return probability_ordinates_; }
    const ProbabilityOrdinates& probability_ordinates() const { return probability_ordinates_; }

    // C# `BayesianAnalysis` (C# 370): the settings holder (PointEstimator + CredibleIntervalWidth).
    // IBayesianAnalysis override.
    corehydro::estimation::BayesianAnalysis& bayesian_analysis() override { return bayesian_analysis_; }
    const corehydro::estimation::BayesianAnalysis& bayesian_analysis() const {
        return bayesian_analysis_;
    }

    // C# `AnalysisResults` (C# 390): the composite frequency results (null until estimated).
    // IBayesianAnalysis override.
    const UncertaintyAnalysisResults* analysis_results() const override {
        return analysis_results_ ? &*analysis_results_ : nullptr;
    }

    // --- Lifecycle -------------------------------------------------------------------------

    // C# `ClearResults` (C# 503).
    void clear_results() {
        analysis_results_.reset();
        composite_mode_.reset();
        composite_samples_.clear();
        set_is_estimated(false);
    }

    // C# `EstimateModelWeights` (C# 566): only runs for ModelAverage. Equal -> 1/N; else gather the
    // GoF scalar over ONLY the successfully-fit children (all others default to weight 0), feed
    // RMSE -> rmse_weights / everything else -> aic_weights, then normalize to sum 1.
    void estimate_model_weights() {
        if (composite_distribution_type_ != CompositeType::ModelAverage || analyses_.empty()) return;

        double sum = 0.0;
        if (model_average_method_ == AverageMethod::Equal) {
            double w = 1.0 / static_cast<double>(analyses_.size());
            for (auto& wua : analyses_) {
                wua.set_weight(w);
                sum += w;
            }
        } else {
            // Only successfully-fit children contribute; passing default-zero entries for unfit
            // children to aic_weights would assign them the BEST weight (0 is the min) and
            // contaminate the composite (C# 591-605).
            std::vector<std::size_t> valid_indices;
            for (std::size_t i = 0; i < analyses_.size(); ++i) {
                IUnivariateAnalysis* ua = analyses_[i].univariate_analysis();
                if (ua != nullptr && ua->analysis_results() != nullptr && ua->is_estimated())
                    valid_indices.push_back(i);
            }

            // Default every weight to 0; only valid sub-analyses contribute.
            for (auto& wua : analyses_) wua.set_weight(0.0);

            if (valid_indices.empty()) return;  // sum stays 0; the normalization branch is a no-op

            std::vector<double> gof_valid(valid_indices.size());
            for (std::size_t j = 0; j < valid_indices.size(); ++j) {
                IUnivariateAnalysis* ua = analyses_[valid_indices[j]].univariate_analysis();
                double g;
                switch (model_average_method_) {
                    case AverageMethod::AIC:
                        g = ua->analysis_results()->aic;
                        break;
                    case AverageMethod::BIC:
                        g = ua->analysis_results()->bic;
                        break;
                    case AverageMethod::DIC:
                        g = ua->bayesian_analysis().dic();
                        break;
                    case AverageMethod::WAIC:
                        g = ua->bayesian_analysis().waic();
                        break;
                    case AverageMethod::LOOIC:
                        g = ua->bayesian_analysis().looic();
                        break;
                    case AverageMethod::RMSE:
                        g = ua->analysis_results()->rmse;
                        break;
                    default:
                        g = 0.0;
                        break;
                }
                gof_valid[j] = g;
            }

            std::vector<double> valid_weights = model_average_method_ == AverageMethod::RMSE
                                                    ? GoodnessOfFit::rmse_weights(gof_valid)
                                                    : GoodnessOfFit::aic_weights(gof_valid);

            for (std::size_t j = 0; j < valid_indices.size(); ++j) {
                analyses_[valid_indices[j]].set_weight(valid_weights[j]);
                sum += valid_weights[j];
            }
        }

        // Normalize proportionally so the weights sum to 1 (C# 638-646).
        if (!analyses_.empty() && std::fabs(sum - 1.0) > 1e-10 && sum > 0.0) {
            for (auto& wua : analyses_) wua.set_weight(wua.weight() / sum);
        }
    }

    // C# `RunAsync` (C# 653), synchronous. Validate guard -> require every child fitted -> clear ->
    // estimate weights -> build the composite frequency results (deviation 5: exceptions propagate).
    void run() override {
        corehydro::models::ValidationResult validation = validate();
        if (!validation.is_valid) {
            std::string joined;
            for (std::size_t i = 0; i < validation.validation_messages.size(); ++i) {
                if (i > 0) joined += "; ";
                joined += validation.validation_messages[i];
            }
            throw std::runtime_error(joined);
        }

        // Composite is a weighted aggregation over already-fitted children; refuse to run unless
        // every child is successfully estimated (C# 664-673).
        for (std::size_t i = 0; i < analyses_.size(); ++i) {
            IUnivariateAnalysis* ua = analyses_[i].univariate_analysis();
            if (ua == nullptr || !ua->is_estimated() || ua->analysis_results() == nullptr) {
                throw std::runtime_error(
                    "CompositeAnalysis cannot run: sub-analysis #" + std::to_string(i) +
                    " has not been successfully fit. Run all sub-analyses first.");
            }
        }

        clear_results();
        estimate_model_weights();
        create_frequency_analysis_results();
        set_is_estimated(true);
    }

    // C# `CreateFrequencyAnalysisResultsAsync` (C# 740), synchronous. Builds the parent `mode` and
    // the per-realisation array of composite distributions, then the UncertaintyAnalysisResults via
    // the compute-ctor DIRECTLY (deviation 1).
    void create_frequency_analysis_results() {
        analysis_results_.reset();
        composite_mode_.reset();
        composite_samples_.clear();
        if (analyses_.empty()) return;

        // Minimum number of realisations across children (C# 752).
        int realz = std::numeric_limits<int>::max();
        for (const auto& wua : analyses_) {
            IUnivariateAnalysis* ua = wua.univariate_analysis();
            int ol = ua != nullptr ? ua->bayesian_analysis().output_length() : 0;
            realz = std::min(realz, ol);
        }
        if (realz <= 0) return;

        composite_samples_.resize(static_cast<std::size_t>(realz));

        if (composite_distribution_type_ == CompositeType::CompetingRisks) {
            // Mode (point estimate) over the child point-estimate distributions (C# 773-787).
            std::vector<UnivariateDistributionBase*> mode_dists;
            mode_dists.reserve(analyses_.size());
            for (auto& wua : analyses_) {
                UnivariateDistributionBase* d = get_child_point_estimate_distribution(
                    wua.univariate_analysis(), bayesian_analysis_.point_estimator());
                if (d == nullptr) return;
                mode_dists.push_back(d);
            }
            composite_mode_ = make_competing_risks(mode_dists);

            // Uncertainty samples (C# 791-824; Parallel.For -> serial).
            for (int idx = 0; idx < realz; ++idx) {
                std::vector<UnivariateDistributionBase*> u_dists;
                u_dists.reserve(analyses_.size());
                for (auto& wua : analyses_)
                    u_dists.push_back(wua.univariate_analysis()->get_distribution(idx));
                composite_samples_[static_cast<std::size_t>(idx)] = make_competing_risks(u_dists);
            }
        } else {  // Mixture or ModelAverage
            // Mode (point estimate) as a weighted Mixture over the child point estimates (C# 828-849).
            double sum = 0.0;
            std::vector<double> weights;
            std::vector<UnivariateDistributionBase*> mode_dists;
            weights.reserve(analyses_.size());
            mode_dists.reserve(analyses_.size());
            for (auto& wua : analyses_) {
                sum += wua.weight();
                weights.push_back(wua.weight());
                UnivariateDistributionBase* d = get_child_point_estimate_distribution(
                    wua.univariate_analysis(), bayesian_analysis_.point_estimator());
                if (d == nullptr) return;
                mode_dists.push_back(d);
            }
            composite_mode_ = make_mixture(weights, mode_dists, sum);

            // Uncertainty samples (C# 853-878; Parallel.For -> serial).
            for (int idx = 0; idx < realz; ++idx) {
                std::vector<UnivariateDistributionBase*> u_dists;
                u_dists.reserve(analyses_.size());
                for (auto& wua : analyses_)
                    u_dists.push_back(wua.univariate_analysis()->get_distribution(idx));
                composite_samples_[static_cast<std::size_t>(idx)] =
                    make_mixture(weights, u_dists, sum);
            }
        }

        // Non-exceedance flip (C# 886-889).
        std::vector<double> probs;
        probs.reserve(probability_ordinates_.count());
        for (std::size_t i = 0; i < probability_ordinates_.count(); ++i)
            probs.push_back(1.0 - probability_ordinates_[i]);

        // Compute-ctor directly (deviation 1; replaces the C# BootstrapAnalysis.Estimate call).
        std::vector<const UnivariateDistributionBase*> sampled;
        sampled.reserve(composite_samples_.size());
        for (const auto& s : composite_samples_) sampled.push_back(s.get());

        double alpha = 1.0 - bayesian_analysis_.credible_interval_width();
        analysis_results_.emplace(*composite_mode_, sampled, probs, alpha, 0.001, 1.0 - 1e-9,
                                  /*recordParameterSets=*/false);
    }

    // C# `GetDistribution(int)` (C# 902): composite distributions have no single component.
    UnivariateDistributionBase* get_distribution(int /*index*/) override { return nullptr; }

    // C# `GetPointEstimateDistribution()` (C# 910).
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_analysis_.point_estimator());
    }

    // C# `GetPointEstimateDistribution(PointEstimateType)` (C# 913): the composite point-estimate
    // distribution (a CompetingRisks or Mixture over the child point estimates). Owned in the cache.
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        if (analyses_.empty()) return nullptr;

        if (composite_distribution_type_ == CompositeType::CompetingRisks) {
            std::vector<UnivariateDistributionBase*> mode_dists;
            mode_dists.reserve(analyses_.size());
            for (auto& wua : analyses_) {
                UnivariateDistributionBase* d = get_child_point_estimate_distribution(
                    wua.univariate_analysis(), point_estimator);
                if (d == nullptr) return nullptr;
                mode_dists.push_back(d);
            }
            point_estimate_cache_.push_back(make_competing_risks(mode_dists));
            return point_estimate_cache_.back().get();
        }

        double sum = 0.0;
        std::vector<double> weights;
        std::vector<UnivariateDistributionBase*> mode_dists;
        weights.reserve(analyses_.size());
        mode_dists.reserve(analyses_.size());
        for (auto& wua : analyses_) {
            UnivariateDistributionBase* d =
                get_child_point_estimate_distribution(wua.univariate_analysis(), point_estimator);
            if (d == nullptr) return nullptr;
            sum += wua.weight();
            weights.push_back(wua.weight());
            mode_dists.push_back(d);
        }
        point_estimate_cache_.push_back(make_mixture(weights, mode_dists, sum));
        return point_estimate_cache_.back().get();
    }

    // C# `Validate` (C# 993): children present + valid, per-type weight checks, ModelAverage/B17C
    // incompatibility, and the ascending [0,1] probability-ordinate checks.
    corehydro::models::ValidationResult validate() const override {
        corehydro::models::ValidationResult result;

        if (analyses_.empty()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: No univariate analyses defined for the composite analysis.");
        } else {
            double sum = 0.0;
            for (const auto& wua : analyses_) {
                // Belt-and-suspenders guard against composite-of-composite (C# 1013-1019).
                if (is_composite_analysis(wua.univariate_analysis())) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: a child analysis is itself a CompositeAnalysis; nesting composites "
                        "is not supported.");
                }

                auto [ok, msg] = wua.validate();
                if (!ok) {
                    result.is_valid = false;
                    result.validation_messages.push_back(msg);
                }

                if (composite_distribution_type_ == CompositeType::Mixture &&
                    (wua.weight() <= 0.0 || wua.weight() >= 1.0)) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Weights for each univariate analysis must be between 0 and 1.");
                }
                sum += wua.weight();
            }

            if (composite_distribution_type_ == CompositeType::Mixture &&
                sum - 1.0 > corehydro::numerics::kDoubleMachineEpsilon * 2.0) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: The sum of the weights is greater than 1.");
            }

            // B17C is fit by GMM, not MCMC, so it has no DIC/WAIC/LOO-CV; reject ModelAverage
            // weighted by any of those when a child is a Bulletin17CAnalysis (C# 1048-1060).
            if (composite_distribution_type_ == CompositeType::ModelAverage &&
                (model_average_method_ == AverageMethod::DIC ||
                 model_average_method_ == AverageMethod::WAIC ||
                 model_average_method_ == AverageMethod::LOOIC)) {
                bool any_b17c = false;
                for (const auto& wua : analyses_) {
                    if (dynamic_cast<Bulletin17CAnalysis*>(wua.univariate_analysis()) != nullptr) {
                        any_b17c = true;
                        break;
                    }
                }
                if (any_b17c) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Model Averaging with the selected criterion is not supported when "
                        "any child is a Bulletin17CAnalysis. B17C uses Generalized Method of Moments "
                        "rather than an MCMC chain, so DIC, WAIC, and LOO-CV are not defined for it. "
                        "Choose AIC, BIC, RMSE, or Equal weighting instead.");
                }
            }
        }

        if (probability_ordinates_.count() == 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: At least one probability ordinate is required.");
        } else {
            for (std::size_t i = 0; i < probability_ordinates_.count(); ++i) {
                if (probability_ordinates_[i] < 0.0 || probability_ordinates_[i] > 1.0) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Probability ordinates must be between 0 and 1.");
                    break;
                }
                if (i > 0 && probability_ordinates_[i] <= probability_ordinates_[i - 1]) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Probability ordinates must be in ascending order.");
                    break;
                }
            }
        }

        return result;
    }

   private:
    // C# `GetChildPointEstimateDistribution` (C# 981): the child point-estimate distribution for the
    // estimator, or null when the child is unavailable / not Bayesian-estimated.
    static UnivariateDistributionBase* get_child_point_estimate_distribution(
        IUnivariateAnalysis* analysis, PointEstimateType point_estimator) {
        if (analysis == nullptr || !analysis->bayesian_analysis().is_estimated() ||
            !analysis->bayesian_analysis().results()) {
            return nullptr;
        }
        return analysis->get_point_estimate_distribution(point_estimator);
    }

    // Builds a CompetingRisks over cloned component pointers with the composite's mode settings and
    // an empirical CDF (C# 780-787 / 812-819).
    std::unique_ptr<UnivariateDistributionBase> make_competing_risks(
        const std::vector<UnivariateDistributionBase*>& dists) const {
        auto cr = std::make_unique<CompetingRisks>(dists);
        cr->minimum_of_random_variables = !is_maximum_;
        cr->dependency = dependency_;
        cr->x_transform = Transform::Logarithmic;
        cr->probability_transform = Transform::NormalZ;
        cr->create_empirical_cdf();
        return cr;
    }

    // Builds a Mixture over cloned component pointers with zero-inflation when the weights sum to
    // less than 1, and an empirical CDF (C# 839-849 / 863-873).
    std::unique_ptr<UnivariateDistributionBase> make_mixture(
        const std::vector<double>& weights, const std::vector<UnivariateDistributionBase*>& dists,
        double sum) const {
        auto mix = std::make_unique<Mixture>(weights, dists);
        mix->x_transform = Transform::Logarithmic;
        mix->probability_transform = Transform::NormalZ;
        if (sum < 1.0) {
            mix->set_is_zero_inflated(true);
            mix->set_zero_weight(1.0 - sum);
        }
        mix->create_empirical_cdf();
        return mix;
    }

    // Owned placeholder model (deviation 2). Declared BEFORE bayesian_analysis_ so it is constructed
    // first (BayesianAnalysis stores a reference into it).
    std::unique_ptr<corehydro::models::UnivariateDistributionModel> placeholder_model_;
    corehydro::estimation::BayesianAnalysis bayesian_analysis_;

    std::vector<WeightedUnivariateAnalysis> analyses_;
    CompositeType composite_distribution_type_ = CompositeType::CompetingRisks;
    AverageMethod model_average_method_ = AverageMethod::DIC;
    DependencyType dependency_ = DependencyType::Independent;
    bool is_maximum_ = true;
    ProbabilityOrdinates probability_ordinates_;

    std::optional<UncertaintyAnalysisResults> analysis_results_;

    // Owns the aggregated parent mode (the UAR parent, held by non-owning pointer) and the sampled
    // composite distributions for the lifetime of the results; plus the point-estimate distributions
    // handed out by get_point_estimate_distribution (deviation 2).
    std::unique_ptr<UnivariateDistributionBase> composite_mode_;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> composite_samples_;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> point_estimate_cache_;
};

// Defines the WeightedUnivariateAnalysis composite-child reject predicate now that CompositeAnalysis
// is complete (see weighted_univariate_analysis.hpp header).
inline bool is_composite_analysis(const IUnivariateAnalysis* analysis) {
    return dynamic_cast<const CompositeAnalysis*>(analysis) != nullptr;
}

}  // namespace corehydro::analyses
