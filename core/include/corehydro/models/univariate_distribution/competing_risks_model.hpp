// ported from: src/RMC.BestFit/Models/UnivariateDistribution/CompetingRisksModel.cs @ fc28c0c
//
// Competing risks model for univariate distributions: a set of 1-3 competing univariate
// parent distributions combined through a competing risks structure (for example, multiple
// processes that can generate an annual maximum), wrapped around the ported Numerics
// `CompetingRisks` (numerics/distributions/competing_risks.hpp). Supports censored, interval,
// threshold, and uncertain data, as well as Bayesian priors on parameters and a single
// quantile of the parent competing risks distribution. Derives from the M8
// `UnivariateDistributionModelBase` (DataFrame owner, Jeffreys toggle, IQuantilePriors state)
// and implements `ISimulatable<std::vector<double>>` (C# ISimulatable<double[]>). NOTE: the
// C# class does NOT implement IUnivariateModel, so neither does this port (unlike the M9/M10
// siblings). Sibling of the M10 MixtureModel; mirrors the C# method-for-method.
//
// Nullability mapping:
//   - C# `CompetingRisks?` is genuinely nullable ("CompetingRisks = null" is a tested
//     state), so the wrapped distribution is a `std::unique_ptr<CompetingRisks>` with
//     nullptr == C# null; `competing_risks()` returns the raw pointer (the M10 convention).
//   - C# `DataFrame = null` maps to the base's never-set optional (see the base header).
//
// Unlike the Mixture sibling, CompetingRisks.SetParameters performs NO weight normalization
// (it distributes the flat array across the components in order), so the C# `ref`-mutation
// asymmetry documented in mixture_model.hpp does not arise here: every likelihood method
// evaluates at the caller's parameters unchanged.
//
// Clone(): C# aliases the DataFrame into `new CompetingRisksModel(DataFrame, CompetingRisks!)`;
// the value-typed move-only C++ frame is DEEP-COPIED instead (M9/M10 precedent), and a model
// with no frame or a null distribution cannot be cloned (std::invalid_argument -- the C#
// would throw ArgumentNullException from the chained ctor on a null distribution). No
// end-state re-sync is needed (M9 lesson checked): the ctor path clones the distribution
// with all its state (min/max rule, dependency, correlation matrix) via the Numerics
// Clone(), and the object-initializer field writes below overwrite everything the setter
// side effects touched (Parameters, QuantilePriors).
//
// SKIPPED (project-wide deferrals): the XElement ctor (C# line 101) and ToXElement (line
// 802) -- XML serialization; INotifyPropertyChanged / RaisePropertyChange / event
// (un)subscription; the [Category]/[DisplayName]/[Description]/[Browsable] attributes.
// The C# DataFrame_PropertyChanged handler (line 257) ports as the explicit-invalidation
// data_frame_property_changed() override (the M4->M8 cadence; see the base header).
// ModelParameter.Name for component parameters: the C# reads
// `ParametersToString[j, 0]`; ParametersToString is not on the ported distribution base
// (Phase 4 decision, display-only), so component-parameter names stay "" while OwnerName
// ("D1", "D2", "D3") ports fully. Same reason trims the pointwise Jeffreys label
// "Jeffreys Scale: D<j>.<scaleName>" to "Jeffreys Scale: D<j>" (ParameterNames not ported).
//
// EXCEPTION-TYPE MAPPING for THIS file: C# ArgumentNullException/ArgumentException ->
// std::invalid_argument; ArgumentOutOfRangeException -> std::out_of_range;
// InvalidOperationException (and the C# `CompetingRisks!` NullReferenceException path in
// SetParameterValues) -> std::runtime_error.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/models/support/quantile_prior.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "corehydro/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/competing_risks.hpp"
#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/ln_normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/math/integration/integration.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class CompetingRisksModel : public UnivariateDistributionModelBase,
                            public ISimulatable<std::vector<double>> {
   public:
    using DistributionBase = numerics::distributions::UnivariateDistributionBase;
    using DistributionType = numerics::distributions::UnivariateDistributionType;
    using CompetingRisks = numerics::distributions::CompetingRisks;

    // --- Construction (C# region, lines 34-149) --------------------------------------------

    // C# parameterless ctor (line 39): default to a single quantile prior for this model.
    CompetingRisksModel() { use_single_quantile_ = true; }

    // C# `CompetingRisksModel(DataFrame, List<UnivariateDistributionType>)` (line 50):
    // argument guards FIRST (C# statement order), then DataFrame assignment (setter side
    // effects), a second ProcessThresholdSeries (C# line 61 -- the C# really does reprocess
    // twice on this path), factory-constructed components, and the CompetingRisks setter.
    CompetingRisksModel(DataFrame data_frame,
                        const std::vector<DistributionType>& distribution_types) {
        if (distribution_types.empty())
            throw std::invalid_argument("At least one distribution type is required.");

        if (distribution_types.size() > 3)
            throw std::invalid_argument(
                "There cannot be more than three distributions in the competing risks "
                "model.");

        use_single_quantile_ = true;
        set_data_frame(std::move(data_frame));

        this->data_frame().process_threshold_series();

        std::vector<std::unique_ptr<DistributionBase>> distributions;
        distributions.reserve(distribution_types.size());
        for (std::size_t i = 0; i < distribution_types.size(); ++i) {
            distributions.push_back(
                numerics::distributions::create_distribution(distribution_types[i]));
        }

        set_competing_risks(std::make_unique<CompetingRisks>(std::move(distributions)));
    }

    // C# `CompetingRisksModel(DataFrame, CompetingRisks)` (line 78). The C# null-checks then
    // clones the passed distribution; the C++ reference cannot be null (the guard is the
    // type system), and the clone keeps the argument untouched.
    CompetingRisksModel(DataFrame data_frame, const CompetingRisks& distribution) {
        use_single_quantile_ = true;
        set_data_frame(std::move(data_frame));
        set_competing_risks(clone_competing_risks(distribution));
    }

    // SKIPPED: the XElement ctor (C# line 101) -- XML serialization is a project-wide
    // deferral.

    // Move-only, like the base (deep copies go through clone()).
    CompetingRisksModel(CompetingRisksModel&&) = default;
    CompetingRisksModel& operator=(CompetingRisksModel&&) = default;

    // --- Members (C# region, lines 151-252) ------------------------------------------------

    // C# `IsSupportedDistributionType` (line 180): the 15-family whitelist (the C#
    // _supportedDistributionTypes HashSet, line 156).
    static bool is_supported_distribution_type(DistributionType distribution_type) {
        switch (distribution_type) {
            case DistributionType::Exponential:
            case DistributionType::GammaDistribution:
            case DistributionType::GeneralizedExtremeValue:
            case DistributionType::GeneralizedLogistic:
            case DistributionType::GeneralizedNormal:
            case DistributionType::GeneralizedPareto:
            case DistributionType::Gumbel:
            case DistributionType::KappaFour:
            case DistributionType::LnNormal:
            case DistributionType::Logistic:
            case DistributionType::LogNormal:
            case DistributionType::LogPearsonTypeIII:
            case DistributionType::Normal:
            case DistributionType::PearsonTypeIII:
            case DistributionType::Weibull:
                return true;
            default:
                return false;
        }
    }

    // C# `DataFrame` override (line 188): store, reprocess thresholds at the model boundary
    // (the M4->M8 cadence), then SetDefaultParameters when UseDefaultFlatPriors. (The C#
    // null-assignment branch has no C++ trigger -- see the base header's nullability note.)
    void set_data_frame(DataFrame data_frame) override {
        data_frame_ = std::move(data_frame);
        data_frame_->process_threshold_series();

        if (use_default_flat_priors()) set_default_parameters();
    }

    // C# `CompetingRisks` property (line 214; nullable -- nullptr == C# null).
    bool has_competing_risks() const { return competing_risks_ != nullptr; }
    CompetingRisks* competing_risks() { return competing_risks_.get(); }
    const CompetingRisks* competing_risks() const { return competing_risks_.get(); }

    // C# setter (line 217): assign, then SetDefaultParameters + SetDefaultQuantilePriors
    // (unconditionally, even for null).
    void set_competing_risks(std::unique_ptr<CompetingRisks> competing_risks) {
        competing_risks_ = std::move(competing_risks);
        set_default_parameters();
        set_default_quantile_priors();
    }

    // C# `UseSingleQuantile` override (line 234): the competing risks model currently
    // supports only a single quantile prior; attempts to disable are ignored.
    bool use_single_quantile() const override { return true; }
    void set_use_single_quantile(bool value) override {
        // Model is restricted to a single quantile prior.
        if (!value) return;

        if (!use_single_quantile_) {
            use_single_quantile_ = true;
            set_default_quantile_priors();
        }
    }

    // --- Methods (C# region, lines 254-996) ------------------------------------------------

    // C# `SetDefaultParameters` override (line 276): per component the constraint-driven
    // ModelParameters (OwnerName "D<i>", Uniform(lower, upper) prior). Unlike the Mixture
    // sibling there are no weight parameters and no IsPositive flag in the C#.
    void set_default_parameters() override {
        // (Handler removal/re-add is INPC plumbing, skipped.)
        parameters_.clear();

        if (competing_risks_ == nullptr || !has_data_frame() ||
            !data_frame().validate().is_valid || competing_risks_->component_count() == 0) {
            return;
        }

        // Set the list of model parameters.
        std::vector<double> sample;
        sample.reserve(data_frame().exact_series().count());
        for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i)
            sample.push_back(data_frame().exact_series()[i].value());

        for (int i = 0; i < competing_risks_->component_count(); ++i) {
            // Get constraints.
            const DistributionBase& component = competing_risks_->component(i);
            const auto& mle_component =
                dynamic_cast<const numerics::distributions::IMaximumLikelihoodEstimation&>(
                    component);

            std::vector<double> initials, lowers, uppers;
            mle_component.get_parameter_constraints(sample, initials, lowers, uppers);

            // Name stays "" -- ParametersToString is not on the ported distribution base
            // (see the file header).
            for (int j = 0; j < component.number_of_parameters(); ++j) {
                std::size_t sj = static_cast<std::size_t>(j);
                parameters_.emplace_back(
                    "D" + std::to_string(i + 1), "", initials[sj], lowers[sj], uppers[sj],
                    std::make_unique<numerics::distributions::Uniform>(lowers[sj],
                                                                       uppers[sj]));
            }
        }
    }

    // C# `SetDefaultQuantilePriors` override (line 329): null distribution -> no-op;
    // disabled -> clear both lists; enabled -> exactly ONE LnNormal prior at alpha = 0.1
    // with mu = Round(InverseCDF(1 - alpha), 2), sigma = Round(0.15 * mu, 2); an existing
    // list is shrunk from the back (the extend branch is ported but unreachable with
    // qCount == 1 and a non-empty list). Ends with ProcessQuantilePriors(), like the M10
    // sibling.
    void set_default_quantile_priors() override {
        if (competing_risks_ == nullptr) return;

        // (Handler removal for the old priors is INPC plumbing, skipped.)

        // If quantile priors are not enabled, clear the list and return.
        if (!enable_quantile_priors()) {
            quantile_priors_.clear();
            quantile_priors_true_.clear();
            return;
        }

        std::vector<QuantilePrior> priors;
        const std::size_t q_count = 1;  // competing-risks model uses a single quantile prior

        if (quantile_priors_.size() >= 1) {
            priors = quantile_priors_;

            while (priors.size() > q_count) priors.pop_back();

            while (priors.size() < q_count) {
                priors.emplace_back(priors.back().alpha() / 10.0,
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu = round_half_even_2(
                    competing_risks_->inverse_cdf(1.0 - priors.back().alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors.back().distribution().set_parameters({mu, sigma});
            }
        } else {
            for (std::size_t i = 1; i <= q_count; ++i) {
                priors.emplace_back(std::pow(10.0, -static_cast<double>(i)),
                                    std::make_unique<numerics::distributions::LnNormal>());
                double mu = round_half_even_2(
                    competing_risks_->inverse_cdf(1.0 - priors[i - 1].alpha()));
                double sigma = round_half_even_2(mu * 0.15);
                priors[i - 1].distribution().set_parameters({mu, sigma});
            }
        }

        // Reset the quantile priors with the new list (handler re-adds skipped).
        quantile_priors_ = std::move(priors);

        process_quantile_priors();
    }

    // C# `ProcessQuantilePriors` override (line 390): the single prior is cloned through.
    // The multi-quantile branch is ported for structure but unreachable -- UseSingleQuantile
    // is hard-true for this model (the C# comment says the branch is "left here for possible
    // future extension but not used").
    void process_quantile_priors() override {
        quantile_priors_true_.clear();

        if (quantile_priors_.size() == 1) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());
            return;
        }

        if (!use_single_quantile() && competing_risks_ != nullptr &&
            static_cast<int>(quantile_priors_.size()) ==
                competing_risks_->number_of_parameters()) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());

            for (std::size_t i = 1; i < quantile_priors_.size(); ++i) {
                double mu_diff = quantile_priors_[i].distribution().mean() -
                                 quantile_priors_[i - 1].distribution().mean();
                double sigma_diff =
                    std::sqrt(quantile_priors_[i].distribution().variance() +
                              quantile_priors_[i - 1].distribution().variance());

                auto gamma_dist =
                    std::make_unique<numerics::distributions::GammaDistribution>();
                gamma_dist->set_parameters(
                    gamma_dist->parameters_from_moments({mu_diff, sigma_diff}));

                quantile_priors_true_.emplace_back(quantile_priors_[i].alpha(),
                                                   std::move(gamma_dist));
            }
        }
    }

    // C# `LogLikelihood` override (line 422): data likelihood (early -inf), plus prior,
    // then the finite collapse.
    double log_likelihood(std::vector<double>& parameters) const override {
        // Get the data likelihood.
        double data_log_lh = data_log_likelihood(parameters);
        if (!numerics::is_finite(data_log_lh)) return -std::numeric_limits<double>::infinity();

        // Get the prior likelihood.
        double prior_log_lh = prior_log_likelihood(parameters);

        // The full likelihood.
        double log_lh = data_log_lh + prior_log_lh;

        return numerics::is_finite(log_lh) ? log_lh
                                           : -std::numeric_limits<double>::infinity();
    }

    // C# `DataLogLikelihood` override (line 439): the full censored likelihood over a
    // working copy of the competing risks distribution. (The C# declares an unused local
    // `int k = model.Distributions.Count;` here, omitted to keep -Wall clean.)
    double data_log_likelihood(std::vector<double>& parameters) const override {
        if (competing_risks_ == nullptr || !has_data_frame())
            return -std::numeric_limits<double>::infinity();

        std::unique_ptr<DistributionBase> model_owner = competing_risks_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        double log_lh = 0.0;
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // Set model parameters.
        model.set_parameters(parameters);

        const DataFrame& df = data_frame();

        // Exact Data.
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            if (!data.is_low_outlier()) {
                log_lh += model.log_likelihood(data.value());
            } else {
                log_lh += model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
            }
        }

        // Uncertain Data.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                // Normalize by retained ME mass from the 1E-8 probability window.
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                log_lh += ep > 0 ? std::log(ep) : neg_inf;
            } else {
                log_lh += neg_inf;
            }
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            log_lh += model.log_likelihood_intervals(data.lower_value(), data.upper_value());
        }

        // Threshold Data.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh +=
                    model.log_likelihood_right_censored(data.value(), data.number_above());
        }
        return log_lh;
    }

    // C# `PointwiseDataLogLikelihood` override (line 506): one entry per observation;
    // threshold entries combine the left and right censored contributions.
    std::vector<double> pointwise_data_log_likelihood(
        const std::vector<double>& parameters) const override {
        if (competing_risks_ == nullptr || !has_data_frame()) return {};

        std::unique_ptr<DistributionBase> model_owner = competing_risks_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // Set model parameters.
        model.set_parameters(parameters);

        std::vector<double> result;
        const DataFrame& df = data_frame();

        // Exact Data.
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            if (!data.is_low_outlier()) {
                result.push_back(model.log_likelihood(data.value()));
            } else {
                result.push_back(
                    model.log_likelihood_left_censored(df.low_outlier_threshold(), 1));
            }
        }

        // Uncertain Data.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                result.push_back(ep > 0 ? std::log(ep) : neg_inf);
            } else {
                result.push_back(neg_inf);
            }
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            result.push_back(
                model.log_likelihood_intervals(data.lower_value(), data.upper_value()));
        }

        // Threshold Data: combine left and right censored contributions for this threshold.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double ll = 0.0;
            if (data.number_below() > 0)
                ll += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                ll += model.log_likelihood_right_censored(data.value(), data.number_above());
            result.push_back(ll);
        }

        return result;
    }

    // C# `PointwiseDataLogLikelihoodComponents` override (line 576).
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& parameters) const override {
        if (competing_risks_ == nullptr || !has_data_frame()) return {};

        std::unique_ptr<DistributionBase> model_owner = competing_risks_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        const double neg_inf = -std::numeric_limits<double>::infinity();

        model.set_parameters(parameters);

        std::vector<DataComponent> result;
        int idx = 0;
        const DataFrame& df = data_frame();

        // Exact Data.
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            double log_lh;
            double value = data.value();

            if (!data.is_low_outlier()) {
                log_lh = model.log_likelihood(data.value());
            } else {
                log_lh = model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
                value = df.low_outlier_threshold();
            }

            result.emplace_back(idx++, log_lh, value, DataComponentType::Exact, 1,
                                std::to_string(data.index()));
        }

        // Uncertain Data.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const UncertainData& data = df.uncertain_series()[i];
            const DistributionBase& dist = data.distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            double log_lh = neg_inf;
            if (numerics::is_finite(a) && numerics::is_finite(b) && numerics::is_finite(mass) &&
                mass > 0.0 && a < b) {
                double ep = numerics::math::integration::Integration::gauss_legendre20(
                                [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                            mass;
                log_lh = ep > 0 ? std::log(ep) : neg_inf;
            }
            result.emplace_back(idx++, log_lh, dist.mean(), DataComponentType::Uncertain, 1,
                                std::to_string(data.index()));
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            double log_lh =
                model.log_likelihood_intervals(data.lower_value(), data.upper_value());
            double midpoint = (data.lower_value() + data.upper_value()) / 2.0;

            result.emplace_back(idx++, log_lh, midpoint, DataComponentType::Interval, 1,
                                std::to_string(data.index()));
        }

        // Threshold Data.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double ll = 0.0;
            if (data.number_below() > 0)
                ll += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                ll += model.log_likelihood_right_censored(data.value(), data.number_above());

            int total_count = data.number_below() + data.number_above();
            result.emplace_back(idx++, ll, data.value(), DataComponentType::LeftCensored,
                                total_count, "Threshold " + std::to_string(i + 1));
        }

        return result;
    }

    // C# `PriorLogLikelihood` override (line 654): parameter priors, the Jeffreys 1/scale
    // term per component (Gamma/Weibull carry the scale in parameter 0, all other supported
    // families in parameter 1), and the single quantile prior. Like the C#, the raw sum is
    // returned (no finite collapse; LogLikelihood collapses). The C# `Parameters == null`
    // guard has no C++ analogue (the vector always exists).
    double prior_log_likelihood(std::vector<double>& parameters) const override {
        if (competing_risks_ == nullptr) return -std::numeric_limits<double>::infinity();

        std::unique_ptr<DistributionBase> model_owner = competing_risks_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        int k = model.component_count();
        double log_lh = 0.0;

        // Set model parameters.
        model.set_parameters(parameters);

        // Parameter Priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(parameters[i]);
        }

        // Jeffreys rule on scale parameters for each component.
        if (use_jeffreys_rule_for_scale()) {
            for (int j = 0; j < k; ++j) {
                const DistributionBase& dist = model.component(j);
                double scale;

                if (dist.type() == DistributionType::GammaDistribution ||
                    dist.type() == DistributionType::Weibull) {
                    scale = dist.get_parameters()[0];
                } else {
                    scale = dist.get_parameters()[1];
                }
                log_lh -= scale > 0 ? std::log(scale)
                                    : std::numeric_limits<double>::infinity();
            }
        }

        // Single quantile prior.
        if (enable_quantile_priors() && quantile_priors_true_.size() == 1) {
            log_lh += quantile_priors_true_[0].distribution().log_pdf(
                model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha()));
        }

        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood` override (line 703): per-parameter prior components,
    // per-component Jeffreys components, and the single quantile-prior component.
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& parameters) const override {
        std::vector<PriorComponent> result;

        if (competing_risks_ == nullptr) return result;

        std::unique_ptr<DistributionBase> model_owner = competing_risks_->clone();
        CompetingRisks& model = static_cast<CompetingRisks&>(*model_owner);
        int k = model.component_count();

        // Set model parameters.
        model.set_parameters(parameters);

        // Parameter Priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(parameters[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }

        // Jeffreys rule on scale parameters for each component. Label: the C# appends
        // ".<scaleName>" from dist.ParameterNames, which is not on the ported base (file
        // header); the label carries the component tag only.
        if (use_jeffreys_rule_for_scale()) {
            for (int j = 0; j < k; ++j) {
                const DistributionBase& dist = model.component(j);
                double scale;

                if (dist.type() == DistributionType::GammaDistribution ||
                    dist.type() == DistributionType::Weibull) {
                    scale = dist.get_parameters()[0];
                } else {
                    scale = dist.get_parameters()[1];
                }
                double ll = scale > 0 ? -std::log(scale)
                                      : -std::numeric_limits<double>::infinity();
                result.emplace_back("Jeffreys Scale: D" + std::to_string(j + 1), ll,
                                    PriorComponentType::JeffreysScalePrior);
            }
        }

        // Single quantile prior. Label: the C# formats alpha with :G4 (general, four
        // significant digits); approximated with printf %.4g (display-only).
        if (enable_quantile_priors() && quantile_priors_true_.size() == 1) {
            double quantile = model.inverse_cdf(1.0 - quantile_priors_true_[0].alpha());
            double ll = quantile_priors_true_[0].distribution().log_pdf(quantile);
            char alpha_buf[64];
            std::snprintf(alpha_buf, sizeof alpha_buf, "%.4g",
                          quantile_priors_true_[0].alpha());
            result.emplace_back(std::string("Quantile Prior: p=") + alpha_buf, ll,
                                PriorComponentType::QuantilePrior);
        }

        return result;
    }

    // C# `SetParameterValues` override (line 761): length guard, write the ModelParameter
    // values, then push the proposal into the wrapped distribution. The C# null-parameters
    // guard (ArgumentNullException) has no C++ analogue (a vector cannot be null). The C#
    // dereferences `CompetingRisks!` unguarded (NullReferenceException on a null
    // distribution); the C++ throws std::runtime_error instead of crashing.
    void set_parameter_values(const std::vector<double>& parameters) override {
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters()))
            throw std::invalid_argument("The length of the parameter list is incorrect.");

        for (std::size_t i = 0; i < parameters.size(); ++i)
            parameters_[i].set_value(parameters[i]);

        if (competing_risks_ == nullptr)
            throw std::runtime_error("Competing risks distribution is null.");  // C# NRE guard
        std::vector<double> parms_copy = parameters;  // C# parameters.ToArray()
        competing_risks_->set_parameters(parms_copy);
    }

    // C# `Clone()` override (line 777): deep, independent copy -- cloned Parameters and
    // QuantilePriors, the flag fields, then ProcessQuantilePriors(). See the file header
    // for the DataFrame deep-copy divergence.
    CompetingRisksModel clone() const {
        if (!has_data_frame())
            throw std::invalid_argument("dataFrame");  // C++ divergence: frame required
        if (competing_risks_ == nullptr)
            throw std::invalid_argument(
                "distribution");  // C# ctor would throw ArgumentNullException

        std::vector<ModelParameter> parms;
        parms.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i)
            parms.push_back(parameters_[i].clone());

        std::vector<QuantilePrior> quants;
        quants.reserve(quantile_priors_.size());
        for (std::size_t i = 0; i < quantile_priors_.size(); ++i)
            quants.push_back(quantile_priors_[i].clone());

        CompetingRisksModel result(data_frame().clone(), *competing_risks_);
        // C# object-initializer field writes (no setter side effects).
        result.use_default_flat_priors_ = use_default_flat_priors_;
        result.use_jeffreys_rule_for_scale_ = use_jeffreys_rule_for_scale_;
        result.enable_quantile_priors_ = enable_quantile_priors_;
        result.use_single_quantile_ = true;
        result.parameters_ = std::move(parms);
        result.quantile_priors_ = std::move(quants);

        result.process_quantile_priors();
        return result;
    }

    // SKIPPED: ToXElement (C# line 802) -- XML serialization is a project-wide deferral.

    // C# `Validate` override (line 831).
    ValidationResult validate() const override {
        ValidationResult result;

        // Data frame.
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: Data frame is null.");
            return result;
        }

        // Competing risks distribution.
        if (competing_risks_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Competing risks distribution is null.");
            return result;
        }

        // Validate data frame.
        ValidationResult data_valid = data_frame().validate();
        if (!data_valid.is_valid) {
            result.is_valid = false;
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
        }

        if (competing_risks_->component_count() == 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Competing risks distribution has no component distributions.");
        }

        if (competing_risks_->component_count() > 3) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: Competing risks model currently supports at most 3 component "
                "distributions.");
        }

        // Validate uncertain-data ME bounds before likelihood evaluation. The uncertain
        // contribution is normalized by retained probability mass over this 1E-8 window.
        for (std::size_t i = 0; i < data_frame().uncertain_series().count(); ++i) {
            const UncertainData& data = data_frame().uncertain_series()[i];
            const DistributionBase& dist = data.distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            // Validate the retained quantile window directly; endpoints at 0/1 are
            // infinite for common ME distributions such as Normal.
            double lower = dist.inverse_cdf(lower_probability);
            double upper = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;

            if (!numerics::is_finite(lower) || !numerics::is_finite(upper) ||
                !numerics::is_finite(mass) || mass <= 0.0 || lower >= upper) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Uncertain data at index " + std::to_string(data.index()) +
                    " has invalid measurement-error integration bounds at the 1E-8 "
                    "probability window.");
            }
        }

        // Component distribution type check. (The C# message quotes the enum name; the
        // ported enum has no name surface, so the message carries the index only.)
        for (int i = 0; i < competing_risks_->component_count(); ++i) {
            if (!is_supported_distribution_type(competing_risks_->component(i).type())) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Component distribution " + std::to_string(i + 1) +
                    " has an unsupported type.");
            }

            if (competing_risks_->component(i).type() == DistributionType::LnNormal ||
                competing_risks_->component(i).type() == DistributionType::LogNormal ||
                competing_risks_->component(i).type() ==
                    DistributionType::LogPearsonTypeIII) {
                for (std::size_t j = 0; j < data_frame().uncertain_series().count(); ++j) {
                    const UncertainData& data = data_frame().uncertain_series()[j];
                    const DistributionBase& dist = data.distribution();
                    double lower_probability = 1E-8;
                    double lower = dist.inverse_cdf(lower_probability);

                    // Log components require positive retained ME support; checking the
                    // ME mean alone can miss a negative lower tail.
                    if (!numerics::is_finite(lower) || lower <= 0.0) {
                        result.is_valid = false;
                        result.validation_messages.push_back(
                            "Error: Component distribution " + std::to_string(i + 1) +
                            " is log-based but uncertain data at index " +
                            std::to_string(data.index()) +
                            " has non-positive retained support.");
                    }
                }
            }
        }

        // Parameter priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            ValidationResult valid = parameters_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }
        }

        // Quantile priors.
        for (std::size_t i = 0; i < quantile_priors_.size(); ++i) {
            ValidationResult valid = quantile_priors_[i].validate();
            if (!valid.is_valid) {
                result.is_valid = false;
                result.validation_messages.insert(result.validation_messages.end(),
                                                  valid.validation_messages.begin(),
                                                  valid.validation_messages.end());
            }

            if (i >= 1) {
                if (quantile_priors_[i].alpha() >= quantile_priors_[i - 1].alpha()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile priors must have strictly decreasing exceedance "
                        "probabilities (Alpha).");
                }

                if (quantile_priors_[i].distribution().mean() <=
                    quantile_priors_[i - 1].distribution().mean()) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile prior means must increase with decreasing "
                        "exceedance probability.");
                }

                if (quantile_priors_[i].distribution().inverse_cdf(0.05) <=
                        quantile_priors_[i - 1].distribution().inverse_cdf(0.05) ||
                    quantile_priors_[i].distribution().inverse_cdf(0.95) <=
                        quantile_priors_[i - 1].distribution().inverse_cdf(0.95)) {
                    result.is_valid = false;
                    result.validation_messages.push_back(
                        "Error: Quantile prior 5 percent and 95 percent bounds must "
                        "increase with decreasing exceedance probability.");
                }
            }
        }

        return result;
    }

    // C# `GenerateRandomValues(int sampleSize, int seed = -1)` (line 986, ISimulatable):
    // guards, then delegates to CompetingRisks.GenerateRandomValues -- the ported
    // per-component min/max Mersenne Twister sampler (seed > 0 deterministic, else
    // clock-seeded).
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size <= 0)
            throw std::out_of_range("Sample size must be positive.");
        if (competing_risks_ == nullptr)
            throw std::runtime_error(
                "CompetingRisks distribution cannot be null when generating random values.");

        return competing_risks_->generate_random_values(sample_size, seed);
    }

   protected:
    // C# `DataFrame_PropertyChanged` override (line 257): the explicit-invalidation
    // equivalent for in-place frame mutations (see the base header's cadence note; the
    // PlottingParameter/PlottingPosition filtering has no C++ trigger to filter).
    // Reprocess thresholds, then SetDefaultParameters when flat priors. NEVER called from
    // the likelihood paths.
    void data_frame_property_changed() override {
        if (!has_data_frame()) return;

        data_frame().process_threshold_series();

        if (use_default_flat_priors()) set_default_parameters();
    }

   private:
    // C# `CompetingRisks = (CompetingRisks)distribution.Clone()`.
    static std::unique_ptr<CompetingRisks> clone_competing_risks(
        const CompetingRisks& distribution) {
        std::unique_ptr<DistributionBase> base = distribution.clone();
        return std::unique_ptr<CompetingRisks>(static_cast<CompetingRisks*>(base.release()));
    }

    // C# `Math.Round(value, 2)` (MidpointRounding.ToEven); the
    // univariate_distribution_model.hpp precedent.
    static double round_half_even_2(double value) {
        return std::nearbyint(value * 100.0) / 100.0;
    }

    std::unique_ptr<CompetingRisks> competing_risks_;  // C# _competingRisks (line 185)
};

}  // namespace corehydro::models
