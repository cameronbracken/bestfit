// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/UnivariateDistribution.cs
// @ fc28c0c -- STATIONARY path (Phase 5, M8; the Phase 4 T6 slice was exact-data-only).
//
// The univariate distribution model: a parent distribution plus a DataFrame of exact /
// uncertain / interval / threshold observations, with the FULL stationary censored
// likelihood (M8). The C++ class keeps the Phase 4 name `UnivariateDistributionModel`
// (the C# class is `UnivariateDistribution`; the suffix avoids colliding with the Numerics
// distribution layer). It derives from `UnivariateDistributionModelBase` (base/...) which
// supplies the DataFrame, UseJeffreysRuleForScale, and IQuantilePriors state.
//
// Ported surface (stationary):
//   - Ctors: parameterless (C# line 32: LogPearsonTypeIII, no data), (DataFrame,
//     distribution) (line 46), (DataFrame, type) (line 63), plus two Phase 4 compatibility
//     shims taking a plain std::vector<double> of exact data (they build a DataFrame with an
//     ExactSeries internally; the fixture runner and the R/Python glue construct through
//     them and their semantics are unchanged).
//   - `IsSupportedDistributionType` / `CreateDistribution` statics (lines 263, 534): the
//     15-family whitelist. create_distribution delegates to the Numerics factory after the
//     whitelist check (same construction result as the C# if-chain of `new X()` defaults;
//     avoids 15 concrete includes here). Unsupported -> std::out_of_range (C#
//     ArgumentOutOfRangeException).
//   - `Distribution` property (line 328): set_distribution runs SetDefaultParameters +
//     SetDefaultQuantilePriors like the C# setter (its TrendModels reset is M9; the
//     _isDeserializing XML suppression is out of scope). `DistributionType` (line 368).
//   - `SetDefaultParameters` (line 571), stationary: when a valid DataFrame with exact data
//     is present, `GetParameterConstraints` supplies (initials, lowers, uppers) and one
//     ModelParameter per distribution parameter gets Value/bounds/Uniform prior; otherwise
//     (the C# null/empty-frame skip path) one default ModelParameter per distribution
//     parameter (upstream these come from the default ConstantTrend per parameter; the
//     trend layer joins this class in M9 -- observably identical for the stationary case,
//     with Name cleared to "" as the C# does).
//     RETAINED PHASE 4 DEVIATION: after the constraint path the distribution itself is set
//     to `initials`; the C# leaves the distribution at its previous parameters until
//     SetParameterValues. Kept because the Phase 4 tests/fixtures pin it and no ported
//     compute path reads the stored parameters before setting them (the likelihoods all set
//     a working copy from the proposal first).
//   - `LogLikelihood` (1116), `PriorLogLikelihood` (1143), `StationaryData_LogLikelihood`
//     (1181), `DataLogLikelihood` (1361), `PointwiseDataLogLikelihood` (1373) ->
//     `StationaryPointwiseLogLikelihood` (1632), `PointwiseDataLogLikelihoodComponents`
//     (1385) -> `StationaryPointwiseLogLikelihoodComponents` (1402), `Prior_LogLikelihood`
//     (1822), `PointwisePriorLogLikelihood` (1882), `SetParameterValues` (1977),
//     `GetParameterValues` (2006), `SetDistributionParameterValues` (2020), `Validate`
//     (2127) -- each stationary branch term-for-term; see the per-method comments.
//   - The uncertain-data (measurement-error) branch integrates
//     `dist.PDF(q) * model.PDF(q)` over the 1e-8 probability window with
//     `Integration::gauss_legendre20` and normalizes by the retained mass, with the
//     finiteness/positivity guards exactly as the C# writes them.
//
// Validity-check mechanism: the C# `model.ValidateParameters(parameters, false)` returns
// null when valid / a message when invalid, and the model is only `SetParameters`ed when
// valid. The C++ distribution base has no validate-without-set; every concrete
// `set_parameters` sets the `parameters_valid_` flag instead. This port therefore probes on
// a THROWAWAY clone (set, then read `parameters_valid()`) and only sets the working copy
// when the probe is valid -- functionally identical, and it preserves the C# semantic that
// the working copy keeps its previous parameters on an invalid proposal (which matters for
// the Jeffreys term below, read off the working copy's parameters as the C# does).
//
// Divergence note (retained from Phase 4): `Normal::set_parameters` clamps a tiny
// non-negative scale up to 1E-16 rather than marking it invalid (mirroring the C#
// `Normal.SetParameters` clamp), so a proposed sigma in [0, 1E-16) rounds up identically on
// both sides of the port.
//
// Deliberately NOT ported in M8 (M9 unless noted):
//   - Nonstationary: IsNonstationary, ParameterTimeIndex, Alpha, TrendModels, SetTrendModel,
//     NonstationaryData_LogLikelihood, NonstationaryPointwise* (C# 1256, 1509, 1716),
//     GetNonstationaryReturnLevel (2035), and the nonstationary branches of the ctor /
//     DataFrame setter / SetDefaultParameters / SetParameterValues / Validate /
//     DataFrame_PropertyChanged. The stationary methods here are the
//     `IsNonstationary == false` sides of the C# ternaries.
//   - Quantile-prior BODIES: SetDefaultQuantilePriors (1024) beyond its disabled early-exit
//     (ProcessQuantilePriors (1089) IS ported -- both branches are self-contained), and the
//     quantile-prior terms of Prior_LogLikelihood (1853-1875) / PointwisePriorLogLikelihood
//     (1938-1972) -- the multi-quantile branch needs IStandardError.QuantileJacobian, not
//     yet on the ported distribution base. Unreachable while EnableQuantilePriors is false;
//     enabling it throws from the M9 stub in set_default_quantile_priors.
//   - ISimulatable / GenerateRandomValues (2303) -- M9.
//   - IModel.Clone (2058) -- M9 (deep-copies TrendModels/QuantilePriors/nonstationary state;
//     also the C# clone ALIASES the DataFrame, which the value-typed C++ frame cannot).
//   - IUnivariateModel -- M9 (its `distribution()` pointer accessor collides with this
//     class's Phase 4 reference accessor; resolved when the bivariate consumers arrive).
//   - XML (XElement ctor, ToXElement), INotifyPropertyChanged, [attributes], and the
//     parameter/owner display-name wiring from Distribution.ParameterNames (WPF display
//     concerns; ModelParameter owner_name()/name() stay empty, Phase 4 decision).
//
// EXCEPTION-TYPE MAPPING for THIS file: C# ArgumentNullException/ArgumentException ->
// std::invalid_argument; ArgumentOutOfRangeException -> std::out_of_range;
// InvalidOperationException -> std::runtime_error.
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/gamma_distribution.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/math/integration/integration.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::models {

class UnivariateDistributionModel : public UnivariateDistributionModelBase {
   public:
    using DistributionBase = numerics::distributions::UnivariateDistributionBase;
    using DistributionType = numerics::distributions::UnivariateDistributionType;

    // C# parameterless ctor (line 32): Log-Pearson Type III and no data.
    UnivariateDistributionModel() {
        set_distribution(create_distribution(DistributionType::LogPearsonTypeIII));
    }

    // C# `UnivariateDistribution(DataFrame, UnivariateDistributionBase)` (line 46). The C#
    // clones the passed distribution; here the caller transfers sole ownership instead.
    UnivariateDistributionModel(DataFrame data_frame,
                                std::unique_ptr<DistributionBase> distribution) {
        if (distribution == nullptr)
            throw std::invalid_argument("distribution");  // C# ArgumentNullException
        set_distribution(std::move(distribution));
        set_data_frame(std::move(data_frame));
    }

    // C# `UnivariateDistribution(DataFrame, UnivariateDistributionType)` (line 63): builds
    // the distribution through the model's supported-type factory.
    UnivariateDistributionModel(DataFrame data_frame, DistributionType distribution_type) {
        set_distribution(create_distribution(distribution_type));
        set_data_frame(std::move(data_frame));
    }

    // Phase 4 compatibility shims (bestfit additions): exact observations as a plain vector.
    // A DataFrame with an ExactSeries (sequential 0-based indexes) is built internally, so
    // the fixture/binding construction paths keep their exact Phase 4 semantics.
    UnivariateDistributionModel(DistributionType distribution_type, std::vector<double> exact_data)
        : UnivariateDistributionModel(make_exact_data_frame(exact_data), distribution_type) {}

    UnivariateDistributionModel(std::unique_ptr<DistributionBase> distribution,
                                std::vector<double> exact_data)
        : UnivariateDistributionModel(make_exact_data_frame(exact_data), std::move(distribution)) {}

    // --- Statics -----------------------------------------------------------------------

    // C# `IsSupportedDistributionType` (line 263): the 15-family whitelist this model
    // supports (the C# _supportedDistributionTypes HashSet).
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

    // C# `CreateDistribution` (line 534): whitelist-checked construction. Delegates to the
    // Numerics factory (identical default-constructed results as the C# `new X()` chain).
    static std::unique_ptr<DistributionBase> create_distribution(
        DistributionType distribution_type) {
        if (!is_supported_distribution_type(distribution_type)) {
            // C# ArgumentOutOfRangeException("Unsupported distribution type.").
            throw std::out_of_range("Unsupported distribution type.");
        }
        return numerics::distributions::create_distribution(distribution_type);
    }

    // --- Distribution property (C# line 328) ---------------------------------------------
    DistributionBase& distribution() { return *distribution_; }
    const DistributionBase& distribution() const { return *distribution_; }

    // C# setter: null check, assign, reset trend models (M9), then SetDefaultParameters()
    // and SetDefaultQuantilePriors() (unconditionally -- not gated on UseDefaultFlatPriors;
    // the XML `_isDeserializing` suppression is out of scope).
    void set_distribution(std::unique_ptr<DistributionBase> distribution) {
        if (distribution == nullptr)
            throw std::invalid_argument("Distribution");  // C# ArgumentNullException
        distribution_ = std::move(distribution);
        set_default_parameters();
        set_default_quantile_priors();
    }

    // C# `DistributionType` property (line 368).
    DistributionType distribution_type() const { return distribution_->type(); }
    void set_distribution_type(DistributionType distribution_type) {
        if (distribution_ != nullptr && distribution_->type() == distribution_type) return;
        set_distribution(create_distribution(distribution_type));
    }

    // --- SetDefaultParameters (C# line 571, stationary path) ------------------------------
    void set_default_parameters() override {
        if (distribution_ == nullptr) return;

        // (Handler removal and the TrendModels bookkeeping are INPC/M9 concerns.)
        // C# guard: DataFrame != null && DataFrame.Validate().IsValid &&
        //           ExactSeries != null && ExactSeries.Count > 0.
        if (has_data_frame() && data_frame().validate().is_valid &&
            data_frame().exact_series().count() > 0) {
            auto* ml_estimator =
                dynamic_cast<numerics::distributions::IMaximumLikelihoodEstimation*>(
                    distribution_.get());
            if (ml_estimator == nullptr) {
                // C# would fail the (IMaximumLikelihoodEstimation) cast, wrapped by the
                // catch into InvalidOperationException.
                throw std::runtime_error(
                    "UnivariateDistributionModel: distribution does not implement "
                    "IMaximumLikelihoodEstimation (GetParameterConstraints unavailable)");
            }

            std::vector<double> initials;
            std::vector<double> lowers;
            std::vector<double> uppers;
            ml_estimator->get_parameter_constraints(
                data_frame().exact_series().values_to_list(), initials, lowers, uppers);

            // Stationary: one ModelParameter per distribution parameter with
            // Value=initials[i], bounds, and a Uniform(lower, upper) prior; the Name is
            // cleared for the stationary case (C# 756-760).
            parameters_.clear();
            parameters_.reserve(initials.size());
            for (std::size_t i = 0; i < initials.size(); ++i) {
                parameters_.emplace_back(
                    /*owner_name=*/"", /*name=*/"", initials[i], lowers[i], uppers[i],
                    std::make_unique<numerics::distributions::Uniform>(lowers[i], uppers[i]));
            }

            // Retained Phase 4 deviation (see file header): the C# leaves the distribution
            // at its previous parameters here.
            distribution_->set_parameters(initials);
        } else {
            // C# skip path (null/invalid/empty frame): parameters come from the default
            // ConstantTrend per distribution parameter -- a single default ModelParameter
            // each -- with Name cleared for the stationary case (trend layer: M9).
            parameters_.clear();
            parameters_.reserve(static_cast<std::size_t>(distribution_->number_of_parameters()));
            for (int i = 0; i < distribution_->number_of_parameters(); ++i) {
                ModelParameter p;
                p.set_name("");
                parameters_.push_back(std::move(p));
            }
        }
    }

    // --- Quantile priors (bodies; see the file-header deferral notes) ---------------------

    // C# `SetDefaultQuantilePriors` (line 1024). Only the disabled early-exit is ported in
    // M8: with EnableQuantilePriors == false both lists are cleared (C# 1037-1043). The
    // enabled branch (default LnNormal priors from the distribution quantiles) is M9.
    void set_default_quantile_priors() override {
        if (distribution_ == nullptr) return;
        if (!enable_quantile_priors()) {
            quantile_priors_.clear();
            quantile_priors_true_.clear();
            return;
        }
        // M9 stub: the default-prior construction (LnNormal at alpha = 10^-i off the
        // distribution quantiles, C# 1045-1085). Throwing keeps the M8 surface honest --
        // enabling quantile priors cannot silently misfit.
        throw std::logic_error(
            "UnivariateDistributionModel: default quantile priors are not ported yet (M9)");
    }

    // C# `ProcessQuantilePriors` (line 1089): the first prior stays; the remaining priors
    // convert to Gamma distributions of the difference in random quantiles.
    void process_quantile_priors() override {
        quantile_priors_true_.clear();
        if (quantile_priors_.size() == 1) {
            quantile_priors_true_.push_back(quantile_priors_[0].clone());
            return;
        }
        if (!use_single_quantile() &&
            static_cast<int>(quantile_priors_.size()) == distribution_->number_of_parameters()) {
            // The first quantile prior remains the same.
            quantile_priors_true_.push_back(quantile_priors_[0].clone());
            for (std::size_t i = 1; i < quantile_priors_.size(); ++i) {
                double mu_diff = quantile_priors_[i].distribution().mean() -
                                 quantile_priors_[i - 1].distribution().mean();
                double sigma_diff = std::sqrt(quantile_priors_[i].distribution().variance() +
                                              quantile_priors_[i - 1].distribution().variance());

                auto gamma_dist = std::make_unique<numerics::distributions::GammaDistribution>();
                gamma_dist->set_parameters(
                    gamma_dist->parameters_from_moments({mu_diff, sigma_diff}));
                quantile_priors_true_.emplace_back(quantile_priors_[i].alpha(),
                                                   std::move(gamma_dist));
            }
        }
    }

    // --- Likelihood surface ----------------------------------------------------------------

    // C# `LogLikelihood` override (line 1116): ONE working copy of the distribution is
    // threaded through the data likelihood and then the prior likelihood (so the Jeffreys
    // term sees the parameters the data step set), then the finite collapse.
    double log_likelihood(const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error("Distribution must be set before computing log likelihood.");

        std::unique_ptr<DistributionBase> model = distribution_->clone();
        double data_log_lh = stationary_data_log_likelihood(*model, p);  // IsNonstationary: M9
        double prior_log_lh = prior_log_likelihood(*model, p);

        double log_lh = data_log_lh + prior_log_lh;
        return numerics::is_finite(log_lh) ? log_lh : -std::numeric_limits<double>::infinity();
    }

    // C# `PriorLogLikelihood` override (line 1143): priors are evaluated against the
    // distribution at the supplied parameters (stationary; the nonstationary
    // last-time-step path is M9). Invalid proposals leave the working copy at its previous
    // parameters (see the validity-check note in the file header).
    double prior_log_likelihood(const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing prior log likelihood.");
        if (p.size() != parameters_.size()) return -std::numeric_limits<double>::infinity();

        std::unique_ptr<DistributionBase> model = distribution_->clone();
        std::vector<double> dist_params(
            p.begin(), p.begin() + static_cast<std::ptrdiff_t>(model->number_of_parameters()));

        std::unique_ptr<DistributionBase> probe = distribution_->clone();
        probe->set_parameters(dist_params);
        if (probe->parameters_valid()) model->set_parameters(dist_params);

        return prior_log_likelihood(*model, p);
    }

    // C# `StationaryData_LogLikelihood` (line 1181): the full stationary censored
    // likelihood -- the type-switch over the four series.
    double stationary_data_log_likelihood(DistributionBase& model,
                                          const std::vector<double>& p) const {
        if (!has_data_frame())
            throw std::runtime_error("DataFrame must be set before computing log likelihood.");

        // Check if proposed parameters are valid (probe clone; see file header). The C#
        // returns -inf WITHOUT setting the working copy.
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(p);
        if (!probe->parameters_valid()) return -std::numeric_limits<double>::infinity();

        // Set model parameters.
        model.set_parameters(p);

        const DataFrame& df = data_frame();
        double log_lh = 0.0;

        // Exact Data (low outliers -> left-censored at the frame's LowOutlierThreshold).
        for (std::size_t i = 0; i < df.exact_series().count(); ++i) {
            const ExactData& data = df.exact_series()[i];
            if (!data.is_low_outlier()) {
                log_lh += model.log_likelihood(data.value());
            } else {
                log_lh += model.log_likelihood_left_censored(df.low_outlier_threshold(), 1);
            }
        }

        // Uncertain Data (measurement error): integrate dist.PDF(q) * model.PDF(q) over the
        // 1e-8 probability window and normalize by the retained mass, with the C# guards.
        for (std::size_t i = 0; i < df.uncertain_series().count(); ++i) {
            const DistributionBase& dist = df.uncertain_series()[i].distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
            double a = dist.inverse_cdf(lower_probability);
            double b = dist.inverse_cdf(upper_probability);
            double mass = upper_probability - lower_probability;
            if (!numerics::is_finite(a) || !numerics::is_finite(b) ||
                !numerics::is_finite(mass) || mass <= 0.0 || a >= b) {
                log_lh += -std::numeric_limits<double>::infinity();
                continue;
            }

            // Normalize by retained ME mass so the likelihood remains conditional on the
            // same 1E-8 probability window used across uncertain-data models.
            double ep = numerics::math::integration::Integration::gauss_legendre20(
                            [&](double q) { return dist.pdf(q) * model.pdf(q); }, a, b) /
                        mass;
            log_lh += ep > 0 ? std::log(ep) : -std::numeric_limits<double>::infinity();
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            log_lh += model.log_likelihood_intervals(data.lower_value(), data.upper_value());
        }

        // Threshold Data (a zero count on either censored side contributes exactly zero).
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh += model.log_likelihood_right_censored(data.value(), data.number_above());
        }

        return log_lh;
    }

    // C# `DataLogLikelihood` override (line 1361).
    double data_log_likelihood(const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing data log likelihood.");
        std::unique_ptr<DistributionBase> model = distribution_->clone();
        double log_lh = stationary_data_log_likelihood(*model, p);  // IsNonstationary: M9
        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwiseDataLogLikelihood` override (line 1373).
    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing pointwise log likelihood.");
        std::unique_ptr<DistributionBase> model = distribution_->clone();
        return stationary_pointwise_log_likelihood(*model, p);  // IsNonstationary: M9
    }

    // C# `PointwiseDataLogLikelihoodComponents` override (line 1385).
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing pointwise log likelihood components.");
        std::unique_ptr<DistributionBase> model = distribution_->clone();
        return stationary_pointwise_log_likelihood_components(*model, p);  // IsNonstationary: M9
    }

    // C# `Prior_LogLikelihood(model, parameters)` (line 1822): parameter priors + the
    // Jeffreys 1/scale term read off the WORKING COPY's parameters (not the raw proposal --
    // this is why the callers set `model` first). The quantile-prior terms (1853-1875) are
    // M9 and unreachable while EnableQuantilePriors is false.
    double prior_log_likelihood(const DistributionBase& model,
                                const std::vector<double>& p) const {
        double log_lh = 0.0;

        // Parameter priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(p[i]);
        }

        if (use_jeffreys_rule_for_scale()) {
            std::vector<double> model_params = model.get_parameters();
            std::size_t scale_index = scale_parameter_index(model);
            // C# divergence (retained from Phase 4): for a genuine 1-parameter family C#
            // indexes GetParameters[1] unguarded and would throw; this port skips the
            // Jeffreys term instead (intentional, untested upstream -- see
            // docs/upstream-csharp-issues.md).
            if (scale_index < model_params.size()) {
                double scale_param = model_params[scale_index];
                // Jeffreys prior requires a positive scale parameter: return -inf directly
                // rather than subtracting +inf (C# early-return, line 1849).
                if (scale_param <= 0) return -std::numeric_limits<double>::infinity();
                log_lh -= std::log(scale_param);
            }
        }

        // Quantile priors: M9 (see method comment).

        if (!numerics::is_finite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood` override (line 1882): the per-parameter prior
    // components plus one JeffreysScale component -- appended only when the proposed
    // distribution parameters are VALID (C# `valid is null`, line 1917), unlike the scalar
    // Prior_LogLikelihood which evaluates the term unconditionally off the working copy.
    // The quantile-prior components (1938-1972) are M9. Component label: the C# suffixes
    // the scale ParameterNames entry ("Jeffreys Scale: {name}"); ParameterNames is not on
    // the ported distribution base, so the label stays "Jeffreys Scale" (display-only).
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& p) const override {
        if (distribution_ == nullptr)
            throw std::runtime_error(
                "Distribution must be set before computing pointwise prior log likelihood.");

        std::vector<PriorComponent> result;
        // Length guard (deviation, documented: the C# indexes parameters[i] unguarded and
        // would throw on a mismatch; empty mirrors the ModelBase base behavior).
        if (p.size() != parameters_.size()) return result;

        std::unique_ptr<DistributionBase> model = distribution_->clone();
        std::vector<double> dist_params(
            p.begin(), p.begin() + static_cast<std::ptrdiff_t>(model->number_of_parameters()));
        std::unique_ptr<DistributionBase> probe = distribution_->clone();
        probe->set_parameters(dist_params);
        bool valid = probe->parameters_valid();
        if (valid) model->set_parameters(dist_params);

        // Parameter priors.
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(p[i]);
            const std::string& param_name = parameters_[i].owner_name().empty()
                                                ? parameters_[i].name()
                                                : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll,
                                PriorComponentType::ParameterPrior);
        }

        // Jeffreys rule for scale parameter (only when the proposal is valid, per the C#).
        if (use_jeffreys_rule_for_scale() && valid) {
            std::vector<double> model_params = model->get_parameters();
            std::size_t scale_index = scale_parameter_index(*model);
            if (scale_index < model_params.size()) {  // 1-parameter-family guard (see above)
                double scale = model_params[scale_index];
                double ll =
                    scale > 0 ? -std::log(scale) : -std::numeric_limits<double>::infinity();
                result.emplace_back("Jeffreys Scale", ll, PriorComponentType::JeffreysScalePrior);
            }
        }

        // Quantile priors: M9 (see method comment).

        return result;
    }

    // C# `SetParameterValues` override (line 1977): writes the model parameters, updates
    // the trend models (M9), then sets the distribution parameters via
    // SetDistributionParameterValues(ParameterTimeIndex) -- stationary, the ConstantTrend
    // chain collapses to the parameter values themselves.
    void set_parameter_values(const std::vector<double>& p) override {
        if (p.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("The length of the parameter list is incorrect.");
        }
        for (std::size_t i = 0; i < p.size(); ++i) parameters_[i].set_value(p[i]);
        set_distribution_parameter_values(0);  // stationary ParameterTimeIndex (M9)
    }

    // C# `GetParameterValues(int index)` (line 2006): the distribution parameters at a time
    // step. Stationary: every trend model is the ConstantTrend whose Predict(index) is the
    // parameter value, so the index is unused until M9.
    std::vector<double> get_parameter_values(int /*index*/) const {
        std::vector<double> values(
            static_cast<std::size_t>(distribution_->number_of_parameters()));
        for (std::size_t i = 0; i < values.size(); ++i) values[i] = parameters_[i].value();
        return values;
    }

    // C# `SetDistributionParameterValues(int index)` (line 2020).
    void set_distribution_parameter_values(int index) {
        distribution_->set_parameters(get_parameter_values(index));
    }

    // C# `Validate` override (line 2127), stationary checks. The nonstationary block
    // (ParameterTimeIndex range, Alpha in (0,1); C# 2274-2293) is M9.
    ValidationResult validate() const override {
        ValidationResult result;

        // DataFrame checks (C# null -> error + early return; the C++ "null" is the
        // never-set optional -- see the base header's nullability note).
        if (!has_data_frame()) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The input DataFrame is null.");
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

        // Distribution checks (defensive; every C++ construction path sets it).
        if (distribution_ == nullptr) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The parent distribution is null.");
            return result;
        }

        // Distribution type check.
        if (!is_supported_distribution_type(distribution_type())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The distribution type is not supported.");
        }

        // Uncertain-data quadrature bounds at the 1E-8 probability window (C# 2165-2185).
        for (std::size_t i = 0; i < data_frame().uncertain_series().count(); ++i) {
            const UncertainData& data = data_frame().uncertain_series()[i];
            const DistributionBase& dist = data.distribution();
            double lower_probability = 1E-8;
            double upper_probability = 1.0 - 1E-8;
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

        // Log distribution support checks (C# 2187-2224).
        if (distribution_type() == DistributionType::LnNormal ||
            distribution_type() == DistributionType::LogNormal ||
            distribution_type() == DistributionType::LogPearsonTypeIII) {
            bool non_positive_exact = false;
            for (std::size_t i = 0; i < data_frame().exact_series().count(); ++i) {
                const ExactData& data = data_frame().exact_series()[i];
                if (!data.is_low_outlier() && data.value() <= 0.0) {
                    non_positive_exact = true;
                    break;
                }
            }

            bool non_positive_uncertain = false;
            for (std::size_t i = 0; i < data_frame().uncertain_series().count(); ++i) {
                const DistributionBase& dist = data_frame().uncertain_series()[i].distribution();
                double lower = dist.inverse_cdf(1E-8);
                // A positive ME mean is not enough; the retained 1E-8 lower support must be
                // positive.
                if (!numerics::is_finite(lower) || lower <= 0.0) {
                    non_positive_uncertain = true;
                    break;
                }
            }

            if (non_positive_exact || non_positive_uncertain) {
                result.is_valid = false;
                result.validation_messages.push_back(
                    "Error: Log based distributions cannot be used because some data values "
                    "are non positive.");
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

        // Quantile priors (C# 2241-2271): per-prior validation plus the strictly-decreasing
        // alpha / increasing mean / increasing 5-95 bound ordering rules.
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

        // Nonstationary checks (C# 2274-2293): M9.

        return result;
    }

   private:
    // Builds the exact-data frame the Phase 4 vector ctors wrap (sequential 0-based
    // indexes, exactly what ExactSeries(values) constructs).
    static DataFrame make_exact_data_frame(const std::vector<double>& values) {
        DataFrame df;
        df.set_exact_series(ExactSeries(values));
        return df;
    }

    // Index of the scale parameter for the Jeffreys 1/scale prior (C# Prior_LogLikelihood,
    // 1834-1843): Gamma/Weibull scale is parameter 0, every other family's is parameter 1.
    static std::size_t scale_parameter_index(const DistributionBase& model) {
        DistributionType type = model.type();
        return (type == DistributionType::GammaDistribution || type == DistributionType::Weibull)
                   ? 0
                   : 1;
    }

    // C# `StationaryPointwiseLogLikelihood` (line 1632).
    std::vector<double> stationary_pointwise_log_likelihood(DistributionBase& model,
                                                            const std::vector<double>& p) const {
        if (!has_data_frame())
            throw std::runtime_error(
                "DataFrame must be set before computing pointwise log likelihood.");

        const DataFrame& df = data_frame();

        // Invalid parameters -> one -inf per observation across the four series.
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(p);
        if (!probe->parameters_valid()) {
            std::size_t n = df.exact_series().count() + df.uncertain_series().count() +
                            df.interval_series().count() + df.threshold_series().count();
            return std::vector<double>(n, -std::numeric_limits<double>::infinity());
        }

        model.set_parameters(p);

        std::vector<double> result;

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
                result.push_back(ep > 0 ? std::log(ep)
                                        : -std::numeric_limits<double>::infinity());
            } else {
                result.push_back(-std::numeric_limits<double>::infinity());
            }
        }

        // Interval Data.
        for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
            const IntervalData& data = df.interval_series()[i];
            result.push_back(
                model.log_likelihood_intervals(data.lower_value(), data.upper_value()));
        }

        // Threshold Data -- each threshold contributes one observation.
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double log_lh = 0.0;
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh += model.log_likelihood_right_censored(data.value(), data.number_above());
            result.push_back(log_lh);
        }

        return result;
    }

    // C# `StationaryPointwiseLogLikelihoodComponents` (line 1402).
    std::vector<DataComponent> stationary_pointwise_log_likelihood_components(
        DistributionBase& model, const std::vector<double>& p) const {
        if (!has_data_frame())
            throw std::runtime_error(
                "DataFrame must be set before computing pointwise log likelihood components.");

        const DataFrame& df = data_frame();
        std::vector<DataComponent> result;
        int idx = 0;
        const double neg_inf = -std::numeric_limits<double>::infinity();

        // Invalid parameters -> list of invalid components with per-type placeholder values
        // (C# 1410-1431).
        std::unique_ptr<DistributionBase> probe = model.clone();
        probe->set_parameters(p);
        if (!probe->parameters_valid()) {
            for (std::size_t i = 0; i < df.exact_series().count(); ++i)
                result.emplace_back(idx++, neg_inf, df.exact_series()[i].value(),
                                    DataComponentType::Exact);
            for (std::size_t i = 0; i < df.uncertain_series().count(); ++i)
                result.emplace_back(idx++, neg_inf,
                                    df.uncertain_series()[i].distribution().mean(),
                                    DataComponentType::Uncertain);
            for (std::size_t i = 0; i < df.interval_series().count(); ++i) {
                const IntervalData& interval = df.interval_series()[i];
                result.emplace_back(idx++, neg_inf,
                                    (interval.lower_value() + interval.upper_value()) / 2.0,
                                    DataComponentType::Interval);
            }
            for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
                const ThresholdData& threshold = df.threshold_series()[i];
                result.emplace_back(idx++, neg_inf, threshold.value(),
                                    DataComponentType::LeftCensored,
                                    threshold.number_below() + threshold.number_above(),
                                    std::to_string(threshold.start_index()) + "-" +
                                        std::to_string(threshold.end_index()));
            }
            return result;
        }

        model.set_parameters(p);

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

        // Threshold Data (LeftCensored type with the combined count, per the C#).
        for (std::size_t i = 0; i < df.threshold_series().count(); ++i) {
            const ThresholdData& data = df.threshold_series()[i];
            double log_lh = 0.0;
            if (data.number_below() > 0)
                log_lh += model.log_likelihood_left_censored(data.value(), data.number_below());
            if (data.number_above() > 0)
                log_lh += model.log_likelihood_right_censored(data.value(), data.number_above());

            int total_count = data.number_below() + data.number_above();
            result.emplace_back(idx++, log_lh, data.value(), DataComponentType::LeftCensored,
                                total_count,
                                std::to_string(data.start_index()) + "-" +
                                    std::to_string(data.end_index()));
        }

        return result;
    }

    std::unique_ptr<DistributionBase> distribution_;
};

}  // namespace bestfit::models
