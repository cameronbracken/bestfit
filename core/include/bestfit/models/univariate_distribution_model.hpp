// ported from: RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/UnivariateDistribution.cs
// @ fc28c0c -- STATIONARY, EXACT-DATA-ONLY subset (Phase 4, Task T6).
//
// Minimal `ModelBase`-derived model: a distribution plus a plain `std::vector<double>` of
// exact observations. This is the smallest slice of the real C# `UnivariateDistribution`
// model needed so the MLE/MAP/Bayesian estimators (T7-T9) have something to run on, and so
// the T12 oracle emitter (which drives the REAL C# model) can be reproduced: this class
// MUST match the C# model's parameter BOUNDS, initial VALUES, default PRIORS, and
// DataLogLikelihood value for exact stationary data.
//
// Ported methods (exact-stationary branch only):
//   - `StationaryData_LogLikelihood` (C# line 1181, exact-data branch ~1195-1207): validate
//     params on a clone -> -inf if invalid; else set params and sum `LogLikelihood(value)`
//     (== log_pdf(value)) over the exact series. Wrapped by `DataLogLikelihood` (1361)'s
//     clone + finite-guard -> `data_log_likelihood` below.
//   - `StationaryPointwiseLogLikelihood` (1632, exact-data branch): invalid params -> a
//     vector of -inf the same length as the data; else one `log_pdf(value)` per value.
//     Wrapped by `PointwiseDataLogLikelihood` (1373) -> `pointwise_data_log_likelihood`.
//   - `StationaryPointwiseLogLikelihoodComponents` (1402, exact-data branch): invalid params
//     -> all components -inf; else one `DataComponent(idx, ll, value, Exact)` per value.
//     Wrapped by `PointwiseDataLogLikelihoodComponents` (1385) ->
//     `pointwise_data_log_likelihood_components`.
//   - `SetDefaultParameters` (571), stationary path: `((IMaximumLikelihoodEstimation)
//     Distribution).GetParameterConstraints(exactValues)` returns (initials, lowers,
//     uppers); one `ModelParameter` per distribution parameter with Value=initials[i],
//     LowerBound=lowers[i], UpperBound=uppers[i], PriorDistribution=Uniform(lowers[i],
//     uppers[i]); the distribution itself is then set to `initials`. Ported as
//     `set_default_parameters`.
//
// Validity-check mechanism: the C# `model.ValidateParameters(parameters, false)` returns
// null when valid / a message when invalid. The C++ distribution base has no equivalent
// return-a-message method; instead every concrete distribution's `set_parameters` sets the
// protected `parameters_valid_` flag (see normal.hpp's `set_parameters`), exposed read-only
// via `UnivariateDistributionBase::parameters_valid()`. This port replicates the C# check by
// calling `set_parameters` on a clone and then reading `parameters_valid()` -- functionally
// identical (both "attempt to set, see if the result is valid"), just inverted control flow
// (C# validates BEFORE setting; this validates AFTER setting, on a throwaway clone, so no
// caller-visible state is ever left in an invalid configuration).
//
// Deliberately NOT ported in this slice (see the brief; desktop-app / not-yet-scoped
// concerns):
//   - Nonstationary / TrendModels / time-index / link functions (IsNonstationary branch,
//     GetParameterValues, SetTrendModel, ParameterTimeIndex).
//   - Uncertain / interval / threshold / low-outlier data and the full `DataFrame` type --
//     this model holds only a plain `std::vector<double>` of exact observations
//     (`exact_data_`); the full DataFrame (Phase 6) is a separate, larger port.
//   - `ISimulatable`, `IUnivariateModel`, XML (de)serialization, `INotifyPropertyChanged`,
//     quantile priors (`EnableQuantilePriors`, `UseSingleQuantile`, `QuantilePriors`).
//   - Parameter/owner display names (C# clears `Name` to "" for the stationary case and
//     wires `OwnerName` from `Distribution.ParameterNames`, which has no C++ port); these are
//     WPF-display concerns with no effect on the compute surface this class exists for, so
//     `ModelParameter::owner_name()`/`name()` are left at their default (empty) values here.
//
// Divergence note: `Normal::set_parameters` clamps a tiny/non-negative scale up to 1E-16
// rather than marking it invalid (mirroring the C# `Normal.SetParameters` clamp). A proposed
// sigma in (0, 1E-16) is therefore silently rounded up rather than rejected on either side of
// the port, so this does not change oracle reproducibility -- both C# and this C++ port
// clamp identically.
#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/numerics/distributions/base/i_maximum_likelihood_estimation.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_factory.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"

namespace bestfit::models {

class UnivariateDistributionModel : public ModelBase {
   public:
    // Builds the distribution via the factory (C# `UnivariateDistribution(DataFrame,
    // UnivariateDistributionType)` ctor, line 63). `exact_data` must be non-empty: default
    // parameters are derived from it (mirrors the C# assumption that
    // `DataFrame.ExactSeries.Count > 0` before `GetParameterConstraints` is called).
    UnivariateDistributionModel(numerics::distributions::UnivariateDistributionType type,
                                 std::vector<double> exact_data)
        : distribution_(numerics::distributions::create_distribution(type)),
          exact_data_(std::move(exact_data)) {
        set_default_parameters();
    }

    // Builds from an already-constructed distribution (C# `UnivariateDistribution(DataFrame,
    // UnivariateDistributionBase)` ctor, line 46). Takes sole ownership of `distribution`.
    UnivariateDistributionModel(
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution,
        std::vector<double> exact_data)
        : distribution_(std::move(distribution)), exact_data_(std::move(exact_data)) {
        set_default_parameters();
    }

    // --- Accessors ---
    numerics::distributions::UnivariateDistributionBase& distribution() { return *distribution_; }
    const numerics::distributions::UnivariateDistributionBase& distribution() const {
        return *distribution_;
    }
    numerics::distributions::UnivariateDistributionType distribution_type() const {
        return distribution_->type();
    }
    const std::vector<double>& exact_data() const { return exact_data_; }

    // Jeffreys 1/scale prior toggle (C# `UseJeffreysRuleForScale`,
    // UnivariateDistributionModelBase.cs:60 -- defaults TRUE). See `prior_log_likelihood`.
    bool use_jeffreys_rule_for_scale() const { return use_jeffreys_rule_for_scale_; }
    void set_use_jeffreys_rule_for_scale(bool value) { use_jeffreys_rule_for_scale_ = value; }

    // C# `DataLogLikelihood` (1361) -> `StationaryData_LogLikelihood` (1181), exact-data
    // branch only.
    double data_log_likelihood(const std::vector<double>& p) const override {
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> model =
            distribution_->clone();
        model->set_parameters(p);
        if (!model->parameters_valid()) return -std::numeric_limits<double>::infinity();

        double log_lh = 0.0;
        for (double value : exact_data_) log_lh += model->log_pdf(value);
        if (!std::isfinite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwiseDataLogLikelihood` (1373) -> `StationaryPointwiseLogLikelihood` (1632),
    // exact-data branch only.
    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>& p) const override {
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> model =
            distribution_->clone();
        model->set_parameters(p);

        std::vector<double> result;
        result.reserve(exact_data_.size());
        if (!model->parameters_valid()) {
            result.assign(exact_data_.size(), -std::numeric_limits<double>::infinity());
            return result;
        }
        for (double value : exact_data_) result.push_back(model->log_pdf(value));
        return result;
    }

    // C# `PointwiseDataLogLikelihoodComponents` (1385) ->
    // `StationaryPointwiseLogLikelihoodComponents` (1402), exact-data branch only.
    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& p) const override {
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> model =
            distribution_->clone();
        model->set_parameters(p);
        bool valid = model->parameters_valid();

        std::vector<DataComponent> result;
        result.reserve(exact_data_.size());
        for (std::size_t i = 0; i < exact_data_.size(); ++i) {
            double ll =
                valid ? model->log_pdf(exact_data_[i]) : -std::numeric_limits<double>::infinity();
            result.emplace_back(static_cast<int>(i), ll, exact_data_[i], DataComponentType::Exact);
        }
        return result;
    }

    // C# `Prior_LogLikelihood` (UnivariateDistribution.cs:1822), reached via this model's
    // `LogLikelihood` override (C# 1116-1132: dataLL + Prior_LogLikelihood, collapsed to -inf if
    // non-finite) and its `PriorLogLikelihood` override (C# 1143-1172). The base
    // `ModelBase::prior_log_likelihood` only sums the per-parameter priors; the real univariate
    // model ALSO applies a Jeffreys 1/scale prior on the scale parameter when
    // `UseJeffreysRuleForScale` is set (true by default), which pulls the scale down relative to
    // the pure MLE -- this is exactly why the MAP/Bayesian oracle values diverge from the MLE
    // (see fixtures/estimation/map_normal.json). Quantile priors are NOT applied
    // (`EnableQuantilePriors` defaults false; the quantile-prior surface is not ported -- T6).
    // Scale-parameter index follows the C# source: Gamma/Weibull use parameter 0, every other
    // family uses parameter 1.
    double prior_log_likelihood(const std::vector<double>& p) const override {
        if (p.size() != parameters_.size()) return -std::numeric_limits<double>::infinity();

        double log_lh = 0.0;
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(p[i]);
        }

        if (use_jeffreys_rule_for_scale_) {
            std::size_t scale_index = scale_parameter_index();
            if (scale_index < p.size()) {
                double scale = p[scale_index];
                // C# returns -inf directly for a non-positive scale (Jeffreys 1/scale requires
                // scale > 0), rather than subtracting +inf.
                if (scale <= 0.0) return -std::numeric_limits<double>::infinity();
                log_lh -= std::log(scale);
            }
        }

        if (!std::isfinite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood` (UnivariateDistribution.cs:1884): the per-parameter prior
    // components PLUS a single JeffreysScale component when `UseJeffreysRuleForScale` is set, so
    // the pointwise sum stays consistent with `prior_log_likelihood` above. (Not on any dumped
    // oracle path -- DIC/WAIC/LOOIC use the DATA pointwise likelihood -- but kept faithful so the
    // two prior surfaces agree.)
    std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& p) const override {
        std::vector<PriorComponent> result = ModelBase::pointwise_prior_log_likelihood(p);
        if (result.empty()) return result;  // length mismatch -> base returned empty

        if (use_jeffreys_rule_for_scale_) {
            std::size_t scale_index = scale_parameter_index();
            if (scale_index < p.size()) {
                double scale = p[scale_index];
                double ll = scale > 0.0 ? -std::log(scale) : -std::numeric_limits<double>::infinity();
                result.emplace_back("Jeffreys Scale", ll, PriorComponentType::JeffreysScalePrior);
            }
        }
        return result;
    }

    // C# `SetDefaultParameters` (571), stationary path only.
    void set_default_parameters() override {
        auto* ml_estimator = dynamic_cast<numerics::distributions::IMaximumLikelihoodEstimation*>(
            distribution_.get());
        if (ml_estimator == nullptr) {
            throw std::runtime_error(
                "UnivariateDistributionModel: distribution does not implement "
                "IMaximumLikelihoodEstimation (GetParameterConstraints unavailable)");
        }

        std::vector<double> initials;
        std::vector<double> lowers;
        std::vector<double> uppers;
        ml_estimator->get_parameter_constraints(exact_data_, initials, lowers, uppers);

        parameters_.clear();
        parameters_.reserve(initials.size());
        for (std::size_t i = 0; i < initials.size(); ++i) {
            parameters_.emplace_back(
                /*owner_name=*/"", /*name=*/"", initials[i], lowers[i], uppers[i],
                std::make_unique<numerics::distributions::Uniform>(lowers[i], uppers[i]));
        }

        distribution_->set_parameters(initials);
    }

   private:
    // Index of the scale parameter for the Jeffreys 1/scale prior (C# `Prior_LogLikelihood`,
    // ~1836-1843): Gamma/Weibull scale is parameter 0, every other family's is parameter 1.
    // C# divergence: for a genuine 1-parameter family (Poisson/Bernoulli/Geometric/Deterministic)
    // this returns an out-of-range index 1; C# indexes `GetParameters[1]` unguarded there and
    // would throw IndexOutOfRangeException if MAP/Bayesian ever ran against one, while the two
    // callers here guard `scale_index < p.size()` and silently skip the Jeffreys term instead
    // (intentional, untested -- see docs/upstream-csharp-issues.md).
    std::size_t scale_parameter_index() const {
        using T = numerics::distributions::UnivariateDistributionType;
        T type = distribution_->type();
        return (type == T::GammaDistribution || type == T::Weibull) ? 0 : 1;
    }

    std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution_;
    std::vector<double> exact_data_;
    bool use_jeffreys_rule_for_scale_ = true;
};

}  // namespace bestfit::models
