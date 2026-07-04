// ported from: RMC.BestFit/Models/Support/{IModel,ModelBase}.cs @ fc28c0c
//
// C++ convention (per .claude/CLAUDE.md): this codebase collapses "pure interface + abstract
// base" into ONE abstract base class (mirroring how UnivariateDistributionBase plays both
// IUnivariateDistribution and the base). So IModel + ModelBase port here as a SINGLE abstract
// class, bestfit::models::ModelBase. Estimators (T7-T9) take a `ModelBase&`; concrete models
// (T6+) derive from it and implement the pure-virtual data-likelihood surface.
//
// Scope of this slice (T5): the state (parameters_, UseDefaultFlatPriors) and the compute
// defaults ModelBase.cs supplies bodies for -- LogLikelihood (C# 104-110), PriorLogLikelihood
// (C# 122-134), PointwisePriorLogLikelihood (C# 137-156), SetParameterValues (C# 159-164) --
// plus the pure-virtual data-likelihood surface concrete models implement in T6:
// DataLogLikelihood, PointwiseDataLogLikelihood, PointwiseDataLogLikelihoodComponents,
// SetDefaultParameters.
//
// Deliberately NOT ported in this slice (desktop-app / XML / WPF concerns; the Estimation
// layer never calls these on IModel):
//   - IModel.Clone() -- deep-copy semantics belong with the concrete models, which know
//     their own non-ModelBase state (every Phase 5 concrete model, M9-M12, implements its
//     own clone()).
//   - ToXElement() -- XML serialization, no compute-layer caller.
//   - INotifyPropertyChanged / PropertyChanged / RaisePropertyChange / Parameter_PropertyChanged
//     -- WPF data-binding plumbing; `parameters_` is a plain std::vector here, no change
//     notification is threaded through it. XML/INPC remain project-wide non-ports.
// Validate() (deferred in the Phase 4 slice) is ported as of M8: the C# abstract
// `(bool IsValid, List<string> ValidationMessages) Validate()` is the pure-virtual
// `validate()` below, returning the shared ValidationResult
// (models/support/validation_result.hpp).
//
// UseDefaultFlatPriors (C# 56-70): ported as a plain bool getter/setter for forward
// compatibility with concrete models (T6+) that read it, but this base class does NOT wire it
// into any compute path -- the C# setter's side effect (calling SetDefaultParameters() when
// re-enabled) is a concrete-model concern deferred to T6, since ModelBase's
// SetDefaultParameters is pure virtual here and calling it from a base-class setter during
// construction of a not-yet-fully-constructed derived object would be unsafe. Note this only:
// no simulated side effect is ported in this slice.
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/support/prior_component.hpp"
#include "bestfit/models/support/validation_result.hpp"

namespace bestfit::models {

class ModelBase {
   public:
    virtual ~ModelBase() = default;

    // --- Parameters: mutable + const accessors (C# `Parameters` property, C# 30-51). ---
    // Estimators read bounds/values off elements; SetParameterValues writes .value(). The
    // setter-with-PropertyChanged-rewiring from the C# property is not ported (see header
    // comment on INotifyPropertyChanged).
    std::vector<ModelParameter>& parameters() { return parameters_; }
    const std::vector<ModelParameter>& parameters() const { return parameters_; }

    // C# `NumberOfParameters => Parameters.Count` (C# 54); C# `int`, so this returns `int`.
    int number_of_parameters() const { return static_cast<int>(parameters_.size()); }

    // C# `UseDefaultFlatPriors` (C# 59-70): plain property here, default `true` matching the
    // C# field initializer `_useDefaultFlatPriors = true`. See header comment: not wired into
    // any compute path in this base.
    bool use_default_flat_priors() const { return use_default_flat_priors_; }
    void set_use_default_flat_priors(bool use_default_flat_priors) {
        use_default_flat_priors_ = use_default_flat_priors;
    }

    // --- Compute: virtual defaults mirroring ModelBase.cs bodies. ---

    // C# `LogLikelihood(double[] parameters)` (C# 104-110): data + prior; -inf if non-finite.
    //
    // MUTABLE-PARAMETER SEMANTICS (M14, C#-fidelity): C#'s likelihood surface takes
    // `double[] parameters`, and .NET arrays are reference types -- a model MAY write back
    // into the caller's array. RMC.BestFit's MixtureModel does exactly that: its
    // DataLogLikelihood and PriorLogLikelihood call `Mixture.SetParameters(ref parameters)`,
    // which normalizes the weight entries IN PLACE, re-projecting the optimizer's (and the
    // numerical differentiator's) own working vectors onto the normalized-weights manifold
    // and thereby steering the search path/finite differences. So `log_likelihood`,
    // `data_log_likelihood`, and `prior_log_likelihood` take a NON-CONST reference here; the
    // pointwise variants keep const (the C# pointwise methods normalize a private
    // `parmsCopy`, never the caller's array). Data-then-prior evaluation order is explicit
    // below because for a mutating model the prior must see the data call's write-back,
    // exactly like the C# left-to-right `DataLogLikelihood(parameters) +
    // PriorLogLikelihood(parameters)`.
    virtual double log_likelihood(std::vector<double>& p) const {
        double data_log_lh = data_log_likelihood(p);
        double log_lh = data_log_lh + prior_log_likelihood(p);
        if (!std::isfinite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PriorLogLikelihood(double[] parameters)` (C# 122-134): -inf on length mismatch;
    // else sum of per-parameter prior LogPDF; -inf if the sum is non-finite. Non-const `p`:
    // see the MUTABLE-PARAMETER note above (MixtureModel's override writes back).
    virtual double prior_log_likelihood(std::vector<double>& p) const {
        if (p.size() != parameters_.size()) return -std::numeric_limits<double>::infinity();

        double log_lh = 0.0;
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            log_lh += parameters_[i].prior_distribution().log_pdf(p[i]);
        }
        if (!std::isfinite(log_lh)) return -std::numeric_limits<double>::infinity();
        return log_lh;
    }

    // C# `PointwisePriorLogLikelihood(double[] parameters)` (C# 137-156): empty on length
    // mismatch; else one PriorComponent per parameter, named from OwnerName (falling back to
    // Name when OwnerName is empty, matching the C# ternary).
    virtual std::vector<PriorComponent> pointwise_prior_log_likelihood(
        const std::vector<double>& p) const {
        std::vector<PriorComponent> result;
        if (p.size() != parameters_.size()) return result;

        result.reserve(parameters_.size());
        for (std::size_t i = 0; i < parameters_.size(); ++i) {
            double ll = parameters_[i].prior_distribution().log_pdf(p[i]);
            const std::string& param_name =
                parameters_[i].owner_name().empty() ? parameters_[i].name() : parameters_[i].owner_name();
            result.emplace_back("Parameter Prior: " + param_name, ll, PriorComponentType::ParameterPrior);
        }
        return result;
    }

    // C# `SetParameterValues(IList<double> parameters)` (C# 159-164): throws on length
    // mismatch (C# ArgumentException -> std::invalid_argument), else writes each value.
    virtual void set_parameter_values(const std::vector<double>& p) {
        if (p.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("The list of parameter values are the wrong length");
        }
        for (int i = 0; i < number_of_parameters(); ++i) {
            parameters_[static_cast<std::size_t>(i)].set_value(p[static_cast<std::size_t>(i)]);
        }
    }

    // --- Pure virtual: concrete models (T6+) implement these. ---

    // C# `DataLogLikelihood(double[] parameters)`. Non-const `p`: see the MUTABLE-PARAMETER
    // note above (MixtureModel's override writes back).
    virtual double data_log_likelihood(std::vector<double>& p) const = 0;

    // C# `PointwiseDataLogLikelihood(double[] parameters)`.
    virtual std::vector<double> pointwise_data_log_likelihood(const std::vector<double>& p) const = 0;

    // C# `PointwiseDataLogLikelihoodComponents(double[] parameters)`.
    virtual std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& p) const = 0;

    // C# `SetDefaultParameters()`.
    virtual void set_default_parameters() = 0;

    // C# `Validate()` (ModelBase.cs line 173, abstract): validates the current state of the
    // model and reports any issues found.
    virtual ValidationResult validate() const = 0;

   protected:
    std::vector<ModelParameter> parameters_;
    bool use_default_flat_priors_ = true;
};

}  // namespace bestfit::models
