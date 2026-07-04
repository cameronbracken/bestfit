// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/IUnivariateModel.cs @ fc28c0c
//
// The minimal contract a univariate model must satisfy to serve as a marginal in a bivariate
// composite model. In C# this is implemented by UnivariateDistribution, Bulletin17CDistribution,
// PointProcessModel, and MixtureModel; it is intentionally minimal so classes with different
// base hierarchies can satisfy it without a base-class refactor. Ported as a pure-virtual
// mixin (the codebase's convention for C# capability interfaces -- see
// numerics/distributions/base/i_estimation.hpp); M8/M9 make UnivariateDistributionModel
// implement it.
//
// DataFrame: the C++ `DataFrame` class does not exist yet -- M4 defines it (in this same
// bestfit::models namespace). Only a forward declaration is needed here because the accessors
// return by reference; implementors written before M4 can only throw from them (and the C#
// property is non-nullable, so a reference -- not a pointer -- is the faithful mapping).
//
// Distribution: the C# property is the NULLABLE `UnivariateDistributionBase?` ("null if the
// model is not yet estimated"), so it ports as a raw non-owning pointer, nullptr meaning
// not-yet-estimated. Any type deriving from UnivariateDistributionBase is acceptable,
// including Mixture and CompetingRisks (what BivariateCopula's marginals expect).
#pragma once
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"

namespace bestfit::models {

class DataFrame;  // defined in M4 (Models/DataFrame port)

class IUnivariateModel {
   public:
    virtual ~IUnivariateModel() = default;

    // The observed data backing the marginal (C# `DataFrame DataFrame { get; }`).
    // Mutable + const accessors, matching the ModelBase::parameters() precedent.
    virtual DataFrame& data_frame() = 0;
    virtual const DataFrame& data_frame() const = 0;

    // The fitted Numerics univariate distribution, or nullptr if the model is not yet
    // estimated (C# `UnivariateDistributionBase? Distribution { get; }`). Non-owning.
    virtual const numerics::distributions::UnivariateDistributionBase* distribution() const = 0;

    // Whether the marginal carries a nonstationary trend on one or more parameters.
    // Models without a trend concept (Bulletin17C, Mixture) return false.
    virtual bool is_nonstationary() const = 0;

    // Standard model validation (C# `(bool IsValid, List<string> ValidationMessages) Validate()`).
    virtual ValidationResult validate() const = 0;
};

}  // namespace bestfit::models
