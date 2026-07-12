// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/QuantilePrior.cs @ fc28c0c
//
// A quantile prior: an exceedance probability Alpha paired with a prior distribution on the
// corresponding quantile (e.g. "the 1% AEP flood is Normal(75000, 15000) cfs"). Like the C#
// original this is a mutable model object (plain getters/setters; the repo's "never mutate"
// rule is relaxed for these, per .claude/CLAUDE.md).
//
// Deliberately NOT ported (project-wide deferrals -- desktop-app / XML / WPF concerns):
//   - INotifyPropertyChanged / PropertyChanged (both C# property setters raise it)
//   - ToXElement() and the XElement constructor
//
// Ownership of Distribution: same treatment as ModelParameter's PriorDistribution (see
// models/support/model_parameter.hpp for the full rationale). The C# property holds a plain
// reference; C++ has no polymorphic value type, so QuantilePrior takes SOLE OWNERSHIP via
// `std::unique_ptr<UnivariateDistributionBase>`, moved in at construction or via
// set_distribution. Copy construction/assignment DEEP-COPY the distribution via clone(), so
// `std::vector<QuantilePrior>` elements (the IQuantilePriors list) copy independently;
// clone() mirrors the C# `Clone()` (which explicitly deep-copies via Distribution.Clone()).
//
// Deviations forced by the C++ port surface:
//   - Validate()'s distribution check: C# calls
//     `Distribution.ValidateParameters(Distribution.GetParameters, false) != null`; the C++
//     distribution base ports that surface as the `parameters_valid()` flag (set by every
//     set_parameters), so the check is `!distribution_->parameters_valid()`.
//   - The alpha error message spells "alpha" without the Greek letter (ASCII-only source);
//     the upstream test only requires the message to contain "alpha" or "exceedance".
#pragma once
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class QuantilePrior {
   public:
    // Empty constructor (C# line 26): Alpha 0 and a Uniform(double.MinValue, double.MaxValue)
    // prior, i.e. Uniform(lowest, max) here.
    QuantilePrior()
        : distribution_(std::make_unique<numerics::distributions::Uniform>(
              std::numeric_limits<double>::lowest(), std::numeric_limits<double>::max())) {}

    // Constructs a quantile prior (C# line 36). `distribution` is moved in; QuantilePrior
    // takes sole ownership (see header comment).
    QuantilePrior(double alpha,
                  std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution)
        : alpha_(alpha), distribution_(std::move(distribution)) {}

    // Deep-copies the distribution (see header comment on ownership).
    QuantilePrior(const QuantilePrior& other)
        : alpha_(other.alpha_),
          distribution_(other.distribution_ ? other.distribution_->clone() : nullptr) {}

    QuantilePrior& operator=(const QuantilePrior& other) {
        if (this == &other) return *this;
        alpha_ = other.alpha_;
        distribution_ = other.distribution_ ? other.distribution_->clone() : nullptr;
        return *this;
    }

    QuantilePrior(QuantilePrior&&) noexcept = default;
    QuantilePrior& operator=(QuantilePrior&&) noexcept = default;
    ~QuantilePrior() = default;

    // --- Alpha: the quantile exceedance probability. ---
    double alpha() const { return alpha_; }
    void set_alpha(double alpha) { alpha_ = alpha; }

    // --- Distribution: the quantile prior distribution; sole ownership (see header). ---
    numerics::distributions::UnivariateDistributionBase& distribution() { return *distribution_; }
    const numerics::distributions::UnivariateDistributionBase& distribution() const {
        return *distribution_;
    }
    void set_distribution(
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution) {
        distribution_ = std::move(distribution);
    }

    // Gets the 95th percentile of the distribution (C# `UpperValue`).
    double upper_value() const { return distribution_->inverse_cdf(0.95); }

    // Gets the 5th percentile of the distribution (C# `LowerValue`).
    double lower_value() const { return distribution_->inverse_cdf(0.05); }

    // Gets the mean of the distribution (C# `MeanValue`).
    double mean_value() const { return distribution_->mean(); }

    // Create a deep copy of the quantile prior (C# `Clone()`, which rebuilds from
    // (Alpha, Distribution.Clone()) -- exactly what the copy constructor does here).
    QuantilePrior clone() const { return QuantilePrior(*this); }

    // Validates the current state of the object and reports any issues found (C# line 139).
    ValidationResult validate() const {
        ValidationResult result;

        if (!numerics::is_finite(alpha_) || alpha_ <= 0.0 || alpha_ >= 1.0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The exceedance probability alpha must be between 0 and 1.");
        }
        // C#: Distribution.ValidateParameters(Distribution.GetParameters, false) != null
        // (see header comment on the parameters_valid() mapping).
        if (!distribution_->parameters_valid()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The quantile prior distribution is invalid.");
        }
        return result;
    }

   private:
    double alpha_ = 0.0;
    std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution_;
};

}  // namespace corehydro::models
