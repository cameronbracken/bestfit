// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/ModelParameter.cs @ fc28c0c
//
// A model parameter: a named value with bounds, positivity/fixed flags, and a prior
// distribution. Unlike DataComponent/PriorComponent, the C# `ModelParameter` is a mutable
// WPF binding object (`INotifyPropertyChanged`); this port keeps it mutable with plain
// getters/setters (the repo's "never mutate" rule is relaxed for these model objects, per
// `.claude/CLAUDE.md`) but drops the WPF plumbing itself.
//
// Deliberately NOT ported (desktop-app / XML / WPF concerns, not called by the Estimation
// layer this class exists to support):
//   - INotifyPropertyChanged / PropertyChanged / RaisePropertyChange
//   - ToXElement() and the XElement constructor
//   - Validate() (bounds/prior/positivity validation returning (bool, List<string>))
//
// Ownership of PriorDistribution: the C# property holds a plain `UnivariateDistributionBase`
// reference with reference semantics (assigning `PriorDistribution = x` aliases `x`; cloning
// is a separate, explicit operation the caller doesn't get for free). C++ has no polymorphic
// value type to mirror that directly, so ModelParameter instead takes SOLE OWNERSHIP of its
// prior via `std::unique_ptr<UnivariateDistributionBase>`, moved in at construction (or via
// `set_prior_distribution`, also move-only) -- the caller transfers, not shares, the
// distribution object. Given ownership, ModelParameter needs copy semantics of its own: it
// defines a copy constructor and copy-assignment operator that DEEP-COPY the prior via
// `clone()`, so `ModelParameter b = a;` yields a `b` whose prior is independently mutable
// from `a`'s. Move construction/assignment are defaulted (defining the copy special members
// suppresses the implicit ones). `clone()` is provided as an explicit, C#-signature-mirroring
// alternative to the copy constructor for callers that prefer a named method. This matches
// the brief's guidance: a model will hold `std::vector<ModelParameter>`, which needs the type
// to be at least movable, and deep-copy-on-copy is the safest default for forward use by the
// Estimation/Models layers that build on this.
#pragma once
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"

namespace bestfit::models {

class ModelParameter {
   public:
    // Constructs an empty model parameter (C# line 26): default bounds and a
    // Uniform(LowerBound, UpperBound) prior.
    ModelParameter()
        : prior_distribution_(
              std::make_unique<numerics::distributions::Uniform>(lower_bound_, upper_bound_)) {}

    // Full constructor (C# line 42). `prior_distribution` is moved in; ModelParameter takes
    // sole ownership (see header comment).
    ModelParameter(std::string owner_name, std::string name, double value, double lower_bound,
                   double upper_bound,
                   std::unique_ptr<numerics::distributions::UnivariateDistributionBase> prior_distribution,
                   bool is_positive = false, bool is_fixed = false)
        : owner_name_(std::move(owner_name)),
          name_(std::move(name)),
          value_(value),
          lower_bound_(lower_bound),
          upper_bound_(upper_bound),
          is_positive_(is_positive),
          is_fixed_(is_fixed),
          prior_distribution_(std::move(prior_distribution)) {}

    // Deep-copies the prior (see header comment on ownership).
    ModelParameter(const ModelParameter& other)
        : owner_name_(other.owner_name_),
          name_(other.name_),
          value_(other.value_),
          lower_bound_(other.lower_bound_),
          upper_bound_(other.upper_bound_),
          is_positive_(other.is_positive_),
          is_fixed_(other.is_fixed_),
          prior_distribution_(other.prior_distribution_ ? other.prior_distribution_->clone() : nullptr) {}

    ModelParameter& operator=(const ModelParameter& other) {
        if (this == &other) return *this;
        owner_name_ = other.owner_name_;
        name_ = other.name_;
        value_ = other.value_;
        lower_bound_ = other.lower_bound_;
        upper_bound_ = other.upper_bound_;
        is_positive_ = other.is_positive_;
        is_fixed_ = other.is_fixed_;
        prior_distribution_ = other.prior_distribution_ ? other.prior_distribution_->clone() : nullptr;
        return *this;
    }

    ModelParameter(ModelParameter&&) noexcept = default;
    ModelParameter& operator=(ModelParameter&&) noexcept = default;
    ~ModelParameter() = default;

    // Create a copy of the model parameter (C# `Clone()`), with an independently-owned deep
    // copy of the prior distribution.
    ModelParameter clone() const { return ModelParameter(*this); }

    // --- OwnerName: the name of the owning model component. ---
    const std::string& owner_name() const { return owner_name_; }
    void set_owner_name(std::string owner_name) { owner_name_ = std::move(owner_name); }

    // --- Name: the parameter name. ---
    const std::string& name() const { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    // --- DisplayName (derived, C# line 131): no owner -> Name; no name -> OwnerName;
    // both -> "OwnerName Name". ---
    std::string display_name() const {
        if (owner_name_.empty()) return name_;
        if (name_.empty()) return owner_name_;
        return owner_name_ + " " + name_;
    }

    // --- Value: the parameter value (point estimate). ---
    double value() const { return value_; }
    void set_value(double value) { value_ = value; }

    // --- LowerBound: default Double.MinValue -> std::numeric_limits<double>::lowest(). ---
    double lower_bound() const { return lower_bound_; }
    void set_lower_bound(double lower_bound) { lower_bound_ = lower_bound; }

    // --- UpperBound: default Double.MaxValue -> std::numeric_limits<double>::max(). ---
    double upper_bound() const { return upper_bound_; }
    void set_upper_bound(double upper_bound) { upper_bound_ = upper_bound; }

    // --- IsPositive: must the parameter be strictly greater than zero. ---
    bool is_positive() const { return is_positive_; }
    void set_is_positive(bool is_positive) { is_positive_ = is_positive; }

    // --- IsFixed: held fixed during model estimation. ---
    bool is_fixed() const { return is_fixed_; }
    void set_is_fixed(bool is_fixed) { is_fixed_ = is_fixed; }

    // --- PriorDistribution: sole ownership (see header comment). ---
    numerics::distributions::UnivariateDistributionBase& prior_distribution() { return *prior_distribution_; }
    const numerics::distributions::UnivariateDistributionBase& prior_distribution() const {
        return *prior_distribution_;
    }
    void set_prior_distribution(
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> prior_distribution) {
        prior_distribution_ = std::move(prior_distribution);
    }

   private:
    std::string owner_name_ = "";
    std::string name_ = "Parameter";
    double value_ = 0.0;
    double lower_bound_ = std::numeric_limits<double>::lowest();
    double upper_bound_ = std::numeric_limits<double>::max();
    bool is_positive_ = false;
    bool is_fixed_ = false;
    std::unique_ptr<numerics::distributions::UnivariateDistributionBase> prior_distribution_;
};

}  // namespace bestfit::models
