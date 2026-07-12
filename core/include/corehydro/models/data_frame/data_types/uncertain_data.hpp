// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataTypes/UncertainData.cs @ fc28c0c
//
// Uncertain data ordinate: an observation represented by a full measurement-error
// distribution. Value is pinned to the distribution mean; LowerValue/UpperValue are the
// 5th/95th percentiles.
//
// Deliberately NOT ported (project-wide deferrals):
//   - ToXElement() and the XElement constructor (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged (the C# Distribution/Value setters raise it)
//   - the Debug.WriteLine warning inside the C# Value setter (the no-op semantics ARE ported)
//
// Ownership of Distribution: same treatment as QuantilePrior's Distribution (see
// models/support/quantile_prior.hpp for the full rationale). The C# property holds a plain
// reference; C++ has no polymorphic value type, so UncertainData takes SOLE OWNERSHIP via
// `std::unique_ptr<UnivariateDistributionBase>`, moved in at construction or via
// set_distribution. Copy construction/assignment DEEP-COPY the distribution via clone(),
// so containers of UncertainData copy independently; clone() mirrors the C# `Clone()`
// (which explicitly deep-copies via Distribution.Clone()).
//
// Deviation forced by the C++ port surface: Validate()'s distribution check maps the C#
// `Distribution.ValidateParameters(Distribution.GetParameters, false) != null` to the
// distribution base's `parameters_valid()` flag, as in QuantilePrior.
#pragma once
#include <memory>
#include <utility>

#include "corehydro/models/data_frame/data_types/data.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_base.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class UncertainData : public Data {
   public:
    // Construct an empty uncertain data ordinate (C# line 24): the field initializer gives
    // it a standard Normal distribution (C# `new Normal()`).
    UncertainData() : distribution_(std::make_unique<numerics::distributions::Normal>()) {}

    // Constructs a new uncertain data ordinate (C# line 32): the base constructor receives
    // distribution.Mean as the value. `distribution` is moved in; UncertainData takes sole
    // ownership (see header comment).
    UncertainData(int index,
                  std::unique_ptr<numerics::distributions::UnivariateDistributionBase>
                      distribution,
                  double plotting_position = 0.0)
        : Data(index, distribution->mean(), plotting_position),
          distribution_(std::move(distribution)) {}

    // Deep-copies the distribution (see header comment on ownership).
    UncertainData(const UncertainData& other)
        : Data(other),
          distribution_(other.distribution_ ? other.distribution_->clone() : nullptr) {}

    UncertainData& operator=(const UncertainData& other) {
        if (this == &other) return *this;
        Data::operator=(other);
        distribution_ = other.distribution_ ? other.distribution_->clone() : nullptr;
        return *this;
    }

    UncertainData(UncertainData&&) noexcept = default;
    UncertainData& operator=(UncertainData&&) noexcept = default;
    ~UncertainData() override = default;

    // --- Distribution: the uncertain data distribution; sole ownership (see header).
    // The C# setter refreshes _value from the new distribution's mean (C# line 63). ---
    numerics::distributions::UnivariateDistributionBase& distribution() {
        return *distribution_;
    }
    const numerics::distributions::UnivariateDistributionBase& distribution() const {
        return *distribution_;
    }
    void set_distribution(
        std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution) {
        distribution_ = std::move(distribution);
        value_ = distribution_->mean();
    }

    // Gets the 95th percentile of the distribution (C# line 77).
    double upper_value() const { return distribution_->inverse_cdf(0.95); }

    // Gets the 5th percentile of the distribution (C# line 82).
    double lower_value() const { return distribution_->inverse_cdf(0.05); }

    // Returns the log base 10 transform of the upper value (C# line 87, Tools.Log10).
    double log10_upper_value() const { return numerics::clamped_log10(upper_value()); }

    // Returns the log base 10 transform of the lower value (C# line 92, Tools.Log10).
    double log10_lower_value() const { return numerics::clamped_log10(lower_value()); }

    // Gets the mean of the distribution (C# override, line 97). The setter is a no-op by
    // design: the value is determined by the distribution's mean -- use set_distribution
    // instead (C# line 100; the Debug.WriteLine warning is not ported).
    double value() const override { return distribution_->mean(); }
    void set_value(double /*value*/) override { value_ = distribution_->mean(); }

    // Validates the current state of the uncertain data and reports any issues found
    // (C# line 125).
    ValidationResult validate() const {
        ValidationResult result;

        if (index() < -100000 || index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The index must be between -100,000 and +100,000.");
        }

        // C#: Distribution.ValidateParameters(Distribution.GetParameters, false) != null
        // (see header comment on the parameters_valid() mapping).
        if (!distribution_->parameters_valid()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The distribution parameters are invalid.");
        }

        return result;
    }

    // Returns a deep copy of the data ordinate (C# line 149, which rebuilds from
    // (Index, Distribution.Clone(), PlottingPosition)). Hides SeriesOrdinate::clone(),
    // like the C# `new virtual`.
    UncertainData clone() const {
        return UncertainData(index(), distribution_->clone(), plotting_position());
    }

   private:
    std::unique_ptr<numerics::distributions::UnivariateDistributionBase> distribution_;
};

}  // namespace corehydro::models
