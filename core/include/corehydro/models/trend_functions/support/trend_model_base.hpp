// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/Support/TrendModelBase.cs @ fc28c0c
//
// Base class for trend models that provides common storage. Logic mirrors the C# source
// member-for-member.
//
// Deliberately NOT ported (per the porting policy for this layer):
//   - the XElement constructor and ToXElement() (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged / RaisePropertyChange (WPF binding plumbing);
//     the setters keep the C# value-changed guards but the notification calls are dropped.
//
// C++ mapping notes:
//   - The C# parameterless base constructor calls SetDefaultParameters() (a virtual).
//     C++ cannot dispatch to a derived override during base construction, so each
//     concrete trend's default constructor calls set_default_parameters() itself.
//   - C# `Parameters { protected set }` does `_parameters = value ?? new List<...>()`.
//     A std::vector cannot be null, so the null branch is unrepresentable; assigning an
//     empty vector is the equivalent reset.
//   - C# `SetParameterValues(IList<double>)` throws ArgumentNullException for null (a
//     const reference cannot be null in C++, so that check is unrepresentable) and
//     ArgumentException for a wrong-length list -> std::invalid_argument here.
#pragma once
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/trend_functions/support/i_trend_model.hpp"
#include "corehydro/models/trend_functions/support/trend_model_type.hpp"

namespace corehydro::models::trend_functions {

class TrendModelBase : public ITrendModel {
   public:
    // --- OwnerName ---
    const std::string& owner_name() const override { return owner_name_; }
    void set_owner_name(const std::string& value) override {
        if (owner_name_ != value) {
            owner_name_ = value;
            for (std::size_t i = 0; i < parameters_.size(); i++) {
                parameters_[i].set_owner_name(owner_name_);
            }
            // RaisePropertyChange(nameof(OwnerName)) skipped (INPC not ported).
        }
    }

    // --- Type (abstract) ---
    TrendModelType type() const override = 0;

    // --- StartIndex ---
    int start_index() const override { return start_index_; }
    void set_start_index(int value) override {
        if (start_index_ != value) {
            start_index_ = value;
            // RaisePropertyChange(nameof(StartIndex)) skipped (INPC not ported).
        }
    }

    // --- Parameters ---
    std::vector<ModelParameter>& parameters() override { return parameters_; }
    const std::vector<ModelParameter>& parameters() const override { return parameters_; }

    // --- NumberOfParameters ---
    int number_of_parameters() const override { return static_cast<int>(parameters_.size()); }

    // --- UseDefaultFlatPriors ---
    bool use_default_flat_priors() const override { return use_default_flat_priors_; }
    void set_use_default_flat_priors(bool value) override {
        if (use_default_flat_priors_ != value) {
            use_default_flat_priors_ = value;
            // RaisePropertyChange(nameof(UseDefaultFlatPriors)) skipped (INPC not ported).
            if (use_default_flat_priors_) {
                set_default_parameters();
            }
        }
    }

    // --- SetDefaultParameters (abstract) ---
    void set_default_parameters() override = 0;

    // --- SetParameterValues ---
    void set_parameter_values(const std::vector<double>& parameters) override {
        // C#'s ArgumentNullException branch is unrepresentable (const reference).
        if (parameters.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("The list of parameter values has the wrong length.");
        }
        for (std::size_t i = 0; i < parameters.size(); i++) {
            parameters_[i].set_value(parameters[i]);
        }
        // RaisePropertyChange(nameof(Parameters)) skipped (INPC not ported).
    }

    // --- Predict / Clone (abstract) ---
    double predict(int index) const override = 0;
    std::unique_ptr<ITrendModel> clone() const override = 0;

   protected:
    // See the header note: concrete trends call set_default_parameters() in their own
    // default constructors (C++ cannot virtual-dispatch from a base constructor).
    TrendModelBase() = default;

    // C# `Parameters { protected set }`: replaces the parameter list wholesale (the
    // `value ?? new List<...>()` null branch maps to assigning an empty vector).
    void set_parameters(std::vector<ModelParameter> value) { parameters_ = std::move(value); }

    std::string owner_name_ = "";
    bool use_default_flat_priors_ = true;
    int start_index_ = 0;
    std::vector<ModelParameter> parameters_;
};

}  // namespace corehydro::models::trend_functions
