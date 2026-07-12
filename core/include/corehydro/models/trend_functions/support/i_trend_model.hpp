// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/Support/ITrendModel.cs @ fc28c0c
//
// Interface for parameter trend models used to describe how a distribution parameter
// varies with time (index). The nonstationary UnivariateDistribution (M9) consumes these
// via `TrendModels`/`SetTrendModel`.
//
// Deliberately NOT ported (XML serialization is out of scope for this port):
//   - ToXElement()
//
// C++ mapping notes:
//   - C# `List<ModelParameter> Parameters { get; }` returns a reference to the mutable
//     list; the C++ port exposes `parameters()` as a mutable (and a const) reference to
//     `std::vector<ModelParameter>` for the same effect.
//   - C# `ITrendModel Clone()` maps to `std::unique_ptr<ITrendModel> clone() const`
//     (polymorphic deep copy; ownership is explicit in C++).
//   - Getters are const; C# has no const-correctness to mirror.
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/trend_functions/support/trend_model_type.hpp"

namespace corehydro::models::trend_functions {

class ITrendModel {
   public:
    virtual ~ITrendModel() = default;

    // Gets or sets the name of the owning distribution parameter. Typically matches one
    // of the parent distribution parameter names.
    virtual const std::string& owner_name() const = 0;
    virtual void set_owner_name(const std::string& value) = 0;

    // Gets the trend model function type.
    virtual TrendModelType type() const = 0;

    // Gets or sets the starting time index for the trend. This is usually set to the
    // first index of the full time series to improve numerical conditioning in the trend
    // function.
    virtual int start_index() const = 0;
    virtual void set_start_index(int value) = 0;

    // Gets the list of model parameters that define the trend. The number of parameters
    // is given by number_of_parameters().
    virtual std::vector<ModelParameter>& parameters() = 0;
    virtual const std::vector<ModelParameter>& parameters() const = 0;

    // Gets the number of model parameters. This is a convenience wrapper for
    // Parameters.Count.
    virtual int number_of_parameters() const = 0;

    // Gets or sets a value indicating whether default "flat" priors should be applied
    // when initializing the trend parameters. When set to true, set_default_parameters()
    // is typically called to reset parameter definitions.
    virtual bool use_default_flat_priors() const = 0;
    virtual void set_use_default_flat_priors(bool value) = 0;

    // Sets the model parameter values from the supplied list. The length must match
    // number_of_parameters().
    virtual void set_parameter_values(const std::vector<double>& parameters) = 0;

    // Set the default parameters and priors for this model.
    virtual void set_default_parameters() = 0;

    // Evaluates the trend function at the specified time index.
    virtual double predict(int index) const = 0;

    // Creates a deep copy of the trend model, including its parameters.
    virtual std::unique_ptr<ITrendModel> clone() const = 0;
};

}  // namespace corehydro::models::trend_functions
