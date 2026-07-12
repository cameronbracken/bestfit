// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/SpatialCorrelation/ICorrelationModel.cs @ fc28c0c
//
// The contract every spatial correlation kernel satisfies (BasicExponential,
// PoweredExponential, Spherical). Ported as an abstract base class -- the codebase convention
// for C# capability interfaces (see models/support/i_univariate_model.hpp and
// models/support/i_gmm_model.hpp) -- rather than a header-only free interface.
//
// DRY decision (base-vs-impl): the C# repeats the identical `Parameters`, `NumberOfParameters`,
// and `SetParameterValues` bodies verbatim in all three concrete classes. Because the bodies are
// byte-identical and operate only on the shared `List<ModelParameter> Parameters`, this port
// hoists them onto the base: the base owns a protected `std::vector<ModelParameter> parameters_`
// (populated by each impl's constructor) and provides concrete `parameters()`,
// `number_of_parameters()`, and `set_parameter_values()`. Each impl then supplies only what
// genuinely differs -- `type()`, `evaluate()`, and `clone()` (pure virtual). This produces
// behavior identical to the C#, mirrors the ModelBase precedent (which likewise keeps
// `parameters()` / `number_of_parameters()` / `set_parameter_values()` on the base), and keeps
// the analytic leaf logic the only per-class code.
//
// Ownership: `clone()` returns `std::unique_ptr<ICorrelationModel>` (value-type + unique_ptr
// factory convention; the C# `ICorrelationModel Clone()` deep-copies). The never-mutate rule is
// RELAXED for these model objects -- they mirror the C# mutable stateful API.
//
// Deliberately NOT ported (project-wide convention): `ToXElement()` and the `Type`
// serialization-dispatch role (XML / serialization only).
//
// SetParameterValues: the C# `ArgumentNullException` null-guard is vacuous here -- the port
// takes a `const std::vector<double>&` (mirroring ModelBase::set_parameter_values), which
// cannot be null. Only the length guard is meaningful; it throws `std::invalid_argument`
// (the C# `ArgumentException`), matching ModelBase.
#pragma once
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

#include "corehydro/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "corehydro/models/support/model_parameter.hpp"

namespace corehydro::models::spatial_extremes {

class ICorrelationModel {
   public:
    virtual ~ICorrelationModel() = default;

    // The list of model parameters (C# `List<ModelParameter> Parameters`). Mutable + const
    // accessors, matching the ModelBase::parameters() precedent (downstream S2/S3 read and
    // write parameter values and bounds).
    std::vector<ModelParameter>& parameters() { return parameters_; }
    const std::vector<ModelParameter>& parameters() const { return parameters_; }

    // The number of model parameters (C# `NumberOfParameters => Parameters.Count`).
    int number_of_parameters() const { return static_cast<int>(parameters_.size()); }

    // The correlation model type identifier (C# `CorrelationFunctionType Type`).
    virtual CorrelationFunctionType type() const = 0;

    // Set the model parameter values (C# `SetParameterValues(IList<double> values)`):
    // length guard (C# ArgumentException -> std::invalid_argument), else per-index write.
    void set_parameter_values(const std::vector<double>& values) {
        if (values.size() != static_cast<std::size_t>(number_of_parameters())) {
            throw std::invalid_argument("The list of values has incorrect length.");
        }
        for (std::size_t i = 0; i < values.size(); ++i) {
            parameters_[i].set_value(values[i]);
        }
    }

    // Evaluate the correlation value rho(h) at distance h (C# `Evaluate(double h)`).
    virtual double evaluate(double h) const = 0;

    // Returns a deep copy of the correlation model (C# `ICorrelationModel Clone()`).
    virtual std::unique_ptr<ICorrelationModel> clone() const = 0;

   protected:
    std::vector<ModelParameter> parameters_;
};

}  // namespace corehydro::models::spatial_extremes
