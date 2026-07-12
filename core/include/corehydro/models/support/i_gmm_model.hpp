// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/IGMMModel.cs @ fc28c0c
//
// Model interface for the Generalized Method of Moments (GMM) estimation method -- the GMM
// analog of IModel. Where IModel defines likelihood-based estimation contracts
// (LogLikelihood, DataLogLikelihood, ...), IGMMModel defines moment-based estimation
// contracts (MomentConditionFunction, PointwiseMomentConditions, ...). The C# interface
// ports as an abstract class, the codebase's convention for C# capability interfaces (see
// models/support/i_univariate_model.hpp).
//
// The delegate accessors return the std::function aliases from estimation/gmm_delegates.hpp
// BY VALUE: the C# properties are computed getters (Bulletin17CDistribution returns a method
// group / a stored field), and a value return lets implementations build a bound closure on
// demand. The three C# nullable delegates (`JacobianFunction?` / `PenaltyFunction?` /
// `PointwiseMomentConditionFunction?`) use the EMPTY std::function as the null state,
// matching the gmm_delegates.hpp convention; MomentConditionFunction is required (the GMM
// constructor throws if it is empty, mirroring the C# null check).
//
// Clone(): the C# `IGMMModel Clone()` reference return ports as a
// std::unique_ptr<IGMMModel> factory, per the repo's value-types + unique_ptr convention.
//
// Deliberately NOT ported (project-wide deferrals): the INotifyPropertyChanged base
// interface and ToXElement() (XML serialization).
#pragma once
#include <memory>
#include <vector>

#include "corehydro/estimation/gmm_delegates.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/validation_result.hpp"

namespace corehydro::models {

class IGMMModel {
   public:
    virtual ~IGMMModel() = default;

    // The list of model parameters (C# `List<ModelParameter> Parameters { get; }`).
    // Mutable + const accessors, matching the ModelBase::parameters() precedent.
    virtual std::vector<ModelParameter>& parameters() = 0;
    virtual const std::vector<ModelParameter>& parameters() const = 0;

    // Returns the number of model parameters.
    virtual int number_of_parameters() const = 0;

    // Returns the number of moment conditions used in the GMM estimation.
    virtual int number_of_moment_conditions() const = 0;

    // Gets the total sample size (number of observations) used in GMM estimation.
    virtual int sample_size() const = 0;

    // Gets the moment condition function that returns the sample mean of moment conditions
    // and their covariance matrix. Required: must not be empty.
    virtual estimation::MomentConditionFunction moment_condition_function() const = 0;

    // Gets the optional analytical Jacobian function (dg/dtheta). Returns the empty
    // std::function if no analytical Jacobian is available (numerical differentiation
    // will be used).
    virtual estimation::JacobianFunction jacobian_function() const = 0;

    // Gets the optional penalty function for regularization. Returns the empty
    // std::function if no penalty is used.
    virtual estimation::PenaltyFunction penalty_function() const = 0;

    // Gets the optional pointwise moment condition function that returns per-observation
    // moment condition vectors: an [n x q] matrix whose row i is observation i's
    // contribution g_i(theta); the sample mean of all rows equals the G vector from
    // moment_condition_function(). Required for GMM influence diagnostics (observation
    // influence, Cook's distance). Returns the empty std::function if pointwise
    // decomposition is not available for this model.
    virtual estimation::PointwiseMomentConditionFunction pointwise_moment_conditions()
        const = 0;

    // Set the model parameter values (C# `SetParameterValues(IList<double>)`).
    virtual void set_parameter_values(const std::vector<double>& parameters) = 0;

    // Set the default parameters and priors for this model.
    virtual void set_default_parameters() = 0;

    // Return a deep copy of the model (C# `IGMMModel Clone()`; see the header note).
    virtual std::unique_ptr<IGMMModel> clone() const = 0;

    // Validates the current state of the object and reports any issues found (C#
    // `(bool IsValid, List<string> ValidationMessages) Validate()` -> the shared
    // ValidationResult).
    virtual ValidationResult validate() const = 0;
};

}  // namespace corehydro::models
