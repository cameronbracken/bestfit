// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/ParameterPenalty.cs @ fc28c0c
//
// A penalty term for a distribution parameter in Bulletin 17C analysis. Parameter penalties
// incorporate prior information about distribution parameters (e.g. regional skewness) into
// the Generalized Method of Moments estimation: the penalty adds a term proportional to
// (parameter - mean)^2 / (MSE * n) to the objective function, pulling the estimate toward the
// prior mean with strength inversely proportional to the prior variance (MSE).
//
// Like the C# original this is a mutable model object (plain getters/setters; the repo's
// "never mutate" rule is relaxed for these, per .claude/CLAUDE.md). The C# setters carry
// `if (_x != value)` guards purely to gate PropertyChanged events; with INPC dropped they
// port as plain assignments.
//
// Deliberately NOT ported (project-wide deferrals -- desktop-app / XML / WPF concerns):
//   - INotifyPropertyChanged / RaisePropertyChanged (all C# property setters raise it)
//   - ToXElement() and the XElement constructor
//
// Validate(): the C# returns the anonymous tuple `(bool IsValid, string Message)`; the port
// returns the shared corehydro::models::ValidationResult (message list empty when valid, else
// holding the C#'s single message string verbatim).
#pragma once
#include <cmath>
#include <limits>
#include <string>

#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class ParameterPenalty {
   public:
    // Initializes a new instance of the ParameterPenalty class (C# line 36).
    ParameterPenalty() = default;

    // --- Enabled: whether this penalty is applied during estimation. ---
    bool enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // --- Name: the name of the parameter being penalized (e.g. "Skewness"). ---
    const std::string& name() const { return name_; }
    void set_name(std::string name) { name_ = std::move(name); }

    // --- Mean: the prior mean value for the parameter (e.g. regional skewness). ---
    double mean() const { return mean_; }
    void set_mean(double mean) { mean_ = mean; }

    // --- MSE: the mean squared error (variance) of the prior. Larger values indicate
    // greater uncertainty in the prior and result in weaker regularization. ---
    double mse() const { return mse_; }
    void set_mse(double mse) { mse_ = mse; }

    // --- UseLog: whether the penalty is computed in natural log space. Mean and MSE are
    // always specified in real space regardless of this setting; function() converts via the
    // delta method (ln(theta0) for the mean, MSE/theta0^2 for the variance), equivalent to a
    // lognormal prior. Appropriate for scale parameters bounded below by zero. ---
    bool use_log() const { return use_log_; }
    void set_use_log(bool use_log) { use_log_ = use_log; }

    // Gets the 95th percentile of the parameter prior distribution (C# `UpperValue`).
    double upper_value() const {
        if (!use_log())
            return mean() + numerics::distributions::Normal::standard_z(0.95) * std::sqrt(mse());
        // Lognormal upper bound: exp(ln(Mean) + z95 * sqrt(MSE/Mean^2))
        double log_sd = std::sqrt(mse()) / mean();
        return mean() * std::exp(numerics::distributions::Normal::standard_z(0.95) * log_sd);
    }

    // Gets the 5th percentile of the parameter prior distribution (C# `LowerValue`).
    double lower_value() const {
        if (!use_log())
            return mean() + numerics::distributions::Normal::standard_z(0.05) * std::sqrt(mse());
        // Lognormal lower bound: exp(ln(Mean) + z05 * sqrt(MSE/Mean^2))
        double log_sd = std::sqrt(mse()) / mean();
        return mean() * std::exp(numerics::distributions::Normal::standard_z(0.05) * log_sd);
    }

    // Whether the penalty parameters are valid: Mean finite, MSE positive and finite, and
    // (under UseLog) Mean positive (C# `IsValid`, line 234).
    bool is_valid() const {
        return numerics::is_finite(mean()) &&                          //
               numerics::is_finite(mse()) && mse() > 0.0 &&            //
               (!use_log() || mean() > 0.0);
    }

    // Computes the penalty function value for a given parameter value in real space
    // (C# `Function`, line 273). Real space: (1/2)(theta - theta0)^2 / (MSE * n). Under
    // UseLog, Mean/MSE are converted to log space via the delta method and the penalty is
    // (1/2)(ln(theta) - ln(theta0))^2 / ((MSE/theta0^2) * n). Returns 0 if the penalty is
    // not enabled, parameters are invalid, sampleSize <= 0, or parameterValue <= 0 under
    // UseLog. The 1/2 factor is the half-quadratic convention used in penalized GMM.
    double function(double parameter_value, int sample_size) const {
        if (!enabled() || !is_valid() || sample_size <= 0) return 0.0;

        if (use_log()) {
            // Guard against non-positive values for log transform
            if (parameter_value <= 0.0 || mean() <= 0.0) return 0.0;

            // Convert real-space Mean/MSE to log-space via delta method:
            // Var(ln(X)) ~= Var(X) / E[X]^2
            double log_param = std::log(parameter_value);
            double log_mean = std::log(mean());
            double log_mse = mse() / (mean() * mean());
            return 0.5 * numerics::sqr(log_param - log_mean) / (log_mse * sample_size);
        } else {
            return 0.5 * numerics::sqr(parameter_value - mean()) / (mse() * sample_size);
        }
    }

    // Validates the penalty configuration (C# `Validate`, line 303; tuple -> the shared
    // ValidationResult, message strings verbatim).
    ValidationResult validate() const {
        ValidationResult result;
        if (!enabled()) return result;

        if (std::isnan(mean()) || std::isinf(mean())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Parameter penalty '" + name() + "': Mean value is not a valid number.");
            return result;
        }
        if (std::isnan(mse()) || std::isinf(mse())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Parameter penalty '" + name() + "': MSE is not a valid number.");
            return result;
        }
        if (mse() <= 0.0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Parameter penalty '" + name() + "': MSE must be greater than zero.");
            return result;
        }
        if (use_log() && mean() <= 0.0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Parameter penalty '" + name() +
                "': Mean must be positive when UseLog is enabled.");
            return result;
        }
        return result;
    }

    // Creates a deep copy of this penalty (C# `Clone`, line 327).
    ParameterPenalty clone() const {
        ParameterPenalty copy;
        copy.set_enabled(enabled());
        copy.set_name(name());
        copy.set_use_log(use_log());
        copy.set_mean(mean());
        copy.set_mse(mse());
        return copy;
    }

   private:
    bool enabled_ = false;
    bool use_log_ = false;
    std::string name_;
    double mean_ = std::numeric_limits<double>::quiet_NaN();
    double mse_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace corehydro::models
