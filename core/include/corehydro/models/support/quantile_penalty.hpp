// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/QuantilePenalty.cs @ fc28c0c
//
// A penalty term for a specific quantile in Bulletin 17C analysis. Quantile penalties
// incorporate prior information about specific flood quantiles (historical flood estimates,
// regional regression equations at specific return periods) into the Generalized Method of
// Moments estimation: the penalty adds a term proportional to (quantile - mean)^2 / (MSE * n)
// to the objective function. When UseLog10 is true, Mean and MSE are specified in log10 space
// (appropriate for flood discharges) and the penalty is computed there.
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
// holding the C#'s single message string). One formatting deviation: the AEP range message
// interpolates the AEP value via C#'s default double.ToString() (shortest round-trip); the
// port renders it with printf "%g" -- the upstream test asserts only the validity flag, not
// the rendered number.
#pragma once
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models {

class QuantilePenalty {
   public:
    // Initializes a new instance of the QuantilePenalty class (C# line 36).
    QuantilePenalty() = default;

    // --- Enabled: whether this penalty is applied during estimation. ---
    bool enabled() const { return enabled_; }
    void set_enabled(bool enabled) { enabled_ = enabled; }

    // --- AEP: the annual exceedance probability for this quantile (e.g. 0.01 for the 1% AEP
    // flood). Must be between 0 and 1. Default 0.01 (C# field initializer, line 79). ---
    double aep() const { return aep_; }
    void set_aep(double aep) { aep_ = aep; }

    // --- UseLog10: whether Mean and MSE are specified in log10 space. Common for flood
    // frequency analysis because discharges are typically log-normally distributed. ---
    bool use_log10() const { return use_log10_; }
    void set_use_log10(bool use_log10) { use_log10_ = use_log10; }

    // --- Mean: the prior mean value for the quantile (the expected log10 value when
    // UseLog10 is true). ---
    double mean() const { return mean_; }
    void set_mean(double mean) { mean_ = mean; }

    // --- MSE: the mean squared error (variance) of the prior (in log10 space when UseLog10
    // is true). ---
    double mse() const { return mse_; }
    void set_mse(double mse) { mse_ = mse; }

    // Gets the mean value in real space (C# `MeanValue`, line 195): Mean directly when
    // UseLog10 is false, else the log10 -> real-space lognormal transform with bias
    // correction.
    double mean_value() const {
        if (!use_log10()) return mean();

        double ln_b = std::log(10.0);
        return std::exp((mean() + 0.5 * mse() * ln_b) * ln_b);
    }

    // Gets the MSE value in real space (C# `MSEValue`, line 214): MSE directly when UseLog10
    // is false, else the log10-space variance -> real-space variance transform.
    double mse_value() const {
        if (!use_log10()) return mse();

        double ln_b = std::log(10.0);
        double a = mse() * ln_b;
        double log_prefactor = (2.0 * mean() + a) * ln_b;
        double exp_a = std::exp(a * ln_b);
        return std::exp(log_prefactor) * (exp_a - 1.0);
    }

    // Gets the 95th percentile of the quantile prior distribution (C# `UpperValue`).
    double upper_value() const {
        if (!use_log10())
            return mean() + numerics::distributions::Normal::standard_z(0.95) * std::sqrt(mse());
        return std::pow(10.0, mean() + numerics::distributions::Normal::standard_z(0.95) *
                                           std::sqrt(mse()));
    }

    // Gets the 5th percentile of the quantile prior distribution (C# `LowerValue`).
    double lower_value() const {
        if (!use_log10())
            return mean() + numerics::distributions::Normal::standard_z(0.05) * std::sqrt(mse());
        return std::pow(10.0, mean() + numerics::distributions::Normal::standard_z(0.05) *
                                           std::sqrt(mse()));
    }

    // Whether the penalty parameters are valid: AEP strictly inside (0, 1), Mean finite, MSE
    // positive and finite (C# `IsValid`, line 267).
    bool is_valid() const {
        return aep() > 0.0 && aep() < 1.0 &&                //
               numerics::is_finite(mean()) &&               //
               numerics::is_finite(mse()) && mse() > 0.0;
    }

    // Computes the penalty function value for a given quantile value (C# `Function`,
    // line 293): (1/2)(quantile - Mean)^2 / (MSE * n), performed in log10 space when
    // UseLog10 is true (via Tools.Log10 = clamped_log10, after the non-positive guard).
    // Returns 0 if the penalty is not enabled, parameters are invalid, or sampleSize <= 0.
    // The 1/2 factor is the half-quadratic convention used in penalized GMM.
    double function(double quantile_value, int sample_size) const {
        if (!enabled() || !is_valid() || sample_size <= 0) return 0.0;

        double value;
        if (use_log10()) {
            // Guard against non-positive values for log10
            if (quantile_value <= 0.0) return 0.0;
            value = numerics::clamped_log10(quantile_value);
        } else {
            value = quantile_value;
        }
        return 0.5 * numerics::sqr(value - mean()) / (mse() * sample_size);
    }

    // Validates the penalty configuration (C# `Validate`, line 319; tuple -> the shared
    // ValidationResult, message strings verbatim -- see the header note on AEP formatting).
    ValidationResult validate() const {
        ValidationResult result;
        if (!enabled()) return result;

        if (aep() <= 0.0 || aep() >= 1.0) {
            result.is_valid = false;
            char aep_text[32];
            std::snprintf(aep_text, sizeof(aep_text), "%g", aep());
            result.validation_messages.push_back(
                std::string("Quantile penalty: AEP must be between 0 and 1 (got ") + aep_text +
                ").");
            return result;
        }
        if (std::isnan(mean()) || std::isinf(mean())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Quantile penalty: Mean value is not a valid number.");
            return result;
        }
        if (std::isnan(mse()) || std::isinf(mse())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Quantile penalty: MSE is not a valid number.");
            return result;
        }
        if (mse() <= 0.0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Quantile penalty: MSE must be greater than zero.");
            return result;
        }
        return result;
    }

    // Creates a deep copy of this penalty (C# `Clone`, line 343).
    QuantilePenalty clone() const {
        QuantilePenalty copy;
        copy.set_enabled(enabled());
        copy.set_use_log10(use_log10());
        copy.set_aep(aep());
        copy.set_mean(mean());
        copy.set_mse(mse());
        return copy;
    }

   private:
    bool enabled_ = false;
    bool use_log10_ = false;
    double aep_ = 0.01;
    double mean_ = std::numeric_limits<double>::quiet_NaN();
    double mse_ = std::numeric_limits<double>::quiet_NaN();
};

}  // namespace corehydro::models
