// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataTypes/IntervalData.cs @ fc28c0c
//
// Interval censored data ordinate: an observation with known bounds but an uncertain exact
// value (Value is the most-likely value inside [LowerValue, UpperValue]).
//
// Deliberately NOT ported (project-wide deferrals):
//   - ToXElement() and the XElement constructor (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged (both C# bound setters raise it)
//
// Log10Lower/UpperValue use Tools.Log10 (the clamped variant), NOT Math.Log10 -- that split
// exists in the C# source (contrast Data.Log10Value).
#pragma once
#include "bestfit/models/data_frame/data_types/data.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::models {

class IntervalData : public Data {
   public:
    // Construct an empty interval censored data ordinate (C# line 24).
    IntervalData() = default;

    // Constructs a new interval censored data ordinate (C# line 34).
    IntervalData(int index, double lower_value, double value, double upper_value,
                 double plotting_position = 0.0)
        : Data(index, value, plotting_position),
          lower_value_(lower_value),
          upper_value_(upper_value) {}

    // --- LowerValue: the lower bound of the interval (C# line 67). ---
    double lower_value() const { return lower_value_; }
    void set_lower_value(double lower_value) { lower_value_ = lower_value; }

    // --- UpperValue: the upper bound of the interval (C# line 83). ---
    double upper_value() const { return upper_value_; }
    void set_upper_value(double upper_value) { upper_value_ = upper_value; }

    // Returns the log base 10 transform of the upper value (C# line 99, Tools.Log10).
    double log10_upper_value() const { return numerics::clamped_log10(upper_value()); }

    // Returns the log base 10 transform of the lower value (C# line 104, Tools.Log10).
    double log10_lower_value() const { return numerics::clamped_log10(lower_value()); }

    // Validates the current state of the interval data and reports any issues found
    // (C# line 120).
    ValidationResult validate() const {
        ValidationResult result;

        if (index() < -100000 || index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The index must be between -100,000 and +100,000.");
        }

        if (!numerics::is_finite(lower_value())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The lower value must be a number.");
        }

        if (!numerics::is_finite(value())) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The most likely value must be a number.");
        }

        if (!numerics::is_finite(upper_value())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The upper value must be a number.");
        }

        if (lower_value() >= value()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The lower value must be less than the most likely value.");
        }

        if (value() >= upper_value()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The upper value must be greater than the most likely value.");
        }

        if (lower_value() >= upper_value()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The lower value must be less than the upper value.");
        }

        return result;
    }

    // Returns a copy of the data ordinate (C# line 173). Hides SeriesOrdinate::clone(),
    // like the C# `new virtual`.
    IntervalData clone() const {
        return IntervalData(index(), lower_value(), value(), upper_value(),
                            plotting_position());
    }

   private:
    double lower_value_ = 0.0;
    double upper_value_ = 0.0;
};

}  // namespace bestfit::models
