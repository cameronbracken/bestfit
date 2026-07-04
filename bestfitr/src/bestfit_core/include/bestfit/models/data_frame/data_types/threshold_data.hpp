// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataTypes/ThresholdData.cs @ fc28c0c
//
// Threshold censored data ordinate: a perception threshold over the window
// [StartIndex, EndIndex] with counts of observations above/below the threshold value.
//
// Deliberately NOT ported (project-wide deferrals):
//   - ToXElement() and the XElement constructor (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged (the C# bound setters raise it)
//
// NumberBelow: in C# the setter is `internal` -- a derived value computed by
// DataFrame.ProcessThresholdSeries as Duration - NumberAbove - (overlapping data points),
// reserved for DataFrame and XML deserialization; users only set NumberAbove. C++ has no
// `internal`, so set_number_below() is public here with the same intent: it exists for
// DataFrame (M4) and for tests, not for end users.
#pragma once
#include "bestfit/models/data_frame/data_types/data.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::models {

class ThresholdData : public Data {
   public:
    // Construct an empty threshold censored data ordinate (C# line 23).
    ThresholdData() = default;

    // Constructs a new threshold censored data ordinate (C# line 31). The base
    // constructor call makes Index track StartIndex.
    ThresholdData(int start_index, int end_index, double value)
        : Data(start_index, value), start_index_(start_index), end_index_(end_index) {}

    // --- StartIndex: the start index of the threshold (C# line 67). The C# setter also
    // assigns the inherited _index. ---
    int start_index() const { return start_index_; }
    void set_start_index(int start_index) {
        index_ = start_index;
        start_index_ = start_index;
    }

    // --- EndIndex: the end index of the threshold (C# line 84). ---
    int end_index() const { return end_index_; }
    void set_end_index(int end_index) { end_index_ = end_index; }

    // --- NumberBelow: the number of data points below the threshold during the threshold
    // window (C# line 110; derived value, see the header note). ---
    int number_below() const { return number_below_; }
    void set_number_below(int number_below) { number_below_ = number_below; }

    // --- NumberAbove: the number of data points above the threshold (C# line 119). ---
    int number_above() const { return number_above_; }
    void set_number_above(int number_above) { number_above_ = number_above; }

    // Returns the duration of the threshold (C# line 135): EndIndex - StartIndex + 1.
    int duration() const { return end_index() - start_index() + 1; }

    // Validates the current state of the threshold data and reports any issues found
    // (C# line 154).
    ValidationResult validate() const {
        ValidationResult result;

        if (index() < -100000 || index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The index must be between -100,000 and +100,000.");
        }

        if (start_index() < -100000 || start_index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The start index must be between -100,000 and +100,000.");
        }

        if (end_index() < -100000 || end_index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The end index must be between -100,000 and +100,000.");
        }

        if (start_index() > end_index()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The start index must be less than or equal to the end index.");
        }

        if (!numerics::is_finite(value())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The value must be a number.");
        }

        if (number_below() < 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The number below cannot be negative.");
        }

        if (number_above() < 0) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The number above cannot be negative.");
        }

        if (number_above() > duration()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The number above must be less than or equal to the threshold "
                "duration.");
        }

        if (number_below() + number_above() > duration()) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The sum of number above and number below cannot exceed the "
                "threshold duration.");
        }

        return result;
    }

    // Returns a copy of the data ordinate (C# line 219: object initializer over the
    // three-argument constructor). Hides SeriesOrdinate::clone(), like the C# `new virtual`.
    ThresholdData clone() const {
        ThresholdData copy(start_index(), end_index(), value());
        copy.set_number_below(number_below());
        copy.set_number_above(number_above());
        copy.set_plotting_position(plotting_position());
        return copy;
    }

   private:
    int start_index_ = 0;
    int end_index_ = 0;
    int number_below_ = 0;
    int number_above_ = 0;
};

}  // namespace bestfit::models
