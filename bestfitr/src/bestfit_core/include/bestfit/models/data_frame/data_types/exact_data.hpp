// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataTypes/ExactData.cs @ fc28c0c
//
// Exact data ordinate: a precisely measured observation (systematic record).
//
// Deliberately NOT ported (project-wide deferrals, per the M3 brief):
//   - the DateTime member, the DateTime constructor overload, and the DateTime property
//     (seasonal path only; deferred project-wide)
//   - ToXElement() and the XElement constructor (XML serialization)
//   - INotifyPropertyChanged / PropertyChanged (the C# IsLowOutlier setter raises it)
#pragma once
#include "bestfit/models/data_frame/data_types/data.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::models {

class ExactData : public Data {
   public:
    // Construct an empty exact data ordinate (C# line 25).
    ExactData() = default;

    // Constructs a new exact data ordinate (C# line 48).
    ExactData(int index, double value, double plotting_position = 0.0,
              bool is_low_outlier = false)
        : Data(index, value, plotting_position), is_low_outlier_(is_low_outlier) {}

    // --- IsLowOutlier: whether the data value is a low outlier (C# line 93). ---
    bool is_low_outlier() const { return is_low_outlier_; }
    void set_is_low_outlier(bool is_low_outlier) { is_low_outlier_ = is_low_outlier; }

    // Validates the current state of the exact data and reports any issues found
    // (C# line 120).
    ValidationResult validate() const {
        ValidationResult result;

        if (index() < -100000 || index() > 100000) {
            result.is_valid = false;
            result.validation_messages.push_back(
                "Error: The index must be between -100,000 and +100,000.");
        }

        if (!numerics::is_finite(value())) {
            result.is_valid = false;
            result.validation_messages.push_back("Error: The value must be a number.");
        }

        return result;
    }

    // Returns a copy of the data ordinate (C# line 143; the DateTime carry-over is not
    // ported). Hides SeriesOrdinate::clone(), like the C# `new virtual`.
    ExactData clone() const {
        return ExactData(index(), value(), plotting_position(), is_low_outlier());
    }

   private:
    bool is_low_outlier_ = false;
};

}  // namespace bestfit::models
