// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataTypes/Data.cs @ fc28c0c
//
// Abstract base class for censored data ordinates: adds the plotting position and the
// log10 / standardized transforms to SeriesOrdinate<int, double>. Like the C# original
// this is a mutable model object (plain getters/setters; the repo's "never mutate" rule
// is relaxed for these, per .claude/CLAUDE.md).
//
// Deliberately NOT ported (project-wide deferrals -- desktop-app concerns):
//   - INotifyPropertyChanged / PropertyChanged (the C# PlottingPosition setter raises it)
//
// Faithful C# quirks kept on purpose:
//   - PlottingPosition clamps to [0, 1] in the GETTER; the backing field stores the raw
//     assigned value (Data.cs line 42).
//   - Log10Value: Value < 0 -> NaN; Value == 0 -> Math.Log10(0.001); else Math.Log10(Value)
//     (plain Math.Log10 here, NOT Tools.Log10 -- that split exists in the C# source).
//     It reads the virtual Value property, so UncertainData's override flows through.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>

#include "corehydro/numerics/data/series_ordinate.hpp"

namespace corehydro::models {

class Data : public numerics::data::SeriesOrdinate<int, double> {
   public:
    virtual ~Data() = default;

    // The plotting position of the data ordinate; the getter clamps to [0, 1] (C# line 42).
    double plotting_position() const {
        return std::max(0.0, std::min(1.0, plotting_position_));
    }
    void set_plotting_position(double plotting_position) {
        plotting_position_ = plotting_position;
    }

    // Returns the complement of the plotting position (C# line 58).
    double plotting_position_complement() const { return 1.0 - plotting_position(); }

    // Returns the log base 10 transform of the data value (C# line 66).
    double log10_value() const {
        double v = value();
        if (v < 0.0) return std::numeric_limits<double>::quiet_NaN();
        return v == 0.0 ? std::log10(0.001) : std::log10(v);
    }

    // --- StandardizedValue (C# auto-property, line 72). ---
    double standardized_value() const { return standardized_value_; }
    void set_standardized_value(double standardized_value) {
        standardized_value_ = standardized_value;
    }

    // --- StandardizedLog10Value (C# auto-property, line 77). ---
    double standardized_log10_value() const { return standardized_log10_value_; }
    void set_standardized_log10_value(double standardized_log10_value) {
        standardized_log10_value_ = standardized_log10_value;
    }

   protected:
    // The C# class is abstract; protected constructors keep this one non-instantiable.
    Data() = default;

    // Constructs a new censored data ordinate (C# line 28).
    Data(int index, double value, double plotting_position = 0.0)
        : SeriesOrdinate<int, double>(index, value), plotting_position_(plotting_position) {}

    // Protected backing field (C# `_plottingPosition`).
    double plotting_position_ = 0.0;

   private:
    double standardized_value_ = 0.0;
    double standardized_log10_value_ = 0.0;
};

}  // namespace corehydro::models
