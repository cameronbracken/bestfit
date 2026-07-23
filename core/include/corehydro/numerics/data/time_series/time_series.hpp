// ported from: Numerics/Data/Time Series/TimeSeries.cs @ 2a0357a
//
// A THIN adapter over the Numerics `TimeSeries : Series<DateTime,double>` (2,334 lines). Only
// the surface the ported ModelBase families consume is ported -- the AR/MA/ARIMA/ARIMAX
// TimeSeries models (RMC-BestFit Models/TimeSeries/*.cs) and the RatingCurve model
// (Models/RatingCurve/RatingCurve.cs). The full container is OUT OF SCOPE.
//
// Index representation. The C# ordinate index is `DateTime`. Scouting the model consumers
// shows the index is NEVER used in arithmetic: the TimeSeries models touch only `.Value`,
// `Count`, `operator[]`, `Clone`, `ValuesToArray`, `ValuesToList`, and the differenced arrays
// (confirmed AutoRegressive.cs: `ValuesToArray().Subset(...)`, `this[i].Value`,
// `this[i].Clone()`), and RatingCurve.GetAlignedObservations uses `[j].Index` only as a
// dictionary/equality JOIN KEY to inner-join stage vs discharge by date (RatingCurve.cs
// ~505-520). A plain integer index satisfies both readings, so the adapter uses
// `SeriesOrdinate<long, double>` (a day-count offset). NO calendar math is ported: the models
// never call AddTimeInterval / ShiftDates* / StartDate arithmetic on the fit/likelihood path,
// so the date-fill constructors below advance the integer index by +1 per step rather than by
// a calendar interval. If a future model path needs true calendar arithmetic, stop and add a
// dedicated date type rather than overloading this offset.
//
// DEFERRED (documented, not ported -- the full 2,334-line container is out of scope; only the
// model-consumed surface lands here):
//   - interpolation, file I/O, hypothesis tests
//   - ResampleWithBlockBootstrap / ResampleWithKNN
//   - the CollectionChanged / SuppressCollectionChanged auto-retrain event machinery and
//     INotifyPropertyChanged / WPF attributes
//   - XML (de)serialization (the XElement ctor / ToXElement); Clone() therefore does a direct
//     deep copy instead of the C#'s ToXElement round-trip
//   - calendar/AddTimeInterval / ShiftDates* / ConvertTimeInterval / MovingAverage /
//     MovingSum / CumulativeSum / smoothing / SortByTime
//   - MaxValue and the descriptive-stat extras not consumed by the models (only MinValue /
//     MeanValue / StandardDeviation are ported)
//   - the DateTime start/end/fixed-value and XElement constructors (the integer-index start
//     range ctor below covers the model-consumed shape)
#pragma once
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/series_ordinate.hpp"
#include "corehydro/numerics/data/time_series/support/time_interval.hpp"

namespace corehydro::numerics::data {

// A time-series: a collection of time-series ordinates. Mutable, mirroring the C# stateful
// binding API (the never-mutate rule is relaxed for these model/binding objects).
class TimeSeries {
   public:
    // The integer index type (a day-count offset; see the header note). C# uses DateTime.
    using IndexType = long;
    using Ordinate = SeriesOrdinate<IndexType, double>;

    // Constructs an empty time-series (C# TimeSeries() -- default interval OneDay per the C#
    // `_timeInterval` field initializer).
    TimeSeries() = default;

    // Constructs an empty time-series with a specified time interval (C# 38-41).
    explicit TimeSeries(TimeInterval time_interval) : time_interval_(time_interval) {}

    // Constructs an empty time-series spanning [start_index, end_index] filled with NaN
    // (C# 49-63, with the DateTime range replaced by a +1 integer walk; no AddTimeInterval).
    TimeSeries(TimeInterval time_interval, IndexType start_index, IndexType end_index)
        : time_interval_(time_interval) {
        if (time_interval == TimeInterval::Irregular)
            throw std::invalid_argument(
                "The time interval cannot be irregular with this constructor.");
        if (start_index > end_index)
            throw std::invalid_argument("Start date must be less than or equal to end date.");
        add(Ordinate(start_index, std::numeric_limits<double>::quiet_NaN()));
        while (ordinates_.back().index() < end_index)
            add(Ordinate(ordinates_.back().index() + 1,
                         std::numeric_limits<double>::quiet_NaN()));
    }

    // Constructs a time-series from a start index and a list of data values (C# 94-106, with
    // the DateTime walk replaced by a +1 integer walk). Used by Difference.
    TimeSeries(TimeInterval time_interval, IndexType start_index, const std::vector<double>& data)
        : time_interval_(time_interval) {
        if (time_interval == TimeInterval::Irregular)
            throw std::invalid_argument(
                "The time interval cannot be irregular with this constructor.");
        if (data.empty()) return;
        add(Ordinate(start_index, data[0]));
        for (std::size_t i = 1; i < data.size(); ++i)
            add(Ordinate(ordinates_[i - 1].index() + 1, data[i]));
    }

    // --- Series base surface (Series.cs). ---

    // The number of ordinates (C# Count).
    int count() const { return static_cast<int>(ordinates_.size()); }

    // Indexer -> a mutable ordinate reference (C# this[int]; models write this[i].Value = ...).
    Ordinate& operator[](int index) { return ordinates_[static_cast<std::size_t>(index)]; }
    const Ordinate& operator[](int index) const {
        return ordinates_[static_cast<std::size_t>(index)];
    }

    // Appends an ordinate (C# Series.Add). The CollectionChanged event is a deferred no-op.
    void add(const Ordinate& item) { ordinates_.push_back(item); }

    // Returns the series values as a vector (C# ValuesToArray / ValuesToList, both a projection
    // of the ordinate values; ported to one body since C++ has a single vector type).
    std::vector<double> values_to_array() const {
        std::vector<double> out;
        out.reserve(ordinates_.size());
        for (const auto& o : ordinates_) out.push_back(o.value());
        return out;
    }
    std::vector<double> values_to_list() const { return values_to_array(); }

    // The time interval (C# TimeInterval getter).
    TimeInterval time_interval() const { return time_interval_; }

    // Start index (C# StartDate: min index, or default when empty). With a monotonically
    // increasing integer index the first ordinate is the minimum.
    IndexType start_date() const {
        if (ordinates_.empty()) return IndexType{};
        IndexType m = ordinates_.front().index();
        for (const auto& o : ordinates_)
            if (o.index() < m) m = o.index();
        return m;
    }

    // First / last ordinate accessors (C# _seriesOrdinates.First()/Last()).
    Ordinate& first() { return ordinates_.front(); }
    Ordinate& last() { return ordinates_.back(); }

    // --- Summary statistics (TimeSeries.cs). ---

    // Min value, skipping NaN (C# 1338).
    double min_value() const {
        double min = std::numeric_limits<double>::max();
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v) && v < min) min = v;
        }
        return min;
    }

    // Mean of the values, skipping NaN (C# 1370).
    double mean_value() const {
        if (count() == 0) return std::numeric_limits<double>::quiet_NaN();
        double mean = 0.0;
        int n = 0;
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v)) {
                mean += v;
                n += 1;
            }
        }
        return mean / n;
    }

    // Sample standard deviation via the C# incremental algorithm, skipping NaN (C# 1389).
    double standard_deviation() const {
        if (count() < 2) return std::numeric_limits<double>::quiet_NaN();
        double variance = 0.0;
        double t = 0.0;
        int start_idx = 0;
        for (int i = 0; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v)) {
                t = v;
                start_idx = i + 1;
                break;
            }
        }
        double n = 1;
        for (int i = start_idx; i < count(); ++i) {
            double v = (*this)[i].value();
            if (!std::isnan(v)) {
                t += v;
                double diff = (i + 1) * v - t;
                variance += diff * diff / ((i + 1.0) * i);
                n += 1;
            }
        }
        return std::sqrt(variance / (n - 1));
    }

    // Returns a time-series of the successive differences (C# 496-515). Copies the value
    // array, repeats `differences` passes each computing temp[i] = result[i+lag] - result[i]
    // over result.size()-lag, throwing when result.size() <= lag; returns a new series.
    TimeSeries difference(int lag = 1, int differences = 1) const {
        std::vector<double> result = values_to_array();
        for (int d = 0; d < differences; ++d) {
            if (static_cast<int>(result.size()) <= lag)
                throw std::invalid_argument("The length of the array must be greater than the lag.");
            std::vector<double> temp(result.size() - static_cast<std::size_t>(lag));
            for (std::size_t i = 0; i < result.size() - static_cast<std::size_t>(lag); ++i)
                temp[i] = result[i + static_cast<std::size_t>(lag)] - result[i];
            result = temp;
        }
        return TimeSeries(time_interval_, start_date(), result);
    }

    // Creates a deep copy (C# 2329 Clone(), which round-trips through ToXElement; XML is
    // deferred, so this copies the interval and every ordinate directly).
    TimeSeries clone() const {
        TimeSeries out(time_interval_);
        out.ordinates_ = ordinates_;
        return out;
    }

   private:
    TimeInterval time_interval_ = TimeInterval::OneDay;
    std::vector<Ordinate> ordinates_;
};

}  // namespace corehydro::numerics::data
