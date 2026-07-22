// ported from: Numerics/Data/Time Series/Support/SeriesOrdinate.cs @ 2a0357a
//
// A series ordinate: a minimal generic (Index, Value) pair. The BestFit DataFrame data
// types derive from SeriesOrdinate<int, double>; Phase 7's TimeSeries will reuse the same
// template with <DateTime, double>.
//
// Deliberately NOT ported (project-wide deferrals -- desktop-app concerns):
//   - INotifyPropertyChanged / PropertyChanged / RaisePropertyChanged (both C# property
//     setters raise it)
//   - GetHashCode (no hashing consumer in this port's scope)
// The C# Index/Value properties are virtual (UncertainData overrides Value); the accessors
// here are virtual for the same reason, and the destructor is virtual because the Data
// hierarchy is used polymorphically.
#pragma once

namespace corehydro::numerics::data {

template <typename TIndex, typename TValue>
class SeriesOrdinate {
   public:
    // Constructs a new series ordinate (C# line 23): value-initialized index and value
    // (the C# fields default to default(T)).
    SeriesOrdinate() = default;

    // Constructs a new series ordinate from an index and value (C# line 30).
    SeriesOrdinate(TIndex index, TValue value) : index_(index), value_(value) {}

    virtual ~SeriesOrdinate() = default;
    SeriesOrdinate(const SeriesOrdinate&) = default;
    SeriesOrdinate& operator=(const SeriesOrdinate&) = default;
    SeriesOrdinate(SeriesOrdinate&&) noexcept = default;
    SeriesOrdinate& operator=(SeriesOrdinate&&) noexcept = default;

    // --- Index: the index of the series ordinate (C# virtual property). ---
    virtual TIndex index() const { return index_; }
    virtual void set_index(TIndex index) { index_ = index; }

    // --- Value: the value of the series ordinate (C# virtual property). ---
    virtual TValue value() const { return value_; }
    virtual void set_value(TValue value) { value_ = value; }

    // Equality compares index and value only (C# Equals / operator==).
    bool operator==(const SeriesOrdinate& other) const {
        return index_ == other.index_ && value_ == other.value_;
    }
    bool operator!=(const SeriesOrdinate& other) const { return !(*this == other); }

    // Returns a copy of the series ordinate (C# Clone()). Like the C# original this
    // returns the base type; derived classes hide it with their own clone().
    SeriesOrdinate clone() const { return SeriesOrdinate(index(), value()); }

   protected:
    // Protected backing fields (C# `_index` / `_value`).
    TIndex index_{};
    TValue value_{};
};

}  // namespace corehydro::numerics::data
