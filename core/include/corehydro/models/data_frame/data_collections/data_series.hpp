// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataCollections/DataSeries.cs @ fc28c0c
//
// Abstract typed-list base over polymorphic Data ordinates. The C# class implements
// IList<Data> over a protected List<Data>; the C++ port wraps a
// std::vector<std::unique_ptr<Data>> (the items are polymorphic Data subclasses, so the
// container owns them through pointers). The IList surface maps to natural container
// methods:
//   this[int]            -> operator[](std::size_t) returning Data& (concrete series
//                           shadow it with a typed reference)
//   Count                -> count()
//   Add/Insert           -> add()/insert() taking std::unique_ptr<Data> (concrete series
//                           add typed by-value overloads that copy the item in)
//   Remove/RemoveAt      -> remove() (first occurrence by object identity, like the C#
//                           reference-equality Remove) / remove_at()
//   Clear/Contains/IndexOf -> clear()/contains()/index_of()
//   GetEnumerator        -> index-based iteration via count() + operator[]
//
// Deliberately NOT ported (project-wide deferrals -- desktop-app concerns):
//   - INotifyCollectionChanged / CollectionChanged / RaiseCollectionChangedReset and the
//     SuppressCollectionChanged flag (WPF event plumbing; see the invalidation-strategy
//     note in data_frame.hpp for what replaces it)
//   - CopyTo(Data[], int) and IsReadOnly (IList<T> interop plumbing)
//   - the indexer SETTER (`this[int] = value`; only used by the C# event plumbing)
//
// ValuesToList/ValuesToArray (and the PlottingPositions/Indices pairs) collapse into a
// single std::vector-returning method each -- C#'s List<T> and T[] both map to
// std::vector here.
#pragma once
#include <algorithm>
#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_types/data.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"

namespace corehydro::models {

class DataSeries {
   public:
    virtual ~DataSeries() = default;

    // Owning container: move-only (concrete series provide deep clone()).
    DataSeries() = default;
    DataSeries(const DataSeries&) = delete;
    DataSeries& operator=(const DataSeries&) = delete;
    DataSeries(DataSeries&&) noexcept = default;
    DataSeries& operator=(DataSeries&&) noexcept = default;

    // Gets the element at the specific index (C# indexer getter, line 21).
    Data& operator[](std::size_t index) { return *series_ordinates_[index]; }
    const Data& operator[](std::size_t index) const { return *series_ordinates_[index]; }

    // Gets the number of elements contained in the collection (C# line 44).
    std::size_t count() const { return series_ordinates_.size(); }

    // Add element to the collection (C# line 55).
    void add(std::unique_ptr<Data> item) { series_ordinates_.push_back(std::move(item)); }

    // Inserts an element into the collection at the specified index (C# line 67).
    void insert(std::size_t index, std::unique_ptr<Data> item) {
        series_ordinates_.insert(
            series_ordinates_.begin() + static_cast<std::ptrdiff_t>(index),
            std::move(item));
    }

    // Removes the first occurrence of the specified object -- object identity, like the
    // C# reference-equality Remove (C# line 78).
    bool remove(const Data& item) {
        for (std::size_t i = 0; i < series_ordinates_.size(); i++) {
            if (series_ordinates_[i].get() == &item) {
                remove_at(i);
                return true;
            }
        }
        return false;
    }

    // Remove element at the specified index of the collection (C# line 94).
    void remove_at(std::size_t index) {
        series_ordinates_.erase(series_ordinates_.begin() +
                                static_cast<std::ptrdiff_t>(index));
    }

    // Remove all elements from the collection (C# line 102; the C# reverse Remove loop
    // only exists to manage events).
    void clear() { series_ordinates_.clear(); }

    // Determines whether an element is in the collection (C# line 121; identity).
    bool contains(const Data& item) const { return index_of(item) >= 0; }

    // Returns the zero-based index of the first occurrence, or -1 (C# line 140; identity).
    int index_of(const Data& item) const {
        for (std::size_t i = 0; i < series_ordinates_.size(); i++) {
            if (series_ordinates_[i].get() == &item) return static_cast<int>(i);
        }
        return -1;
    }

    // Returns the list of series values (C# ValuesToList line 167 / ValuesToArray
    // line 175 -- one method here, see the header note).
    std::vector<double> values_to_list() const {
        std::vector<double> result;
        result.reserve(series_ordinates_.size());
        for (const auto& item : series_ordinates_) result.push_back(item->value());
        return result;
    }

    // Returns the list of series plotting positions (C# lines 184/192).
    std::vector<double> plotting_positions_to_list() const {
        std::vector<double> result;
        result.reserve(series_ordinates_.size());
        for (const auto& item : series_ordinates_)
            result.push_back(item->plotting_position());
        return result;
    }

    // Returns the list of series indices (C# lines 201/209).
    std::vector<int> indices_to_list() const {
        std::vector<int> result;
        result.reserve(series_ordinates_.size());
        for (const auto& item : series_ordinates_) result.push_back(item->index());
        return result;
    }

   protected:
    // Internal list (C# `_seriesOrdinates`, line 14).
    std::vector<std::unique_ptr<Data>> series_ordinates_;

    // Shared comparator plumbing for the concrete series' Sort/SortByIndex. The C#
    // List<T>.Sort is unstable with unspecified tie order; std::stable_sort keeps ties
    // deterministic (and therefore reproducible across R/Python bindings).
    template <typename Key>
    void stable_sort_by(numerics::data::SortOrder order, Key key) {
        bool ascending = order == numerics::data::SortOrder::Ascending;
        std::stable_sort(series_ordinates_.begin(), series_ordinates_.end(),
                         [&](const std::unique_ptr<Data>& x, const std::unique_ptr<Data>& y) {
                             return ascending ? key(*x) < key(*y) : key(*y) < key(*x);
                         });
    }

    // Shared min/max plumbing for the concrete series' MinimumValue/MaximumValue/
    // MinimumIndex/MaximumIndex (each C# formula supplies its own key and empty
    // sentinel; the sentinels mirror the C# early returns).
    template <typename T, typename Key>
    T minimum_by(T empty_sentinel, Key key) const {
        if (series_ordinates_.empty()) return empty_sentinel;
        T result = key(*series_ordinates_[0]);
        for (std::size_t i = 1; i < series_ordinates_.size(); i++)
            result = std::min(result, key(*series_ordinates_[i]));
        return result;
    }
    template <typename T, typename Key>
    T maximum_by(T empty_sentinel, Key key) const {
        if (series_ordinates_.empty()) return empty_sentinel;
        T result = key(*series_ordinates_[0]);
        for (std::size_t i = 1; i < series_ordinates_.size(); i++)
            result = std::max(result, key(*series_ordinates_[i]));
        return result;
    }

    // Duplicate index keys in first-occurrence order, mirroring the C# duplicate-check
    // pipeline `GroupBy(d => d.Index).Where(g => g.Count() > 1).Select(g => g.Key)`
    // shared by the ExactSeries / IntervalSeries / UncertainSeries Validate methods.
    std::vector<int> duplicate_indices() const {
        std::vector<int> duplicates;
        for (std::size_t i = 0; i < series_ordinates_.size(); i++) {
            int idx = series_ordinates_[i]->index();
            bool seen_before = false;
            for (std::size_t j = 0; j < i; j++) {
                if (series_ordinates_[j]->index() == idx) {
                    seen_before = true;
                    break;
                }
            }
            if (seen_before) continue;
            int occurrences = 0;
            for (const auto& item : series_ordinates_) {
                if (item->index() == idx) occurrences++;
            }
            if (occurrences > 1) duplicates.push_back(idx);
        }
        return duplicates;
    }
};

}  // namespace corehydro::models
