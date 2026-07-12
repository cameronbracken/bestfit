// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataCollections/ExactSeries.cs @ fc28c0c
//
// Exact data series: the collection of systematic-record observations.
//
// Deliberately NOT ported:
//   - ToXElement() and the XElement constructor (XML serialization, project-wide)
//   - Autocorrelation() / PartialAutocorrelation() (thin facades over
//     Numerics.Data.Statistics.Autocorrelation, which is unported; the exact-data
//     hypothesis-test facade is deferred with it)
//
// The typed add()/insert() overloads copy the item into owned storage (the observable
// C# difference -- Add stores the caller's reference -- only matters for the unported
// INPC plumbing). Clone(), like the C# (new ExactSeries(ToList())), does NOT carry the
// EnforceUniqueIndex flag.
#pragma once
#include <algorithm>
#include <cstddef>
#include <limits>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/data_frame/data_collections/data_series.hpp"
#include "corehydro/models/data_frame/data_types/exact_data.hpp"
#include "corehydro/models/support/validation_result.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"

namespace corehydro::models {

class ExactSeries : public DataSeries {
   public:
    // Constructs an empty series (C# line 26).
    ExactSeries() = default;

    // Constructs series based on a list of data; entries are cloned (C# line 32).
    explicit ExactSeries(const std::vector<ExactData>& data) {
        for (const auto& item : data) add(item.clone());
    }

    // Constructs series based on a list of values with sequential 0-based indexes
    // (C# line 42).
    explicit ExactSeries(const std::vector<double>& values) {
        for (std::size_t i = 0; i < values.size(); i++)
            add(ExactData(static_cast<int>(i), values[i]));
    }

    // Typed element access (shadows the Data& base indexer).
    ExactData& operator[](std::size_t index) {
        return static_cast<ExactData&>(DataSeries::operator[](index));
    }
    const ExactData& operator[](std::size_t index) const {
        return static_cast<const ExactData&>(DataSeries::operator[](index));
    }

    // Typed add/insert (copy the item into owned storage; see the header note).
    void add(ExactData item) {
        DataSeries::add(std::make_unique<ExactData>(std::move(item)));
    }
    void insert(std::size_t index, ExactData item) {
        DataSeries::insert(index, std::make_unique<ExactData>(std::move(item)));
    }

    // --- EnforceUniqueIndex: whether Validate rejects duplicate indexes (C# line 62). ---
    bool enforce_unique_index() const { return enforce_unique_index_; }
    void set_enforce_unique_index(bool enforce) { enforce_unique_index_ = enforce; }

    // Gets the median value of the series: middle value for odd counts, average of the
    // two middle values for even counts (C# property, line 71).
    double median_value() const {
        if (count() == 0) return 0.0;
        if (count() == 1) return (*this)[0].value();
        std::vector<double> data = values_to_list();
        std::sort(data.begin(), data.end());
        std::size_t mid = count() / 2;
        if (count() % 2 == 0) return (data[mid - 1] + data[mid]) / 2.0;
        return data[mid];
    }

    // Gets the smallest value in the upper half of the sorted series: the value at sorted
    // index Count/2 (C# property, line 99). Used as the upper bound for the low outlier
    // threshold validation (see the C# remarks).
    double upper_middle_value() const {
        if (count() == 0) return 0.0;
        if (count() == 1) return (*this)[0].value();
        std::vector<double> data = values_to_list();
        std::sort(data.begin(), data.end());
        return data[count() / 2];
    }

    // Gets the minimum value of the series (C# line 116).
    double minimum_value() const {
        return minimum_by(std::numeric_limits<double>::max(),
                          [](const Data& d) { return d.value(); });
    }

    // Gets the maximum value of the series (C# line 125; C# double.MinValue is the
    // most-negative finite double -> lowest()).
    double maximum_value() const {
        return maximum_by(std::numeric_limits<double>::lowest(),
                          [](const Data& d) { return d.value(); });
    }

    // Gets the minimum index of the series (C# line 134).
    int minimum_index() const {
        return minimum_by(-100000, [](const Data& d) { return d.index(); });
    }

    // Gets the maximum index of the series (C# line 143).
    int maximum_index() const {
        return maximum_by(100000, [](const Data& d) { return d.index(); });
    }

    // Gets the number of unique indexes (C# line 152).
    int unique_indices() const {
        std::vector<int> indices = indices_to_list();
        std::set<int> distinct(indices.begin(), indices.end());
        return static_cast<int>(distinct.size());
    }

    // Gets the index span (C# line 160): MaximumIndex - MinimumIndex + 1, 0 when empty.
    int index_span() const {
        if (count() == 0) return 0;
        return maximum_index() - minimum_index() + 1;
    }

    // Returns the series as a list of cloned items (C# line 169).
    std::vector<ExactData> to_list() const {
        std::vector<ExactData> result;
        result.reserve(count());
        for (std::size_t i = 0; i < count(); i++) result.push_back((*this)[i].clone());
        return result;
    }

    // Create a clone of the series (C# line 180).
    ExactSeries clone() const { return ExactSeries(to_list()); }

    // Sorts the elements in the collection by index (C# line 189; stable, see the base).
    void sort_by_index(
        numerics::data::SortOrder order = numerics::data::SortOrder::Ascending) {
        stable_sort_by(order, [](const Data& d) { return d.index(); });
    }

    // Sorts the elements in the collection by value (C# line 205).
    void sort(numerics::data::SortOrder order = numerics::data::SortOrder::Ascending) {
        stable_sort_by(order, [](const Data& d) { return d.value(); });
    }

    // Validates the current state of the exact series and reports any issues found
    // (C# line 231). The duplicate-index check runs once after the loop, only when
    // EnforceUniqueIndex is set.
    ValidationResult validate() const {
        ValidationResult result;

        for (std::size_t i = 0; i < count(); i++) {
            ValidationResult data_valid = (*this)[i].validate();
            if (!data_valid.is_valid) {
                result.validation_messages.insert(result.validation_messages.end(),
                                                  data_valid.validation_messages.begin(),
                                                  data_valid.validation_messages.end());
                result.is_valid = false;
            }
        }

        if (enforce_unique_index_) {
            std::vector<int> duplicates = duplicate_indices();
            if (!duplicates.empty()) {
                std::string joined;
                for (std::size_t i = 0; i < duplicates.size(); i++) {
                    if (i > 0) joined += ", ";
                    joined += std::to_string(duplicates[i]);
                }
                result.validation_messages.push_back("Error: Duplicate indexes found: " +
                                                     joined);
                result.is_valid = false;
            }
        }

        return result;
    }

   private:
    bool enforce_unique_index_ = false;
};

}  // namespace corehydro::models
