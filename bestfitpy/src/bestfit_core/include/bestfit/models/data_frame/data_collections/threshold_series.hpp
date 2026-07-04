// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataCollections/ThresholdSeries.cs @ fc28c0c
//
// Threshold data series: historical perception-threshold periods. MinimumIndex /
// MaximumIndex reflect the period span (StartIndex / EndIndex) rather than a single
// observation index -- different from the other series classes.
//
// Deliberately NOT ported: ToXElement() and the XElement constructor (XML, project-wide).
#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/models/data_frame/data_collections/data_series.hpp"
#include "bestfit/models/data_frame/data_types/threshold_data.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/data/interpolation/sort_order.hpp"

namespace bestfit::models {

class ThresholdSeries : public DataSeries {
   public:
    // Constructs an empty series (C# line 24).
    ThresholdSeries() = default;

    // Constructs series based on a list of data; entries are cloned (C# line 30).
    explicit ThresholdSeries(const std::vector<ThresholdData>& data) {
        for (const auto& item : data) add(item.clone());
    }

    // Typed element access (shadows the Data& base indexer).
    ThresholdData& operator[](std::size_t index) {
        return static_cast<ThresholdData&>(DataSeries::operator[](index));
    }
    const ThresholdData& operator[](std::size_t index) const {
        return static_cast<const ThresholdData&>(DataSeries::operator[](index));
    }

    // Typed add/insert (copy the item into owned storage).
    void add(ThresholdData item) {
        DataSeries::add(std::make_unique<ThresholdData>(std::move(item)));
    }
    void insert(std::size_t index, ThresholdData item) {
        DataSeries::insert(index, std::make_unique<ThresholdData>(std::move(item)));
    }

    // Gets the minimum value of the series (C# line 53).
    double minimum_value() const {
        return minimum_by(std::numeric_limits<double>::max(),
                          [](const Data& d) { return d.value(); });
    }

    // Gets the maximum value of the series (C# line 62).
    double maximum_value() const {
        return maximum_by(std::numeric_limits<double>::lowest(),
                          [](const Data& d) { return d.value(); });
    }

    // Gets the minimum index of the series -- the earliest StartIndex (C# line 71).
    int minimum_index() const {
        return minimum_by(-100000, [](const Data& d) {
            return static_cast<const ThresholdData&>(d).start_index();
        });
    }

    // Gets the maximum index of the series -- the latest EndIndex (C# line 80).
    int maximum_index() const {
        return maximum_by(100000, [](const Data& d) {
            return static_cast<const ThresholdData&>(d).end_index();
        });
    }

    // Returns the series as a list of cloned items (C# line 89).
    std::vector<ThresholdData> to_list() const {
        std::vector<ThresholdData> result;
        result.reserve(count());
        for (std::size_t i = 0; i < count(); i++) result.push_back((*this)[i].clone());
        return result;
    }

    // Create a clone of the series (C# line 100).
    ThresholdSeries clone() const { return ThresholdSeries(to_list()); }

    // Sorts the elements in the collection by StartIndex (C# line 109; stable, see the
    // base).
    void sort_by_index(
        numerics::data::SortOrder order = numerics::data::SortOrder::Ascending) {
        stable_sort_by(order, [](const Data& d) {
            return static_cast<const ThresholdData&>(d).start_index();
        });
    }

    // Sorts the elements in the collection by value (C# line 125).
    void sort(numerics::data::SortOrder order = numerics::data::SortOrder::Ascending) {
        stable_sort_by(order, [](const Data& d) { return d.value(); });
    }

    // Validates the current state of the threshold series and reports any issues found
    // (C# line 151): per-item validation plus the pairwise period-overlap rule.
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

            // Check for overlapping thresholds.
            for (std::size_t j = 0; j < count(); j++) {
                if (i == j) continue;
                if ((*this)[i].start_index() >= (*this)[j].start_index() &&
                    (*this)[i].start_index() <= (*this)[j].end_index()) {
                    result.validation_messages.push_back(
                        "Error: The threshold periods cannot overlap.");
                    result.is_valid = false;
                    break;
                }
            }
        }

        return result;
    }
};

}  // namespace bestfit::models
