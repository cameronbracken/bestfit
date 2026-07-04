// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataCollections/UncertainSeries.cs @ fc28c0c
//
// Uncertain data series: observations carrying a measurement-error distribution.
// MinimumValue / MaximumValue use the distributions' 5th/95th percentiles
// (LowerValue / UpperValue).
//
// Deliberately NOT ported: ToXElement() and the XElement constructor (XML, project-wide).
//
// Validate(DataFrame) cross-references the parent frame's exact series, so it is declared
// here against a forward-declared DataFrame and DEFINED inline in data_frame.hpp (same
// arrangement as IntervalSeries::validate).
#pragma once
#include <cstddef>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/models/data_frame/data_collections/data_series.hpp"
#include "bestfit/models/data_frame/data_types/uncertain_data.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/data/interpolation/sort_order.hpp"

namespace bestfit::models {

class DataFrame;  // the cross-reference parameter of validate(); defined in data_frame.hpp

class UncertainSeries : public DataSeries {
   public:
    // Constructs an empty series (C# line 25).
    UncertainSeries() = default;

    // Constructs series based on a list of data; entries are cloned (C# line 31).
    explicit UncertainSeries(const std::vector<UncertainData>& data) {
        for (const auto& item : data) add(item.clone());
    }

    // Typed element access (shadows the Data& base indexer).
    UncertainData& operator[](std::size_t index) {
        return static_cast<UncertainData&>(DataSeries::operator[](index));
    }
    const UncertainData& operator[](std::size_t index) const {
        return static_cast<const UncertainData&>(DataSeries::operator[](index));
    }

    // Typed add/insert (the item is moved into owned storage; UncertainData copies
    // deep-clone their distribution, so pass-by-value stays safe).
    void add(UncertainData item) {
        DataSeries::add(std::make_unique<UncertainData>(std::move(item)));
    }
    void insert(std::size_t index, UncertainData item) {
        DataSeries::insert(index, std::make_unique<UncertainData>(std::move(item)));
    }

    // Gets the minimum value of the series -- the smallest LowerValue, i.e. 5th
    // percentile (C# line 54).
    double minimum_value() const {
        return minimum_by(std::numeric_limits<double>::max(), [](const Data& d) {
            return static_cast<const UncertainData&>(d).lower_value();
        });
    }

    // Gets the maximum value of the series -- the largest UpperValue, i.e. 95th
    // percentile (C# line 63).
    double maximum_value() const {
        return maximum_by(std::numeric_limits<double>::lowest(), [](const Data& d) {
            return static_cast<const UncertainData&>(d).upper_value();
        });
    }

    // Gets the minimum index of the series (C# line 72).
    int minimum_index() const {
        return minimum_by(-100000, [](const Data& d) { return d.index(); });
    }

    // Gets the maximum index of the series (C# line 81).
    int maximum_index() const {
        return maximum_by(100000, [](const Data& d) { return d.index(); });
    }

    // Returns the series as a list of cloned items (C# line 90).
    std::vector<UncertainData> to_list() const {
        std::vector<UncertainData> result;
        result.reserve(count());
        for (std::size_t i = 0; i < count(); i++) result.push_back((*this)[i].clone());
        return result;
    }

    // Create a clone of the series (C# line 101).
    UncertainSeries clone() const { return UncertainSeries(to_list()); }

    // Sorts the elements in the collection by index (C# line 110; stable, see the base).
    void sort_by_index(
        numerics::data::SortOrder order = numerics::data::SortOrder::Ascending) {
        stable_sort_by(order, [](const Data& d) { return d.index(); });
    }

    // Sorts the elements in the collection by value (C# line 126).
    void sort(numerics::data::SortOrder order = numerics::data::SortOrder::Ascending) {
        stable_sort_by(order, [](const Data& d) { return d.value(); });
    }

    // Validates the current state of the uncertain series, cross-referencing the parent
    // data frame for overlaps with exact data (C# line 153). Defined in data_frame.hpp;
    // see the header note. The C# quirk of running the duplicate and overlap checks
    // INSIDE the per-item loop is preserved there.
    ValidationResult validate(const DataFrame* data_frame) const;
};

}  // namespace bestfit::models
