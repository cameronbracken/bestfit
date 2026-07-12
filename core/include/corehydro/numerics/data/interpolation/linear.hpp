// ported from: Numerics/Data/Interpolation/Linear.cs @ a2c4dbf
//
// 1D linear interpolation with optional log10/normal-z transforms on x and/or y before
// interpolating (and the inverse transform applied to the result). Designed to be
// compatible with the VBA Macro 'Interpolate Version 2.0.0' (USACE Risk Management
// Center), per the C# source.
//
// BaseInterpolate/Extrapolate use the clamped corehydro::numerics::clamped_log10 (ported
// from Tools.Log10); Bilinear (bilinear.hpp) instead uses plain std::log10 for its own
// transforms -- that split exists in the C# source itself (Linear.cs calls Tools.Log10,
// Bilinear.cs calls Math.Log10 directly) and is preserved here rather than "fixed". See
// docs/upstream-csharp-issues.md.
#pragma once
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/data/interpolation/interpolater.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data {

class Linear : public Interpolater {
   public:
    Linear(std::vector<double> x_values, std::vector<double> y_values,
           SortOrder sort_order = SortOrder::Ascending)
        : Interpolater(std::move(x_values), std::move(y_values), sort_order) {}

    // The transform for the x-values and y-values. Default = None.
    Transform x_transform = Transform::None;
    Transform y_transform = Transform::None;

    double base_interpolate(double x, int index) const override {
        // See if x is out of range
        if ((sort_order() == SortOrder::Ascending && x <= x_values()[0]) ||
            (sort_order() == SortOrder::Descending && x >= x_values()[0]))
            return y_values()[0];
        std::size_t last = static_cast<std::size_t>(count() - 1);
        if ((sort_order() == SortOrder::Ascending && x >= x_values()[last]) ||
            (sort_order() == SortOrder::Descending && x <= x_values()[last]))
            return y_values()[last];

        double y, x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
        std::size_t xlo = static_cast<std::size_t>(index), xhi = xlo + 1;

        // Get X transform
        if (x_transform == Transform::None) {
            x1 = x_values()[xlo];
            x2 = x_values()[xhi];
        } else if (x_transform == Transform::Logarithmic) {
            x = clamped_log10(x);
            x1 = clamped_log10(x_values()[xlo]);
            x2 = clamped_log10(x_values()[xhi]);
        } else if (x_transform == Transform::NormalZ) {
            x = distributions::Normal::standard_z(x);
            x1 = distributions::Normal::standard_z(x_values()[xlo]);
            x2 = distributions::Normal::standard_z(x_values()[xhi]);
        }

        // Get Y transform
        if (y_transform == Transform::None) {
            y1 = y_values()[xlo];
            y2 = y_values()[xhi];
        } else if (y_transform == Transform::Logarithmic) {
            y1 = clamped_log10(y_values()[xlo]);
            y2 = clamped_log10(y_values()[xhi]);
        } else if (y_transform == Transform::NormalZ) {
            y1 = distributions::Normal::standard_z(y_values()[xlo]);
            y2 = distributions::Normal::standard_z(y_values()[xhi]);
        }

        // Interpolate (check for division by zero)
        if ((x2 - x1) == 0.0) {
            y = y1;
        } else {
            y = y1 + (x - x1) / (x2 - x1) * (y2 - y1);
        }

        if (y_transform == Transform::None) return y;
        if (y_transform == Transform::Logarithmic) return std::pow(10.0, y);
        if (y_transform == Transform::NormalZ) return distributions::Normal::standard_cdf(y);

        // return NaN if we get to here
        return std::numeric_limits<double>::quiet_NaN();
    }

    // Given a value x, returns an extrapolated value.
    double extrapolate(double x) const {
        double y, x1, x2, y1, y2;
        std::size_t last = static_cast<std::size_t>(count() - 1);

        // See if x is out of range
        if ((sort_order() == SortOrder::Ascending && x < x_values()[0]) ||
            (sort_order() == SortOrder::Descending && x > x_values()[0])) {
            x1 = x_values()[0];
            x2 = x_values()[1];
            y1 = y_values()[0];
            y2 = y_values()[1];
        } else if ((sort_order() == SortOrder::Ascending && x > x_values()[last]) ||
                   (sort_order() == SortOrder::Descending && x < x_values()[last])) {
            x1 = x_values()[last];
            x2 = x_values()[last - 1];
            y1 = y_values()[last];
            y2 = y_values()[last - 1];
        } else {
            return interpolate(x);
        }

        // Get X transform
        if (x_transform == Transform::Logarithmic) {
            x = clamped_log10(x);
            x1 = clamped_log10(x1);
            x2 = clamped_log10(x2);
        } else if (x_transform == Transform::NormalZ) {
            x = distributions::Normal::standard_z(x);
            x1 = distributions::Normal::standard_z(x1);
            x2 = distributions::Normal::standard_z(x2);
        }

        // Get Y transform
        if (y_transform == Transform::Logarithmic) {
            y1 = clamped_log10(y1);
            y2 = clamped_log10(y2);
        } else if (y_transform == Transform::NormalZ) {
            y1 = distributions::Normal::standard_z(y1);
            y2 = distributions::Normal::standard_z(y2);
        }

        // Extrapolate (check for division by zero)
        if ((x2 - x1) == 0.0) {
            y = y1;
        } else {
            y = y1 - (x1 - x) * (y2 - y1) / (x2 - x1);
        }

        if (y_transform == Transform::None) return y;
        if (y_transform == Transform::Logarithmic) return std::pow(10.0, y);
        if (y_transform == Transform::NormalZ) return distributions::Normal::standard_cdf(y);

        // return NaN if we get to here
        return std::numeric_limits<double>::quiet_NaN();
    }
};

}  // namespace corehydro::numerics::data
