// ported from: Numerics/Data/Interpolation/Bilinear.cs @ 2a0357a
//
// 2D interpolation via repeated 1D linear interpolation, with independent optional
// transforms (None/Logarithmic/NormalZ) on x1, x2, and y. Designed to be compatible with
// the VBA Macro 'Interpolate Version 2.0.0' (USACE Risk Management Center), per the C#
// source.
//
// Reuses two internal Linear instances purely as sorted-array search helpers (their own
// BaseInterpolate formula is never invoked here): x1_li_ is built from (x1_values,
// x1_values) and x2_li_ from (x2_values, x2_values), so each instance's x_values() and
// y_values() are identical -- this mirrors the C# source's reuse pattern exactly,
// including reading X2LI.YValues (not XValues) for the x2 range below.
//
// v2.1.4 sync (Numerics 33dc1af): FIXED, not mirrored -- every `Math.Log10` call in this
// class's Logarithmic-transform branches now calls the guarded `Tools.Log10` instead
// (matching what Linear.cs already did). Previously this class's Logarithmic transform
// used plain, unguarded log10 while Linear::base_interpolate/extrapolate used the clamped
// corehydro::numerics::clamped_log10 -- a real divergence between the two interpolation
// classes (see docs/upstream-csharp-issues.md, marked RESOLVED) that could produce NaN for
// a grid coordinate/ordinate at or near zero (log10(0) = -inf, and -inf - -inf = NaN once
// the interpolation formula subtracts two such transformed values). This class now calls
// `corehydro::numerics::clamped_log10` at every one of the twelve log10 call sites below,
// exactly like Linear does; see fixtures/special_functions/bilinear.json's
// `log_floor_*` cases (ported from the new v2.1.4
// Test_LogarithmicFloorMatchesLinearInterpolation) for the regression pin.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/data/interpolation/linear.hpp"
#include "corehydro/numerics/data/interpolation/sort_order.hpp"
#include "corehydro/numerics/data/interpolation/transform.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::numerics::data {

class Bilinear {
   public:
    // yValues[i][j] = y(x1_values[i], x2_values[j]); outer size must match x1_values,
    // each inner row's size must match x2_values.
    Bilinear(std::vector<double> x1_values, std::vector<double> x2_values,
             std::vector<std::vector<double>> y_values, SortOrder sort_order = SortOrder::Ascending)
        : x1_values_(x1_values),
          x2_values_(x2_values),
          y_values_(std::move(y_values)),
          x1_li_(x1_values, x1_values, sort_order),
          x2_li_(x2_values, x2_values, sort_order),
          sort_order_(sort_order) {
        if (y_values_.size() != x1_values_.size())
            throw std::invalid_argument(
                "The number of columns in the y-array must be the same length as the x1 array.");
        for (const auto& row : y_values_) {
            if (row.size() != x2_values_.size())
                throw std::invalid_argument(
                    "The number of rows in the y-array must be the same length as the x2 array.");
        }
    }

    Transform x1_transform = Transform::None;
    Transform x2_transform = Transform::None;
    Transform y_transform = Transform::None;

    const std::vector<double>& x1_values() const { return x1_values_; }
    const std::vector<double>& x2_values() const { return x2_values_; }
    const std::vector<std::vector<double>>& y_values() const { return y_values_; }

    SortOrder sort_order() const { return sort_order_; }

    bool use_smart_search() const { return use_smart_search_; }
    void set_use_smart_search(bool value) {
        use_smart_search_ = value;
        x1_li_.set_use_smart_search(value);
        x2_li_.set_use_smart_search(value);
    }

    // Given a value x1 and x2, returns an interpolated value for y.
    double interpolate(double x1, double x2) const {
        bool ascending = sort_order_ == SortOrder::Ascending;
        std::size_t i1_last = static_cast<std::size_t>(x1_li_.count() - 1);
        std::size_t i2_last = static_cast<std::size_t>(x2_li_.count() - 1);

        // Handle edge cases where x1 and/or x2 are out of range.
        if ((ascending && x1 < x1_values_[0] && x2 < x2_values_[0]) ||
            (!ascending && x1 > x1_values_[0] && x2 > x2_values_[0]))
            return y_values_[0][0];  // Top left
        if ((ascending && x1 < x1_values_[0] && x2 > x2_values_[i2_last]) ||
            (!ascending && x1 > x1_values_[0] && x2 < x2_values_[i2_last]))
            return y_values_[0][i2_last];  // Top Right
        if ((ascending && x1 > x1_values_[i1_last] && x2 > x2_values_[i2_last]) ||
            (!ascending && x1 < x1_values_[i1_last] && x2 < x2_values_[i2_last]))
            return y_values_[i1_last][i2_last];  // Bottom Right
        if ((ascending && x1 > x1_values_[i1_last] && x2 < x2_values_[0]) ||
            (!ascending && x1 < x1_values_[i1_last] && x2 > x2_values_[0]))
            return y_values_[i1_last][0];  // Bottom Left

        double y, t, u, x1i, x1ii, x2j, x2jj, yij, yiij, yiijj, yijj, x1lb, x1ub, x2lb, x2ub;
        int i = x1_li_.search(x1);
        int j = x2_li_.search(x2);
        std::size_t ui = static_cast<std::size_t>(i), uj = static_cast<std::size_t>(j);

        // Get x1 transform
        x1i = x1_li_.x_values()[ui];
        x1ii = x1_li_.x_values()[ui + 1];
        x1lb = x1_values_[0];
        x1ub = x1_values_[i1_last];
        if (x1_transform == Transform::Logarithmic) {
            x1 = clamped_log10(x1);
            x1i = clamped_log10(x1i);
            x1ii = clamped_log10(x1ii);
            x1lb = clamped_log10(x1lb);
            x1ub = clamped_log10(x1ub);
        } else if (x1_transform == Transform::NormalZ) {
            x1 = distributions::Normal::standard_z(x1);
            x1i = distributions::Normal::standard_z(x1i);
            x1ii = distributions::Normal::standard_z(x1ii);
            x1lb = distributions::Normal::standard_z(x1lb);
            x1ub = distributions::Normal::standard_z(x1ub);
        }

        // Get x2 transform (note: reads X2LI.YValues, matching the C# source exactly --
        // X2LI was built from (x2_values, x2_values), so XValues and YValues are identical)
        x2j = x2_li_.y_values()[uj];
        x2jj = x2_li_.y_values()[uj + 1];
        x2lb = x2_values_[0];
        x2ub = x2_values_[i2_last];
        if (x2_transform == Transform::Logarithmic) {
            x2 = clamped_log10(x2);
            x2j = clamped_log10(x2j);
            x2jj = clamped_log10(x2jj);
            x2lb = clamped_log10(x2lb);
            x2ub = clamped_log10(x2ub);
        } else if (x2_transform == Transform::NormalZ) {
            x2 = distributions::Normal::standard_z(x2);
            x2j = distributions::Normal::standard_z(x2j);
            x2jj = distributions::Normal::standard_z(x2jj);
            x2lb = distributions::Normal::standard_z(x2lb);
            x2ub = distributions::Normal::standard_z(x2ub);
        }

        // Get y transform
        yij = y_values_[ui][uj];
        yiij = y_values_[ui + 1][uj];
        yiijj = y_values_[ui + 1][uj + 1];
        yijj = y_values_[ui][uj + 1];
        if (y_transform == Transform::Logarithmic) {
            yij = clamped_log10(yij);
            yiij = clamped_log10(yiij);
            yiijj = clamped_log10(yiijj);
            yijj = clamped_log10(yijj);
        } else if (y_transform == Transform::NormalZ) {
            yij = distributions::Normal::standard_z(yij);
            yiij = distributions::Normal::standard_z(yiij);
            yiijj = distributions::Normal::standard_z(yiijj);
            yijj = distributions::Normal::standard_z(yijj);
        }

        // Interpolate
        t = (x1 - x1i) / (x1ii - x1i);
        u = (x2 - x2j) / (x2jj - x2j);
        y = (1.0 - t) * (1.0 - u) * yij + t * (1.0 - u) * yiij + t * u * yiijj + (1.0 - t) * u * yijj;

        // x1 out of range - 1D linear interpolation
        if ((ascending && x1 < x1lb) || (!ascending && x1 > x1lb))
            y = yij + (x2 - x2j) / (x2jj - x2j) * (yijj - yij);
        if ((ascending && x1 > x1ub) || (!ascending && x1 < x1ub))
            y = yiij + (x2 - x2j) / (x2jj - x2j) * (yiijj - yiij);

        // x2 out of range - 1D linear interpolation
        if ((ascending && x2 < x2lb) || (!ascending && x2 > x2lb))
            y = yij + (x1 - x1i) / (x1ii - x1i) * (yiij - yij);
        if ((ascending && x2 > x2ub) || (!ascending && x2 < x2ub))
            y = yijj + (x1 - x1i) / (x1ii - x1i) * (yiijj - yijj);

        // Back transform y
        if (y_transform == Transform::Logarithmic) {
            y = std::pow(10.0, y);
        } else if (y_transform == Transform::NormalZ) {
            y = distributions::Normal::standard_cdf(y);
        }

        return y;
    }

   private:
    std::vector<double> x1_values_;
    std::vector<double> x2_values_;
    std::vector<std::vector<double>> y_values_;
    Linear x1_li_;
    Linear x2_li_;
    bool use_smart_search_ = true;
    SortOrder sort_order_ = SortOrder::Ascending;
};

}  // namespace corehydro::numerics::data
