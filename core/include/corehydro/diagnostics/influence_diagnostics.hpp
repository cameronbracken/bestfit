// ported from: src/RMC.BestFit/Diagnostics/InfluenceDiagnostics.cs @ fc28c0c
//
// Provides influence diagnostics for individual observations in a Bayesian analysis. The key
// diagnostic is the Pareto k value from PSIS-LOO (Pareto Smoothed Importance Sampling
// Leave-One-Out cross-validation), which indicates how influential each observation is on the
// posterior. Categories (Category property, C# 497): k < 0.5 Good, 0.5 <= k < 0.7 OK,
// 0.7 <= k < 1.0 Bad, k >= 1.0 VeryBad. This class is a LIGHT wrapper: it does NOT recompute the
// Pareto-k / elpd_loo values -- those are produced upstream in `bayesian_analysis.hpp`
// (`compute_psis_loo()`) and passed into the primary constructor. See the C# class remarks for
// references (Vehtari, Gelman & Gabry 2017; Vehtari et al. 2022).
//
// C++ mapping notes (structural mirroring):
//   - The nested C# `readonly struct ObservationInfluence` ports as a getter-only value class
//     with non-`const` members (same rationale as data_component.hpp / leverage_diagnostics.hpp:
//     `const` members would delete copy/move assignment that `std::vector` sorting/erase needs).
//   - The C# optional `string? Name` ports as `std::optional<std::string>`.
//   - `int topN = int.MaxValue` ports as `int top_n = INT_MAX`.
//   - The C# reliability-summary strings use the Unicode "GREATER-THAN OR EQUAL TO" glyph; this
//     port uses the ASCII ">=" (portability rule: no non-keyboard Unicode in the core). The
//     upstream tests only assert the leading keyword substrings, which are preserved verbatim.
//
// SKIPPED (deliberate): the `XElement` constructors on InfluenceDiagnostics / ObservationInfluence
// (C# 117 / 413) and `ToXElement()` on both (C# 347 / 512) -- XML, there is no XElement port in
// this core. `ObservationInfluence.ToString()` (C# 532) -- presentation-only (`:G4`/`:F3`/`:F2`
// + Category name assembly), not oracle-checked and not called by any compute path; skipped.
#pragma once
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/support/data_component.hpp"

namespace corehydro::diagnostics {

// Categorizes the Pareto k diagnostic value (C# `enum ParetoKCategory`, C# 542).
enum class ParetoKCategory {
    Good,     // k < 0.5
    OK,       // 0.5 <= k < 0.7
    Bad,      // 0.7 <= k < 1.0
    VeryBad,  // k >= 1.0
};

class InfluenceDiagnostics {
   public:
    // ------------------------------------------------------------------------------------
    // Nested value type (C# `public readonly struct ObservationInfluence`, C# 385)
    // ------------------------------------------------------------------------------------

    // Represents influence diagnostics for a single observation. The full-field constructor
    // (C# 397) defaults `value` to NaN, `data_type` to Exact, `count` to 1, `name` to none.
    class ObservationInfluence {
       public:
        ObservationInfluence(
            int index, double pareto_k, double elpd_loo,
            double value = std::numeric_limits<double>::quiet_NaN(),
            corehydro::models::DataComponentType data_type = corehydro::models::DataComponentType::Exact,
            int count = 1, std::optional<std::string> name = std::nullopt)
            : index_(index),
              pareto_k_(pareto_k),
              elpd_loo_(elpd_loo),
              value_(value),
              data_type_(data_type),
              count_(count),
              name_(std::move(name)) {}

        int index() const { return index_; }
        double pareto_k() const { return pareto_k_; }
        double elpd_loo() const { return elpd_loo_; }
        double value() const { return value_; }
        corehydro::models::DataComponentType data_type() const { return data_type_; }
        int count() const { return count_; }
        const std::optional<std::string>& name() const { return name_; }

        // The diagnostic category based on the Pareto k value (C# `Category`, C# 497). The
        // comparisons are ported verbatim (half-open bins).
        ParetoKCategory category() const {
            if (pareto_k_ < 0.5) return ParetoKCategory::Good;
            if (pareto_k_ < 0.7) return ParetoKCategory::OK;
            if (pareto_k_ < 1.0) return ParetoKCategory::Bad;
            return ParetoKCategory::VeryBad;
        }

       private:
        int index_;
        double pareto_k_;
        double elpd_loo_;
        double value_;
        corehydro::models::DataComponentType data_type_;
        int count_;
        std::optional<std::string> name_;
    };

    // ------------------------------------------------------------------------------------
    // Construction
    // ------------------------------------------------------------------------------------

    // Creates an empty influence diagnostics instance (C# 57): empty observations, rollups NaN.
    InfluenceDiagnostics()
        : mean_pareto_k_(std::numeric_limits<double>::quiet_NaN()),
          max_pareto_k_(std::numeric_limits<double>::quiet_NaN()) {}

    // Creates influence diagnostics from computed observation metrics (C# 69). C#'s
    // ArgumentNullException guard has no analogue (C++ vectors are never null).
    explicit InfluenceDiagnostics(std::vector<ObservationInfluence> observations)
        : observations_(std::move(observations)) {
        compute_summary_statistics();
    }

    // Creates influence diagnostics from PSIS-LOO results (C# 83). Throws std::invalid_argument
    // (C# ArgumentException) when the array lengths don't match, or when the optional data
    // components' count doesn't match. `data_components` == nullopt mirrors the C#
    // `List<DataComponent>? dataComponents = null` (no per-observation metadata).
    InfluenceDiagnostics(const std::vector<double>& pareto_k, const std::vector<double>& elpd_loo,
                         std::optional<std::vector<corehydro::models::DataComponent>> data_components =
                             std::nullopt) {
        if (pareto_k.size() != elpd_loo.size())
            throw std::invalid_argument("Array lengths must match.");
        if (data_components.has_value() && data_components->size() != pareto_k.size())
            throw std::invalid_argument("DataComponents count must match array lengths.");

        std::size_t n = pareto_k.size();
        observations_.reserve(n);
        for (std::size_t i = 0; i < n; ++i) {
            if (data_components.has_value()) {
                const corehydro::models::DataComponent& dc = (*data_components)[i];
                observations_.emplace_back(static_cast<int>(i), pareto_k[i], elpd_loo[i], dc.value(),
                                           dc.type(), dc.count(), dc.name());
            } else {
                observations_.emplace_back(static_cast<int>(i), pareto_k[i], elpd_loo[i]);
            }
        }
        compute_summary_statistics();
    }

    // ------------------------------------------------------------------------------------
    // Properties (C# 156-217)
    // ------------------------------------------------------------------------------------

    const std::vector<ObservationInfluence>& observations() const { return observations_; }
    int count() const { return static_cast<int>(observations_.size()); }
    double mean_pareto_k() const { return mean_pareto_k_; }
    double max_pareto_k() const { return max_pareto_k_; }
    int count_pareto_k_above_05() const { return count_pareto_k_above_05_; }
    int count_pareto_k_above_07() const { return count_pareto_k_above_07_; }
    int count_pareto_k_above_10() const { return count_pareto_k_above_10_; }

    // Proportion of observations with Pareto k >= 0.7 (C# 208). Empty -> 0.
    double proportion_problematic() const {
        return count() > 0 ? static_cast<double>(count_pareto_k_above_07_) / count() : 0.0;
    }

    // Whether the PSIS-LOO estimates are reliable (C# 217): fewer than 1% of observations with
    // k >= 0.7 and no observation with k >= 1.0.
    bool is_reliable() const {
        return proportion_problematic() < 0.01 && count_pareto_k_above_10_ == 0;
    }

    // ------------------------------------------------------------------------------------
    // Methods (C# 278-341)
    // ------------------------------------------------------------------------------------

    // Observations sorted by descending Pareto k, truncated to `top_n` (C# 278).
    std::vector<ObservationInfluence> get_most_influential_observations(int top_n = INT_MAX) const {
        if (observations_.empty()) return {};

        std::vector<ObservationInfluence> sorted = observations_;
        std::stable_sort(sorted.begin(), sorted.end(),
                         [](const ObservationInfluence& a, const ObservationInfluence& b) {
                             return a.pareto_k() > b.pareto_k();
                         });
        int take = std::min(top_n, static_cast<int>(sorted.size()));
        if (take < 0) take = 0;
        sorted.erase(sorted.begin() + take, sorted.end());
        return sorted;
    }

    // Observations with Pareto k >= threshold, sorted by descending Pareto k (C# 294).
    std::vector<ObservationInfluence> get_problematic_observations(double threshold = 0.7) const {
        if (observations_.empty()) return {};

        std::vector<ObservationInfluence> result;
        for (const auto& o : observations_)
            if (o.pareto_k() >= threshold) result.push_back(o);
        std::stable_sort(result.begin(), result.end(),
                         [](const ObservationInfluence& a, const ObservationInfluence& b) {
                             return a.pareto_k() > b.pareto_k();
                         });
        return result;
    }

    // Reliability category as a human-readable string (C# 309). Pure string assembly, no side
    // effects; the leading keyword ("No observations" / "UNRELIABLE" / "CAUTION" / "OK" / "GOOD")
    // is preserved verbatim (the ">=" glyph replaces the C# Unicode "GREATER-THAN OR EQUAL TO").
    std::string get_reliability_summary() const {
        if (count() == 0) return "No observations available for diagnostics.";

        if (count_pareto_k_above_10_ > 0) {
            return "UNRELIABLE: " + std::to_string(count_pareto_k_above_10_) +
                   " observation(s) have Pareto k >= 1.0. The PSIS-LOO estimates are unreliable. "
                   "Consider using exact LOO-CV or WAIC.";
        }

        if (proportion_problematic() >= 0.01) {
            return "CAUTION: " + std::to_string(count_pareto_k_above_07_) +
                   " observation(s) have Pareto k >= 0.7. PSIS-LOO estimates may be biased. "
                   "Consider using exact LOO-CV for problematic points.";
        }

        if (count_pareto_k_above_05_ > 0) {
            return "OK: " + std::to_string(count_pareto_k_above_05_) +
                   " observation(s) have moderate influence (0.5 <= k < 0.7). PSIS-LOO estimates "
                   "should be reasonably accurate.";
        }

        return "GOOD: All observations have Pareto k < 0.5. PSIS-LOO estimates are reliable.";
    }

    // The observation at the given index (C# `this[int]`, C# 341).
    const ObservationInfluence& operator[](int index) const {
        return observations_[static_cast<std::size_t>(index)];
    }

   private:
    // Computes summary statistics from the observation data (C# 226), skipping NaN Pareto-k
    // values; if every value is NaN the mean/max stay NaN.
    void compute_summary_statistics() {
        if (observations_.empty()) {
            mean_pareto_k_ = std::numeric_limits<double>::quiet_NaN();
            max_pareto_k_ = std::numeric_limits<double>::quiet_NaN();
            count_pareto_k_above_05_ = 0;
            count_pareto_k_above_07_ = 0;
            count_pareto_k_above_10_ = 0;
            return;
        }

        double sum = 0.0;
        double max = -std::numeric_limits<double>::infinity();
        int count05 = 0, count07 = 0, count10 = 0;
        int valid_count = 0;

        for (const auto& o : observations_) {
            double k = o.pareto_k();
            if (!std::isnan(k)) {
                sum += k;
                ++valid_count;
                if (k > max) max = k;
                if (k >= 0.5) ++count05;
                if (k >= 0.7) ++count07;
                if (k >= 1.0) ++count10;
            }
        }

        if (valid_count == 0) {
            mean_pareto_k_ = std::numeric_limits<double>::quiet_NaN();
            max_pareto_k_ = std::numeric_limits<double>::quiet_NaN();
        } else {
            mean_pareto_k_ = sum / valid_count;
            max_pareto_k_ = max;
        }
        count_pareto_k_above_05_ = count05;
        count_pareto_k_above_07_ = count07;
        count_pareto_k_above_10_ = count10;
    }

    // ------------------------------------------------------------------------------------
    // State
    // ------------------------------------------------------------------------------------
    std::vector<ObservationInfluence> observations_;
    double mean_pareto_k_ = std::numeric_limits<double>::quiet_NaN();
    double max_pareto_k_ = std::numeric_limits<double>::quiet_NaN();
    int count_pareto_k_above_05_ = 0;
    int count_pareto_k_above_07_ = 0;
    int count_pareto_k_above_10_ = 0;
};

}  // namespace corehydro::diagnostics
