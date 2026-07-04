// ported from: RMC-BestFit/src/RMC.BestFit/Models/DataFrame/DataFrame.cs @ fc28c0c
//
// The input data frame containing exact, uncertain, interval, and threshold data series --
// the data container every univariate model consumes. This class satisfies the forward
// declaration in models/support/i_univariate_model.hpp.
//
// INVALIDATION STRATEGY (replaces the C# INPC event plumbing). Upstream, every series
// CollectionChanged / item PropertyChanged event re-ran CalculatePlottingPositions (which
// first runs ProcessThresholdSeries) and, for the exact series, CalculateLambda. The C++
// port has no events; the replacement is:
//   1. full_time_series() recomputes LAZILY on access: it rebuilds via
//      create_full_time_series() whenever the cached list size no longer matches
//      TotalRecordLength() -- exactly the C# getter's own rebuild condition. The upstream
//      double-checked-lock / Volatile.Read machinery simplifies to this plain lazy rebuild
//      because the C++ port is single-threaded by design.
//   2. Threshold-derived state (ThresholdData::NumberBelow, and the zeroing of
//      NumberAbove for fully covered windows) is recomputed EXPLICITLY: call
//      process_threshold_series() after mutating any series or threshold counts. Upstream
//      this ran automatically on every collection change; M5's
//      calculate_plotting_positions() will call it first, restoring the upstream trigger
//      point for plotting-position consumers.
//   3. Lambda is recomputed EXPLICITLY via calculate_lambda() (upstream: triggered by
//      exact-series collection-changed events) or pinned via set_lambda().
//
// Ported surface: the four owned series, FullTimeSeries/CreateFullTimeSeries,
// ProcessThresholdSeries, TotalRecordLength, ZeroValueRelativeFrequency, Lambda
// (SetLambda/CalculateLambda), the low-outlier surface (NumberOfLowOutliers,
// LowOutlierThreshold, ClearLowOutliers, SetLowOutliersFromMGBT,
// SetLowOutliersFromThreshold), the plain PlottingParameter property, Validate(), and a
// direct deep Clone() (the C# clones via an XElement round trip; the direct clone has the
// same observable result, including the empty lazily-rebuilt full series noted in the C#
// remarks).
//
// Deliberately NOT ported:
//   - CalculatePlottingPositions / ApplyLangbeinConversion (Hirsch-Stedinger plotting
//     positions are M5; calculate_plotting_positions() below is a marked no-op stub
//     because the ported PlottingParameter setter calls it)
//   - the hypothesis-test facade (JarqueBera/LjungBox/EqualVarianceTtest/
//     UnequalVarianceTtest/Ftest/LinearTrend/Unimodality/WaldWolfowitz/MannWhitney/
//     MannKendall/SummaryHypothesisTest -- Numerics HypothesisTests is unported)
//   - the summary-statistics / Q-Q surface (SummaryStatisticsExactDataOnly,
//     SummaryStatisticsAllData, GetNonparametricMoments, GetNonparametricMomentsROS,
//     SetStandardizedValues -- they need EmpiricalDistribution/LinearRegression facades
//     and the M5 plotting positions)
//   - CreateBlockSeries / CreatePeaksOverThresholdSeries (need the unported TimeSeries
//     container), CreateFromUSGS + USGSRawText (network import)
//   - the bootstrap/resampling surface (JackKnife, Resample, BootstrapDataFrame,
//     ShiftDistribution -- follow-up alongside the estimation consumers)
//   - XML (ToXElement / XElement constructor), INotifyPropertyChanged, and the
//     concurrency machinery (_syncRoot, Volatile, SnapshotNonNull)
#pragma once
#include <algorithm>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "bestfit/models/data_frame/data_collections/data_series.hpp"
#include "bestfit/models/data_frame/data_collections/exact_series.hpp"
#include "bestfit/models/data_frame/data_collections/interval_series.hpp"
#include "bestfit/models/data_frame/data_collections/threshold_series.hpp"
#include "bestfit/models/data_frame/data_collections/uncertain_series.hpp"
#include "bestfit/models/data_frame/data_types/data.hpp"
#include "bestfit/models/data_frame/data_types/exact_data.hpp"
#include "bestfit/models/data_frame/data_types/interval_data.hpp"
#include "bestfit/models/data_frame/data_types/threshold_data.hpp"
#include "bestfit/models/data_frame/data_types/uncertain_data.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/numerics/data/multiple_grubbs_beck_test.hpp"

namespace bestfit::models {

class DataFrame {
   public:
    // Constructs an empty data frame (C# line 32; the event hookups are not ported).
    DataFrame() = default;

    // The series are owning containers, so the frame is move-only; use clone() for a
    // deep copy (mirrors the C# Clone()).
    DataFrame(const DataFrame&) = delete;
    DataFrame& operator=(const DataFrame&) = delete;
    DataFrame(DataFrame&&) noexcept = default;
    DataFrame& operator=(DataFrame&&) noexcept = default;

    // --- The exact data series collection (C# property, line 128). ---
    ExactSeries& exact_series() { return exact_series_; }
    const ExactSeries& exact_series() const { return exact_series_; }
    void set_exact_series(ExactSeries series) { exact_series_ = std::move(series); }

    // --- The uncertain data series collection (C# property, line 143). ---
    UncertainSeries& uncertain_series() { return uncertain_series_; }
    const UncertainSeries& uncertain_series() const { return uncertain_series_; }
    void set_uncertain_series(UncertainSeries series) {
        uncertain_series_ = std::move(series);
    }

    // --- The interval data series collection (C# property, line 158). ---
    IntervalSeries& interval_series() { return interval_series_; }
    const IntervalSeries& interval_series() const { return interval_series_; }
    void set_interval_series(IntervalSeries series) { interval_series_ = std::move(series); }

    // --- The threshold data series collection (C# property, line 173). ---
    ThresholdSeries& threshold_series() { return threshold_series_; }
    const ThresholdSeries& threshold_series() const { return threshold_series_; }
    void set_threshold_series(ThresholdSeries series) {
        threshold_series_ = std::move(series);
    }

    // The full time series in chronological order (C# property, line 203). Lazy: rebuilds
    // when the cached size no longer matches TotalRecordLength() -- the C# getter's own
    // rebuild condition, minus the locking (see the invalidation-strategy note). The
    // returned list must be treated as read-only.
    const std::vector<std::unique_ptr<Data>>& full_time_series() {
        int total = total_record_length();
        if (total == 0 || static_cast<int>(full_time_series_.size()) == total)
            return full_time_series_;
        create_full_time_series();
        return full_time_series_;
    }

    // Returns the number of low outliers in the exact data series (C# line 224).
    int number_of_low_outliers() const { return number_of_low_outliers_; }

    // --- The low outlier threshold value (C# property, line 236). ---
    double low_outlier_threshold() const { return low_outlier_threshold_; }
    void set_low_outlier_threshold(double threshold) { low_outlier_threshold_ = threshold; }

    // --- The plotting position parameter; default 0.0 = Weibull (C# property, line 256;
    // alternatives: 0.40 Cunnane, 0.44 Gringorten, 0.50 Hazen). The C# setter recomputes
    // plotting positions on change; today that call hits the M5 stub below. ---
    double plotting_parameter() const { return plotting_parameter_; }
    void set_plotting_parameter(double plotting_parameter) {
        if (plotting_parameter_ != plotting_parameter) {
            plotting_parameter_ = plotting_parameter;
            calculate_plotting_positions();
        }
    }

    // The average number of events per index (C# line 273).
    double lambda() const { return lambda_; }

    // Sets the lambda value directly without calculation (C# line 1938).
    void set_lambda(double lambda) { lambda_ = lambda; }

    // Calculates the average number of events per index (C# line 1947): exact-series
    // count over its index span, 0 when either is empty. Explicit trigger -- see the
    // invalidation-strategy note.
    void calculate_lambda() {
        double events = static_cast<double>(exact_series_.count());
        double span = static_cast<double>(exact_series_.index_span());
        lambda_ = (events <= 0.0 || span <= 0.0) ? 0.0 : events / span;
    }

    // M5 stub -- Hirsch-Stedinger plotting positions (C# CalculatePlottingPositions,
    // line 1140) arrive in M5; deliberately a no-op until then. Kept because the ported
    // PlottingParameter setter calls it, preserving the C# call shape.
    void calculate_plotting_positions() { /* M5 */ }

    // Validates the current state of the data frame and reports any issues found
    // (C# line 527): plotting-parameter range plus the four series validations, in the
    // C# order.
    ValidationResult validate() const {
        ValidationResult result;

        if (plotting_parameter_ < 0.0 || plotting_parameter_ > 1.0) {
            result.validation_messages.push_back(
                "Error: The plotting parameter must be between 0 and 1.");
            result.is_valid = false;
        }

        append(result, exact_series_.validate());
        append(result, uncertain_series_.validate(this));
        append(result, interval_series_.validate(this));
        append(result, threshold_series_.validate());

        return result;
    }

    // Returns the total record length of the data frame (C# line 573): explicit data
    // points plus each threshold's NumberBelow + NumberAbove (call
    // process_threshold_series() first after mutations).
    int total_record_length() const {
        int n = static_cast<int>(exact_series_.count() + uncertain_series_.count() +
                                 interval_series_.count());
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            n += threshold_series_[i].number_below() + threshold_series_[i].number_above();
        }
        return n;
    }

    // Computes the relative frequency of data points less than or equal to zero
    // (C# line 587).
    double zero_value_relative_frequency() const {
        double total_count = 0;
        double total_zero_count = 0;
        if (exact_series_.count() > 0) {
            total_count = static_cast<double>(exact_series_.count());
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                if (exact_series_[i].value() <= 0.0) total_zero_count += 1;
        }
        if (uncertain_series_.count() > 0) {
            total_count += static_cast<double>(uncertain_series_.count());
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                if (uncertain_series_[i].value() <= 0.0) total_zero_count += 1;
        }
        if (interval_series_.count() > 0) {
            total_count += static_cast<double>(interval_series_.count());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                if (interval_series_[i].value() <= 0.0) total_zero_count += 1;
        }
        if (total_count == 0) return 0.0;
        return total_zero_count / total_count;
    }

    // Process the threshold data to ensure exclusivity by adjusting counts for
    // overlapping data (C# line 618): NumberBelow = Duration - NumberAbove - (explicit
    // interval/uncertain/exact points inside the window), clamped at 0; NumberAbove is
    // zeroed when the explicit points account for every remaining year (nBelow == 0).
    void process_threshold_series() {
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            ThresholdData& threshold_data = threshold_series_[i];
            int n_above = threshold_data.number_above();
            int n_below = threshold_data.duration() - n_above;
            // Check interval data
            for (std::size_t j = 0; j < interval_series_.count(); j++) {
                if (interval_series_[j].index() >= threshold_data.start_index() &&
                    interval_series_[j].index() <= threshold_data.end_index()) {
                    n_below -= 1;
                }
            }
            // Check uncertain data
            for (std::size_t j = 0; j < uncertain_series_.count(); j++) {
                if (uncertain_series_[j].index() >= threshold_data.start_index() &&
                    uncertain_series_[j].index() <= threshold_data.end_index()) {
                    n_below -= 1;
                }
            }
            // Check exact data
            for (std::size_t j = 0; j < exact_series_.count(); j++) {
                if (exact_series_[j].index() >= threshold_data.start_index() &&
                    exact_series_[j].index() <= threshold_data.end_index()) {
                    n_below -= 1;
                }
            }
            // Zero out NumberAbove when all years are accounted for by explicit data
            threshold_data.set_number_above(n_below == 0 ? 0 : n_above);
            threshold_data.set_number_below(std::max(0, n_below));
        }
    }

    // Creates a full time series in chronological order by expanding threshold data into
    // per-index left/right-censored clones and combining all series (C# line 669; the
    // concurrency snapshot/retry machinery is not ported).
    void create_full_time_series() {
        std::vector<std::unique_ptr<Data>> new_list;

        // Occupied indexes (Exact, Interval, Uncertain), built once per call as upstream.
        std::unordered_set<int> occupied;
        for (std::size_t k = 0; k < exact_series_.count(); k++)
            occupied.insert(exact_series_[k].index());
        for (std::size_t k = 0; k < interval_series_.count(); k++)
            occupied.insert(interval_series_[k].index());
        for (std::size_t k = 0; k < uncertain_series_.count(); k++)
            occupied.insert(uncertain_series_[k].index());

        // Threshold data
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            const ThresholdData& threshold = threshold_series_[i];
            // Left (below) thresholds
            for (int j = threshold.start_index();
                 j <= threshold.end_index() - threshold.number_above(); j++) {
                if (occupied.count(j) != 0) continue;
                auto t_data = std::make_unique<ThresholdData>(threshold.clone());
                t_data->set_start_index(j);
                t_data->set_end_index(j);
                t_data->set_number_above(0);
                t_data->set_number_below(1);
                new_list.push_back(std::move(t_data));
            }
            // Right (above) thresholds
            for (int j = threshold.end_index() - threshold.number_above() + 1;
                 j <= threshold.end_index(); j++) {
                if (occupied.count(j) != 0) continue;
                auto t_data = std::make_unique<ThresholdData>(threshold.clone());
                t_data->set_start_index(j);
                t_data->set_end_index(j);
                t_data->set_number_above(1);
                t_data->set_number_below(0);
                new_list.push_back(std::move(t_data));
            }
        }
        // Exact data
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            new_list.push_back(std::make_unique<ExactData>(exact_series_[i].clone()));
        // Uncertain data
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            new_list.push_back(std::make_unique<UncertainData>(uncertain_series_[i].clone()));
        // Interval data
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            new_list.push_back(std::make_unique<IntervalData>(interval_series_[i].clone()));

        // Sort data by index (stable, so equal indexes keep the threshold -> exact ->
        // uncertain -> interval insertion order deterministically; the C# unstable
        // List<T>.Sort leaves tie order unspecified).
        std::stable_sort(new_list.begin(), new_list.end(),
                         [](const std::unique_ptr<Data>& x, const std::unique_ptr<Data>& y) {
                             return x->index() < y->index();
                         });

        full_time_series_ = std::move(new_list);
    }

    // Clear the low outlier results (C# line 794).
    void clear_low_outliers() {
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            exact_series_[i].set_is_low_outlier(false);
        number_of_low_outliers_ = 0;
    }

    // Estimates and sets the low outliers using the Multiple Grubbs Beck Test (MGBT);
    // exact data only (C# line 805). Throws std::invalid_argument (C# ArgumentException)
    // when the exact series has errors or fewer than 10 items.
    void set_low_outliers_from_mgbt() {
        if (!exact_series_.validate().is_valid)
            throw std::invalid_argument("The exact data series has errors.");
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before evaluating low "
                "outliers.");

        clear_low_outliers();
        low_outlier_threshold_ = 0;

        std::vector<double> values = exact_series_.values_to_list();

        // Compute the number of low outliers using the Multiple Grubbs Beck Test.
        number_of_low_outliers_ = numerics::data::MultipleGrubbsBeckTest::function(values);

        // Set the threshold value as the first value larger than N.
        std::sort(values.begin(), values.end());
        low_outlier_threshold_ =
            number_of_low_outliers_ > 0
                ? values[static_cast<std::size_t>(number_of_low_outliers_)]
                : 0.0;

        // Flag every exact data point below the threshold.
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            exact_series_[i].set_is_low_outlier(exact_series_[i].value() <
                                                low_outlier_threshold_);
    }

    // Estimates and sets the low outliers using the low outlier threshold value; exact
    // data only (C# line 852). Throws std::invalid_argument (C# ArgumentException) when
    // the exact series has errors, has fewer than 10 items, or the threshold would censor
    // more than 50 percent of the values.
    void set_low_outliers_from_threshold() {
        if (!exact_series_.validate().is_valid)
            throw std::invalid_argument("The exact data series has errors.");
        if (exact_series_.count() < 10)
            throw std::invalid_argument(
                "The exact data series must have at least 10 items before evaluating low "
                "outliers.");
        if (low_outlier_threshold_ > exact_series_.upper_middle_value())
            throw std::invalid_argument(
                "The low outlier threshold value cannot be set to a value that would "
                "censor more than 50 percent of the values.");

        number_of_low_outliers_ = 0;
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            bool is_low = exact_series_[i].value() < low_outlier_threshold_;
            exact_series_[i].set_is_low_outlier(is_low);
            if (is_low) number_of_low_outliers_ += 1;
        }
    }

    // Create a deep copy of the data frame (C# Clone, line 1907, which round-trips
    // through XElement; the direct deep clone here preserves the same state: the four
    // series, NumberOfLowOutliers, LowOutlierThreshold, PlottingParameter, and Lambda).
    // Like the C#, the clone starts with an empty full time series; the first
    // full_time_series() access triggers a lazy rebuild on the clone.
    DataFrame clone() const {
        DataFrame copy;
        copy.exact_series_ = exact_series_.clone();
        copy.uncertain_series_ = uncertain_series_.clone();
        copy.interval_series_ = interval_series_.clone();
        copy.threshold_series_ = threshold_series_.clone();
        copy.lambda_ = lambda_;
        copy.number_of_low_outliers_ = number_of_low_outliers_;
        copy.low_outlier_threshold_ = low_outlier_threshold_;
        copy.plotting_parameter_ = plotting_parameter_;
        return copy;
    }

   private:
    static void append(ValidationResult& result, const ValidationResult& partial) {
        if (partial.is_valid) return;
        result.validation_messages.insert(result.validation_messages.end(),
                                          partial.validation_messages.begin(),
                                          partial.validation_messages.end());
        result.is_valid = false;
    }

    ExactSeries exact_series_;
    UncertainSeries uncertain_series_;
    IntervalSeries interval_series_;
    ThresholdSeries threshold_series_;
    std::vector<std::unique_ptr<Data>> full_time_series_;

    double lambda_ = 1.0;
    int number_of_low_outliers_ = 0;
    double low_outlier_threshold_ = 0;
    double plotting_parameter_ = 0.0;
};

// ---------------------------------------------------------------------------
// Out-of-line series Validate definitions (declared in interval_series.hpp /
// uncertain_series.hpp; they cross-reference the DataFrame defined above). Both keep the
// C# quirk of running the duplicate-index and overlap checks INSIDE the per-item loop
// (so an empty series skips them and a multi-item series repeats them).
// ---------------------------------------------------------------------------

// C# IntervalSeries.Validate(DataFrame), line 153.
inline ValidationResult IntervalSeries::validate(const DataFrame* data_frame) const {
    ValidationResult result;

    for (std::size_t i = 0; i < count(); i++) {
        ValidationResult data_valid = (*this)[i].validate();
        if (!data_valid.is_valid) {
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
            result.is_valid = false;
        }

        // Check for duplicate indexes (in-loop, per the C#).
        std::vector<int> duplicates = duplicate_indices();
        if (!duplicates.empty()) {
            std::string joined;
            for (std::size_t k = 0; k < duplicates.size(); k++) {
                if (k > 0) joined += ", ";
                joined += std::to_string(duplicates[k]);
            }
            result.validation_messages.push_back(
                "Error: Duplicate indexes found in interval series: " + joined);
            result.is_valid = false;
        }

        // Check for overlaps with the exact and uncertain series.
        if (data_frame != nullptr) {
            std::unordered_set<int> exact_indexes;
            for (std::size_t k = 0; k < data_frame->exact_series().count(); k++)
                exact_indexes.insert(data_frame->exact_series()[k].index());
            std::unordered_set<int> uncertain_indexes;
            for (std::size_t k = 0; k < data_frame->uncertain_series().count(); k++)
                uncertain_indexes.insert(data_frame->uncertain_series()[k].index());

            for (std::size_t k = 0; k < count(); k++) {
                int idx = (*this)[k].index();
                if (exact_indexes.count(idx) != 0) {
                    result.validation_messages.push_back(
                        "Error: Interval data at index " + std::to_string(idx) +
                        " overlaps with exact data.");
                    result.is_valid = false;
                }
                if (uncertain_indexes.count(idx) != 0) {
                    result.validation_messages.push_back(
                        "Error: Interval data at index " + std::to_string(idx) +
                        " overlaps with uncertain data.");
                    result.is_valid = false;
                }
            }
        }
    }

    return result;
}

// C# UncertainSeries.Validate(DataFrame), line 153.
inline ValidationResult UncertainSeries::validate(const DataFrame* data_frame) const {
    ValidationResult result;

    for (std::size_t i = 0; i < count(); i++) {
        ValidationResult data_valid = (*this)[i].validate();
        if (!data_valid.is_valid) {
            result.validation_messages.insert(result.validation_messages.end(),
                                              data_valid.validation_messages.begin(),
                                              data_valid.validation_messages.end());
            result.is_valid = false;
        }

        // Check for duplicate indexes (in-loop, per the C#).
        std::vector<int> duplicates = duplicate_indices();
        if (!duplicates.empty()) {
            std::string joined;
            for (std::size_t k = 0; k < duplicates.size(); k++) {
                if (k > 0) joined += ", ";
                joined += std::to_string(duplicates[k]);
            }
            result.validation_messages.push_back(
                "Error: Duplicate indexes found in uncertain series: " + joined);
            result.is_valid = false;
        }

        // Check for overlaps with the exact series.
        if (data_frame != nullptr) {
            std::unordered_set<int> exact_indexes;
            for (std::size_t k = 0; k < data_frame->exact_series().count(); k++)
                exact_indexes.insert(data_frame->exact_series()[k].index());

            for (std::size_t k = 0; k < count(); k++) {
                int idx = (*this)[k].index();
                if (exact_indexes.count(idx) != 0) {
                    result.validation_messages.push_back(
                        "Error: Uncertain data at index " + std::to_string(idx) +
                        " overlaps with exact data.");
                    result.is_valid = false;
                }
            }
        }
    }

    return result;
}

}  // namespace bestfit::models
