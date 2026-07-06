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
//      this ran automatically on every collection change; calculate_plotting_positions()
//      calls it first (as the C# method itself does, lines 1142-1144), restoring the
//      upstream trigger point for plotting-position consumers.
//   3. Lambda is recomputed EXPLICITLY via calculate_lambda() (upstream: triggered by
//      exact-series collection-changed events) or pinned via set_lambda().
//
// Ported surface: the four owned series, FullTimeSeries/CreateFullTimeSeries,
// ProcessThresholdSeries, TotalRecordLength, ZeroValueRelativeFrequency, Lambda
// (SetLambda/CalculateLambda), the low-outlier surface (NumberOfLowOutliers,
// LowOutlierThreshold, ClearLowOutliers, SetLowOutliersFromMGBT,
// SetLowOutliersFromThreshold), the plain PlottingParameter property, Validate(), a
// direct deep Clone() (the C# clones via an XElement round trip; the direct clone has the
// same observable result, including the empty lazily-rebuilt full series noted in the C#
// remarks), (M5) CalculatePlottingPositions / ApplyLangbeinConversion -- the
// Hirsch-Stedinger censored plotting positions, defined out-of-line in
// data_frame_plotting.hpp (included at the bottom of this file) -- and (A3) the
// bootstrap/resampling surface: JackKnife, Resample, BootstrapDataFrame, and
// ShiftDistribution (the `#region Bootstrap Methods`, DataFrame.cs 2059-2543).
// Invalidation contract: BootstrapDataFrame and Resample call process_threshold_series()
// as the last (or only-when-create_full_time_series) step matching the C#; the C#
// SuppressCollectionChanged event-suppression lines have no C++ equivalent (no events) and
// are dropped. ShiftDistribution is C# `private static`; the C++ port exposes it
// `public static` (access modifier only, per the ThresholdData::set_number_below
// internal->public precedent) so the arm-by-arm shift is directly testable.
//
// Deliberately NOT ported:
//   - the hypothesis-test facade (JarqueBera/LjungBox/EqualVarianceTtest/
//     UnequalVarianceTtest/Ftest/LinearTrend/Unimodality/WaldWolfowitz/MannWhitney/
//     MannKendall/SummaryHypothesisTest -- Numerics HypothesisTests is unported)
//   - the summary-statistics / Q-Q surface (SummaryStatisticsExactDataOnly,
//     SummaryStatisticsAllData, SetStandardizedValues -- they need further
//     EmpiricalDistribution facades). GetNonparametricMoments and
//     GetNonparametricMomentsROS were ported ADDITIVELY in B9 (Bulletin17CDistribution's
//     SetInitialParameters/SetDefaultParameters call them); the C# `double[]?` null
//     return maps to std::optional<std::vector<double>> (empty optional == C# null).
//   - CreateBlockSeries / CreatePeaksOverThresholdSeries (need the unported TimeSeries
//     container), CreateFromUSGS + USGSRawText (network import)
//   - XML (ToXElement / XElement constructor), INotifyPropertyChanged, and the
//     concurrency machinery (_syncRoot, Volatile, SnapshotNonNull)
//
// FOLLOW-UPS (M5 ledger finding): create_full_time_series() below sorts the combined series
// with std::stable_sort, while M5 proved the .NET List<T>.Sort introsort tie order IS
// oracle-visible in plotting positions (data_frame_plotting.hpp carries the faithful
// detail::dotnet_list_sort port for its three internal sorts for exactly that reason). No
// tie-sensitive oracle exercises THIS sort today, but a future oracle pinned to a frame with
// equal-index ties could trip on the tie order; if one does, switch this sort to the
// dotnet_list_sort port rather than loosening the fixture. Do not change the sort without a
// tie-sensitive oracle proving which order the C# produces here.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
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
#include "bestfit/numerics/data/regression/linear_regression.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/binomial.hpp"
#include "bestfit/numerics/distributions/empirical_distribution.hpp"
#include "bestfit/numerics/distributions/gamma_distribution.hpp"
#include "bestfit/numerics/distributions/generalized_beta.hpp"
#include "bestfit/numerics/distributions/ln_normal.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/pert.hpp"
#include "bestfit/numerics/distributions/student_t.hpp"
#include "bestfit/numerics/distributions/triangular.hpp"
#include "bestfit/numerics/distributions/truncated_normal.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "bestfit/numerics/tools.hpp"
#include "bestfit/numerics/utilities/extension_methods.hpp"

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
    // returned list must be treated as read-only. Const (M9): the C# getter is reachable
    // from read-only paths (the nonstationary likelihoods evaluate it on every call), so
    // the cache member is `mutable` and the getter/rebuild are logically const.
    const std::vector<std::unique_ptr<Data>>& full_time_series() const {
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
    // alternatives: 0.40 Cunnane, 0.44 Gringorten, 0.50 Hazen). The setter recomputes
    // plotting positions on change, mirroring the C#. ---
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

    // Hirsch-Stedinger censored plotting positions (C# CalculatePlottingPositions,
    // line 1140). Defined in data_frame_plotting.hpp (included below); calls
    // process_threshold_series() first, exactly like the C#.
    void calculate_plotting_positions();

    // Apply the Langbein conversion to the plotting positions (C#
    // ApplyLangbeinConversion, line 1458). Defined in data_frame_plotting.hpp.
    void apply_langbein_conversion(double lambda);

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
    // concurrency snapshot/retry machinery is not ported). Const (M9): only refreshes the
    // mutable full-time-series cache, so the lazy const getter above can call it.
    void create_full_time_series() const {
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

    // Computes nonparametric central moments [mean, stdDev, skewness, kurtosis] of the
    // data, optionally log10-transformed (C# GetNonparametricMoments, line 1659; ported
    // additively in B9). Combines exact, uncertain, and interval data with their
    // Hirsch-Stedinger plotting position complements into an EmpiricalDistribution, then
    // computes central moments via numerical integration with 1000 points. Returns an
    // empty optional (the C# null) when there are fewer than 4 data points. Reference:
    // Hirsch, R.M. and Stedinger, J.R. (1987). Plotting positions for historical floods
    // and their precision. Water Resources Research, 23(4), 715-727.
    std::optional<std::vector<double>> get_nonparametric_moments(
        bool use_log10_values = false) const {
        if (exact_series_.count() < 4) return std::nullopt;

        int total_count = static_cast<int>(exact_series_.count() + uncertain_series_.count() +
                                           interval_series_.count());
        if (total_count < 4) return std::nullopt;

        // Build sorted values
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(total_count));
        if (use_log10_values) {
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                values.push_back(exact_series_[i].log10_value());
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].log10_value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].log10_value());
        } else {
            for (std::size_t i = 0; i < exact_series_.count(); i++)
                values.push_back(exact_series_[i].value());
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].value());
        }
        std::sort(values.begin(), values.end());

        // Build sorted plotting position complements
        std::vector<double> probs = plotting_position_complements();

        numerics::distributions::EmpiricalDistribution dist(std::move(values),
                                                            std::move(probs));
        return dist.central_moments(1000);
    }

    // Computes nonparametric central moments using Regression on Order Statistics (ROS)
    // to impute values for low outliers below the censoring threshold (C#
    // GetNonparametricMomentsROS, line 1734; ported additively in B9). Algorithm: fit a
    // simple linear regression of value vs. standard normal quantile z = Phi^-1(pp)
    // through the uncensored exact points only, then replace each low outlier's value
    // with the regression prediction at its z before computing the empirical moments.
    // Falls back to get_nonparametric_moments() when there are no low outliers or fewer
    // than 2 uncensored exact points; returns an empty optional (the C# null) when there
    // are fewer than 4 data points. References: Helsel, D.R. and Cohn, T.A. (1988).
    // Estimation of descriptive statistics for multiply censored water quality data.
    // Water Resources Research, 24(12), 1997-2004; Helsel, D.R. (2005). Nondetects and
    // Data Analysis. Wiley, New York.
    std::optional<std::vector<double>> get_nonparametric_moments_ros(
        bool use_log10_values = false) const {
        // Fall back to standard method when there are no low outliers to impute
        if (number_of_low_outliers_ == 0) return get_nonparametric_moments(use_log10_values);

        if (exact_series_.count() < 4) return std::nullopt;

        int total_count = static_cast<int>(exact_series_.count() + uncertain_series_.count() +
                                           interval_series_.count());
        if (total_count < 4) return std::nullopt;

        // Separate exact data into uncensored and censored (low outlier) sets
        std::vector<double> uncensored_values;
        std::vector<double> uncensored_quantiles;
        numerics::distributions::Normal std_normal(0, 1);

        // Build paired (value, quantile) lists for exact series
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            double value = use_log10_values ? exact_series_[i].log10_value()
                                            : exact_series_[i].value();
            double z = std_normal.inverse_cdf(exact_series_[i].plotting_position_complement());

            if (!exact_series_[i].is_low_outlier()) {
                uncensored_values.push_back(value);
                uncensored_quantiles.push_back(z);
            }
        }

        // Need at least 2 uncensored points to fit a regression line
        if (uncensored_values.size() < 2) return get_nonparametric_moments(use_log10_values);

        // Fit linear regression: value = a + b * z using uncensored points only
        // (the C# single-column Matrix(double[]) ctor is not on the ported Matrix; the
        // same layout is built through the (rows, cols) ctor).
        numerics::math::linalg::Matrix x_matrix(
            static_cast<int>(uncensored_quantiles.size()), 1);
        for (std::size_t i = 0; i < uncensored_quantiles.size(); i++)
            x_matrix(static_cast<int>(i), 0) = uncensored_quantiles[i];
        numerics::math::linalg::Vector y_vector(uncensored_values);
        numerics::data::regression::LinearRegression regression(x_matrix, y_vector, true);
        double intercept = regression.parameters()[0];
        double slope = regression.parameters()[1];

        // Build combined values list with ROS-imputed values for low outliers
        std::vector<double> values;
        values.reserve(static_cast<std::size_t>(total_count));
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            if (exact_series_[i].is_low_outlier()) {
                // Impute from regression line at this point's normal quantile
                double z =
                    std_normal.inverse_cdf(exact_series_[i].plotting_position_complement());
                values.push_back(intercept + slope * z);
            } else {
                values.push_back(use_log10_values ? exact_series_[i].log10_value()
                                                  : exact_series_[i].value());
            }
        }

        // Add uncertain and interval series values (not subject to low-outlier imputation)
        if (use_log10_values) {
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].log10_value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].log10_value());
        } else {
            for (std::size_t i = 0; i < uncertain_series_.count(); i++)
                values.push_back(uncertain_series_[i].value());
            for (std::size_t i = 0; i < interval_series_.count(); i++)
                values.push_back(interval_series_[i].value());
        }
        std::sort(values.begin(), values.end());

        // Build sorted plotting position complements
        std::vector<double> probs = plotting_position_complements();

        numerics::distributions::EmpiricalDistribution dist(std::move(values),
                                                            std::move(probs));
        return dist.central_moments(1000);
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

    // ===================== Bootstrap Methods (A3) =====================
    // Ported from the C# `#region Bootstrap Methods` (DataFrame.cs 2059-2543). The C#
    // suppresses collection-changed events for speed inside these methods
    // (SuppressCollectionChanged = true / false); the C++ port has no events, so those
    // lines drop (the invalidation strategy is documented at the top of this file --
    // recompute explicitly via process_threshold_series() / lazily via full_time_series()).

    // Generate a jackknife data frame by leaving out one observation (C# JackKnife,
    // line 2066). Returns a new reduced frame: clone() the frame, then remove the single
    // observation at the global `index`. The C# searches the series in order Exact ->
    // Uncertain -> Interval to remove the ordinate; for a Threshold hit it decrements the
    // covering threshold's NumberAbove/NumberBelow by the sub-range (above region
    // [StartIndex, StartIndex + NumberAbove); below region [StartIndex + NumberAbove,
    // EndIndex]) rather than erasing a stored point. Consumed by A8 AccelerationConstants.
    DataFrame JackKnife(int index) {
        DataFrame dataframe = clone();

        // Exact data
        for (std::size_t i = 0; i < dataframe.exact_series_.count(); i++) {
            if (dataframe.exact_series_[i].index() == index) {
                dataframe.exact_series_.remove_at(i);
                return dataframe;
            }
        }
        // Uncertain data
        for (std::size_t i = 0; i < dataframe.uncertain_series_.count(); i++) {
            if (dataframe.uncertain_series_[i].index() == index) {
                dataframe.uncertain_series_.remove_at(i);
                return dataframe;
            }
        }
        // Interval data
        for (std::size_t i = 0; i < dataframe.interval_series_.count(); i++) {
            if (dataframe.interval_series_[i].index() == index) {
                dataframe.interval_series_.remove_at(i);
                return dataframe;
            }
        }
        // Threshold data
        for (std::size_t t = 0; t < dataframe.threshold_series_.count(); t++) {
            ThresholdData& data = dataframe.threshold_series_[t];
            // Number above: [StartIndex, StartIndex + NumberAbove)
            for (int i = data.start_index(); i < data.start_index() + data.number_above();
                 i++) {
                if (index == i) {
                    data.set_number_above(data.number_above() - 1);
                    return dataframe;
                }
            }
            // Number below: [StartIndex + NumberAbove, EndIndex]
            for (int i = data.start_index() + data.number_above(); i <= data.end_index();
                 i++) {
                if (index == i) {
                    data.set_number_below(data.number_below() - 1);
                    return dataframe;
                }
            }
        }

        return dataframe;
    }

    // Generate a resampled data frame using nonparametric bootstrap resampling with
    // replacement (C# Resample, line 2164). Draws indices via the ranged
    // next_integers(prng, startIndex, endIndex + 1, FullTimeSeries.Count, replace = true),
    // sorts them, then rebuilds the frame by index lookup across all four series (including
    // the threshold bins above/below). The above/below split uses the C# Resample
    // convention: above region [StartIndex, StartIndex + NumberAbove); below region
    // [StartIndex + NumberAbove, EndIndex].
    DataFrame Resample(numerics::sampling::MersenneTwister& prng,
                       bool create_full_time_series = false) {
        // FullTimeSeries getter rebuilds lazily when empty (C# CreateFullTimeSeries guard).
        const std::vector<std::unique_ptr<Data>>& full = full_time_series();
        DataFrame dataframe;
        if (full.empty()) return dataframe;

        int start_index = full.front()->index();
        int end_index = full.back()->index();
        std::vector<int> indexes = numerics::utilities::next_integers(
            prng, start_index, end_index + 1, static_cast<int>(full.size()), true);
        std::sort(indexes.begin(), indexes.end());

        for (std::size_t k = 0; k < indexes.size(); k++) {
            int idx = indexes[k];
            bool found = false;

            // Exact data
            for (std::size_t i = 0; i < exact_series_.count(); i++) {
                if (exact_series_[i].index() == idx) {
                    dataframe.exact_series_.add(exact_series_[i].clone());
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // Uncertain data
            for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
                if (uncertain_series_[i].index() == idx) {
                    dataframe.uncertain_series_.add(uncertain_series_[i].clone());
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // Interval data
            for (std::size_t i = 0; i < interval_series_.count(); i++) {
                if (interval_series_[i].index() == idx) {
                    dataframe.interval_series_.add(interval_series_[i].clone());
                    found = true;
                    break;
                }
            }
            if (found) continue;

            // Threshold data - check each threshold period
            for (std::size_t i = 0; i < threshold_series_.count(); i++) {
                const ThresholdData& threshold = threshold_series_[i];
                // Above-threshold region
                if (idx >= threshold.start_index() &&
                    idx < threshold.start_index() + threshold.number_above()) {
                    ThresholdData* existing = find_threshold_bin(
                        dataframe.threshold_series_, threshold.start_index(),
                        threshold.end_index());
                    if (existing != nullptr) {
                        existing->set_number_above(existing->number_above() + 1);
                    } else {
                        ThresholdData new_threshold = threshold.clone();
                        new_threshold.set_number_above(1);
                        new_threshold.set_number_below(0);
                        dataframe.threshold_series_.add(std::move(new_threshold));
                    }
                    found = true;
                    break;
                }
                // Below-threshold region
                if (idx >= threshold.start_index() + threshold.number_above() &&
                    idx <= threshold.end_index()) {
                    ThresholdData* existing = find_threshold_bin(
                        dataframe.threshold_series_, threshold.start_index(),
                        threshold.end_index());
                    if (existing != nullptr) {
                        existing->set_number_below(existing->number_below() + 1);
                    } else {
                        ThresholdData new_threshold = threshold.clone();
                        new_threshold.set_number_above(0);
                        new_threshold.set_number_below(1);
                        dataframe.threshold_series_.add(std::move(new_threshold));
                    }
                    found = true;
                    break;
                }
            }
        }

        if (create_full_time_series) dataframe.create_full_time_series();

        return dataframe;
    }

    // Generates a parametric bootstrap data frame by simulating the physical observation
    // process (C# BootstrapDataFrame, line 2336). Per-series behaviour mirrors the C#:
    // exact -> unconditional inverse_cdf(U); uncertain -> shift the measurement-error
    // distribution to center on a simulated value; interval -> reclassify against the
    // original bounds; threshold -> clone systematic (NumberBelow == 0) bins, else resample
    // the exceedance count from Binomial(n, 1 - F(threshold)). Finally re-flags low
    // outliers and calls process_threshold_series() (the C# ordering). Consumed by A8's
    // parametric bootstrap. Reference: Davison, A.C. and Hinkley, D.V. (1997). Bootstrap
    // Methods and Their Application, Sections 3.5 and 7.3.
    DataFrame BootstrapDataFrame(
        const numerics::distributions::UnivariateDistributionBase& distribution,
        numerics::sampling::MersenneTwister& prng, bool create_full_time_series = false) {
        DataFrame dataframe;
        bool filter_low_outliers = number_of_low_outliers_ > 0;

        // Exact data: unconditional draw from the fitted distribution.
        for (std::size_t i = 0; i < exact_series_.count(); i++) {
            double simulated_value = distribution.inverse_cdf(prng.next_double());
            dataframe.exact_series_.add(ExactData(exact_series_[i].index(), simulated_value));
        }

        // Uncertain data: draw a "true" magnitude, then shift the error distribution.
        for (std::size_t i = 0; i < uncertain_series_.count(); i++) {
            double simulated_value = distribution.inverse_cdf(prng.next_double());
            auto shifted_dist =
                shift_distribution(uncertain_series_[i].distribution(), simulated_value);
            dataframe.uncertain_series_.add(
                UncertainData(uncertain_series_[i].index(), std::move(shifted_dist)));
        }

        // Interval data: unconditional draw, then re-classify against original bounds.
        for (std::size_t i = 0; i < interval_series_.count(); i++) {
            const IntervalData& data = interval_series_[i];
            double simulated_value = distribution.inverse_cdf(prng.next_double());
            if (simulated_value < data.lower_value()) {
                double upper = data.lower_value();
                double lower =
                    std::min(upper - 1E-8,
                             distribution.inverse_cdf(numerics::kDoubleMachineEpsilon));
                double mid = 0.5 * (lower + upper);
                dataframe.interval_series_.add(
                    IntervalData(data.index(), lower, mid, upper));
            } else if (simulated_value > data.upper_value()) {
                double lower = data.upper_value();
                double upper = std::max(
                    lower + 1E-8,
                    distribution.inverse_cdf(1.0 - numerics::kDoubleMachineEpsilon));
                double mid = 0.5 * (lower + upper);
                dataframe.interval_series_.add(
                    IntervalData(data.index(), lower, mid, upper));
            } else {
                dataframe.interval_series_.add(IntervalData(
                    data.index(), data.lower_value(),
                    0.5 * (data.lower_value() + data.upper_value()), data.upper_value()));
            }
        }

        // Threshold data: systematic clone vs. historical Binomial resample.
        for (std::size_t i = 0; i < threshold_series_.count(); i++) {
            const ThresholdData& data = threshold_series_[i];
            if (data.number_below() == 0) {
                // Systematic threshold: clone with original counts to avoid spurious
                // NumberAbove double-counting the exact observations.
                dataframe.threshold_series_.add(data.clone());
            } else {
                // Historical threshold: resample the exceedance count from Binomial(n, p),
                // p = P(X > threshold) under the fitted distribution.
                double p = 1.0 - distribution.cdf(data.value());
                int n = data.duration() - data.number_above();
                int n_above;
                if (n <= 0 || p <= 0.0) {
                    n_above = 0;
                } else if (p >= 1.0) {
                    n_above = n;
                } else {
                    numerics::distributions::Binomial binomial_dist(p, n);
                    n_above = std::max(
                        0, static_cast<int>(std::floor(
                               binomial_dist.inverse_cdf(prng.next_double()))));
                    n_above = std::min(n_above, n);
                }
                int n_below = data.duration() - n_above;
                ThresholdData new_threshold(data.start_index(), data.end_index(),
                                            data.value());
                new_threshold.set_number_above(n_above);
                new_threshold.set_number_below(n_below);
                dataframe.threshold_series_.add(std::move(new_threshold));
            }
        }

        // Low outliers: re-flag resampled exact values below the threshold. Inline marking
        // (not set_low_outliers_from_threshold) to avoid the >50%-censored validation guard.
        if (filter_low_outliers) {
            dataframe.low_outlier_threshold_ = low_outlier_threshold_;
            dataframe.number_of_low_outliers_ = 0;
            for (std::size_t i = 0; i < dataframe.exact_series_.count(); i++) {
                if (dataframe.exact_series_[i].value() < low_outlier_threshold_) {
                    dataframe.exact_series_[i].set_is_low_outlier(true);
                    dataframe.number_of_low_outliers_++;
                }
            }
        }

        // Post-processing (mirror the C# ordering of the final steps).
        dataframe.process_threshold_series();
        if (create_full_time_series) dataframe.create_full_time_series();

        return dataframe;
    }

    // Creates a new distribution of the same type, shifted so that its center is at
    // new_center while preserving the original measurement error spread (C# ShiftDistribution,
    // line 2485). For additive-error families the distribution is shifted by
    // `new_center - original.Mean`; for multiplicative-error families (LogNormal, Gamma) a
    // ratio-based shift preserves the CV. Unrecognized types fall back to a clone.
    //
    // The C# declares this `private static`; the C++ port exposes it `public static`
    // following the ThresholdData::set_number_below precedent (a C# `internal`/`private`
    // widened for DataFrame and for the C++-only test harness, not for end users) so the
    // arm-by-arm shift behaviour is directly testable. Access modifier only -- no numerical
    // deviation.
    static std::unique_ptr<numerics::distributions::UnivariateDistributionBase>
    shift_distribution(
        const numerics::distributions::UnivariateDistributionBase& original,
        double new_center) {
        namespace nd = numerics::distributions;
        double original_mean = original.mean();
        double shift = new_center - original_mean;

        // Guard against degenerate cases (C#: double.IsNaN || double.IsInfinity).
        if (std::isnan(shift) || std::isinf(shift)) return original.clone();

        switch (original.type()) {
            case nd::UnivariateDistributionType::Normal: {
                const auto& n = static_cast<const nd::Normal&>(original);
                return std::make_unique<nd::Normal>(n.mu() + shift, n.sigma());
            }
            case nd::UnivariateDistributionType::TruncatedNormal: {
                const auto& tn = static_cast<const nd::TruncatedNormal&>(original);
                return std::make_unique<nd::TruncatedNormal>(
                    tn.mu() + shift, tn.sigma(), tn.min_param() + shift,
                    tn.max_param() + shift);
            }
            case nd::UnivariateDistributionType::StudentT: {
                const auto& st = static_cast<const nd::StudentT&>(original);
                return std::make_unique<nd::StudentT>(st.mu() + shift, st.sigma(),
                                                      st.degrees_of_freedom());
            }
            case nd::UnivariateDistributionType::LogNormal: {
                const auto& ln = static_cast<const nd::LogNormal&>(original);
                // Multiplicative shift in log-space preserves the CV.
                double ratio = (original_mean > 0 && new_center > 0)
                                   ? new_center / original_mean
                                   : 1.0;
                double new_mu = ln.mu() + std::log(std::max(ratio, 1e-12));
                return std::make_unique<nd::LogNormal>(new_mu, ln.sigma());
            }
            case nd::UnivariateDistributionType::LnNormal: {
                const auto& lnn = static_cast<const nd::LnNormal&>(original);
                // C# uses the real-space (Mean, StandardDeviation) constructor.
                return std::make_unique<nd::LnNormal>(lnn.mean() + shift,
                                                      lnn.standard_deviation());
            }
            case nd::UnivariateDistributionType::GammaDistribution: {
                const auto& g = static_cast<const nd::GammaDistribution&>(original);
                // Scale shift preserves shape (and CV).
                double ratio = (original_mean > 0 && new_center > 0)
                                   ? new_center / original_mean
                                   : 1.0;
                double new_scale = g.theta() * ratio;
                return std::make_unique<nd::GammaDistribution>(std::max(new_scale, 1e-12),
                                                               g.kappa());
            }
            case nd::UnivariateDistributionType::Uniform: {
                const auto& u = static_cast<const nd::Uniform&>(original);
                return std::make_unique<nd::Uniform>(u.min() + shift, u.max() + shift);
            }
            case nd::UnivariateDistributionType::Triangular: {
                const auto& t = static_cast<const nd::Triangular&>(original);
                return std::make_unique<nd::Triangular>(
                    t.min_val() + shift, t.most_likely() + shift, t.max_val() + shift);
            }
            case nd::UnivariateDistributionType::Pert: {
                const auto& p = static_cast<const nd::Pert&>(original);
                return std::make_unique<nd::Pert>(p.min_val() + shift,
                                                  p.most_likely() + shift,
                                                  p.max_val() + shift);
            }
            case nd::UnivariateDistributionType::GeneralizedBeta: {
                const auto& gb = static_cast<const nd::GeneralizedBeta&>(original);
                return std::make_unique<nd::GeneralizedBeta>(
                    gb.alpha(), gb.beta(), gb.min_val() + shift, gb.max_val() + shift);
            }
            default:
                // Unrecognized distribution type -- clone as-is. The C# logs a
                // Debug.WriteLine here; ported as a silent no-throw path per the global
                // constraint.
                return original.clone();
        }
    }

   private:
    // Finds the threshold bin in `series` matching (start_index, end_index) -- the C#
    // FirstOrDefault(t => t.StartIndex == ... && t.EndIndex == ...) used by Resample.
    // Returns nullptr when no bin matches.
    static ThresholdData* find_threshold_bin(ThresholdSeries& series, int start_index,
                                             int end_index) {
        for (std::size_t i = 0; i < series.count(); i++) {
            if (series[i].start_index() == start_index &&
                series[i].end_index() == end_index) {
                return &series[i];
            }
        }
        return nullptr;
    }

    // The sorted plotting-position complements of the exact + uncertain + interval series
    // (the shared tail of both C# GetNonparametricMoments methods; B9).
    std::vector<double> plotting_position_complements() const {
        std::vector<double> probs;
        probs.reserve(exact_series_.count() + uncertain_series_.count() +
                      interval_series_.count());
        for (std::size_t i = 0; i < exact_series_.count(); i++)
            probs.push_back(exact_series_[i].plotting_position_complement());
        for (std::size_t i = 0; i < uncertain_series_.count(); i++)
            probs.push_back(uncertain_series_[i].plotting_position_complement());
        for (std::size_t i = 0; i < interval_series_.count(); i++)
            probs.push_back(interval_series_[i].plotting_position_complement());
        std::sort(probs.begin(), probs.end());
        return probs;
    }

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
    // Mutable: a logically-const cache, lazily rebuilt by the const full_time_series()
    // getter exactly as the C# property getter does (see the M9 note there).
    mutable std::vector<std::unique_ptr<Data>> full_time_series_;

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

// Out-of-line definitions of DataFrame::calculate_plotting_positions() and
// DataFrame::apply_langbein_conversion() (split out purely for file size; the header
// must not be included directly).
#include "bestfit/models/data_frame/data_frame_plotting.hpp"  // NOLINT
