// Standalone tests for the DataFrame data types (M3): the SeriesOrdinate generic base and
// the censored-data ordinates Data / ExactData / IntervalData / ThresholdData / UncertainData.
//
// Oracles are the upstream C# test classes under
// upstream/RMC-BestFit/src/RMC.BestFit.Tests/DataFrame/ @ fc28c0c:
//   ExactDataTests.cs, IntervalDataTests.cs, ThresholdDataTests.cs, UncertainDataTests.cs,
// transcribed method-for-method below (same section order). Deliberately NOT transcribed
// (project-wide deferrals, per the ported headers):
//   - the "Serialization Tests" regions and every Test_Constructor_XElement_* /
//     Test_RoundTrip_* method (ToXElement / XElement round-trips -- XML is not ported)
//   - the "PropertyChanged Tests" regions (INotifyPropertyChanged is not ported)
//   - ExactData's DateTime surface: Test_Constructor_WithDateTime_SetsIndexFromYear,
//     Test_DateTime_PreservedFromConstructor, Test_Clone_PreservesDateTime,
//     Test_SystematicRecord (DateTime is deferred project-wide -- seasonal path only)
// ThresholdData tests that used the C# XElement helper to reach the internal NumberBelow
// setter use set_number_below() directly here (the C++ port has no `internal`; see the
// header note in threshold_data.hpp).
//
// SeriesOrdinate<TIndex, TValue> has no upstream test class; its Equals/Clone surface is
// exercised directly below. These are structural/behavioral ports validated against the C#
// source itself, so there is no fixtures/ entry for this file.
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/models/data_frame/data_types/data.hpp"
#include "bestfit/models/data_frame/data_types/exact_data.hpp"
#include "bestfit/models/data_frame/data_types/interval_data.hpp"
#include "bestfit/models/data_frame/data_types/threshold_data.hpp"
#include "bestfit/models/data_frame/data_types/uncertain_data.hpp"
#include "bestfit/numerics/data/series_ordinate.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/triangular.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "check.hpp"

using bestfit::models::ExactData;
using bestfit::models::IntervalData;
using bestfit::models::ThresholdData;
using bestfit::models::UncertainData;
using bestfit::models::ValidationResult;
using bestfit::numerics::data::SeriesOrdinate;
using bestfit::numerics::distributions::LogNormal;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::Triangular;
using bestfit::numerics::distributions::Uniform;
using bestfit::numerics::distributions::UnivariateDistributionBase;
using bestfit::numerics::distributions::UnivariateDistributionType;

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

// Mirrors the C# LINQ `messages.Any(m => m.Contains(substr))`.
bool any_contains(const std::vector<std::string>& messages, const std::string& substr) {
    for (const auto& m : messages) {
        if (m.find(substr) != std::string::npos) return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// SeriesOrdinate<TIndex, TValue>: base contract (no upstream test class)
// ---------------------------------------------------------------------------

void test_series_ordinate_default_constructor() {
    SeriesOrdinate<int, double> ord;
    CHECK_EQ(ord.index(), 0);
    CHECK_NEAR(ord.value(), 0.0, 0.0);
}

void test_series_ordinate_constructor_sets_index_and_value() {
    SeriesOrdinate<int, double> ord(1985, 75000.0);
    CHECK_EQ(ord.index(), 1985);
    CHECK_NEAR(ord.value(), 75000.0, 0.0);
}

void test_series_ordinate_set_and_get() {
    SeriesOrdinate<int, double> ord;
    ord.set_index(2010);
    ord.set_value(50000.0);
    CHECK_EQ(ord.index(), 2010);
    CHECK_NEAR(ord.value(), 50000.0, 0.0);
}

// C# Equals / operator== compare index and value only.
void test_series_ordinate_equality() {
    SeriesOrdinate<int, double> a(1985, 75000.0);
    SeriesOrdinate<int, double> b(1985, 75000.0);
    SeriesOrdinate<int, double> c(1986, 75000.0);
    SeriesOrdinate<int, double> d(1985, 80000.0);
    CHECK_TRUE(a == b);
    CHECK_TRUE(!(a == c));
    CHECK_TRUE(a != c);
    CHECK_TRUE(a != d);
    CHECK_TRUE(!(a != b));
}

void test_series_ordinate_clone() {
    SeriesOrdinate<int, double> original(1985, 75000.0);
    SeriesOrdinate<int, double> clone = original.clone();
    CHECK_TRUE(clone == original);
    original.set_value(1.0);
    CHECK_NEAR(clone.value(), 75000.0, 0.0);
}

// ---------------------------------------------------------------------------
// Data (abstract base, exercised through ExactData): derived getters
// ---------------------------------------------------------------------------

// C# PlottingPosition getter clamps to [0, 1]; the backing field stores the raw value.
void test_data_plotting_position_clamps_on_get() {
    ExactData data(1985, 75000.0, 1.5);
    CHECK_NEAR(data.plotting_position(), 1.0, 0.0);
    data.set_plotting_position(-0.5);
    CHECK_NEAR(data.plotting_position(), 0.0, 0.0);
    data.set_plotting_position(0.25);
    CHECK_NEAR(data.plotting_position(), 0.25, 0.0);
}

void test_data_plotting_position_complement() {
    ExactData data(1985, 75000.0, 0.25);
    CHECK_NEAR(data.plotting_position_complement(), 0.75, 1e-15);
    data.set_plotting_position(1.5);  // clamped to 1 on get
    CHECK_NEAR(data.plotting_position_complement(), 0.0, 0.0);
}

// C# Log10Value: Value < 0 -> NaN; Value == 0 -> Math.Log10(0.001); else Math.Log10(Value).
void test_data_log10_value_edge_cases() {
    ExactData negative(1985, -5.0);
    CHECK_TRUE(std::isnan(negative.log10_value()));

    ExactData zero(1985, 0.0);
    CHECK_NEAR(zero.log10_value(), std::log10(0.001), 1e-15);

    ExactData positive(1985, 1000.0);
    CHECK_NEAR(positive.log10_value(), 3.0, 1e-12);
}

void test_data_standardized_values_set_and_get() {
    ExactData data(1985, 75000.0);
    CHECK_NEAR(data.standardized_value(), 0.0, 0.0);
    CHECK_NEAR(data.standardized_log10_value(), 0.0, 0.0);
    data.set_standardized_value(1.23);
    data.set_standardized_log10_value(-0.5);
    CHECK_NEAR(data.standardized_value(), 1.23, 0.0);
    CHECK_NEAR(data.standardized_log10_value(), -0.5, 0.0);
}

// ---------------------------------------------------------------------------
// ExactData: Constructor Tests
// ---------------------------------------------------------------------------

// C# Test_Constructor_EmptyConstructor_CreatesDefaultInstance
void test_exact_constructor_empty_creates_default_instance() {
    ExactData data;
    CHECK_EQ(data.index(), 0);
    CHECK_NEAR(data.value(), 0.0, 0.0);
    CHECK_NEAR(data.plotting_position(), 0.0, 0.0);
    CHECK_TRUE(!data.is_low_outlier());
}

// C# Test_Constructor_WithIndexAndValue_SetsProperties
void test_exact_constructor_with_index_and_value_sets_properties() {
    ExactData data(1985, 75000.0);
    CHECK_EQ(data.index(), 1985);
    CHECK_NEAR(data.value(), 75000.0, 0.0);
    CHECK_NEAR(data.plotting_position(), 0.0, 0.0);
    CHECK_TRUE(!data.is_low_outlier());
}

// C# Test_Constructor_WithAllParameters_SetsAllProperties
void test_exact_constructor_with_all_parameters_sets_all_properties() {
    ExactData data(1985, 75000.0, 0.5, true);
    CHECK_EQ(data.index(), 1985);
    CHECK_NEAR(data.value(), 75000.0, 0.0);
    CHECK_NEAR(data.plotting_position(), 0.5, 0.0);
    CHECK_TRUE(data.is_low_outlier());
}

// ---------------------------------------------------------------------------
// ExactData: Property Tests
// ---------------------------------------------------------------------------

// C# Test_Index_SetAndGet
void test_exact_index_set_and_get() {
    ExactData data;
    data.set_index(2010);
    CHECK_EQ(data.index(), 2010);
}

// C# Test_Value_SetAndGet
void test_exact_value_set_and_get() {
    ExactData data;
    data.set_value(50000.0);
    CHECK_NEAR(data.value(), 50000.0, 0.0);
}

// C# Test_PlottingPosition_SetAndGet
void test_exact_plotting_position_set_and_get() {
    ExactData data;
    data.set_plotting_position(0.75);
    CHECK_NEAR(data.plotting_position(), 0.75, 0.0);
}

// C# Test_IsLowOutlier_SetAndGet
void test_exact_is_low_outlier_set_and_get() {
    ExactData data;
    data.set_is_low_outlier(true);
    CHECK_TRUE(data.is_low_outlier());
}

// ---------------------------------------------------------------------------
// ExactData: Validation Tests
// ---------------------------------------------------------------------------

// C# Test_Validate_ValidData_ReturnsTrue
void test_exact_validate_valid_data_returns_true() {
    ExactData data(1985, 75000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_NaNValue_ReturnsFalse
void test_exact_validate_nan_value_returns_false() {
    ExactData data(1985, kNaN);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "number"));
}

// C# Test_Validate_InfinityValue_ReturnsFalse
void test_exact_validate_infinity_value_returns_false() {
    ExactData data(1985, kInf);
    CHECK_TRUE(!data.validate().is_valid);
}

// C# Test_Validate_IndexTooLow_ReturnsFalse
void test_exact_validate_index_too_low_returns_false() {
    ExactData data(-200000, 75000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "index"));
}

// C# Test_Validate_IndexTooHigh_ReturnsFalse
void test_exact_validate_index_too_high_returns_false() {
    ExactData data(200000, 75000.0);
    CHECK_TRUE(!data.validate().is_valid);
}

// C# Test_Validate_NegativeValue_IsValid
void test_exact_validate_negative_value_is_valid() {
    // Negative values are valid (e.g., temperature, elevation)
    ExactData data(1985, -10.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_Validate_ZeroValue_IsValid
void test_exact_validate_zero_value_is_valid() {
    ExactData data(1985, 0.0);
    CHECK_TRUE(data.validate().is_valid);
}

// ---------------------------------------------------------------------------
// ExactData: Clone Tests
// ---------------------------------------------------------------------------

// C# Test_Clone_CreatesIndependentCopy
void test_exact_clone_creates_independent_copy() {
    ExactData original(1985, 75000.0, 0.5, true);

    ExactData clone = original.clone();

    // Verify values match
    CHECK_EQ(clone.index(), original.index());
    CHECK_NEAR(clone.value(), original.value(), 0.0);
    CHECK_NEAR(clone.plotting_position(), original.plotting_position(), 0.0);
    CHECK_TRUE(clone.is_low_outlier() == original.is_low_outlier());

    // Verify independence
    original.set_is_low_outlier(false);
    CHECK_TRUE(clone.is_low_outlier());
}

// ---------------------------------------------------------------------------
// ExactData: Flood Frequency Scenarios
// ---------------------------------------------------------------------------

// C# Test_FloodPeakFlow
void test_exact_flood_peak_flow() {
    ExactData data(1993, 350000.0);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_NEAR(data.value(), 350000.0, 0.0);
}

// C# Test_LowOutlier_SmallFlood
void test_exact_low_outlier_small_flood() {
    // Low outlier (censored at perception threshold)
    ExactData data(1988, 5000.0, 0.0, true);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_TRUE(data.is_low_outlier());
}

// C# Test_HistoricalFlood
void test_exact_historical_flood() {
    // Historical flood from 1889
    ExactData data(1889, 500000.0, 0.001);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_EQ(data.index(), 1889);
}

// ---------------------------------------------------------------------------
// ExactData: Edge Cases
// ---------------------------------------------------------------------------

// C# Test_VerySmallValue
void test_exact_very_small_value() {
    ExactData data(1985, 1e-10);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_NEAR(data.value(), 1e-10, 0.0);
}

// C# Test_VeryLargeValue
void test_exact_very_large_value() {
    ExactData data(1985, 1e15);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_NEAR(data.value(), 1e15, 0.0);
}

// C# Test_NegativeIndex_Paleoflood
void test_exact_negative_index_paleoflood() {
    // Very old paleoflood estimate (500 BCE)
    ExactData data(-500, 200000.0);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_EQ(data.index(), -500);
}

// C# Test_PlottingPosition_Range
void test_exact_plotting_position_range() {
    ExactData data(1985, 75000.0, 0.001);
    CHECK_TRUE(data.validate().is_valid);
    CHECK_NEAR(data.plotting_position(), 0.001, 0.0);
}

// ---------------------------------------------------------------------------
// IntervalData: Constructor Tests
// ---------------------------------------------------------------------------

// C# Test_Constructor_EmptyConstructor_CreatesDefaultInstance
void test_interval_constructor_empty_creates_default_instance() {
    IntervalData data;
    CHECK_EQ(data.index(), 0);
    CHECK_NEAR(data.value(), 0.0, 0.0);
    CHECK_NEAR(data.lower_value(), 0.0, 0.0);
    CHECK_NEAR(data.upper_value(), 0.0, 0.0);
    CHECK_NEAR(data.plotting_position(), 0.0, 0.0);
}

// C# Test_Constructor_WithParameters_SetsProperties
void test_interval_constructor_with_parameters_sets_properties() {
    IntervalData data(1850, 50000.0, 75000.0, 100000.0);
    CHECK_EQ(data.index(), 1850);
    CHECK_NEAR(data.lower_value(), 50000.0, 0.0);
    CHECK_NEAR(data.value(), 75000.0, 0.0);
    CHECK_NEAR(data.upper_value(), 100000.0, 0.0);
    CHECK_NEAR(data.plotting_position(), 0.0, 0.0);
}

// C# Test_Constructor_WithPlottingPosition_SetsAllProperties
void test_interval_constructor_with_plotting_position_sets_all_properties() {
    IntervalData data(1850, 50000.0, 75000.0, 100000.0, 0.01);
    CHECK_EQ(data.index(), 1850);
    CHECK_NEAR(data.lower_value(), 50000.0, 0.0);
    CHECK_NEAR(data.value(), 75000.0, 0.0);
    CHECK_NEAR(data.upper_value(), 100000.0, 0.0);
    CHECK_NEAR(data.plotting_position(), 0.01, 0.0);
}

// ---------------------------------------------------------------------------
// IntervalData: Property Tests
// ---------------------------------------------------------------------------

// C# Test_LowerValue_SetAndGet
void test_interval_lower_value_set_and_get() {
    IntervalData data;
    data.set_lower_value(40000.0);
    CHECK_NEAR(data.lower_value(), 40000.0, 0.0);
}

// C# Test_UpperValue_SetAndGet
void test_interval_upper_value_set_and_get() {
    IntervalData data;
    data.set_upper_value(120000.0);
    CHECK_NEAR(data.upper_value(), 120000.0, 0.0);
}

// C# Test_Log10LowerValue_ReturnsCorrectTransform
void test_interval_log10_lower_value_returns_correct_transform() {
    IntervalData data(1850, 100.0, 1000.0, 10000.0);
    CHECK_NEAR(data.log10_lower_value(), 2.0, 1e-10);  // log10(100)
}

// C# Test_Log10UpperValue_ReturnsCorrectTransform
void test_interval_log10_upper_value_returns_correct_transform() {
    IntervalData data(1850, 100.0, 1000.0, 10000.0);
    CHECK_NEAR(data.log10_upper_value(), 4.0, 1e-10);  // log10(10000)
}

// ---------------------------------------------------------------------------
// IntervalData: Validation Tests
// ---------------------------------------------------------------------------

// C# Test_Validate_ValidData_ReturnsTrue
void test_interval_validate_valid_data_returns_true() {
    IntervalData data(1850, 50000.0, 75000.0, 100000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_LowerGreaterThanValue_ReturnsFalse
void test_interval_validate_lower_greater_than_value_returns_false() {
    IntervalData data(1850, 80000.0, 75000.0, 100000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "lower") &&
               any_contains(result.validation_messages, "less"));
}

// C# Test_Validate_ValueGreaterThanUpper_ReturnsFalse
void test_interval_validate_value_greater_than_upper_returns_false() {
    IntervalData data(1850, 50000.0, 110000.0, 100000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "upper") &&
               any_contains(result.validation_messages, "greater"));
}

// C# Test_Validate_LowerGreaterThanUpper_ReturnsFalse
void test_interval_validate_lower_greater_than_upper_returns_false() {
    IntervalData data(1850, 100000.0, 75000.0, 50000.0);
    CHECK_TRUE(!data.validate().is_valid);
}

// C# Test_Validate_NaNLowerValue_ReturnsFalse
void test_interval_validate_nan_lower_value_returns_false() {
    IntervalData data(1850, kNaN, 75000.0, 100000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "lower"));
}

// C# Test_Validate_NaNValue_ReturnsFalse
void test_interval_validate_nan_value_returns_false() {
    IntervalData data(1850, 50000.0, kNaN, 100000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "most likely"));
}

// C# Test_Validate_NaNUpperValue_ReturnsFalse
void test_interval_validate_nan_upper_value_returns_false() {
    IntervalData data(1850, 50000.0, 75000.0, kNaN);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "upper"));
}

// C# Test_Validate_InfinityValue_ReturnsFalse
void test_interval_validate_infinity_value_returns_false() {
    IntervalData data(1850, 50000.0, kInf, 100000.0);
    CHECK_TRUE(!data.validate().is_valid);
}

// C# Test_Validate_IndexOutOfRange_ReturnsFalse
void test_interval_validate_index_out_of_range_returns_false() {
    IntervalData data(-200000, 50000.0, 75000.0, 100000.0);
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "index"));
}

// ---------------------------------------------------------------------------
// IntervalData: Clone Tests
// ---------------------------------------------------------------------------

// C# Test_Clone_CreatesIndependentCopy
void test_interval_clone_creates_independent_copy() {
    IntervalData original(1850, 50000.0, 75000.0, 100000.0, 0.01);

    IntervalData clone = original.clone();

    // Verify values match
    CHECK_EQ(clone.index(), original.index());
    CHECK_NEAR(clone.lower_value(), original.lower_value(), 0.0);
    CHECK_NEAR(clone.value(), original.value(), 0.0);
    CHECK_NEAR(clone.upper_value(), original.upper_value(), 0.0);
    CHECK_NEAR(clone.plotting_position(), original.plotting_position(), 0.0);

    // Verify independence
    original.set_lower_value(99999.0);
    CHECK_NEAR(clone.lower_value(), 50000.0, 0.0);
}

// ---------------------------------------------------------------------------
// IntervalData: Paleoflood Scenarios
// ---------------------------------------------------------------------------

// C# Test_SlackWaterDeposit
void test_interval_slack_water_deposit() {
    // Slack-water deposit: flood peaked between two elevations
    IntervalData data(1500, 50000.0, 75000.0, 100000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_HistoricalHighWaterMark
void test_interval_historical_high_water_mark() {
    IntervalData data(1889, 300000.0, 350000.0, 400000.0, 0.002);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_RatingCurveUncertainty
void test_interval_rating_curve_uncertainty() {
    IntervalData data(1993, 330000.0, 350000.0, 370000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_NonExceedanceBound
void test_interval_non_exceedance_bound() {
    // Non-inundation bound from geomorphic evidence
    IntervalData data(1000, 0.0, 250000.0, 500000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// ---------------------------------------------------------------------------
// IntervalData: Edge Cases
// ---------------------------------------------------------------------------

// C# Test_NarrowInterval
void test_interval_narrow_interval() {
    IntervalData data(1993, 74000.0, 75000.0, 76000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_WideInterval
void test_interval_wide_interval() {
    IntervalData data(1850, 10000.0, 100000.0, 1000000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_NegativeValues_Temperature
void test_interval_negative_values_temperature() {
    IntervalData data(1985, -50.0, -30.0, -10.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_AsymmetricInterval
void test_interval_asymmetric_interval() {
    IntervalData data(1850, 60000.0, 75000.0, 150000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_NegativeIndex_AncientFlood
void test_interval_negative_index_ancient_flood() {
    IntervalData data(-500, 100000.0, 200000.0, 300000.0);
    CHECK_TRUE(data.validate().is_valid);
}

// ---------------------------------------------------------------------------
// ThresholdData: Constructor Tests
// ---------------------------------------------------------------------------

// C# Test_Constructor_EmptyConstructor_CreatesDefaultInstance
void test_threshold_constructor_empty_creates_default_instance() {
    ThresholdData threshold;
    CHECK_EQ(threshold.start_index(), 0);
    CHECK_EQ(threshold.end_index(), 0);
    CHECK_NEAR(threshold.value(), 0.0, 0.0);
    CHECK_EQ(threshold.number_below(), 0);
    CHECK_EQ(threshold.number_above(), 0);
}

// C# Test_Constructor_WithParameters_SetsValues
void test_threshold_constructor_with_parameters_sets_values() {
    ThresholdData threshold(1900, 1950, 100000.0);
    CHECK_EQ(threshold.start_index(), 1900);
    CHECK_EQ(threshold.end_index(), 1950);
    CHECK_NEAR(threshold.value(), 100000.0, 0.0);
    // The C# base(startIndex, value) call makes Index track StartIndex.
    CHECK_EQ(threshold.index(), 1900);
}

// ---------------------------------------------------------------------------
// ThresholdData: Property Tests
// ---------------------------------------------------------------------------

// C# Test_StartIndex_SetAndGet
void test_threshold_start_index_set_and_get() {
    ThresholdData threshold;
    threshold.set_start_index(1920);
    CHECK_EQ(threshold.start_index(), 1920);
    // The C# StartIndex setter also assigns the inherited _index.
    CHECK_EQ(threshold.index(), 1920);
}

// C# Test_EndIndex_SetAndGet
void test_threshold_end_index_set_and_get() {
    ThresholdData threshold;
    threshold.set_end_index(1980);
    CHECK_EQ(threshold.end_index(), 1980);
}

// C# Test_Value_SetAndGet
void test_threshold_value_set_and_get() {
    ThresholdData threshold;
    threshold.set_value(50000.0);
    CHECK_NEAR(threshold.value(), 50000.0, 0.0);
}

// C# Test_NumberAbove_SetAndGet
void test_threshold_number_above_set_and_get() {
    ThresholdData threshold;
    threshold.set_number_above(5);
    CHECK_EQ(threshold.number_above(), 5);
}

// C# Test_Duration_ComputedCorrectly
void test_threshold_duration_computed_correctly() {
    ThresholdData threshold(1900, 1950, 100000.0);
    // Duration = EndIndex - StartIndex + 1 = 1950 - 1900 + 1 = 51
    CHECK_EQ(threshold.duration(), 51);
}

// C# Test_Duration_SingleYear
void test_threshold_duration_single_year() {
    ThresholdData threshold(1950, 1950, 100000.0);
    CHECK_EQ(threshold.duration(), 1);
}

// ---------------------------------------------------------------------------
// ThresholdData: Validation Tests
// ---------------------------------------------------------------------------

// C# Test_Validate_ValidThreshold_ReturnsTrue
void test_threshold_validate_valid_threshold_returns_true() {
    ThresholdData threshold(1900, 1950, 100000.0);
    threshold.set_number_above(6);
    ValidationResult result = threshold.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_StartAfterEnd_ReturnsFalse
void test_threshold_validate_start_after_end_returns_false() {
    ThresholdData threshold(1950, 1900, 100000.0);
    ValidationResult result = threshold.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "start index"));
}

// C# Test_Validate_NegativeNumberAbove_ReturnsFalse
void test_threshold_validate_negative_number_above_returns_false() {
    ThresholdData threshold(1900, 1950, 100000.0);
    threshold.set_number_above(-3);
    ValidationResult result = threshold.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "number above"));
}

// C# Test_Validate_NumberAboveExceedsDuration_ReturnsFalse
void test_threshold_validate_number_above_exceeds_duration_returns_false() {
    ThresholdData threshold(1900, 1950, 100000.0);
    threshold.set_number_above(100);  // Duration is 51
    ValidationResult result = threshold.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "number above"));
}

// Not in C# (unreachable there in isolation: NumberBelow's setter is internal): the
// NumberBelow >= 0 and NumberBelow + NumberAbove <= Duration rules.
void test_threshold_validate_number_below_rules() {
    ThresholdData negative(1900, 1950, 100000.0);
    negative.set_number_below(-1);
    ValidationResult result = negative.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "number below"));

    ThresholdData overfull(1900, 1950, 100000.0);  // Duration 51
    overfull.set_number_above(30);
    overfull.set_number_below(30);  // 30 + 30 > 51
    ValidationResult sum_result = overfull.validate();
    CHECK_TRUE(!sum_result.is_valid);
    CHECK_TRUE(any_contains(sum_result.validation_messages, "sum"));
}

// C# Test_Validate_NaNValue_ReturnsFalse
void test_threshold_validate_nan_value_returns_false() {
    ThresholdData threshold(1900, 1950, kNaN);
    ValidationResult result = threshold.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "number"));
}

// C# Test_Validate_InfinityValue_ReturnsFalse
void test_threshold_validate_infinity_value_returns_false() {
    ThresholdData threshold(1900, 1950, kInf);
    CHECK_TRUE(!threshold.validate().is_valid);
}

// C# Test_Validate_IndexOutOfRange_ReturnsFalse
void test_threshold_validate_index_out_of_range_returns_false() {
    ThresholdData threshold(-200000, 1950, 100000.0);
    ValidationResult result = threshold.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "index"));
}

// ---------------------------------------------------------------------------
// ThresholdData: Clone Tests
// ---------------------------------------------------------------------------

// C# Test_Clone_CreatesIndependentCopy (source built via the XElement helper in C#;
// here the counts are set directly -- see the file header)
void test_threshold_clone_creates_independent_copy() {
    ThresholdData original(1900, 1950, 75000.0);
    original.set_number_above(6);
    original.set_number_below(45);
    original.set_plotting_position(0.882);

    ThresholdData clone = original.clone();

    // Verify values match
    CHECK_EQ(clone.start_index(), original.start_index());
    CHECK_EQ(clone.end_index(), original.end_index());
    CHECK_NEAR(clone.value(), original.value(), 0.0);
    CHECK_EQ(clone.number_below(), original.number_below());
    CHECK_EQ(clone.number_above(), original.number_above());
    CHECK_NEAR(clone.plotting_position(), original.plotting_position(), 0.0);

    // Verify independence: mutating the original must not affect the clone.
    original.set_number_above(99);
    CHECK_EQ(clone.number_above(), 6);
}

// ---------------------------------------------------------------------------
// ThresholdData: Historical Flood Analysis Scenarios
// ---------------------------------------------------------------------------

// C# Test_HistoricalPeriod_TypicalScenario
void test_threshold_historical_period_typical_scenario() {
    // Typical historical period: 1850-1939, perception threshold 50,000 cfs,
    // 3 known floods exceeded threshold, 87 years below.
    ThresholdData threshold(1850, 1939, 50000.0);
    threshold.set_number_above(3);
    threshold.set_number_below(87);

    ValidationResult result = threshold.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(threshold.duration(), 90);
}

// C# Test_HistoricalPeriod_NoExceedances
void test_threshold_historical_period_no_exceedances() {
    ThresholdData threshold(1900, 1950, 100000.0);
    threshold.set_number_above(0);
    CHECK_TRUE(threshold.validate().is_valid);
}

// C# Test_HistoricalPeriod_AllExceedances
void test_threshold_historical_period_all_exceedances() {
    ThresholdData threshold(1900, 1909, 1000.0);
    threshold.set_number_above(10);
    CHECK_TRUE(threshold.validate().is_valid);
    CHECK_EQ(threshold.duration(), 10);
}

// C# Test_HistoricalPeriod_PaleofloodRecord
void test_threshold_historical_period_paleoflood_record() {
    ThresholdData threshold(1500, 1999, 200000.0);
    threshold.set_number_above(4);
    threshold.set_number_below(496);
    CHECK_TRUE(threshold.validate().is_valid);
    CHECK_EQ(threshold.duration(), 500);
}

// ---------------------------------------------------------------------------
// ThresholdData: Edge Cases
// ---------------------------------------------------------------------------

// C# Test_NegativeIndices_Allowed
void test_threshold_negative_indices_allowed() {
    // Years before common era could use negative indices
    ThresholdData threshold(-500, -400, 100000.0);
    threshold.set_number_above(2);
    CHECK_TRUE(threshold.validate().is_valid);
    CHECK_EQ(threshold.duration(), 101);
}

// C# Test_LargeValue
void test_threshold_large_value() {
    ThresholdData threshold(1900, 1950, 1e12);  // 1 trillion
    CHECK_TRUE(threshold.validate().is_valid);
    CHECK_NEAR(threshold.value(), 1e12, 0.0);
}

// C# Test_SmallValue
void test_threshold_small_value() {
    ThresholdData threshold(1900, 1950, 0.001);
    CHECK_TRUE(threshold.validate().is_valid);
    CHECK_NEAR(threshold.value(), 0.001, 0.0);
}

// C# Test_ZeroValue
void test_threshold_zero_value() {
    ThresholdData threshold(1900, 1950, 0.0);
    CHECK_TRUE(threshold.validate().is_valid);
}

// C# Test_NegativeValue
void test_threshold_negative_value() {
    // Negative values could represent temperatures or elevations
    ThresholdData threshold(1900, 1950, -50.0);
    CHECK_TRUE(threshold.validate().is_valid);
    CHECK_NEAR(threshold.value(), -50.0, 0.0);
}

// ---------------------------------------------------------------------------
// UncertainData: Constructor Tests
// ---------------------------------------------------------------------------

// C# Test_Constructor_EmptyConstructor_CreatesDefaultInstance
void test_uncertain_constructor_empty_creates_default_instance() {
    UncertainData data;
    CHECK_EQ(data.index(), 0);
    // C# asserts Distribution is not null; the default is a standard Normal.
    CHECK_TRUE(data.distribution().type() == UnivariateDistributionType::Normal);
}

// C# Test_Constructor_WithIndexAndDistribution_SetsProperties
void test_uncertain_constructor_with_index_and_distribution_sets_properties() {
    auto dist = std::make_unique<Normal>(75000.0, 5000.0);
    const UnivariateDistributionBase* raw = dist.get();
    UncertainData data(1985, std::move(dist));

    CHECK_EQ(data.index(), 1985);
    CHECK_NEAR(data.value(), 75000.0, 1e-10);  // Mean of distribution
    // C# Assert.AreSame -> pointer identity of the owned object.
    CHECK_TRUE(&data.distribution() == raw);
}

// C# Test_Constructor_WithPlottingPosition_SetsAllProperties
void test_uncertain_constructor_with_plotting_position_sets_all_properties() {
    UncertainData data(1985, std::make_unique<Normal>(75000.0, 5000.0), 0.5);
    CHECK_EQ(data.index(), 1985);
    CHECK_NEAR(data.plotting_position(), 0.5, 0.0);
}

// ---------------------------------------------------------------------------
// UncertainData: Property Tests
// ---------------------------------------------------------------------------

// C# Test_Value_ReturnsDistributionMean
void test_uncertain_value_returns_distribution_mean() {
    UncertainData data(1985, std::make_unique<Normal>(100.0, 10.0));
    CHECK_NEAR(data.value(), 100.0, 1e-10);
}

// C# Test_Value_UpdatesWhenDistributionChanges
void test_uncertain_value_updates_when_distribution_changes() {
    UncertainData data(1985, std::make_unique<Normal>(100.0, 10.0));
    data.set_distribution(std::make_unique<Normal>(200.0, 15.0));
    CHECK_NEAR(data.value(), 200.0, 1e-10);
}

// Not in C# as a test, but ported behavior (UncertainData.cs Value setter): setting Value
// directly is a no-op -- the value stays pinned to the distribution mean.
void test_uncertain_value_setter_is_pinned_to_mean() {
    UncertainData data(1985, std::make_unique<Normal>(100.0, 10.0));
    data.set_value(12345.0);
    CHECK_NEAR(data.value(), 100.0, 1e-10);
}

// C# Test_UpperValue_Returns95thPercentile
void test_uncertain_upper_value_returns_95th_percentile() {
    Normal dist(100.0, 10.0);
    UncertainData data(1985, std::make_unique<Normal>(100.0, 10.0));
    CHECK_NEAR(data.upper_value(), dist.inverse_cdf(0.95), 1e-10);
}

// C# Test_LowerValue_Returns5thPercentile
void test_uncertain_lower_value_returns_5th_percentile() {
    Normal dist(100.0, 10.0);
    UncertainData data(1985, std::make_unique<Normal>(100.0, 10.0));
    CHECK_NEAR(data.lower_value(), dist.inverse_cdf(0.05), 1e-10);
}

// C# Test_Log10LowerValue_ReturnsCorrectTransform
void test_uncertain_log10_lower_value_returns_correct_transform() {
    UncertainData data(1985, std::make_unique<Normal>(10000.0, 1000.0));
    double expected = std::log10(data.lower_value());
    CHECK_NEAR(data.log10_lower_value(), expected, 1e-10);
}

// C# Test_Log10UpperValue_ReturnsCorrectTransform
void test_uncertain_log10_upper_value_returns_correct_transform() {
    UncertainData data(1985, std::make_unique<Normal>(10000.0, 1000.0));
    double expected = std::log10(data.upper_value());
    CHECK_NEAR(data.log10_upper_value(), expected, 1e-10);
}

// ---------------------------------------------------------------------------
// UncertainData: Distribution Type Tests
// ---------------------------------------------------------------------------

// C# Test_Distribution_Normal
void test_uncertain_distribution_normal() {
    UncertainData data(1985, std::make_unique<Normal>(75000.0, 5000.0));
    CHECK_TRUE(data.distribution().type() == UnivariateDistributionType::Normal);
    CHECK_NEAR(data.value(), 75000.0, 1e-10);
}

// C# Test_Distribution_LogNormal
void test_uncertain_distribution_log_normal() {
    // Log-Normal often better for flood magnitudes
    LogNormal oracle(11.2, 0.3);
    UncertainData data(1985, std::make_unique<LogNormal>(11.2, 0.3));
    CHECK_TRUE(data.distribution().type() == UnivariateDistributionType::LogNormal);
    CHECK_NEAR(data.value(), oracle.mean(), 1e-5);
}

// C# Test_Distribution_Uniform
void test_uncertain_distribution_uniform() {
    UncertainData data(1985, std::make_unique<Uniform>(50000.0, 100000.0));
    CHECK_TRUE(data.distribution().type() == UnivariateDistributionType::Uniform);
    CHECK_NEAR(data.value(), 75000.0, 1e-10);  // Uniform mean = (a+b)/2
}

// C# Test_Distribution_Triangular
void test_uncertain_distribution_triangular() {
    UncertainData data(1985, std::make_unique<Triangular>(50000.0, 75000.0, 100000.0));
    CHECK_TRUE(data.distribution().type() == UnivariateDistributionType::Triangular);
}

// ---------------------------------------------------------------------------
// UncertainData: Validation Tests
// ---------------------------------------------------------------------------

// C# Test_Validate_ValidData_ReturnsTrue
void test_uncertain_validate_valid_data_returns_true() {
    UncertainData data(1985, std::make_unique<Normal>(75000.0, 5000.0));
    ValidationResult result = data.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_IndexOutOfRange_ReturnsFalse
void test_uncertain_validate_index_out_of_range_returns_false() {
    UncertainData data(-200000, std::make_unique<Normal>(75000.0, 5000.0));
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "index"));
}

// C# Test_Validate_LogNormalWithPositiveValues_IsValid
void test_uncertain_validate_log_normal_with_positive_values_is_valid() {
    UncertainData data(1985, std::make_unique<LogNormal>(11.0, 0.5));
    CHECK_TRUE(data.validate().is_valid);
}

// Not in C# (mirrors the ValidateParameters branch of Validate(), as in QuantilePrior):
// an invalid distribution (Normal with sigma <= 0) must fail validation.
void test_uncertain_validate_invalid_distribution_returns_false() {
    UncertainData data(1985, std::make_unique<Normal>(0.0, -1.0));
    ValidationResult result = data.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(any_contains(result.validation_messages, "distribution"));
}

// ---------------------------------------------------------------------------
// UncertainData: Clone Tests
// ---------------------------------------------------------------------------

// C# Test_Clone_CreatesIndependentCopy
void test_uncertain_clone_creates_independent_copy() {
    UncertainData original(1985, std::make_unique<Normal>(75000.0, 5000.0), 0.5);

    UncertainData clone = original.clone();

    // Verify values match
    CHECK_EQ(clone.index(), original.index());
    CHECK_NEAR(clone.value(), original.value(), 1e-10);
    CHECK_NEAR(clone.plotting_position(), original.plotting_position(), 0.0);

    // Verify distribution independence
    original.set_distribution(std::make_unique<Normal>(999999.0, 1.0));
    CHECK_NEAR(clone.value(), 75000.0, 1e-10);
}

// C# Test_Clone_PreservesDistributionType
void test_uncertain_clone_preserves_distribution_type() {
    UncertainData original(1985, std::make_unique<LogNormal>(11.2, 0.3));
    UncertainData clone = original.clone();
    CHECK_TRUE(clone.distribution().type() == UnivariateDistributionType::LogNormal);
}

// C# Test_Clone_PreservesDistributionParameters
void test_uncertain_clone_preserves_distribution_parameters() {
    UncertainData original(1985, std::make_unique<Normal>(75000.0, 5000.0));
    UncertainData clone = original.clone();
    CHECK_NEAR(clone.lower_value(), original.lower_value(), 1e-5);
    CHECK_NEAR(clone.upper_value(), original.upper_value(), 1e-5);
}

// Not in C# (deep-copy semantics, mirrors the QuantilePrior ownership tests): copy
// construction/assignment must deep-copy the distribution.
void test_uncertain_copy_is_deep() {
    UncertainData original(1985, std::make_unique<Normal>(100.0, 15.0));

    UncertainData copy(original);
    CHECK_TRUE(&copy.distribution() != &original.distribution());
    copy.set_distribution(std::make_unique<Normal>(5.0, 2.0));
    CHECK_NEAR(original.value(), 100.0, 1e-12);

    UncertainData assigned;
    assigned = original;
    CHECK_TRUE(&assigned.distribution() != &original.distribution());
    assigned.set_distribution(std::make_unique<Normal>(9.0, 3.0));
    CHECK_NEAR(original.value(), 100.0, 1e-12);
    CHECK_EQ(assigned.index(), 1985);
}

// ---------------------------------------------------------------------------
// UncertainData: Flood Frequency Scenarios
// ---------------------------------------------------------------------------

// C# Test_HistoricalFloodEstimate
void test_uncertain_historical_flood_estimate() {
    // Historical flood estimate with normal measurement error: 350,000 +- 50,000 cfs
    UncertainData data(1889, std::make_unique<Normal>(350000.0, 50000.0));
    CHECK_TRUE(data.validate().is_valid);
    CHECK_NEAR(data.value(), 350000.0, 1e-10);
}

// C# Test_RatingCurveUncertainty
void test_uncertain_rating_curve_uncertainty() {
    UncertainData data(1993, std::make_unique<Normal>(100000.0, 10000.0));
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_PaleofloodEstimate
void test_uncertain_paleoflood_estimate() {
    UncertainData data(1500, std::make_unique<LogNormal>(12.5, 0.4));
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_HighWaterMarkEstimate
void test_uncertain_high_water_mark_estimate() {
    UncertainData data(1889, std::make_unique<Triangular>(300000.0, 350000.0, 420000.0), 0.002);
    CHECK_TRUE(data.validate().is_valid);
}

// C# Test_ExpertJudgmentEstimate
void test_uncertain_expert_judgment_estimate() {
    // Expert judgment: uniform over plausible range
    UncertainData data(1850, std::make_unique<Uniform>(200000.0, 500000.0));
    CHECK_TRUE(data.validate().is_valid);
    CHECK_NEAR(data.value(), 350000.0, 1e-10);  // Uniform mean
}

// ---------------------------------------------------------------------------
// UncertainData: Edge Cases
// ---------------------------------------------------------------------------

// C# Test_NarrowUncertainty
void test_uncertain_narrow_uncertainty() {
    UncertainData data(1985, std::make_unique<Normal>(75000.0, 100.0));
    double ci_width = data.upper_value() - data.lower_value();
    CHECK_TRUE(ci_width < 400.0);  // 90% CI very narrow
}

// C# Test_WideUncertainty
void test_uncertain_wide_uncertainty() {
    UncertainData data(1850, std::make_unique<Normal>(75000.0, 25000.0));
    double ci_width = data.upper_value() - data.lower_value();
    CHECK_TRUE(ci_width > 80000.0);  // 90% CI very wide
}

// C# Test_SkewedDistribution
void test_uncertain_skewed_distribution() {
    UncertainData data(1985, std::make_unique<LogNormal>(11.0, 0.5));
    // Log-normal: median < mean, upper tail heavier
    CHECK_TRUE(data.upper_value() - data.value() > data.value() - data.lower_value());
}

// C# Test_NegativeIndex
void test_uncertain_negative_index() {
    // Ancient paleoflood
    UncertainData data(-500, std::make_unique<Normal>(200000.0, 50000.0));
    CHECK_TRUE(data.validate().is_valid);
}

}  // namespace

int main() {
    // SeriesOrdinate<int, double> base contract
    test_series_ordinate_default_constructor();
    test_series_ordinate_constructor_sets_index_and_value();
    test_series_ordinate_set_and_get();
    test_series_ordinate_equality();
    test_series_ordinate_clone();

    // Data base getters (through ExactData)
    test_data_plotting_position_clamps_on_get();
    test_data_plotting_position_complement();
    test_data_log10_value_edge_cases();
    test_data_standardized_values_set_and_get();

    // ExactData (transcribed from ExactDataTests.cs)
    test_exact_constructor_empty_creates_default_instance();
    test_exact_constructor_with_index_and_value_sets_properties();
    test_exact_constructor_with_all_parameters_sets_all_properties();
    test_exact_index_set_and_get();
    test_exact_value_set_and_get();
    test_exact_plotting_position_set_and_get();
    test_exact_is_low_outlier_set_and_get();
    test_exact_validate_valid_data_returns_true();
    test_exact_validate_nan_value_returns_false();
    test_exact_validate_infinity_value_returns_false();
    test_exact_validate_index_too_low_returns_false();
    test_exact_validate_index_too_high_returns_false();
    test_exact_validate_negative_value_is_valid();
    test_exact_validate_zero_value_is_valid();
    test_exact_clone_creates_independent_copy();
    test_exact_flood_peak_flow();
    test_exact_low_outlier_small_flood();
    test_exact_historical_flood();
    test_exact_very_small_value();
    test_exact_very_large_value();
    test_exact_negative_index_paleoflood();
    test_exact_plotting_position_range();

    // IntervalData (transcribed from IntervalDataTests.cs)
    test_interval_constructor_empty_creates_default_instance();
    test_interval_constructor_with_parameters_sets_properties();
    test_interval_constructor_with_plotting_position_sets_all_properties();
    test_interval_lower_value_set_and_get();
    test_interval_upper_value_set_and_get();
    test_interval_log10_lower_value_returns_correct_transform();
    test_interval_log10_upper_value_returns_correct_transform();
    test_interval_validate_valid_data_returns_true();
    test_interval_validate_lower_greater_than_value_returns_false();
    test_interval_validate_value_greater_than_upper_returns_false();
    test_interval_validate_lower_greater_than_upper_returns_false();
    test_interval_validate_nan_lower_value_returns_false();
    test_interval_validate_nan_value_returns_false();
    test_interval_validate_nan_upper_value_returns_false();
    test_interval_validate_infinity_value_returns_false();
    test_interval_validate_index_out_of_range_returns_false();
    test_interval_clone_creates_independent_copy();
    test_interval_slack_water_deposit();
    test_interval_historical_high_water_mark();
    test_interval_rating_curve_uncertainty();
    test_interval_non_exceedance_bound();
    test_interval_narrow_interval();
    test_interval_wide_interval();
    test_interval_negative_values_temperature();
    test_interval_asymmetric_interval();
    test_interval_negative_index_ancient_flood();

    // ThresholdData (transcribed from ThresholdDataTests.cs)
    test_threshold_constructor_empty_creates_default_instance();
    test_threshold_constructor_with_parameters_sets_values();
    test_threshold_start_index_set_and_get();
    test_threshold_end_index_set_and_get();
    test_threshold_value_set_and_get();
    test_threshold_number_above_set_and_get();
    test_threshold_duration_computed_correctly();
    test_threshold_duration_single_year();
    test_threshold_validate_valid_threshold_returns_true();
    test_threshold_validate_start_after_end_returns_false();
    test_threshold_validate_negative_number_above_returns_false();
    test_threshold_validate_number_above_exceeds_duration_returns_false();
    test_threshold_validate_number_below_rules();
    test_threshold_validate_nan_value_returns_false();
    test_threshold_validate_infinity_value_returns_false();
    test_threshold_validate_index_out_of_range_returns_false();
    test_threshold_clone_creates_independent_copy();
    test_threshold_historical_period_typical_scenario();
    test_threshold_historical_period_no_exceedances();
    test_threshold_historical_period_all_exceedances();
    test_threshold_historical_period_paleoflood_record();
    test_threshold_negative_indices_allowed();
    test_threshold_large_value();
    test_threshold_small_value();
    test_threshold_zero_value();
    test_threshold_negative_value();

    // UncertainData (transcribed from UncertainDataTests.cs)
    test_uncertain_constructor_empty_creates_default_instance();
    test_uncertain_constructor_with_index_and_distribution_sets_properties();
    test_uncertain_constructor_with_plotting_position_sets_all_properties();
    test_uncertain_value_returns_distribution_mean();
    test_uncertain_value_updates_when_distribution_changes();
    test_uncertain_value_setter_is_pinned_to_mean();
    test_uncertain_upper_value_returns_95th_percentile();
    test_uncertain_lower_value_returns_5th_percentile();
    test_uncertain_log10_lower_value_returns_correct_transform();
    test_uncertain_log10_upper_value_returns_correct_transform();
    test_uncertain_distribution_normal();
    test_uncertain_distribution_log_normal();
    test_uncertain_distribution_uniform();
    test_uncertain_distribution_triangular();
    test_uncertain_validate_valid_data_returns_true();
    test_uncertain_validate_index_out_of_range_returns_false();
    test_uncertain_validate_log_normal_with_positive_values_is_valid();
    test_uncertain_validate_invalid_distribution_returns_false();
    test_uncertain_clone_creates_independent_copy();
    test_uncertain_clone_preserves_distribution_type();
    test_uncertain_clone_preserves_distribution_parameters();
    test_uncertain_copy_is_deep();
    test_uncertain_historical_flood_estimate();
    test_uncertain_rating_curve_uncertainty();
    test_uncertain_paleoflood_estimate();
    test_uncertain_high_water_mark_estimate();
    test_uncertain_expert_judgment_estimate();
    test_uncertain_narrow_uncertainty();
    test_uncertain_wide_uncertainty();
    test_uncertain_skewed_distribution();
    test_uncertain_negative_index();

    return bftest::summary("data_types");
}
