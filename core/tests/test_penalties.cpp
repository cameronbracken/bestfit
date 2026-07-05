// Transcribed C# oracle tests for the Bulletin 17C penalty classes (Task B3):
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/Support/ParameterPenaltyTests.cs @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/Support/QuantilePenaltyTests.cs  @ fc28c0c
//
// Each upstream [TestMethod] is transcribed with its own tolerance exactly as written in the
// C# (the files mix 1e-15, 1e-12, and 1e-10 assertions; none are normalized). These are
// internal-support ports validated against the C# source itself, so there is no fixtures/
// entry for this file (fixtures/ is the public estimation API surface only).
//
// Skipped upstream test methods (exhaustive; XML serialization and INotifyPropertyChanged are
// not ported, per the repo-wide rule):
//   ParameterPenaltyTests.cs (29 methods total = 24 transcribed + 5 skipped):
//     Constructor_XElement_Null_ThrowsArgumentNullException (XElement ctor not ported),
//     XmlRoundTrip_PreservesAllProperties, ToXElement_ElementNameIsParameterPenalty (XML),
//     Enabled_Changed_FiresPropertyChanged, Mean_Changed_FiresPropertyChanged (INPC).
//   QuantilePenaltyTests.cs (28 methods total = 23 transcribed + 5 skipped):
//     Constructor_XElement_Null_ThrowsArgumentNullException (XElement ctor not ported),
//     XmlRoundTrip_PreservesAllProperties, ToXElement_ElementNameIsQuantilePenalty (XML),
//     AEP_Changed_FiresPropertyChanged, MSE_Changed_FiresPropertyChanged (INPC).
//
// The C# `Validate()` returns the anonymous tuple `(bool IsValid, string Message)`; the port
// returns the shared bestfit::models::ValidationResult, so `msg == string.Empty` transcribes
// to `validation_messages.empty()` and `msg.Length > 0` to a non-empty message list.
#include <cmath>

#include "bestfit/models/support/parameter_penalty.hpp"
#include "bestfit/models/support/quantile_penalty.hpp"
#include "check.hpp"

using bestfit::models::ParameterPenalty;
using bestfit::models::QuantilePenalty;

namespace {

// ===================== ParameterPenaltyTests.cs =====================

// --- Constructor Tests ---

// Default constructor creates a penalty with NaN Mean, NaN MSE, and disabled state.
void pp_constructor_default_sets_expected_defaults() {
    ParameterPenalty penalty;

    CHECK_TRUE(!penalty.enabled());
    CHECK_TRUE(std::isnan(penalty.mean()));
    CHECK_TRUE(std::isnan(penalty.mse()));
    CHECK_EQ(penalty.name(), std::string());
    CHECK_TRUE(!penalty.use_log());
}

// --- IsValid Tests ---

// IsValid is false when Mean is NaN.
void pp_is_valid_mean_nan_returns_false() {
    ParameterPenalty penalty;
    penalty.set_mean(std::nan(""));
    penalty.set_mse(0.1);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is false when MSE is zero.
void pp_is_valid_mse_zero_returns_false() {
    ParameterPenalty penalty;
    penalty.set_mean(1.0);
    penalty.set_mse(0.0);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is false when MSE is negative.
void pp_is_valid_mse_negative_returns_false() {
    ParameterPenalty penalty;
    penalty.set_mean(1.0);
    penalty.set_mse(-0.01);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is true when Mean is finite and MSE is positive.
void pp_is_valid_valid_mean_and_mse_returns_true() {
    ParameterPenalty penalty;
    penalty.set_mean(0.3);
    penalty.set_mse(0.04);

    CHECK_TRUE(penalty.is_valid());
}

// IsValid is false when UseLog is true but Mean is non-positive.
void pp_is_valid_use_log_non_positive_mean_returns_false() {
    ParameterPenalty penalty;
    penalty.set_mean(-0.5);
    penalty.set_mse(0.01);
    penalty.set_use_log(true);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is true when UseLog is true and Mean is positive.
void pp_is_valid_use_log_positive_mean_returns_true() {
    ParameterPenalty penalty;
    penalty.set_mean(2.5);
    penalty.set_mse(0.1);
    penalty.set_use_log(true);

    CHECK_TRUE(penalty.is_valid());
}

// --- Validate Tests ---

// Validate returns (true, empty) when not enabled (regardless of other values).
void pp_validate_disabled_always_valid() {
    ParameterPenalty penalty;
    penalty.set_enabled(false);
    penalty.set_mean(std::nan(""));
    penalty.set_mse(-1.0);

    auto result = penalty.validate();

    CHECK_TRUE(result.is_valid);
    CHECK_TRUE(result.validation_messages.empty());
}

// Validate returns error when enabled and Mean is NaN.
void pp_validate_enabled_mean_nan_returns_error() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(std::nan(""));
    penalty.set_mse(0.1);
    penalty.set_name("Skew");

    auto result = penalty.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(!result.validation_messages.empty() && !result.validation_messages[0].empty());
}

// Validate returns error when enabled and MSE is non-positive.
void pp_validate_enabled_mse_zero_returns_error() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(0.3);
    penalty.set_mse(0.0);

    auto result = penalty.validate();

    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(!result.validation_messages.empty() && !result.validation_messages[0].empty());
}

// Validate returns (true, empty) when all parameters are valid and enabled.
void pp_validate_valid_parameters_returns_true() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(0.3);
    penalty.set_mse(0.04);
    penalty.set_name("Skew");

    auto result = penalty.validate();

    CHECK_TRUE(result.is_valid);
    CHECK_TRUE(result.validation_messages.empty());
}

// Validate with UseLog and non-positive Mean returns error.
void pp_validate_use_log_non_positive_mean_returns_error() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(0.0);
    penalty.set_mse(0.01);
    penalty.set_use_log(true);

    auto result = penalty.validate();

    CHECK_TRUE(!result.is_valid);
}

// --- Function Tests ---

// Function returns 0 when penalty is not enabled.
void pp_function_disabled_returns_zero() {
    ParameterPenalty penalty;
    penalty.set_enabled(false);
    penalty.set_mean(0.3);
    penalty.set_mse(0.04);

    double result = penalty.function(0.5, 50);

    CHECK_NEAR(result, 0.0, 1e-15);
}

// Function returns 0 when sampleSize is zero or negative.
void pp_function_zero_sample_size_returns_zero() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(0.3);
    penalty.set_mse(0.04);

    CHECK_NEAR(penalty.function(0.5, 0), 0.0, 1e-15);
    CHECK_NEAR(penalty.function(0.5, -5), 0.0, 1e-15);
}

// Function returns zero when parameterValue equals Mean (no deviation from prior).
void pp_function_at_mean_returns_zero() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(0.3);
    penalty.set_mse(0.04);

    double result = penalty.function(0.3, 50);

    CHECK_NEAR(result, 0.0, 1e-15);
}

// Function in real space: (1/2)*(param - Mean)^2 / (MSE * n).
void pp_function_real_space_matches_formula() {
    double mean = 0.3;
    double mse = 0.04;
    int n = 50;
    double param = 0.5;
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(mean);
    penalty.set_mse(mse);

    double expected = 0.5 * (param - mean) * (param - mean) / (mse * n);
    double actual = penalty.function(param, n);

    CHECK_NEAR(actual, expected, 1e-12);
}

// Function in log space returns zero when parameterValue is non-positive.
void pp_function_log_space_non_positive_param_returns_zero() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(2.0);
    penalty.set_mse(0.1);
    penalty.set_use_log(true);

    CHECK_NEAR(penalty.function(-1.0, 50), 0.0, 1e-15);
    CHECK_NEAR(penalty.function(0.0, 50), 0.0, 1e-15);
}

// Function in log space at mean value returns zero.
void pp_function_log_space_at_mean_returns_zero() {
    double mean = 2.0;
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(mean);
    penalty.set_mse(0.1);
    penalty.set_use_log(true);

    double result = penalty.function(mean, 50);

    CHECK_NEAR(result, 0.0, 1e-12);
}

// Function in log space increases as parameter deviates further from mean.
void pp_function_log_space_larger_deviation_larger_penalty() {
    ParameterPenalty penalty;
    penalty.set_enabled(true);
    penalty.set_mean(2.0);
    penalty.set_mse(0.1);
    penalty.set_use_log(true);

    double p1 = penalty.function(2.5, 50);  // Small deviation
    double p2 = penalty.function(5.0, 50);  // Large deviation

    CHECK_TRUE(p2 > p1);
}

// --- Clone Tests ---

// Clone produces an independent copy with all properties equal.
void pp_clone_produces_independent_copy_with_same_values() {
    ParameterPenalty original;
    original.set_enabled(true);
    original.set_name("Skewness");
    original.set_mean(0.25);
    original.set_mse(0.05);
    original.set_use_log(false);

    ParameterPenalty clone = original.clone();

    CHECK_TRUE(&original != &clone);
    CHECK_EQ(clone.enabled(), original.enabled());
    CHECK_EQ(clone.name(), original.name());
    CHECK_NEAR(clone.mean(), original.mean(), 1e-15);
    CHECK_NEAR(clone.mse(), original.mse(), 1e-15);
    CHECK_EQ(clone.use_log(), original.use_log());
}

// Modifying the clone does not affect the original.
void pp_clone_modification_does_not_affect_original() {
    ParameterPenalty original;
    original.set_mean(0.3);
    original.set_mse(0.04);
    ParameterPenalty clone = original.clone();
    clone.set_mean(99.0);

    CHECK_NEAR(original.mean(), 0.3, 1e-15);
}

// --- UpperValue / LowerValue Tests ---

// UpperValue is greater than LowerValue for valid parameters.
void pp_upper_value_greater_than_lower_value() {
    ParameterPenalty penalty;
    penalty.set_mean(1.0);
    penalty.set_mse(0.25);

    CHECK_TRUE(penalty.upper_value() > penalty.lower_value());
}

// In real space, UpperValue and LowerValue are symmetric around Mean.
void pp_upper_value_lower_value_real_space_symmetric() {
    ParameterPenalty penalty;
    penalty.set_mean(1.0);
    penalty.set_mse(0.25);

    double mid = (penalty.upper_value() + penalty.lower_value()) / 2.0;

    CHECK_NEAR(mid, 1.0, 1e-10);
}

// In log space, UpperValue and LowerValue bracket the mean.
void pp_upper_value_lower_value_log_space_bracket_mean() {
    ParameterPenalty penalty;
    penalty.set_mean(2.0);
    penalty.set_mse(0.1);
    penalty.set_use_log(true);

    CHECK_TRUE(penalty.upper_value() > penalty.mean());
    CHECK_TRUE(penalty.lower_value() < penalty.mean());
}

// ===================== QuantilePenaltyTests.cs =====================

// --- Constructor Tests ---

// Default constructor sets expected defaults: AEP=0.01, Mean=NaN, MSE=NaN, Enabled=false.
void qp_constructor_default_sets_expected_defaults() {
    QuantilePenalty penalty;

    CHECK_NEAR(penalty.aep(), 0.01, 1e-12);
    CHECK_TRUE(std::isnan(penalty.mean()));
    CHECK_TRUE(std::isnan(penalty.mse()));
    CHECK_TRUE(!penalty.enabled());
    CHECK_TRUE(!penalty.use_log10());
}

// --- IsValid Tests ---

// IsValid is false when AEP is 0.
void qp_is_valid_aep_zero_returns_false() {
    QuantilePenalty penalty;
    penalty.set_aep(0.0);
    penalty.set_mean(100.0);
    penalty.set_mse(25.0);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is false when AEP is 1.
void qp_is_valid_aep_one_returns_false() {
    QuantilePenalty penalty;
    penalty.set_aep(1.0);
    penalty.set_mean(100.0);
    penalty.set_mse(25.0);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is false when Mean is NaN.
void qp_is_valid_mean_nan_returns_false() {
    QuantilePenalty penalty;
    penalty.set_aep(0.01);
    penalty.set_mean(std::nan(""));
    penalty.set_mse(25.0);

    CHECK_TRUE(!penalty.is_valid());
}

// IsValid is false when MSE is zero or negative.
void qp_is_valid_mse_zero_or_negative_returns_false() {
    QuantilePenalty p0;
    p0.set_aep(0.01);
    p0.set_mean(100.0);
    p0.set_mse(0.0);
    QuantilePenalty p_neg;
    p_neg.set_aep(0.01);
    p_neg.set_mean(100.0);
    p_neg.set_mse(-1.0);

    CHECK_TRUE(!p0.is_valid());
    CHECK_TRUE(!p_neg.is_valid());
}

// IsValid is true when AEP, Mean, and MSE are all valid.
void qp_is_valid_all_valid_returns_true() {
    QuantilePenalty penalty;
    penalty.set_aep(0.01);
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);

    CHECK_TRUE(penalty.is_valid());
}

// --- Validate Tests ---

// Validate returns (true, empty) when disabled regardless of other values.
void qp_validate_disabled_always_valid() {
    QuantilePenalty penalty;
    penalty.set_enabled(false);
    penalty.set_aep(-1.0);
    penalty.set_mean(std::nan(""));

    auto result = penalty.validate();

    CHECK_TRUE(result.is_valid);
    CHECK_TRUE(result.validation_messages.empty());
}

// Validate returns error when AEP is out of range.
void qp_validate_enabled_bad_aep_returns_error() {
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(1.5);
    penalty.set_mean(100.0);
    penalty.set_mse(25.0);

    auto result = penalty.validate();

    CHECK_TRUE(!result.is_valid);
}

// Validate returns error when MSE is non-positive.
void qp_validate_enabled_mse_zero_returns_error() {
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(100.0);
    penalty.set_mse(0.0);

    auto result = penalty.validate();

    CHECK_TRUE(!result.is_valid);
}

// Validate returns (true, empty) when all parameters are valid.
void qp_validate_valid_parameters_returns_true() {
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);

    auto result = penalty.validate();

    CHECK_TRUE(result.is_valid);
    CHECK_TRUE(result.validation_messages.empty());
}

// --- Function Tests ---

// Function returns 0 when not enabled.
void qp_function_disabled_returns_zero() {
    QuantilePenalty penalty;
    penalty.set_enabled(false);
    penalty.set_aep(0.01);
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);

    CHECK_NEAR(penalty.function(3.0, 50), 0.0, 1e-15);
}

// Function returns 0 when sampleSize is zero.
void qp_function_zero_sample_size_returns_zero() {
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);

    CHECK_NEAR(penalty.function(3.0, 0), 0.0, 1e-15);
}

// Function in real space at mean returns zero.
void qp_function_real_space_at_mean_returns_zero() {
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(100.0);
    penalty.set_mse(25.0);

    double result = penalty.function(100.0, 50);

    CHECK_NEAR(result, 0.0, 1e-12);
}

// Function in real space matches the formula: (1/2)*(q - Mean)^2 / (MSE * n).
void qp_function_real_space_matches_formula() {
    double mean = 2.5;
    double mse = 0.04;
    int n = 30;
    double q = 3.0;
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(mean);
    penalty.set_mse(mse);

    double expected = 0.5 * (q - mean) * (q - mean) / (mse * n);
    double actual = penalty.function(q, n);

    CHECK_NEAR(actual, expected, 1e-12);
}

// Function in log10 space returns zero for non-positive quantile value.
void qp_function_log10_space_non_positive_quantile_returns_zero() {
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);
    penalty.set_use_log10(true);

    CHECK_NEAR(penalty.function(0.0, 50), 0.0, 1e-15);
    CHECK_NEAR(penalty.function(-10.0, 50), 0.0, 1e-15);
}

// Function in log10 space: penalty at 10^Mean is approximately zero.
void qp_function_log10_space_at_mean_value_returns_near_zero() {
    double log_mean = 2.5;  // means q=316.2 in real space
    QuantilePenalty penalty;
    penalty.set_enabled(true);
    penalty.set_aep(0.01);
    penalty.set_mean(log_mean);
    penalty.set_mse(0.04);
    penalty.set_use_log10(true);
    double q_at_mean = std::pow(10.0, log_mean);

    double result = penalty.function(q_at_mean, 50);

    CHECK_NEAR(result, 0.0, 1e-10);
}

// --- Clone Tests ---

// Clone produces a deep copy with all properties equal.
void qp_clone_produces_deep_copy_with_same_values() {
    QuantilePenalty original;
    original.set_enabled(true);
    original.set_aep(0.01);
    original.set_mean(2.5);
    original.set_mse(0.04);
    original.set_use_log10(true);

    QuantilePenalty clone = original.clone();

    CHECK_TRUE(&original != &clone);
    CHECK_EQ(clone.enabled(), original.enabled());
    CHECK_NEAR(clone.aep(), original.aep(), 1e-15);
    CHECK_NEAR(clone.mean(), original.mean(), 1e-15);
    CHECK_NEAR(clone.mse(), original.mse(), 1e-15);
    CHECK_EQ(clone.use_log10(), original.use_log10());
}

// Modifying the clone does not affect the original.
void qp_clone_modification_does_not_affect_original() {
    QuantilePenalty original;
    original.set_aep(0.01);
    original.set_mean(2.5);
    original.set_mse(0.04);
    QuantilePenalty clone = original.clone();
    clone.set_mean(99.0);

    CHECK_NEAR(original.mean(), 2.5, 1e-15);
}

// --- MeanValue / MSEValue Tests ---

// MeanValue returns Mean directly when UseLog10 is false.
void qp_mean_value_no_log_equals_mean() {
    QuantilePenalty penalty;
    penalty.set_mean(100.0);
    penalty.set_mse(25.0);
    penalty.set_use_log10(false);

    CHECK_NEAR(penalty.mean_value(), 100.0, 1e-10);
}

// MSEValue returns MSE directly when UseLog10 is false.
void qp_mse_value_no_log_equals_mse() {
    QuantilePenalty penalty;
    penalty.set_mean(100.0);
    penalty.set_mse(25.0);
    penalty.set_use_log10(false);

    CHECK_NEAR(penalty.mse_value(), 25.0, 1e-10);
}

// MeanValue in log10 space is greater than 10^Mean (due to lognormal bias correction).
void qp_mean_value_log10_greater_than_pow_of_10_mean() {
    QuantilePenalty penalty;
    penalty.set_mean(2.5);
    penalty.set_mse(0.1);
    penalty.set_use_log10(true);

    // MeanValue (real-space) should be > 10^2.5 = 316.2 (positive bias correction)
    CHECK_TRUE(penalty.mean_value() > std::pow(10.0, 2.5));
}

// --- UpperValue / LowerValue Tests ---

// UpperValue is greater than LowerValue for valid parameters.
void qp_upper_value_greater_than_lower_value() {
    QuantilePenalty penalty;
    penalty.set_aep(0.01);
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);

    CHECK_TRUE(penalty.upper_value() > penalty.lower_value());
}

// In real space, UpperValue and LowerValue are symmetric around Mean.
void qp_upper_value_lower_value_real_space_symmetric_around_mean() {
    QuantilePenalty penalty;
    penalty.set_mean(2.5);
    penalty.set_mse(0.04);

    double mid = (penalty.upper_value() + penalty.lower_value()) / 2.0;

    CHECK_NEAR(mid, 2.5, 1e-10);
}

}  // namespace

int main() {
    // ParameterPenaltyTests.cs
    pp_constructor_default_sets_expected_defaults();
    pp_is_valid_mean_nan_returns_false();
    pp_is_valid_mse_zero_returns_false();
    pp_is_valid_mse_negative_returns_false();
    pp_is_valid_valid_mean_and_mse_returns_true();
    pp_is_valid_use_log_non_positive_mean_returns_false();
    pp_is_valid_use_log_positive_mean_returns_true();
    pp_validate_disabled_always_valid();
    pp_validate_enabled_mean_nan_returns_error();
    pp_validate_enabled_mse_zero_returns_error();
    pp_validate_valid_parameters_returns_true();
    pp_validate_use_log_non_positive_mean_returns_error();
    pp_function_disabled_returns_zero();
    pp_function_zero_sample_size_returns_zero();
    pp_function_at_mean_returns_zero();
    pp_function_real_space_matches_formula();
    pp_function_log_space_non_positive_param_returns_zero();
    pp_function_log_space_at_mean_returns_zero();
    pp_function_log_space_larger_deviation_larger_penalty();
    pp_clone_produces_independent_copy_with_same_values();
    pp_clone_modification_does_not_affect_original();
    pp_upper_value_greater_than_lower_value();
    pp_upper_value_lower_value_real_space_symmetric();
    pp_upper_value_lower_value_log_space_bracket_mean();
    // QuantilePenaltyTests.cs
    qp_constructor_default_sets_expected_defaults();
    qp_is_valid_aep_zero_returns_false();
    qp_is_valid_aep_one_returns_false();
    qp_is_valid_mean_nan_returns_false();
    qp_is_valid_mse_zero_or_negative_returns_false();
    qp_is_valid_all_valid_returns_true();
    qp_validate_disabled_always_valid();
    qp_validate_enabled_bad_aep_returns_error();
    qp_validate_enabled_mse_zero_returns_error();
    qp_validate_valid_parameters_returns_true();
    qp_function_disabled_returns_zero();
    qp_function_zero_sample_size_returns_zero();
    qp_function_real_space_at_mean_returns_zero();
    qp_function_real_space_matches_formula();
    qp_function_log10_space_non_positive_quantile_returns_zero();
    qp_function_log10_space_at_mean_value_returns_near_zero();
    qp_clone_produces_deep_copy_with_same_values();
    qp_clone_modification_does_not_affect_original();
    qp_mean_value_no_log_equals_mean();
    qp_mse_value_no_log_equals_mse();
    qp_mean_value_log10_greater_than_pow_of_10_mean();
    qp_upper_value_greater_than_lower_value();
    qp_upper_value_lower_value_real_space_symmetric_around_mean();
    return bftest::summary("test_penalties");
}
