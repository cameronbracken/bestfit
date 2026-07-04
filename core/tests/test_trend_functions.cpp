// Tests for the TrendFunctions layer (M6): ITrendModel/TrendModelBase/TrendModelType plus the
// 10 simple trend models.
//
// Oracles transcribed from upstream/RMC-BestFit/src/RMC.BestFit.Tests/Univariate/TrendFunctions/
// @ fc28c0c: TrendModelBaseTests.cs, ConstantTrendTests.cs, LinearTrendTests.cs,
// QuadraticTrendTests.cs, CubicTrendTests.cs, ExponentialTrendTests.cs, PowerTrendTests.cs,
// ReciprocalTrendTests.cs, LogisticTrendTests.cs, SinusoidalTrendTests.cs, StepFunctionTests.cs.
// GeneralLinearFunctionTests.cs is M7 and deliberately not covered here.
//
// This is a structural/behavioral internal-support port (like test_model_parameter.cpp), so the
// oracle values are transcribed directly from the C# test files rather than routed through
// fixtures/. Skipped upstream methods (XML serialization, INotifyPropertyChanged, null-argument
// cases that are unrepresentable in C++) are listed per section below and in the task report.
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/models/trend_functions/constant_trend.hpp"
#include "bestfit/models/trend_functions/cubic_trend.hpp"
#include "bestfit/models/trend_functions/exponential_trend.hpp"
#include "bestfit/models/trend_functions/linear_trend.hpp"
#include "bestfit/models/trend_functions/logistic_trend.hpp"
#include "bestfit/models/trend_functions/power_trend.hpp"
#include "bestfit/models/trend_functions/quadratic_trend.hpp"
#include "bestfit/models/trend_functions/reciprocal_trend.hpp"
#include "bestfit/models/trend_functions/sinusoidal_trend.hpp"
#include "bestfit/models/trend_functions/step_function.hpp"
#include "bestfit/models/trend_functions/support/i_trend_model.hpp"
#include "bestfit/models/trend_functions/support/trend_model_base.hpp"
#include "bestfit/models/trend_functions/support/trend_model_type.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/tools.hpp"
#include "check.hpp"

using bestfit::models::trend_functions::ConstantTrend;
using bestfit::models::trend_functions::CubicTrend;
using bestfit::models::trend_functions::ExponentialTrend;
using bestfit::models::trend_functions::LinearTrend;
using bestfit::models::trend_functions::LogisticTrend;
using bestfit::models::trend_functions::PowerTrend;
using bestfit::models::trend_functions::QuadraticTrend;
using bestfit::models::trend_functions::ReciprocalTrend;
using bestfit::models::trend_functions::SinusoidalTrend;
using bestfit::models::trend_functions::StepFunction;
using bestfit::models::trend_functions::TrendModelType;
using bestfit::numerics::kE;
using bestfit::numerics::kPi;
using bestfit::numerics::distributions::Uniform;

namespace {

// UTF-8 byte escapes for the upstream Greek parameter names (kept as explicit escapes so the
// literals survive any compiler source-charset handling; see the trend headers).
const std::string kAlpha = "(\xCE\xB1)";              // "(alpha)"
const std::string kBeta = "(\xCE\xB2)";               // "(beta)"
const std::string kGamma = "(\xCE\xB3)";              // "(gamma)"
const std::string kDelta = "(\xCE\xB4)";              // "(delta)"
const std::string kMu1 = "(\xCE\xBC\xE2\x82\x81)";    // "(mu_1)"
const std::string kMu2 = "(\xCE\xBC\xE2\x82\x82)";    // "(mu_2)"
const std::string kTs = "(t\xE2\x82\x9B)";            // "(t_s)"

// =====================================================================================
// TrendModelBase  (TrendModelBaseTests.cs)
//
// Skipped upstream methods:
//   Test_Constructor_XElement_RestoresModel, Test_Constructor_XElement_NullThrows,
//   Test_ToXElement_ContainsAllAttributes, Test_ToXElement_ParametersAreSerialized,
//   Test_RoundTrip_Serialization, Test_EmptyOwnerName_SerializesCorrectly  -- XML not ported.
//   Test_PropertyChanged_OwnerName / _StartIndex / _UseDefaultFlatPriors /
//   _NotFiredWhenValueUnchanged, Test_SetParameterValues_RaisesPropertyChanged -- INPC not
//   ported.
//   Test_OwnerName_NullBecomesEmpty, Test_SetParameterValues_NullThrows -- null string/list is
//   unrepresentable through the C++ value-type signatures.
// =====================================================================================

void base_constructor_empty_initializes_parameters() {
    ConstantTrend model;

    CHECK_EQ(model.number_of_parameters(), 1);
}

void base_owner_name_set_and_get() {
    ConstantTrend model;
    model.set_owner_name("Scale");

    CHECK_EQ(model.owner_name(), std::string("Scale"));
}

void base_start_index_set_and_get() {
    LinearTrend model;
    model.set_start_index(2000);

    CHECK_EQ(model.start_index(), 2000);
}

void base_use_default_flat_priors_set_and_get() {
    ConstantTrend model;
    model.set_use_default_flat_priors(false);

    CHECK_TRUE(!model.use_default_flat_priors());
}

void base_use_default_flat_priors_true_resets_parameters() {
    // Arrange
    LinearTrend model;
    model.set_use_default_flat_priors(false);
    model.parameters()[0].set_value(999.0);
    model.parameters()[1].set_value(0.99);

    // Act - setting to true should call set_default_parameters
    model.set_use_default_flat_priors(true);

    // Assert - default values should be restored
    CHECK_EQ(model.parameters()[0].value(), 0.0);
    CHECK_EQ(model.parameters()[1].value(), 0.0);
}

void base_number_of_parameters_returns_correct_count() {
    ConstantTrend constant;
    LinearTrend linear;

    CHECK_EQ(constant.number_of_parameters(), 1);
    CHECK_EQ(linear.number_of_parameters(), 2);
}

void base_set_parameter_values_updates_all_parameters() {
    LinearTrend model;
    std::vector<double> values = {50.0, 0.25};

    model.set_parameter_values(values);

    CHECK_EQ(model.parameters()[0].value(), 50.0);
    CHECK_EQ(model.parameters()[1].value(), 0.25);
}

void base_set_parameter_values_wrong_length_throws() {
    LinearTrend model;
    std::vector<double> values = {50.0};  // Only 1 value, but model has 2 parameters

    CHECK_THROWS(model.set_parameter_values(values));
}

void base_negative_start_index_allowed() {
    LinearTrend model;
    model.set_start_index(-100);

    CHECK_EQ(model.start_index(), -100);
}

void base_large_start_index_allowed() {
    LinearTrend model;
    model.set_start_index(std::numeric_limits<int>::max());

    CHECK_EQ(model.start_index(), std::numeric_limits<int>::max());
}

// =====================================================================================
// ConstantTrend  (ConstantTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void constant_constructor_empty_creates_default_model() {
    ConstantTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Constant);
    CHECK_EQ(model.number_of_parameters(), 1);
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
}

void constant_type_returns_constant() {
    ConstantTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Constant);
}

void constant_set_default_parameters_creates_one_parameter() {
    ConstantTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(1));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
}

void constant_set_default_parameters_parameter_value_is_zero() {
    ConstantTrend model;

    CHECK_EQ(model.parameters()[0].value(), 0.0);
}

void constant_set_default_parameters_owner_name_is_propagated() {
    ConstantTrend model;
    model.set_owner_name("Scale");
    model.set_default_parameters();

    CHECK_EQ(model.parameters()[0].owner_name(), std::string("Scale"));
}

void constant_predict_returns_constant_value() {
    ConstantTrend model;
    model.parameters()[0].set_value(50.0);

    // Predict should return same value regardless of index
    CHECK_EQ(model.predict(0), 50.0);
    CHECK_EQ(model.predict(100), 50.0);
    CHECK_EQ(model.predict(-50), 50.0);
    CHECK_EQ(model.predict(std::numeric_limits<int>::max()), 50.0);
}

void constant_predict_with_start_index_still_returns_constant() {
    ConstantTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(75.5);

    // StartIndex should not affect constant trend
    CHECK_EQ(model.predict(1950), 75.5);
    CHECK_EQ(model.predict(1900), 75.5);
    CHECK_EQ(model.predict(2000), 75.5);
}

void constant_predict_negative_value() {
    ConstantTrend model;
    model.parameters()[0].set_value(-25.5);

    CHECK_EQ(model.predict(0), -25.5);
}

void constant_predict_zero_value() {
    ConstantTrend model;
    model.parameters()[0].set_value(0.0);

    CHECK_EQ(model.predict(0), 0.0);
}

void constant_predict_large_value() {
    ConstantTrend model;
    model.parameters()[0].set_value(1e100);

    CHECK_EQ(model.predict(0), 1e100);
}

void constant_clone_creates_independent_copy() {
    // Arrange
    ConstantTrend original;
    original.set_owner_name("Location");
    original.set_start_index(1980);
    original.parameters()[0].set_value(123.456);

    // Act
    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<ConstantTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    // Assert - clone has same values
    CHECK_EQ(clone->owner_name(), original.owner_name());
    CHECK_EQ(clone->start_index(), original.start_index());
    CHECK_EQ(clone->parameters()[0].value(), original.parameters()[0].value());

    // Assert - clone is independent
    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 123.456);
}

void constant_clone_copies_use_default_flat_priors() {
    ConstantTrend original;
    original.set_use_default_flat_priors(false);

    auto clone_ptr = original.clone();

    CHECK_EQ(clone_ptr->use_default_flat_priors(), original.use_default_flat_priors());
}

void constant_clone_clones_parameter_properties() {
    ConstantTrend original;
    original.parameters()[0].set_value(50.0);
    original.parameters()[0].set_lower_bound(0.0);
    original.parameters()[0].set_upper_bound(100.0);

    auto clone_ptr = original.clone();

    CHECK_EQ(clone_ptr->parameters()[0].value(), 50.0);
    CHECK_EQ(clone_ptr->parameters()[0].lower_bound(), 0.0);
    CHECK_EQ(clone_ptr->parameters()[0].upper_bound(), 100.0);
}

void constant_predict_with_nan_returns_nan() {
    ConstantTrend model;
    model.parameters()[0].set_value(std::numeric_limits<double>::quiet_NaN());

    CHECK_TRUE(std::isnan(model.predict(0)));
}

void constant_predict_with_infinity_returns_infinity() {
    ConstantTrend model;
    model.parameters()[0].set_value(std::numeric_limits<double>::infinity());

    CHECK_TRUE(std::isinf(model.predict(0)) && model.predict(0) > 0.0);
}

void constant_set_parameter_values_updates_prediction() {
    ConstantTrend model;
    model.set_parameter_values({42.0});

    CHECK_EQ(model.predict(0), 42.0);
}

// =====================================================================================
// LinearTrend  (LinearTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties,
//   Test_RoundTrip_PreservesPrediction -- XML.
// =====================================================================================

void linear_constructor_empty_creates_default_model() {
    LinearTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Linear);
    CHECK_EQ(model.number_of_parameters(), 2);
}

void linear_type_returns_linear() {
    LinearTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Linear);
}

void linear_set_default_parameters_creates_two_parameters() {
    LinearTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
}

void linear_set_default_parameters_slope_has_bounds() {
    LinearTrend model;

    // beta (slope) should have bounds [-1, 1]
    CHECK_EQ(model.parameters()[1].lower_bound(), -1.0);
    CHECK_EQ(model.parameters()[1].upper_bound(), 1.0);
}

void linear_set_default_parameters_slope_has_uniform_prior() {
    LinearTrend model;

    const auto* prior = dynamic_cast<const Uniform*>(&model.parameters()[1].prior_distribution());
    CHECK_TRUE(prior != nullptr);
    CHECK_EQ(prior->min(), -1.0);
    CHECK_EQ(prior->max(), 1.0);
}

void linear_predict_at_start_index_returns_intercept() {
    LinearTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.5);    // beta

    // At t = StartIndex, y = alpha + beta * 0 = alpha
    CHECK_NEAR(model.predict(1950), 100.0, 1e-10);
}

void linear_predict_after_start_index_returns_correct_value() {
    LinearTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.5);    // beta

    // At t = 1960, y = 100 + 0.5 * (1960 - 1950) = 100 + 5 = 105
    CHECK_NEAR(model.predict(1960), 105.0, 1e-10);
}

void linear_predict_before_start_index_returns_correct_value() {
    LinearTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.5);    // beta

    // At t = 1940, y = 100 + 0.5 * (1940 - 1950) = 100 - 5 = 95
    CHECK_NEAR(model.predict(1940), 95.0, 1e-10);
}

void linear_predict_with_negative_slope_decreasing() {
    LinearTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(-2.0);   // beta

    // At t = 10, y = 100 - 2 * 10 = 80
    CHECK_NEAR(model.predict(10), 80.0, 1e-10);
}

void linear_predict_with_zero_slope_returns_constant() {
    LinearTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(50.0);  // alpha
    model.parameters()[1].set_value(0.0);   // beta

    // Should return alpha regardless of index
    CHECK_NEAR(model.predict(0), 50.0, 1e-10);
    CHECK_NEAR(model.predict(100), 50.0, 1e-10);
    CHECK_NEAR(model.predict(-100), 50.0, 1e-10);
}

void linear_predict_large_time_offset() {
    LinearTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);    // alpha
    model.parameters()[1].set_value(0.001);  // beta (small slope)

    // At t = 1000, y = 0 + 0.001 * 1000 = 1.0
    CHECK_NEAR(model.predict(1000), 1.0, 1e-10);
}

void linear_predict_default_start_index_zero() {
    LinearTrend model;
    model.parameters()[0].set_value(10.0);
    model.parameters()[1].set_value(1.0);

    // StartIndex defaults to 0, so at t = 5: y = 10 + 1 * 5 = 15
    CHECK_EQ(model.start_index(), 0);
    CHECK_NEAR(model.predict(5), 15.0, 1e-10);
}

void linear_clone_creates_independent_copy() {
    // Arrange
    LinearTrend original;
    original.set_owner_name("Location");
    original.set_start_index(1980);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(0.25);

    // Act
    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<LinearTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    // Assert - clone has same values
    CHECK_EQ(clone->owner_name(), original.owner_name());
    CHECK_EQ(clone->start_index(), original.start_index());
    CHECK_EQ(clone->parameters()[0].value(), original.parameters()[0].value());
    CHECK_EQ(clone->parameters()[1].value(), original.parameters()[1].value());

    // Assert - clone is independent
    original.parameters()[0].set_value(999.0);
    original.parameters()[1].set_value(0.99);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
    CHECK_EQ(clone->parameters()[1].value(), 0.25);
}

void linear_clone_preserves_prediction() {
    LinearTrend original;
    original.set_start_index(2000);
    original.parameters()[0].set_value(50.0);
    original.parameters()[1].set_value(0.1);

    auto clone_ptr = original.clone();

    // Both should give same prediction
    CHECK_NEAR(clone_ptr->predict(2010), original.predict(2010), 1e-10);
}

void linear_predict_large_slope() {
    LinearTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1000.0);  // Very steep slope

    CHECK_NEAR(model.predict(10), 10000.0, 1e-10);
}

void linear_predict_small_slope() {
    LinearTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(1e-10);  // Nearly flat

    // After 1000 units: y = 100 + 1e-10 * 1000 = 100 + 1e-7
    CHECK_NEAR(model.predict(1000), 100.0000001, 1e-10);
}

void linear_set_parameter_values_updates_prediction() {
    LinearTrend model;
    model.set_start_index(0);
    model.set_parameter_values({20.0, 2.0});

    // y(5) = 20 + 2 * 5 = 30
    CHECK_NEAR(model.predict(5), 30.0, 1e-10);
}

void linear_predict_integer_overflow_handled() {
    // This tests behavior with extreme time offsets
    LinearTrend model;
    model.set_start_index(std::numeric_limits<int>::min());
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(0.0);  // Zero slope to avoid overflow

    // Should still work
    CHECK_NEAR(model.predict(std::numeric_limits<int>::max()), 0.0, 1e-10);
}

// =====================================================================================
// QuadraticTrend  (QuadraticTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void quadratic_constructor_empty_creates_default_model() {
    QuadraticTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Quadratic);
    CHECK_EQ(model.number_of_parameters(), 3);
}

void quadratic_type_returns_quadratic() {
    QuadraticTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Quadratic);
}

void quadratic_set_default_parameters_creates_three_parameters() {
    QuadraticTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(3));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
    CHECK_EQ(model.parameters()[2].name(), kGamma);
}

void quadratic_predict_at_start_index_returns_intercept() {
    QuadraticTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(0.5);
    model.parameters()[2].set_value(0.01);

    CHECK_NEAR(model.predict(1950), 100.0, 1e-10);
}

void quadratic_predict_quadratic_growth() {
    QuadraticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);  // alpha
    model.parameters()[1].set_value(0.0);  // beta
    model.parameters()[2].set_value(1.0);  // gamma (pure quadratic)

    // y(t) = t^2
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    CHECK_NEAR(model.predict(1), 1.0, 1e-10);
    CHECK_NEAR(model.predict(2), 4.0, 1e-10);
    CHECK_NEAR(model.predict(3), 9.0, 1e-10);
    CHECK_NEAR(model.predict(10), 100.0, 1e-10);
}

void quadratic_predict_parabola_with_vertex() {
    // y(t) = 100 - 0.1 * t^2 has vertex at t=0
    QuadraticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.0);    // beta
    model.parameters()[2].set_value(-0.1);   // gamma (downward parabola)

    CHECK_NEAR(model.predict(0), 100.0, 1e-10);  // Vertex
    CHECK_NEAR(model.predict(1), 99.9, 1e-10);
    CHECK_NEAR(model.predict(-1), 99.9, 1e-10);  // Symmetric
    CHECK_NEAR(model.predict(10), 90.0, 1e-10);
}

void quadratic_predict_full_quadratic() {
    QuadraticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(10.0);  // alpha
    model.parameters()[1].set_value(2.0);   // beta
    model.parameters()[2].set_value(0.5);   // gamma

    // y(5) = 10 + 2*5 + 0.5*25 = 10 + 10 + 12.5 = 32.5
    CHECK_NEAR(model.predict(5), 32.5, 1e-10);
}

void quadratic_predict_with_start_index() {
    QuadraticTrend model;
    model.set_start_index(2000);
    model.parameters()[0].set_value(50.0);  // alpha
    model.parameters()[1].set_value(1.0);   // beta
    model.parameters()[2].set_value(0.01);  // gamma

    // At t=2010: y = 50 + 1*10 + 0.01*100 = 50 + 10 + 1 = 61
    CHECK_NEAR(model.predict(2010), 61.0, 1e-10);
}

void quadratic_clone_creates_independent_copy() {
    QuadraticTrend original;
    original.set_start_index(1990);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(0.5);
    original.parameters()[2].set_value(0.01);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<QuadraticTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

// =====================================================================================
// CubicTrend  (CubicTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void cubic_constructor_empty_creates_default_model() {
    CubicTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Cubic);
    CHECK_EQ(model.number_of_parameters(), 4);
}

void cubic_type_returns_cubic() {
    CubicTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Cubic);
}

void cubic_set_default_parameters_creates_four_parameters() {
    CubicTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(4));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
    CHECK_EQ(model.parameters()[2].name(), kGamma);
    CHECK_EQ(model.parameters()[3].name(), kDelta);
}

void cubic_predict_at_start_index_returns_intercept() {
    CubicTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);

    CHECK_NEAR(model.predict(1950), 100.0, 1e-10);
}

void cubic_predict_pure_cubic() {
    CubicTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);  // alpha
    model.parameters()[1].set_value(0.0);  // beta
    model.parameters()[2].set_value(0.0);  // gamma
    model.parameters()[3].set_value(1.0);  // delta (pure cubic)

    // y(t) = t^3
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    CHECK_NEAR(model.predict(1), 1.0, 1e-10);
    CHECK_NEAR(model.predict(2), 8.0, 1e-10);
    CHECK_NEAR(model.predict(3), 27.0, 1e-10);
    CHECK_NEAR(model.predict(-2), -8.0, 1e-10);
}

void cubic_predict_full_cubic() {
    CubicTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);  // alpha
    model.parameters()[1].set_value(2.0);  // beta
    model.parameters()[2].set_value(3.0);  // gamma
    model.parameters()[3].set_value(4.0);  // delta

    // y(2) = 1 + 2*2 + 3*4 + 4*8 = 1 + 4 + 12 + 32 = 49
    CHECK_NEAR(model.predict(2), 49.0, 1e-10);
}

void cubic_predict_inflection_point() {
    // Cubic with inflection at origin
    CubicTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(0.0);
    model.parameters()[2].set_value(0.0);
    model.parameters()[3].set_value(1.0);

    // Inflection point at t=0: second derivative = 0
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    // Odd symmetry through origin
    CHECK_NEAR(model.predict(5), -model.predict(-5), 1e-10);
}

void cubic_clone_creates_independent_copy() {
    CubicTrend original;
    original.set_start_index(1990);
    original.parameters()[0].set_value(100.0);
    original.parameters()[3].set_value(0.001);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<CubicTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

// =====================================================================================
// ExponentialTrend  (ExponentialTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties,
//   Test_RoundTrip_PreservesPrediction -- XML.
// =====================================================================================

void exponential_constructor_empty_creates_default_model() {
    ExponentialTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Exponential);
    CHECK_EQ(model.number_of_parameters(), 2);
}

void exponential_type_returns_exponential() {
    ExponentialTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Exponential);
}

void exponential_set_default_parameters_creates_two_parameters() {
    ExponentialTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
}

void exponential_set_default_parameters_rate_has_bounds() {
    ExponentialTrend model;

    // beta (rate) should have bounds [-1, 1]
    CHECK_EQ(model.parameters()[1].lower_bound(), -1.0);
    CHECK_EQ(model.parameters()[1].upper_bound(), 1.0);
}

void exponential_set_default_parameters_rate_has_uniform_prior() {
    ExponentialTrend model;

    const auto* prior = dynamic_cast<const Uniform*>(&model.parameters()[1].prior_distribution());
    CHECK_TRUE(prior != nullptr);
}

void exponential_predict_at_start_index_returns_alpha() {
    ExponentialTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.05);   // beta

    // At t = StartIndex, y = alpha * exp(0) = alpha
    CHECK_NEAR(model.predict(1950), 100.0, 1e-10);
}

void exponential_predict_exponential_growth() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);  // alpha
    model.parameters()[1].set_value(1.0);  // beta

    // At t = 1, y = 1 * exp(1) = e
    CHECK_NEAR(model.predict(1), kE, 1e-10);
}

void exponential_predict_exponential_decay() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(-0.1);   // beta (negative = decay)

    // At t = 10, y = 100 * exp(-0.1 * 10) = 100 * exp(-1)
    const double expected = 100.0 * std::exp(-1.0);
    CHECK_NEAR(model.predict(10), expected, 1e-10);
}

void exponential_predict_with_zero_rate_returns_constant() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(50.0);  // alpha
    model.parameters()[1].set_value(0.0);   // beta = 0 means exp(0) = 1

    // Should return alpha regardless of index
    CHECK_NEAR(model.predict(0), 50.0, 1e-10);
    CHECK_NEAR(model.predict(100), 50.0, 1e-10);
    CHECK_NEAR(model.predict(-100), 50.0, 1e-10);
}

void exponential_predict_doubling_time() {
    // Test that doubling time formula works
    const double doubling_time = std::log(2.0) / 0.1;  // ~6.93 time units
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(0.1);

    // After one doubling time, value should be ~200.
    // The continuous doubling time is 6.9315; rounding overshoots to t=7, where value =
    // 100*exp(0.7) ~= 201.375 -- an unavoidable ~0.7% overshoot from integer rounding.
    // Tolerance is 2.0 (1% of 200).
    const double value = model.predict(static_cast<int>(std::lround(doubling_time)));
    CHECK_NEAR(value, 200.0, 2.0);
}

void exponential_predict_half_life() {
    // Test that half-life formula works
    const double half_life = std::log(2.0) / 0.1;  // ~6.93 time units
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(-0.1);  // Decay

    // After one half-life, value should be ~50
    const double value = model.predict(static_cast<int>(std::lround(half_life)));
    CHECK_NEAR(value, 50.0, 1.0);  // Allow small error
}

void exponential_predict_large_exponent_propagates_infinity() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(1.0);  // beta = 1

    // At t = 1000, exponent = 1000 which would overflow exp(). Predict propagates the
    // saturation as +inf (since a > 0) so downstream log-likelihood guards reject the
    // unphysical extrapolation, rather than treating a * 1e304 as a valid prediction.
    const double result = model.predict(1000);

    CHECK_TRUE(std::isinf(result) && result > 0.0);
}

void exponential_predict_large_negative_exponent_clamped() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(-1.0);  // beta = -1

    // At t = 1000, exponent = -1000 which underflows to 0
    const double result = model.predict(1000);

    // Should be a very small positive number (or 0)
    CHECK_TRUE(result >= 0.0);
    CHECK_TRUE(!std::isnan(result));
}

void exponential_predict_normal_range_not_affected_by_clamping() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(0.01);

    // At t = 50, exponent = 0.5, well within normal range
    const double expected = 100.0 * std::exp(0.5);
    CHECK_NEAR(model.predict(50), expected, 1e-10);
}

void exponential_clone_creates_independent_copy() {
    // Arrange
    ExponentialTrend original;
    original.set_owner_name("Location");
    original.set_start_index(1980);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(0.02);

    // Act
    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<ExponentialTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    // Assert - clone has same values
    CHECK_EQ(clone->owner_name(), original.owner_name());
    CHECK_EQ(clone->start_index(), original.start_index());
    CHECK_EQ(clone->parameters()[0].value(), original.parameters()[0].value());
    CHECK_EQ(clone->parameters()[1].value(), original.parameters()[1].value());

    // Assert - clone is independent
    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

void exponential_clone_preserves_prediction() {
    ExponentialTrend original;
    original.set_start_index(2000);
    original.parameters()[0].set_value(50.0);
    original.parameters()[1].set_value(0.03);

    auto clone_ptr = original.clone();

    CHECK_NEAR(clone_ptr->predict(2010), original.predict(2010), 1e-10);
}

void exponential_predict_zero_alpha_returns_zero() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(0.1);

    // 0 * exp(anything) = 0
    CHECK_NEAR(model.predict(10), 0.0, 1e-10);
}

void exponential_predict_negative_alpha() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(-100.0);
    model.parameters()[1].set_value(0.1);

    // Negative alpha * exp(positive) = negative
    const double result = model.predict(10);
    CHECK_TRUE(result < 0.0);
    CHECK_NEAR(result, -100.0 * std::exp(1.0), 1e-10);
}

void exponential_set_parameter_values_updates_prediction() {
    ExponentialTrend model;
    model.set_start_index(0);
    model.set_parameter_values({100.0, 0.1});

    const double expected = 100.0 * std::exp(0.5);
    CHECK_NEAR(model.predict(5), expected, 1e-10);
}

void exponential_predict_small_rate_approximately_linear() {
    // For small beta, exp(beta*t) ~= 1 + beta*t
    ExponentialTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(0.001);  // Very small rate

    // At t = 10: exp(0.01) ~= 1.01005
    const double exponential_result = model.predict(10);
    const double linear_approx = 100.0 * (1.0 + 0.001 * 10.0);  // 101.0

    // Should be close to linear approximation
    CHECK_NEAR(exponential_result, linear_approx, 0.1);
}

// =====================================================================================
// PowerTrend  (PowerTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void power_constructor_empty_creates_default_model() {
    PowerTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Power);
    CHECK_EQ(model.number_of_parameters(), 2);
}

void power_type_returns_power() {
    PowerTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Power);
}

void power_set_default_parameters_creates_two_parameters() {
    PowerTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
}

void power_set_default_parameters_exponent_has_bounds() {
    PowerTrend model;

    // beta (exponent) should have bounds [-5, 5]
    CHECK_EQ(model.parameters()[1].lower_bound(), -5.0);
    CHECK_EQ(model.parameters()[1].upper_bound(), 5.0);
}

void power_set_default_parameters_exponent_has_uniform_prior() {
    PowerTrend model;

    const auto* prior = dynamic_cast<const Uniform*>(&model.parameters()[1].prior_distribution());
    CHECK_TRUE(prior != nullptr);
}

void power_predict_at_start_index_returns_zero() {
    PowerTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(2.0);

    // At t = StartIndex, (t - StartIndex) = 0, so y = alpha * 0^beta = 0
    CHECK_NEAR(model.predict(1950), 0.0, 1e-10);
}

void power_predict_square_law() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);  // alpha
    model.parameters()[1].set_value(2.0);  // beta (square)

    // y(t) = t^2
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    CHECK_NEAR(model.predict(1), 1.0, 1e-10);
    CHECK_NEAR(model.predict(2), 4.0, 1e-10);
    CHECK_NEAR(model.predict(3), 9.0, 1e-10);
    CHECK_NEAR(model.predict(10), 100.0, 1e-10);
}

void power_predict_cube_law() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(2.0);  // alpha
    model.parameters()[1].set_value(3.0);  // beta (cube)

    // y(t) = 2 * t^3
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    CHECK_NEAR(model.predict(1), 2.0, 1e-10);
    CHECK_NEAR(model.predict(2), 16.0, 1e-10);  // 2 * 8
    CHECK_NEAR(model.predict(3), 54.0, 1e-10);  // 2 * 27
}

void power_predict_square_root() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);  // alpha
    model.parameters()[1].set_value(0.5);  // beta (square root)

    // y(t) = sqrt(t)
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    CHECK_NEAR(model.predict(1), 1.0, 1e-10);
    CHECK_NEAR(model.predict(2), std::sqrt(2.0), 1e-10);
    CHECK_NEAR(model.predict(9), 3.0, 1e-10);
}

void power_predict_inverse_law() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(-1.0);   // beta (inverse)

    // y(t) = 100 / t
    CHECK_NEAR(model.predict(1), 100.0, 1e-10);
    CHECK_NEAR(model.predict(2), 50.0, 1e-10);
    CHECK_NEAR(model.predict(10), 10.0, 1e-10);
}

void power_predict_linear_case() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(5.0);  // alpha
    model.parameters()[1].set_value(1.0);  // beta = 1 (linear)

    // y(t) = 5t
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);
    CHECK_NEAR(model.predict(1), 5.0, 1e-10);
    CHECK_NEAR(model.predict(10), 50.0, 1e-10);
}

void power_predict_constant_case() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(42.0);  // alpha
    model.parameters()[1].set_value(0.0);   // beta = 0, so t^0 = 1

    // y(t) = 42 * t^0 = 42 (for t > 0)
    // Note: 0^0 is undefined but typically treated as 1 in this context
    CHECK_NEAR(model.predict(1), 42.0, 1e-10);
    CHECK_NEAR(model.predict(10), 42.0, 1e-10);
    CHECK_NEAR(model.predict(100), 42.0, 1e-10);
}

void power_predict_with_start_index() {
    PowerTrend model;
    model.set_start_index(2000);
    model.parameters()[0].set_value(1.0);  // alpha
    model.parameters()[1].set_value(2.0);  // beta

    // At t=2010: y = 1 * (10)^2 = 100
    CHECK_NEAR(model.predict(2010), 100.0, 1e-10);
}

void power_predict_negative_t_clamps_to_zero() {
    PowerTrend model;
    model.set_start_index(2000);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(0.5);

    // At t=1990: t - StartIndex = -10, but should be clamped to 0
    // y = 100 * 0^0.5 = 0
    CHECK_NEAR(model.predict(1990), 0.0, 1e-10);
}

void power_predict_negative_t_avoid_complex_values() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(0.5);  // Square root of negative would be complex

    // For index < StartIndex, the implementation clamps t to 0
    const double result = model.predict(-5);
    CHECK_TRUE(!std::isnan(result));
    CHECK_NEAR(result, 0.0, 1e-10);
}

void power_clone_creates_independent_copy() {
    PowerTrend original;
    original.set_start_index(1990);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(1.5);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<PowerTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

void power_clone_preserves_prediction() {
    PowerTrend original;
    original.set_start_index(2000);
    original.parameters()[0].set_value(50.0);
    original.parameters()[1].set_value(0.5);

    auto clone_ptr = original.clone();

    CHECK_NEAR(clone_ptr->predict(2010), original.predict(2010), 1e-10);
}

void power_predict_sediment_transport() {
    // Sediment transport often follows power law: Q = alpha * V^beta
    // where Q is sediment discharge and V is velocity
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.001);  // alpha (coefficient)
    model.parameters()[1].set_value(2.5);    // beta (typically 2-3)

    // At velocity 10: Q = 0.001 * 10^2.5 ~= 0.316
    const double expected = 0.001 * std::pow(10.0, 2.5);
    CHECK_NEAR(model.predict(10), expected, 1e-10);
}

void power_predict_area_scaling() {
    // Drainage area scaling: Q = alpha * A^beta
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(0.75);  // Regional exponent

    // At area 100: Q = 1.0 * 100^0.75 ~= 31.62
    const double expected = std::pow(100.0, 0.75);
    CHECK_NEAR(model.predict(100), expected, 1e-10);
}

void power_predict_zero_alpha() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(2.0);

    // 0 * anything = 0
    CHECK_NEAR(model.predict(10), 0.0, 1e-10);
}

void power_predict_negative_alpha() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(-1.0);
    model.parameters()[1].set_value(2.0);

    // y(t) = -1 * t^2 (negative parabola)
    CHECK_NEAR(model.predict(2), -4.0, 1e-10);
    CHECK_NEAR(model.predict(10), -100.0, 1e-10);
}

void power_predict_large_power() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(5.0);  // At upper bound

    // y(t) = t^5
    CHECK_NEAR(model.predict(2), std::pow(2.0, 5.0), 1e-10);  // 32
}

void power_predict_negative_power() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(-2.0);

    // y(t) = t^-2 = 1/t^2
    CHECK_NEAR(model.predict(2), 0.25, 1e-10);   // 1/4
    CHECK_NEAR(model.predict(10), 0.01, 1e-10);  // 1/100
}

void power_predict_very_small_power() {
    PowerTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(0.001);

    // y(t) = t^0.001 ~= 1 for reasonable t values
    const double result = model.predict(100);
    const double expected = std::pow(100.0, 0.001);  // ~= 1.0046
    CHECK_NEAR(result, expected, 1e-10);
}

// =====================================================================================
// ReciprocalTrend  (ReciprocalTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void reciprocal_constructor_empty_creates_default_model() {
    ReciprocalTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Reciprocal);
    CHECK_EQ(model.number_of_parameters(), 2);
}

void reciprocal_type_returns_reciprocal() {
    ReciprocalTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Reciprocal);
}

void reciprocal_set_default_parameters_creates_two_parameters() {
    ReciprocalTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
}

void reciprocal_set_default_parameters_slope_has_bounds() {
    ReciprocalTrend model;

    // beta should have bounds [-1, 1]
    CHECK_EQ(model.parameters()[1].lower_bound(), -1.0);
    CHECK_EQ(model.parameters()[1].upper_bound(), 1.0);
}

void reciprocal_set_default_parameters_slope_has_uniform_prior() {
    ReciprocalTrend model;

    const auto* prior = dynamic_cast<const Uniform*>(&model.parameters()[1].prior_distribution());
    CHECK_TRUE(prior != nullptr);
}

void reciprocal_predict_at_start_index_returns_one_over_alpha() {
    ReciprocalTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(2.0);  // alpha
    model.parameters()[1].set_value(0.1);  // beta

    // At t = StartIndex, y = 1 / (alpha + 0) = 1 / alpha = 0.5
    CHECK_NEAR(model.predict(1950), 0.5, 1e-10);
}

void reciprocal_predict_simple_reciprocal() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);  // alpha
    model.parameters()[1].set_value(1.0);  // beta

    // y(t) = 1 / (1 + t)
    CHECK_NEAR(model.predict(0), 1.0, 1e-10);            // 1/1
    CHECK_NEAR(model.predict(1), 0.5, 1e-10);            // 1/2
    CHECK_NEAR(model.predict(2), 1.0 / 3.0, 1e-10);      // 1/3
    CHECK_NEAR(model.predict(9), 0.1, 1e-10);            // 1/10
}

void reciprocal_predict_decreasing_trend() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(10.0);  // alpha
    model.parameters()[1].set_value(1.0);   // beta

    // y(t) = 1 / (10 + t)
    CHECK_NEAR(model.predict(0), 0.1, 1e-10);             // 1/10
    CHECK_NEAR(model.predict(10), 1.0 / 20.0, 1e-10);     // 1/20
    CHECK_NEAR(model.predict(90), 0.01, 1e-10);           // 1/100
}

void reciprocal_predict_increasing_trend() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(10.0);  // alpha
    model.parameters()[1].set_value(-0.1);  // beta (negative)

    // y(t) = 1 / (10 - 0.1t)
    // At t=0: y = 0.1
    // At t=50: y = 1/(10-5) = 0.2
    CHECK_NEAR(model.predict(0), 0.1, 1e-10);
    CHECK_NEAR(model.predict(50), 0.2, 1e-10);
}

void reciprocal_predict_with_start_index() {
    ReciprocalTrend model;
    model.set_start_index(2000);
    model.parameters()[0].set_value(4.0);  // alpha
    model.parameters()[1].set_value(0.1);  // beta

    // At t=2010: y = 1 / (4 + 0.1*10) = 1/5 = 0.2
    CHECK_NEAR(model.predict(2010), 0.2, 1e-10);
}

void reciprocal_predict_zero_slope_returns_constant() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(5.0);  // alpha
    model.parameters()[1].set_value(0.0);  // beta = 0

    // y(t) = 1 / 5 = 0.2 (constant)
    CHECK_NEAR(model.predict(0), 0.2, 1e-10);
    CHECK_NEAR(model.predict(100), 0.2, 1e-10);
    CHECK_NEAR(model.predict(-50), 0.2, 1e-10);
}

void reciprocal_predict_near_zero_denominator_clamped() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);   // alpha
    model.parameters()[1].set_value(-1.0);  // beta

    // At t=1: denominator = 1 - 1 = 0, should be clamped
    const double result = model.predict(1);

    CHECK_TRUE(!std::isinf(result));
    CHECK_TRUE(!std::isnan(result));
}

void reciprocal_predict_very_small_positive_denominator() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1e-13);
    model.parameters()[1].set_value(0.0);

    const double result = model.predict(0);

    // Should clamp to MinDenominatorMagnitude (1e-12) and return ~1e12
    CHECK_TRUE(!std::isinf(result));
    CHECK_TRUE(!std::isnan(result));
    CHECK_TRUE(result > 0.0);
}

void reciprocal_predict_very_small_negative_denominator() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(-1e-13);
    model.parameters()[1].set_value(0.0);

    const double result = model.predict(0);

    // Should clamp to -MinDenominatorMagnitude and return large negative
    CHECK_TRUE(!std::isinf(result));
    CHECK_TRUE(!std::isnan(result));
    CHECK_TRUE(result < 0.0);
}

void reciprocal_predict_exactly_zero_denominator() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(0.0);

    const double result = model.predict(0);

    // Both parameters zero means denominator = 0, should be clamped
    CHECK_TRUE(!std::isinf(result));
    CHECK_TRUE(!std::isnan(result));
}

void reciprocal_clone_creates_independent_copy() {
    ReciprocalTrend original;
    original.set_start_index(1990);
    original.parameters()[0].set_value(5.0);
    original.parameters()[1].set_value(0.1);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<ReciprocalTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 5.0);
}

void reciprocal_clone_preserves_prediction() {
    ReciprocalTrend original;
    original.set_start_index(2000);
    original.parameters()[0].set_value(10.0);
    original.parameters()[1].set_value(0.05);

    auto clone_ptr = original.clone();

    CHECK_NEAR(clone_ptr->predict(2010), original.predict(2010), 1e-10);
}

void reciprocal_predict_hyperbolic_decay() {
    // Many physical processes follow hyperbolic decay: y = 1/(a + bt)
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(0.1);

    // Half-value occurs when denominator doubles
    // 1 + 0.1t = 2 => t = 10
    const double initial = model.predict(0);
    const double half_value = model.predict(10);
    CHECK_NEAR(half_value, initial / 2.0, 1e-10);
}

void reciprocal_predict_rating_curve_approximation() {
    // Rating curve with asymptotic behavior at high stages
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.1);
    model.parameters()[1].set_value(0.01);

    // y(0) = 1/0.1 = 10
    // y(90) = 1/(0.1 + 0.9) = 1
    CHECK_NEAR(model.predict(0), 10.0, 1e-10);
    CHECK_NEAR(model.predict(90), 1.0, 1e-10);
}

void reciprocal_predict_negative_time() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(10.0);
    model.parameters()[1].set_value(1.0);

    // y(-5) = 1/(10 + 1*(-5)) = 1/5 = 0.2
    CHECK_NEAR(model.predict(-5), 0.2, 1e-10);
}

void reciprocal_predict_large_time() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(1.0);
    model.parameters()[1].set_value(0.001);

    // y(1000) = 1/(1 + 1) = 0.5
    CHECK_NEAR(model.predict(1000), 0.5, 1e-10);
}

void reciprocal_predict_negative_intercept() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(-5.0);  // Negative alpha
    model.parameters()[1].set_value(1.0);

    // y(0) = 1/(-5) = -0.2
    CHECK_NEAR(model.predict(0), -0.2, 1e-10);
    // y(10) = 1/(-5 + 10) = 1/5 = 0.2
    CHECK_NEAR(model.predict(10), 0.2, 1e-10);
}

void reciprocal_predict_asymptote() {
    ReciprocalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(0.01);

    // As t -> inf, y -> 0
    const double far_future = model.predict(100000);
    CHECK_TRUE(std::fabs(far_future) < 0.01);
}

// =====================================================================================
// LogisticTrend  (LogisticTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void logistic_constructor_empty_creates_default_model() {
    LogisticTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Logistic);
    CHECK_EQ(model.number_of_parameters(), 2);
}

void logistic_type_returns_logistic() {
    LogisticTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Logistic);
}

void logistic_set_default_parameters_creates_two_parameters() {
    LogisticTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);
    CHECK_EQ(model.parameters()[1].name(), kBeta);
}

void logistic_predict_at_start_index_returns_half_alpha() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.1);    // beta > 0

    // At t=0: y = 100 / (1 + exp(0)) = 100 / 2 = 50
    CHECK_NEAR(model.predict(0), 50.0, 1e-10);
}

void logistic_predict_large_positive_t_approaches_alpha() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.1);    // beta > 0

    // For large positive t, exp(-beta*t) -> 0, so y -> alpha
    const double y_large = model.predict(100);
    CHECK_TRUE(y_large > 99.0);
}

void logistic_predict_large_negative_t_approaches_zero() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(0.1);    // beta > 0

    // For large negative t, exp(-beta*t) -> inf, so y -> 0
    const double y_small = model.predict(-100);
    CHECK_TRUE(y_small < 1.0);
}

void logistic_predict_negative_beta_reverses_direction() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(-0.1);  // beta < 0 (decreasing)

    // For beta < 0, curve is reversed
    const double y_start = model.predict(0);
    const double y_large = model.predict(100);
    const double y_neg = model.predict(-100);

    CHECK_NEAR(y_start, 50.0, 1e-10);
    CHECK_TRUE(y_large < y_start);  // Should decrease with positive t when beta < 0
    CHECK_TRUE(y_neg > y_start);    // Should increase with negative t when beta < 0
}

void logistic_predict_zero_beta_returns_constant() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(0.0);  // beta = 0

    // y = alpha / (1 + exp(0)) = alpha / 2 regardless of t
    CHECK_NEAR(model.predict(0), 50.0, 1e-10);
    CHECK_NEAR(model.predict(100), 50.0, 1e-10);
    CHECK_NEAR(model.predict(-100), 50.0, 1e-10);
}

void logistic_predict_large_beta_steeper_curve() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(1.0);  // Large beta

    // Steeper transition
    CHECK_NEAR(model.predict(0), 50.0, 1e-10);
    CHECK_TRUE(model.predict(5) > 99.0);   // Should be near 100 at t=5 with beta=1
    CHECK_TRUE(model.predict(-5) < 1.0);   // Should be near 0 at t=-5 with beta=1
}

void logistic_predict_extreme_positive_t_no_overflow() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(1.0);

    // Very large t should not cause overflow
    const double result = model.predict(10000);
    CHECK_TRUE(!std::isnan(result));
    CHECK_TRUE(!std::isinf(result));
    CHECK_TRUE(result > 0.0);
}

void logistic_predict_extreme_negative_t_no_underflow() {
    LogisticTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(1.0);

    // Very large negative t should not cause issues
    const double result = model.predict(-10000);
    CHECK_TRUE(!std::isnan(result));
    CHECK_TRUE(!std::isinf(result));
    CHECK_TRUE(result >= 0.0);
}

void logistic_clone_creates_independent_copy() {
    LogisticTrend original;
    original.set_start_index(1950);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(0.1);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<LogisticTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

void logistic_clone_preserves_prediction() {
    LogisticTrend original;
    original.set_start_index(1950);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(0.05);

    auto clone_ptr = original.clone();

    CHECK_NEAR(clone_ptr->predict(1960), original.predict(1960), 1e-10);
}

void logistic_urbanization_scenario() {
    // Model urbanization effect on flooding
    // Gradual transition from rural to urban catchment
    LogisticTrend model;
    model.set_start_index(1960);
    model.parameters()[0].set_value(5000.0);  // Maximum increase in peak flow
    model.parameters()[1].set_value(0.05);    // Gradual transition

    // Before urbanization: minimal effect.
    // With beta=0.05, the curve reaches 10% of asymptote (i.e., value < 500) only at
    // t < -ln(9)/beta ~= -44 years before the inflection (1916). Use 1900 (-60 years),
    // which is unambiguously inside the sub-5% region where "minimal effect" holds:
    // 5000 / (1 + exp(3)) ~= 237.
    CHECK_TRUE(model.predict(1900) < 500.0);

    // Mid transition
    CHECK_NEAR(model.predict(1960), 2500.0, 1.0);  // Half effect at StartIndex

    // After urbanization: near maximum effect
    CHECK_TRUE(model.predict(2020) > 4500.0);
}

// =====================================================================================
// SinusoidalTrend  (SinusoidalTrendTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties,
//   Test_RoundTrip_PreservesPhaseParameter -- XML.
// =====================================================================================

void sinusoidal_constructor_empty_creates_default_model() {
    SinusoidalTrend model;

    CHECK_TRUE(model.type() == TrendModelType::Sinusoidal);
    CHECK_EQ(model.number_of_parameters(), 4);
}

void sinusoidal_type_returns_sinusoidal() {
    SinusoidalTrend model;
    CHECK_TRUE(model.type() == TrendModelType::Sinusoidal);
}

void sinusoidal_set_default_parameters_creates_four_parameters() {
    SinusoidalTrend model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(4));
    CHECK_EQ(model.parameters()[0].name(), kAlpha);  // Mean level
    CHECK_EQ(model.parameters()[1].name(), kBeta);   // Amplitude
    CHECK_EQ(model.parameters()[2].name(), kGamma);  // Frequency
    CHECK_EQ(model.parameters()[3].name(), kDelta);  // Phase
}

void sinusoidal_set_default_parameters_amplitude_has_positive_bounds() {
    SinusoidalTrend model;

    // beta (amplitude) should be non-negative [0, 1]
    CHECK_EQ(model.parameters()[1].lower_bound(), 0.0);
    CHECK_EQ(model.parameters()[1].upper_bound(), 1.0);
}

void sinusoidal_set_default_parameters_frequency_has_positive_bounds() {
    SinusoidalTrend model;

    // gamma (frequency) should be positive (up to Nyquist = 0.5)
    CHECK_TRUE(model.parameters()[2].lower_bound() > 0.0);
    CHECK_EQ(model.parameters()[2].upper_bound(), 0.5);
}

void sinusoidal_set_default_parameters_phase_has_periodic_bounds() {
    SinusoidalTrend model;

    // delta (phase) should be [0, 2*pi]
    CHECK_EQ(model.parameters()[3].lower_bound(), 0.0);
    CHECK_NEAR(model.parameters()[3].upper_bound(), 2.0 * kPi, 1e-10);
}

void sinusoidal_set_default_parameters_all_have_uniform_priors() {
    SinusoidalTrend model;

    CHECK_TRUE(dynamic_cast<const Uniform*>(&model.parameters()[1].prior_distribution()) != nullptr);
    CHECK_TRUE(dynamic_cast<const Uniform*>(&model.parameters()[2].prior_distribution()) != nullptr);
    CHECK_TRUE(dynamic_cast<const Uniform*>(&model.parameters()[3].prior_distribution()) != nullptr);
}

void sinusoidal_predict_at_start_index_with_zero_phase() {
    SinusoidalTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(100.0);  // alpha
    model.parameters()[1].set_value(10.0);   // beta
    model.parameters()[2].set_value(0.1);    // gamma
    model.parameters()[3].set_value(0.0);    // delta (zero phase)

    // At t = StartIndex: y = alpha + beta * sin(0) = alpha = 100
    CHECK_NEAR(model.predict(1950), 100.0, 1e-10);
}

void sinusoidal_predict_simple_sine() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);   // alpha (no offset)
    model.parameters()[1].set_value(1.0);   // beta (unit amplitude)
    model.parameters()[2].set_value(0.25);  // gamma (period = 4 time units)
    model.parameters()[3].set_value(0.0);   // delta (no phase shift)

    // y(t) = sin(2*pi * 0.25 * t) = sin(pi*t/2)
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);    // sin(0) = 0
    CHECK_NEAR(model.predict(1), 1.0, 1e-10);    // sin(pi/2) = 1
    CHECK_NEAR(model.predict(2), 0.0, 1e-10);    // sin(pi) = 0
    CHECK_NEAR(model.predict(3), -1.0, 1e-10);   // sin(3*pi/2) = -1
    CHECK_NEAR(model.predict(4), 0.0, 1e-10);    // sin(2*pi) = 0
}

void sinusoidal_predict_with_mean_level() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(50.0);  // alpha (mean = 50)
    model.parameters()[1].set_value(10.0);  // beta (amplitude = 10)
    model.parameters()[2].set_value(0.25);  // gamma (period = 4)
    model.parameters()[3].set_value(0.0);   // delta

    // Oscillates between 40 and 60
    CHECK_NEAR(model.predict(0), 50.0, 1e-10);  // Mean
    CHECK_NEAR(model.predict(1), 60.0, 1e-10);  // Max
    CHECK_NEAR(model.predict(2), 50.0, 1e-10);  // Mean
    CHECK_NEAR(model.predict(3), 40.0, 1e-10);  // Min
}

void sinusoidal_predict_with_phase_shift() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(0.25);
    model.parameters()[3].set_value(kPi / 2.0);  // 90 degree phase shift (cosine)

    // y(t) = sin(pi*t/2 + pi/2) = cos(pi*t/2)
    CHECK_NEAR(model.predict(0), 1.0, 1e-10);    // cos(0) = 1
    CHECK_NEAR(model.predict(1), 0.0, 1e-10);    // cos(pi/2) = 0
    CHECK_NEAR(model.predict(2), -1.0, 1e-10);   // cos(pi) = -1
    CHECK_NEAR(model.predict(3), 0.0, 1e-10);    // cos(3*pi/2) = 0
}

void sinusoidal_predict_annual_cycle() {
    // Model annual temperature cycle
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(15.0);        // alpha (mean temp 15 C)
    model.parameters()[1].set_value(10.0);        // beta (amplitude 10 C)
    model.parameters()[2].set_value(1.0 / 12.0);  // gamma (annual cycle over 12 months)
    model.parameters()[3].set_value(-kPi / 2.0);  // delta (minimum at month 0, e.g. January)

    // Check range
    double max_temp = -std::numeric_limits<double>::infinity();
    double min_temp = std::numeric_limits<double>::infinity();
    for (int i = 0; i < 12; i++) {
        const double temp = model.predict(i);
        if (temp > max_temp) max_temp = temp;
        if (temp < min_temp) min_temp = temp;
    }

    CHECK_TRUE(max_temp <= 25.0);  // alpha + beta
    CHECK_TRUE(min_temp >= 5.0);   // alpha - beta
}

void sinusoidal_predict_zero_amplitude_returns_constant() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(42.0);  // alpha
    model.parameters()[1].set_value(0.0);   // beta = 0 (no oscillation)
    model.parameters()[2].set_value(0.1);   // gamma
    model.parameters()[3].set_value(1.0);   // delta

    // y(t) = 42 + 0 = 42 (constant)
    CHECK_NEAR(model.predict(0), 42.0, 1e-10);
    CHECK_NEAR(model.predict(100), 42.0, 1e-10);
    CHECK_NEAR(model.predict(-50), 42.0, 1e-10);
}

void sinusoidal_predict_period() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(0.1);  // gamma (period = 10 time units)
    model.parameters()[3].set_value(0.0);

    // Period = 1/gamma = 10
    // Values should repeat every 10 units
    CHECK_NEAR(model.predict(10), model.predict(0), 1e-10);
    CHECK_NEAR(model.predict(20), model.predict(0), 1e-10);
    CHECK_NEAR(model.predict(13), model.predict(3), 1e-10);
}

void sinusoidal_clone_creates_independent_copy() {
    SinusoidalTrend original;
    original.set_start_index(1990);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(20.0);
    original.parameters()[2].set_value(0.05);
    original.parameters()[3].set_value(kPi);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<SinusoidalTrend*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

void sinusoidal_clone_preserves_prediction() {
    SinusoidalTrend original;
    original.set_start_index(2000);
    original.parameters()[0].set_value(50.0);
    original.parameters()[1].set_value(10.0);
    original.parameters()[2].set_value(0.1);
    original.parameters()[3].set_value(0.5);

    auto clone_ptr = original.clone();

    CHECK_NEAR(clone_ptr->predict(2010), original.predict(2010), 1e-10);
}

void sinusoidal_predict_tidal_pattern() {
    // Semi-diurnal tide: period ~= 12.42 hours
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(5.0);          // Mean sea level
    model.parameters()[1].set_value(2.0);          // Tidal range
    model.parameters()[2].set_value(1.0 / 12.42);  // Frequency
    model.parameters()[3].set_value(0.0);

    // Max should be mean +- amplitude
    CHECK_TRUE(model.predict(0) >= model.parameters()[0].value() - model.parameters()[1].value());
    CHECK_TRUE(model.predict(0) <= model.parameters()[0].value() + model.parameters()[1].value());
}

void sinusoidal_predict_seasonal_flood_pattern() {
    // Seasonal variation in flood magnitude
    SinusoidalTrend model;
    model.set_start_index(1);  // Month 1 = January
    model.parameters()[0].set_value(50000.0);     // Mean annual flow
    model.parameters()[1].set_value(30000.0);     // Seasonal amplitude
    model.parameters()[2].set_value(1.0 / 12.0);  // Annual cycle
    model.parameters()[3].set_value(kPi);         // Peak in summer (month 7)

    // Range should be mean +- amplitude
    for (int month = 1; month <= 12; month++) {
        const double flow = model.predict(month);
        CHECK_TRUE(flow >= 20000.0);  // min = 50k - 30k
        CHECK_TRUE(flow <= 80000.0);  // max = 50k + 30k
    }
}

void sinusoidal_predict_multi_decadal_oscillation() {
    // Pacific Decadal Oscillation (PDO) ~ 20-30 year cycle
    SinusoidalTrend model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(0.0);         // Zero mean
    model.parameters()[1].set_value(0.5);         // Normalized index
    model.parameters()[2].set_value(1.0 / 25.0);  // 25-year cycle
    model.parameters()[3].set_value(0.0);

    // Check 25-year periodicity
    CHECK_NEAR(model.predict(1975), model.predict(1950), 1e-10);
    CHECK_NEAR(model.predict(2000), model.predict(1950), 1e-10);
}

void sinusoidal_predict_very_high_frequency() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(0.5);  // Nyquist frequency
    model.parameters()[3].set_value(0.0);

    // At Nyquist, period = 2 time units
    CHECK_NEAR(model.predict(0), 0.0, 1e-10);  // sin(0)
    CHECK_NEAR(model.predict(1), 0.0, 1e-10);  // sin(pi)
    CHECK_NEAR(model.predict(2), 0.0, 1e-10);  // sin(2*pi)
}

void sinusoidal_predict_very_low_frequency() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(0.001);  // Period = 1000 time units
    model.parameters()[3].set_value(0.0);

    // Very slow oscillation - nearly constant over short periods
    const double val0 = model.predict(0);
    const double val10 = model.predict(10);
    CHECK_TRUE(std::fabs(val10 - val0) < 0.1);  // Little change
}

void sinusoidal_predict_negative_time() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(0.25);
    model.parameters()[3].set_value(0.0);

    // sin is odd function, so sin(-x) = -sin(x)
    CHECK_NEAR(model.predict(-1), -model.predict(1), 1e-10);
    CHECK_NEAR(model.predict(-3), -model.predict(3), 1e-10);
}

void sinusoidal_predict_large_time() {
    SinusoidalTrend model;
    model.set_start_index(0);
    model.parameters()[0].set_value(0.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(0.1);
    model.parameters()[3].set_value(0.0);

    // Even at large times, result should be bounded
    const double result = model.predict(1000000);
    CHECK_TRUE(result >= -1.0);
    CHECK_TRUE(result <= 1.0);
    CHECK_TRUE(!std::isnan(result));
}

void sinusoidal_predict_phase_equivalence() {
    SinusoidalTrend model1;
    model1.set_start_index(0);
    model1.parameters()[0].set_value(0.0);
    model1.parameters()[1].set_value(1.0);
    model1.parameters()[2].set_value(0.1);
    model1.parameters()[3].set_value(0.0);

    SinusoidalTrend model2;
    model2.set_start_index(0);
    model2.parameters()[0].set_value(0.0);
    model2.parameters()[1].set_value(1.0);
    model2.parameters()[2].set_value(0.1);
    model2.parameters()[3].set_value(2.0 * kPi);  // Full cycle shift

    // Phase shift by 2*pi should give same result
    CHECK_NEAR(model2.predict(5), model1.predict(5), 1e-10);
}

// =====================================================================================
// StepFunction  (StepFunctionTests.cs)
//
// Skipped upstream methods: Test_Constructor_XElement_RestoresModel,
//   Test_ToXElement_ContainsTypeAttribute, Test_RoundTrip_PreservesAllProperties -- XML.
// =====================================================================================

void step_constructor_empty_creates_default_model() {
    StepFunction model;

    CHECK_TRUE(model.type() == TrendModelType::StepFunction);
    CHECK_EQ(model.number_of_parameters(), 3);
}

void step_type_returns_step_function() {
    StepFunction model;
    CHECK_TRUE(model.type() == TrendModelType::StepFunction);
}

void step_set_default_parameters_creates_three_parameters() {
    StepFunction model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(3));
    CHECK_EQ(model.parameters()[0].name(), kMu1);
    CHECK_EQ(model.parameters()[1].name(), kMu2);
    CHECK_EQ(model.parameters()[2].name(), kTs);
}

void step_predict_before_change_point_returns_mu1() {
    StepFunction model;
    model.set_start_index(1900);
    model.parameters()[0].set_value(50.0);    // mu_1
    model.parameters()[1].set_value(100.0);   // mu_2
    model.parameters()[2].set_value(1950.0);  // t_c (change point)

    // Before change point
    CHECK_NEAR(model.predict(1900), 50.0, 1e-10);
    CHECK_NEAR(model.predict(1949), 50.0, 1e-10);
    CHECK_NEAR(model.predict(1950), 50.0, 1e-10);  // At change point (<=)
}

void step_predict_after_change_point_returns_mu2() {
    StepFunction model;
    model.set_start_index(1900);
    model.parameters()[0].set_value(50.0);    // mu_1
    model.parameters()[1].set_value(100.0);   // mu_2
    model.parameters()[2].set_value(1950.0);  // t_c

    // After change point
    CHECK_NEAR(model.predict(1951), 100.0, 1e-10);
    CHECK_NEAR(model.predict(2000), 100.0, 1e-10);
}

void step_predict_at_exact_change_point() {
    StepFunction model;
    model.set_start_index(0);
    model.parameters()[0].set_value(10.0);  // mu_1
    model.parameters()[1].set_value(20.0);  // mu_2
    model.parameters()[2].set_value(50.0);  // t_c

    // At t=50 (t <= t_c), should return mu_1
    CHECK_NEAR(model.predict(50), 10.0, 1e-10);
    // Just after
    CHECK_NEAR(model.predict(51), 20.0, 1e-10);
}

void step_predict_decrease_after_change_point() {
    StepFunction model;
    model.set_start_index(0);
    model.parameters()[0].set_value(100.0);  // mu_1 (higher)
    model.parameters()[1].set_value(50.0);   // mu_2 (lower)
    model.parameters()[2].set_value(25.0);   // t_c

    CHECK_NEAR(model.predict(20), 100.0, 1e-10);
    CHECK_NEAR(model.predict(30), 50.0, 1e-10);
}

void step_predict_same_value_both_sides() {
    StepFunction model;
    model.set_start_index(0);
    model.parameters()[0].set_value(75.0);  // mu_1
    model.parameters()[1].set_value(75.0);  // mu_2 (same)
    model.parameters()[2].set_value(50.0);  // t_c

    // Should return 75 everywhere
    CHECK_NEAR(model.predict(0), 75.0, 1e-10);
    CHECK_NEAR(model.predict(50), 75.0, 1e-10);
    CHECK_NEAR(model.predict(100), 75.0, 1e-10);
}

void step_predict_change_point_at_start() {
    StepFunction model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(50.0);
    model.parameters()[1].set_value(100.0);
    model.parameters()[2].set_value(1950.0);  // Change at start

    // At start: t = t_c, so returns mu_1
    CHECK_NEAR(model.predict(1950), 50.0, 1e-10);
    CHECK_NEAR(model.predict(1951), 100.0, 1e-10);
}

void step_clone_creates_independent_copy() {
    StepFunction original;
    original.set_start_index(1900);
    original.parameters()[0].set_value(50.0);
    original.parameters()[1].set_value(100.0);
    original.parameters()[2].set_value(1950.0);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<StepFunction*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 50.0);
}

void step_clone_preserves_prediction() {
    StepFunction original;
    original.set_start_index(1900);
    original.parameters()[0].set_value(50.0);
    original.parameters()[1].set_value(100.0);
    original.parameters()[2].set_value(1950.0);

    auto clone_ptr = original.clone();

    CHECK_EQ(clone_ptr->predict(1940), original.predict(1940));
    CHECK_EQ(clone_ptr->predict(1960), original.predict(1960));
}

void step_climate_shift_scenario() {
    // Model a climate shift in 1970
    StepFunction model;
    model.set_start_index(1950);
    model.parameters()[0].set_value(1000.0);  // Pre-1970 mean annual flood
    model.parameters()[1].set_value(1200.0);  // Post-1970 mean (20% increase)
    model.parameters()[2].set_value(1970.0);  // Change year

    CHECK_NEAR(model.predict(1960), 1000.0, 1e-10);
    CHECK_NEAR(model.predict(1970), 1000.0, 1e-10);  // At change point
    CHECK_NEAR(model.predict(1980), 1200.0, 1e-10);
    CHECK_NEAR(model.predict(2020), 1200.0, 1e-10);
}

void step_dam_construction_scenario() {
    // Model flood reduction after dam construction in 1960
    StepFunction model;
    model.set_start_index(1900);
    model.parameters()[0].set_value(50000.0);  // Pre-dam peak flows
    model.parameters()[1].set_value(35000.0);  // Post-dam peak flows (30% reduction)
    model.parameters()[2].set_value(1960.0);

    CHECK_NEAR(model.predict(1950), 50000.0, 1e-10);
    CHECK_NEAR(model.predict(1970), 35000.0, 1e-10);
}

}  // namespace

int main() {
    // TrendModelBase
    base_constructor_empty_initializes_parameters();
    base_owner_name_set_and_get();
    base_start_index_set_and_get();
    base_use_default_flat_priors_set_and_get();
    base_use_default_flat_priors_true_resets_parameters();
    base_number_of_parameters_returns_correct_count();
    base_set_parameter_values_updates_all_parameters();
    base_set_parameter_values_wrong_length_throws();
    base_negative_start_index_allowed();
    base_large_start_index_allowed();

    // ConstantTrend
    constant_constructor_empty_creates_default_model();
    constant_type_returns_constant();
    constant_set_default_parameters_creates_one_parameter();
    constant_set_default_parameters_parameter_value_is_zero();
    constant_set_default_parameters_owner_name_is_propagated();
    constant_predict_returns_constant_value();
    constant_predict_with_start_index_still_returns_constant();
    constant_predict_negative_value();
    constant_predict_zero_value();
    constant_predict_large_value();
    constant_clone_creates_independent_copy();
    constant_clone_copies_use_default_flat_priors();
    constant_clone_clones_parameter_properties();
    constant_predict_with_nan_returns_nan();
    constant_predict_with_infinity_returns_infinity();
    constant_set_parameter_values_updates_prediction();

    // LinearTrend
    linear_constructor_empty_creates_default_model();
    linear_type_returns_linear();
    linear_set_default_parameters_creates_two_parameters();
    linear_set_default_parameters_slope_has_bounds();
    linear_set_default_parameters_slope_has_uniform_prior();
    linear_predict_at_start_index_returns_intercept();
    linear_predict_after_start_index_returns_correct_value();
    linear_predict_before_start_index_returns_correct_value();
    linear_predict_with_negative_slope_decreasing();
    linear_predict_with_zero_slope_returns_constant();
    linear_predict_large_time_offset();
    linear_predict_default_start_index_zero();
    linear_clone_creates_independent_copy();
    linear_clone_preserves_prediction();
    linear_predict_large_slope();
    linear_predict_small_slope();
    linear_set_parameter_values_updates_prediction();
    linear_predict_integer_overflow_handled();

    // QuadraticTrend
    quadratic_constructor_empty_creates_default_model();
    quadratic_type_returns_quadratic();
    quadratic_set_default_parameters_creates_three_parameters();
    quadratic_predict_at_start_index_returns_intercept();
    quadratic_predict_quadratic_growth();
    quadratic_predict_parabola_with_vertex();
    quadratic_predict_full_quadratic();
    quadratic_predict_with_start_index();
    quadratic_clone_creates_independent_copy();

    // CubicTrend
    cubic_constructor_empty_creates_default_model();
    cubic_type_returns_cubic();
    cubic_set_default_parameters_creates_four_parameters();
    cubic_predict_at_start_index_returns_intercept();
    cubic_predict_pure_cubic();
    cubic_predict_full_cubic();
    cubic_predict_inflection_point();
    cubic_clone_creates_independent_copy();

    // ExponentialTrend
    exponential_constructor_empty_creates_default_model();
    exponential_type_returns_exponential();
    exponential_set_default_parameters_creates_two_parameters();
    exponential_set_default_parameters_rate_has_bounds();
    exponential_set_default_parameters_rate_has_uniform_prior();
    exponential_predict_at_start_index_returns_alpha();
    exponential_predict_exponential_growth();
    exponential_predict_exponential_decay();
    exponential_predict_with_zero_rate_returns_constant();
    exponential_predict_doubling_time();
    exponential_predict_half_life();
    exponential_predict_large_exponent_propagates_infinity();
    exponential_predict_large_negative_exponent_clamped();
    exponential_predict_normal_range_not_affected_by_clamping();
    exponential_clone_creates_independent_copy();
    exponential_clone_preserves_prediction();
    exponential_predict_zero_alpha_returns_zero();
    exponential_predict_negative_alpha();
    exponential_set_parameter_values_updates_prediction();
    exponential_predict_small_rate_approximately_linear();

    // PowerTrend
    power_constructor_empty_creates_default_model();
    power_type_returns_power();
    power_set_default_parameters_creates_two_parameters();
    power_set_default_parameters_exponent_has_bounds();
    power_set_default_parameters_exponent_has_uniform_prior();
    power_predict_at_start_index_returns_zero();
    power_predict_square_law();
    power_predict_cube_law();
    power_predict_square_root();
    power_predict_inverse_law();
    power_predict_linear_case();
    power_predict_constant_case();
    power_predict_with_start_index();
    power_predict_negative_t_clamps_to_zero();
    power_predict_negative_t_avoid_complex_values();
    power_clone_creates_independent_copy();
    power_clone_preserves_prediction();
    power_predict_sediment_transport();
    power_predict_area_scaling();
    power_predict_zero_alpha();
    power_predict_negative_alpha();
    power_predict_large_power();
    power_predict_negative_power();
    power_predict_very_small_power();

    // ReciprocalTrend
    reciprocal_constructor_empty_creates_default_model();
    reciprocal_type_returns_reciprocal();
    reciprocal_set_default_parameters_creates_two_parameters();
    reciprocal_set_default_parameters_slope_has_bounds();
    reciprocal_set_default_parameters_slope_has_uniform_prior();
    reciprocal_predict_at_start_index_returns_one_over_alpha();
    reciprocal_predict_simple_reciprocal();
    reciprocal_predict_decreasing_trend();
    reciprocal_predict_increasing_trend();
    reciprocal_predict_with_start_index();
    reciprocal_predict_zero_slope_returns_constant();
    reciprocal_predict_near_zero_denominator_clamped();
    reciprocal_predict_very_small_positive_denominator();
    reciprocal_predict_very_small_negative_denominator();
    reciprocal_predict_exactly_zero_denominator();
    reciprocal_clone_creates_independent_copy();
    reciprocal_clone_preserves_prediction();
    reciprocal_predict_hyperbolic_decay();
    reciprocal_predict_rating_curve_approximation();
    reciprocal_predict_negative_time();
    reciprocal_predict_large_time();
    reciprocal_predict_negative_intercept();
    reciprocal_predict_asymptote();

    // LogisticTrend
    logistic_constructor_empty_creates_default_model();
    logistic_type_returns_logistic();
    logistic_set_default_parameters_creates_two_parameters();
    logistic_predict_at_start_index_returns_half_alpha();
    logistic_predict_large_positive_t_approaches_alpha();
    logistic_predict_large_negative_t_approaches_zero();
    logistic_predict_negative_beta_reverses_direction();
    logistic_predict_zero_beta_returns_constant();
    logistic_predict_large_beta_steeper_curve();
    logistic_predict_extreme_positive_t_no_overflow();
    logistic_predict_extreme_negative_t_no_underflow();
    logistic_clone_creates_independent_copy();
    logistic_clone_preserves_prediction();
    logistic_urbanization_scenario();

    // SinusoidalTrend
    sinusoidal_constructor_empty_creates_default_model();
    sinusoidal_type_returns_sinusoidal();
    sinusoidal_set_default_parameters_creates_four_parameters();
    sinusoidal_set_default_parameters_amplitude_has_positive_bounds();
    sinusoidal_set_default_parameters_frequency_has_positive_bounds();
    sinusoidal_set_default_parameters_phase_has_periodic_bounds();
    sinusoidal_set_default_parameters_all_have_uniform_priors();
    sinusoidal_predict_at_start_index_with_zero_phase();
    sinusoidal_predict_simple_sine();
    sinusoidal_predict_with_mean_level();
    sinusoidal_predict_with_phase_shift();
    sinusoidal_predict_annual_cycle();
    sinusoidal_predict_zero_amplitude_returns_constant();
    sinusoidal_predict_period();
    sinusoidal_clone_creates_independent_copy();
    sinusoidal_clone_preserves_prediction();
    sinusoidal_predict_tidal_pattern();
    sinusoidal_predict_seasonal_flood_pattern();
    sinusoidal_predict_multi_decadal_oscillation();
    sinusoidal_predict_very_high_frequency();
    sinusoidal_predict_very_low_frequency();
    sinusoidal_predict_negative_time();
    sinusoidal_predict_large_time();
    sinusoidal_predict_phase_equivalence();

    // StepFunction
    step_constructor_empty_creates_default_model();
    step_type_returns_step_function();
    step_set_default_parameters_creates_three_parameters();
    step_predict_before_change_point_returns_mu1();
    step_predict_after_change_point_returns_mu2();
    step_predict_at_exact_change_point();
    step_predict_decrease_after_change_point();
    step_predict_same_value_both_sides();
    step_predict_change_point_at_start();
    step_clone_creates_independent_copy();
    step_clone_preserves_prediction();
    step_climate_shift_scenario();
    step_dam_construction_scenario();

    return bftest::summary("trend_functions");
}
