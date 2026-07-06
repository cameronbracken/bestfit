// S1 support ctest (C++-only): the SpatialExtremes leaf correlation kernels
// (BasicExponential / PoweredExponential / Spherical) behind ICorrelationModel. These are
// internal support (not public-API distributions), so hardcoded oracles transcribed from the
// upstream C# test files are correct here (public-API oracle values stay in fixtures/ only).
//
// Oracles transcribed VALUES-UNALTERED from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/SpatialExtremes/CorrelationFunctionTests.cs @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/SpatialExtremes/SpatialCorrelationTests.cs @ fc28c0c
// The C# closed-form expressions (Math.Exp / Math.Pow) are recomputed inline in C++ exactly as
// the C# does (std::exp / std::pow), not pre-rounded. C# deltas are transcribed as-is (1e-10 /
// 1e-12); exact identities that C# asserts without a delta (Evaluate(0)==1.0,
// Spherical Evaluate(>=range)==0.0) are asserted exactly.
//
// Deferred to P5 (documented, no regression): the P4 brief's section-1 "public-path
// corroboration" (dump correlation-model Evaluate spot values through the REAL C# via the oracle
// emitter to back these transcribed leaf oracles) is NOT wired. It is redundant defense-in-depth
// -- the oracles above are transcribed VALUES-UNALTERED from the upstream CorrelationFunctionTests
// / SpatialCorrelationTests literals and recomputed inline from the identical closed-form
// kernels, so they already ARE the C# public-path values. Routing them through the
// emitter/verify_oracles gate (which reproduces FIXTURES) would require either a fixture --
// violating the binding "internal support gets C++-only ctests, public-API oracles live ONLY in
// fixtures/" constraint -- or new four-way harness wiring for a non-distribution support class.
// Tracked as a P5 follow-up.
//
// Skipped C# test methods (documented in task-S1-report.md):
//   - BasicExponential_SetParameterValues_Null_ThrowsException (CorrelationFunctionTests) and
//     BasicExponential_SetParameterValues_Null_Throws (SpatialCorrelationTests): the C#
//     ArgumentNullException null-guard is vacuous here -- set_parameter_values takes a
//     const std::vector<double>& (mirroring ModelBase::set_parameter_values), which cannot be
//     null. Only the length guard is meaningful and is transcribed.
//   - BasicExponential_ToXElement_ContainsTypeAttribute (SpatialCorrelationTests): XML/
//     serialization only -- ToXElement is a project-wide skip.
#include <cmath>
#include <memory>
#include <vector>

#include "bestfit/models/spatial_extremes/spatial_correlation/basic_exponential.hpp"
#include "bestfit/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "bestfit/models/spatial_extremes/spatial_correlation/i_correlation_model.hpp"
#include "bestfit/models/spatial_extremes/spatial_correlation/powered_exponential.hpp"
#include "bestfit/models/spatial_extremes/spatial_correlation/spherical.hpp"
#include "check.hpp"

using bestfit::models::spatial_extremes::BasicExponential;
using bestfit::models::spatial_extremes::CorrelationFunctionType;
using bestfit::models::spatial_extremes::ICorrelationModel;
using bestfit::models::spatial_extremes::PoweredExponential;
using bestfit::models::spatial_extremes::Spherical;

namespace {

// ------------------------------------------------------------------
// CorrelationFunctionTests.cs -- BasicExponential
// ------------------------------------------------------------------

void basic_exponential_constructor_initializes_correctly() {
    BasicExponential corr;
    CHECK_EQ(corr.number_of_parameters(), 1);
    CHECK_EQ(corr.parameters()[0].name(), std::string("Range"));
}

void basic_exponential_zero_distance_returns_one() {
    BasicExponential corr;
    corr.set_parameter_values({1.0});
    CHECK_NEAR(corr.evaluate(0), 1.0, 1e-10);
}

void basic_exponential_known_distance_returns_correct_value() {
    BasicExponential corr;
    double range = 2.0;
    corr.set_parameter_values({range});
    double h = 1.0;
    double expected = std::exp(-h / range);
    CHECK_NEAR(corr.evaluate(h), expected, 1e-10);
}

void basic_exponential_increasing_distance_decreasing_correlation() {
    BasicExponential corr;
    corr.set_parameter_values({5.0});
    double r0 = corr.evaluate(0);
    double r1 = corr.evaluate(1);
    double r5 = corr.evaluate(5);
    double r10 = corr.evaluate(10);
    CHECK_TRUE(r0 > r1);
    CHECK_TRUE(r1 > r5);
    CHECK_TRUE(r5 > r10);
}

void basic_exponential_all_distances_returns_values_between_zero_and_one() {
    BasicExponential corr;
    corr.set_parameter_values({3.0});
    for (double h : {0.0, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0}) {
        double result = corr.evaluate(h);
        CHECK_TRUE(result >= 0 && result <= 1);
    }
}

void basic_exponential_negative_distance_throws() {
    BasicExponential corr;
    corr.set_parameter_values({1.0});
    CHECK_THROWS(corr.evaluate(-1));
}

void basic_exponential_set_parameter_values_updates_range() {
    BasicExponential corr;
    corr.set_parameter_values({5.5});
    CHECK_NEAR(corr.parameters()[0].value(), 5.5, 1e-10);
}

void basic_exponential_set_parameter_values_wrong_count_throws() {
    BasicExponential corr;
    CHECK_THROWS(corr.set_parameter_values({1.0, 2.0}));
}

void basic_exponential_clone_creates_independent_copy() {
    BasicExponential original;
    original.set_parameter_values({3.0});
    auto clone = original.clone();
    original.set_parameter_values({10.0});
    CHECK_NEAR(clone->parameters()[0].value(), 3.0, 1e-10);
}

void basic_exponential_clone_preserves_bounds() {
    BasicExponential original;
    original.parameters()[0].set_lower_bound(0.5);
    original.parameters()[0].set_upper_bound(20.0);
    auto clone = original.clone();
    CHECK_NEAR(clone->parameters()[0].lower_bound(), 0.5, 1e-10);
    CHECK_NEAR(clone->parameters()[0].upper_bound(), 20.0, 1e-10);
}

void basic_exponential_practical_range_is_approximately_3_times_range() {
    BasicExponential corr;
    double range = 10.0;
    corr.set_parameter_values({range});
    double practical_range = 3 * range;
    double correlation_at_practical_range = corr.evaluate(practical_range);
    CHECK_NEAR(correlation_at_practical_range, std::exp(-3), 1e-10);
}

// ------------------------------------------------------------------
// CorrelationFunctionTests.cs -- PoweredExponential
// ------------------------------------------------------------------

void powered_exponential_constructor_initializes_correctly() {
    PoweredExponential corr;
    CHECK_EQ(corr.number_of_parameters(), 2);
    CHECK_EQ(corr.parameters()[0].name(), std::string("Range"));
    CHECK_EQ(corr.parameters()[1].name(), std::string("Smoothness"));
}

void powered_exponential_zero_distance_returns_one() {
    PoweredExponential corr;
    corr.set_parameter_values({1.0, 1.5});
    CHECK_NEAR(corr.evaluate(0), 1.0, 1e-10);
}

void powered_exponential_smoothness1_equals_basic_exponential() {
    double range = 5.0;
    PoweredExponential powered;
    powered.set_parameter_values({range, 1.0});
    BasicExponential basic;
    basic.set_parameter_values({range});
    for (double h : {0.5, 1.0, 2.0, 5.0, 10.0}) {
        CHECK_NEAR(powered.evaluate(h), basic.evaluate(h), 1e-10);
    }
}

void powered_exponential_smoothness2_is_gaussian_correlation() {
    PoweredExponential corr;
    double range = 3.0;
    corr.set_parameter_values({range, 2.0});
    double h = 2.0;
    double expected = std::exp(-std::pow(h / range, 2));
    CHECK_NEAR(corr.evaluate(h), expected, 1e-10);
}

void powered_exponential_increasing_distance_decreasing_correlation() {
    PoweredExponential corr;
    corr.set_parameter_values({5.0, 1.5});
    double r0 = corr.evaluate(0);
    double r1 = corr.evaluate(1);
    double r5 = corr.evaluate(5);
    CHECK_TRUE(r0 > r1);
    CHECK_TRUE(r1 > r5);
}

void powered_exponential_all_distances_returns_values_between_zero_and_one() {
    PoweredExponential corr;
    corr.set_parameter_values({3.0, 1.8});
    for (double h : {0.0, 0.1, 1.0, 5.0, 10.0, 50.0}) {
        double result = corr.evaluate(h);
        CHECK_TRUE(result >= 0 && result <= 1);
    }
}

void powered_exponential_negative_distance_throws() {
    PoweredExponential corr;
    corr.set_parameter_values({1.0, 1.5});
    CHECK_THROWS(corr.evaluate(-1));
}

void powered_exponential_set_parameter_values_updates_both_parameters() {
    PoweredExponential corr;
    corr.set_parameter_values({7.5, 1.8});
    CHECK_NEAR(corr.parameters()[0].value(), 7.5, 1e-10);
    CHECK_NEAR(corr.parameters()[1].value(), 1.8, 1e-10);
}

void powered_exponential_set_parameter_values_wrong_count_throws() {
    PoweredExponential corr;
    CHECK_THROWS(corr.set_parameter_values({1.0}));
}

void powered_exponential_clone_creates_independent_copy() {
    PoweredExponential original;
    original.set_parameter_values({3.0, 1.5});
    auto clone = original.clone();
    original.set_parameter_values({10.0, 2.0});
    CHECK_NEAR(clone->parameters()[0].value(), 3.0, 1e-10);
    CHECK_NEAR(clone->parameters()[1].value(), 1.5, 1e-10);
}

void powered_exponential_higher_smoothness_slower_near_origin_decay() {
    PoweredExponential low_smoothness;
    low_smoothness.set_parameter_values({5.0, 1.0});
    PoweredExponential high_smoothness;
    high_smoothness.set_parameter_values({5.0, 2.0});
    double h = 0.5;
    double r_low = low_smoothness.evaluate(h);
    double r_high = high_smoothness.evaluate(h);
    CHECK_TRUE(r_high > r_low);
}

// ------------------------------------------------------------------
// CorrelationFunctionTests.cs -- Spherical
// ------------------------------------------------------------------

void spherical_constructor_initializes_correctly() {
    Spherical corr;
    CHECK_EQ(corr.number_of_parameters(), 1);
    CHECK_EQ(corr.parameters()[0].name(), std::string("Range"));
}

void spherical_zero_distance_returns_one() {
    Spherical corr;
    corr.set_parameter_values({5.0});
    CHECK_NEAR(corr.evaluate(0), 1.0, 1e-10);
}

void spherical_at_range_returns_zero() {
    Spherical corr;
    double range = 5.0;
    corr.set_parameter_values({range});
    CHECK_NEAR(corr.evaluate(range), 0.0, 1e-10);
}

void spherical_beyond_range_returns_zero() {
    Spherical corr;
    double range = 5.0;
    corr.set_parameter_values({range});
    CHECK_NEAR(corr.evaluate(range + 0.1), 0.0, 1e-10);
    CHECK_NEAR(corr.evaluate(range * 2), 0.0, 1e-10);
    CHECK_NEAR(corr.evaluate(range * 10), 0.0, 1e-10);
}

void spherical_known_distance_returns_correct_value() {
    Spherical corr;
    double range = 10.0;
    corr.set_parameter_values({range});
    double h = 5.0;
    double ratio = h / range;
    double expected = 1.0 - 1.5 * ratio + 0.5 * std::pow(ratio, 3);
    CHECK_NEAR(corr.evaluate(h), expected, 1e-10);
}

void spherical_increasing_distance_within_range_decreasing_correlation() {
    Spherical corr;
    double range = 10.0;
    corr.set_parameter_values({range});
    double r0 = corr.evaluate(0);
    double r2 = corr.evaluate(2);
    double r5 = corr.evaluate(5);
    double r8 = corr.evaluate(8);
    CHECK_TRUE(r0 > r2);
    CHECK_TRUE(r2 > r5);
    CHECK_TRUE(r5 > r8);
}

void spherical_all_distances_returns_values_between_zero_and_one() {
    Spherical corr;
    double range = 10.0;
    corr.set_parameter_values({range});
    for (double h : {0.0, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0, 15.0, 100.0}) {
        double result = corr.evaluate(h);
        CHECK_TRUE(result >= 0 && result <= 1);
    }
}

void spherical_negative_distance_throws() {
    Spherical corr;
    corr.set_parameter_values({5.0});
    CHECK_THROWS(corr.evaluate(-1));
}

void spherical_set_parameter_values_updates_range() {
    Spherical corr;
    corr.set_parameter_values({8.5});
    CHECK_NEAR(corr.parameters()[0].value(), 8.5, 1e-10);
}

void spherical_clone_creates_independent_copy() {
    Spherical original;
    original.set_parameter_values({7.0});
    auto clone = original.clone();
    original.set_parameter_values({15.0});
    CHECK_NEAR(clone->parameters()[0].value(), 7.0, 1e-10);
}

void spherical_midpoint_returns_correct_value() {
    Spherical corr;
    double range = 10.0;
    corr.set_parameter_values({range});
    CHECK_NEAR(corr.evaluate(range / 2), 0.3125, 1e-10);
}

// ------------------------------------------------------------------
// CorrelationFunctionTests.cs -- ICorrelationModel interface
// ------------------------------------------------------------------

void all_correlation_functions_implement_icorrelation_model() {
    std::unique_ptr<ICorrelationModel> basic = std::make_unique<BasicExponential>();
    std::unique_ptr<ICorrelationModel> powered = std::make_unique<PoweredExponential>();
    std::unique_ptr<ICorrelationModel> spherical = std::make_unique<Spherical>();
    CHECK_TRUE(basic != nullptr);
    CHECK_TRUE(powered != nullptr);
    CHECK_TRUE(spherical != nullptr);
    CHECK_EQ(basic->type(), CorrelationFunctionType::Exponential);
    CHECK_EQ(powered->type(), CorrelationFunctionType::PoweredExponential);
    CHECK_EQ(spherical->type(), CorrelationFunctionType::Spherical);
}

void all_correlation_functions_have_required_members() {
    std::vector<std::unique_ptr<ICorrelationModel>> models;
    models.push_back(std::make_unique<BasicExponential>());
    models.push_back(std::make_unique<PoweredExponential>());
    models.push_back(std::make_unique<Spherical>());
    for (auto& model : models) {
        CHECK_TRUE(model->parameters().size() > 0);
        std::vector<double> values;
        for (const auto& p : model->parameters()) values.push_back(p.value());
        model->set_parameter_values(values);
        double result = model->evaluate(1.0);
        CHECK_TRUE(!std::isnan(result));
        auto clone = model->clone();
        CHECK_TRUE(clone != nullptr);
    }
}

// ------------------------------------------------------------------
// CorrelationFunctionTests.cs -- Comparison
// ------------------------------------------------------------------

void all_correlation_functions_zero_distance_return_one() {
    std::vector<std::unique_ptr<ICorrelationModel>> models;
    models.push_back(std::make_unique<BasicExponential>());
    models.push_back(std::make_unique<PoweredExponential>());
    models.push_back(std::make_unique<Spherical>());
    for (auto& model : models) {
        std::vector<double> values;
        for (const auto& p : model->parameters()) values.push_back(p.value());
        model->set_parameter_values(values);
    }
    for (auto& model : models) {
        CHECK_NEAR(model->evaluate(0), 1.0, 1e-10);
    }
}

void all_correlation_functions_are_monotonically_decreasing() {
    BasicExponential basic;
    basic.set_parameter_values({5.0});
    PoweredExponential powered;
    powered.set_parameter_values({5.0, 1.5});
    Spherical spherical;
    spherical.set_parameter_values({10.0});
    ICorrelationModel* models[] = {&basic, &powered, &spherical};
    for (auto* model : models) {
        double prev = std::numeric_limits<double>::max();
        for (double h : {0.0, 1.0, 2.0, 3.0, 4.0}) {
            double current = model->evaluate(h);
            CHECK_TRUE(current <= prev);
            prev = current;
        }
    }
}

void compare_support_spherical_compact_others_asymptotic() {
    double range = 5.0;
    BasicExponential basic;
    basic.set_parameter_values({range});
    PoweredExponential powered;
    powered.set_parameter_values({range, 1.5});
    Spherical spherical;
    spherical.set_parameter_values({range});
    double far_distance = range * 2;
    CHECK_TRUE(basic.evaluate(far_distance) > 0);
    CHECK_TRUE(powered.evaluate(far_distance) > 0);
    CHECK_NEAR(spherical.evaluate(far_distance), 0.0, 1e-10);
}

// ------------------------------------------------------------------
// CorrelationFunctionTests.cs -- Edge cases
// ------------------------------------------------------------------

void all_correlation_functions_very_small_distance_returns_near_one() {
    std::vector<std::unique_ptr<ICorrelationModel>> models;
    models.push_back(std::make_unique<BasicExponential>());
    models.push_back(std::make_unique<PoweredExponential>());
    models.push_back(std::make_unique<Spherical>());
    double tiny_distance = 1e-10;
    for (auto& model : models) {
        std::vector<double> values;
        for (const auto& p : model->parameters()) values.push_back(p.value());
        model->set_parameter_values(values);
        double result = model->evaluate(tiny_distance);
        CHECK_TRUE(result > 0.99);
    }
}

void all_correlation_functions_very_large_distance_returns_near_zero() {
    BasicExponential basic;
    basic.set_parameter_values({1.0});
    PoweredExponential powered;
    powered.set_parameter_values({1.0, 1.5});
    Spherical spherical;
    spherical.set_parameter_values({1.0});
    double large_distance = 1000.0;
    CHECK_TRUE(basic.evaluate(large_distance) < 1e-10);
    CHECK_TRUE(powered.evaluate(large_distance) < 1e-10);
    CHECK_NEAR(spherical.evaluate(large_distance), 0.0, 1e-10);
}

void all_correlation_functions_very_small_range_fast_decay() {
    double tiny_range = 0.001;
    BasicExponential basic;
    basic.set_parameter_values({tiny_range});
    PoweredExponential powered;
    powered.set_parameter_values({tiny_range, 1.5});
    Spherical spherical;
    spherical.set_parameter_values({tiny_range});
    double distance = 1.0;
    CHECK_TRUE(basic.evaluate(distance) < 0.01);
    CHECK_TRUE(powered.evaluate(distance) < 0.01);
    CHECK_NEAR(spherical.evaluate(distance), 0.0, 1e-10);
}

void all_correlation_functions_very_large_range_slow_decay() {
    double large_range = 1000.0;
    BasicExponential basic;
    basic.set_parameter_values({large_range});
    PoweredExponential powered;
    powered.set_parameter_values({large_range, 1.5});
    Spherical spherical;
    spherical.set_parameter_values({large_range});
    double distance = 1.0;
    CHECK_TRUE(basic.evaluate(distance) > 0.99);
    CHECK_TRUE(powered.evaluate(distance) > 0.99);
    CHECK_TRUE(spherical.evaluate(distance) > 0.99);
}

void basic_exponential_zero_range_returns_zero_correlation() {
    BasicExponential corr;
    corr.parameters()[0].set_value(0.0);
    CHECK_NEAR(corr.evaluate(1.0), 0.0, 1e-10);
}

// ------------------------------------------------------------------
// SpatialCorrelationTests.cs -- BasicExponential
// ------------------------------------------------------------------

void sc_basic_exponential_defaults_one_parameter_correct_type() {
    BasicExponential model;
    CHECK_EQ(model.number_of_parameters(), 1);
    CHECK_EQ(model.parameters()[0].name(), std::string("Range"));
    CHECK_EQ(model.type(), CorrelationFunctionType::Exponential);
}

void sc_basic_exponential_at_zero_distance_returns_one() {
    BasicExponential model;
    CHECK_EQ(model.evaluate(0.0), 1.0);
}

void sc_basic_exponential_at_range_distance_returns_exp_minus_one() {
    BasicExponential model;
    model.set_parameter_values({10.0});
    CHECK_NEAR(model.evaluate(10.0), std::exp(-1.0), 1e-12);
}

void sc_basic_exponential_large_distance_decays_toward_zero() {
    BasicExponential model;
    model.set_parameter_values({1.0});
    CHECK_TRUE(model.evaluate(100.0) < 1e-30);
}

void sc_basic_exponential_evaluate_negative_distance_throws() {
    BasicExponential model;
    CHECK_THROWS(model.evaluate(-1.0));
}

void sc_basic_exponential_evaluate_non_positive_range_returns_zero() {
    BasicExponential model;
    model.set_parameter_values({0.0});
    CHECK_EQ(model.evaluate(1.0), 0.0);
}

void sc_basic_exponential_set_parameter_values_wrong_count_throws() {
    BasicExponential model;
    CHECK_THROWS(model.set_parameter_values({1.0, 2.0}));
}

void sc_basic_exponential_clone_independent_values() {
    BasicExponential original;
    original.set_parameter_values({25.0});
    auto clone = original.clone();
    original.set_parameter_values({1.0});
    CHECK_EQ(clone->parameters()[0].value(), 25.0);
}

// ------------------------------------------------------------------
// SpatialCorrelationTests.cs -- PoweredExponential
// ------------------------------------------------------------------

void sc_powered_exponential_defaults_two_parameters_correct_type() {
    PoweredExponential model;
    CHECK_EQ(model.number_of_parameters(), 2);
    CHECK_EQ(model.parameters()[0].name(), std::string("Range"));
    CHECK_EQ(model.parameters()[1].name(), std::string("Smoothness"));
    CHECK_EQ(model.type(), CorrelationFunctionType::PoweredExponential);
}

void sc_powered_exponential_at_zero_distance_returns_one() {
    PoweredExponential model;
    CHECK_EQ(model.evaluate(0.0), 1.0);
}

void sc_powered_exponential_with_smoothness1_reduces_to_exponential() {
    PoweredExponential model;
    model.set_parameter_values({10.0, 1.0});
    BasicExponential basic_exp;
    basic_exp.set_parameter_values({10.0});
    CHECK_NEAR(model.evaluate(5.0), basic_exp.evaluate(5.0), 1e-12);
    CHECK_NEAR(model.evaluate(15.0), basic_exp.evaluate(15.0), 1e-12);
}

void sc_powered_exponential_with_smoothness2_is_gaussian() {
    PoweredExponential model;
    model.set_parameter_values({10.0, 2.0});
    CHECK_NEAR(model.evaluate(10.0), std::exp(-1.0), 1e-12);
}

void sc_powered_exponential_non_positive_range_returns_zero() {
    PoweredExponential model;
    model.set_parameter_values({0.0, 1.5});
    CHECK_EQ(model.evaluate(1.0), 0.0);
}

void sc_powered_exponential_negative_distance_throws() {
    PoweredExponential model;
    CHECK_THROWS(model.evaluate(-1.0));
}

void sc_powered_exponential_clone_independent_values() {
    PoweredExponential original;
    original.set_parameter_values({25.0, 1.8});
    auto clone = original.clone();
    original.set_parameter_values({1.0, 0.5});
    CHECK_EQ(clone->parameters()[0].value(), 25.0);
    CHECK_EQ(clone->parameters()[1].value(), 1.8);
}

// ------------------------------------------------------------------
// SpatialCorrelationTests.cs -- Spherical
// ------------------------------------------------------------------

void sc_spherical_defaults_one_parameter_correct_type() {
    Spherical model;
    CHECK_EQ(model.number_of_parameters(), 1);
    CHECK_EQ(model.parameters()[0].name(), std::string("Range"));
    CHECK_EQ(model.type(), CorrelationFunctionType::Spherical);
}

void sc_spherical_at_zero_distance_returns_one() {
    Spherical model;
    CHECK_EQ(model.evaluate(0.0), 1.0);
}

void sc_spherical_at_or_beyond_range_returns_zero() {
    Spherical model;
    model.set_parameter_values({10.0});
    CHECK_EQ(model.evaluate(10.0), 0.0);
    CHECK_EQ(model.evaluate(15.0), 0.0);
}

void sc_spherical_half_range_matches_closed_form() {
    Spherical model;
    model.set_parameter_values({10.0});
    CHECK_NEAR(model.evaluate(5.0), 0.3125, 1e-12);
}

void sc_spherical_non_positive_range_returns_zero() {
    Spherical model;
    model.set_parameter_values({0.0});
    CHECK_EQ(model.evaluate(1.0), 0.0);
}

void sc_spherical_monotonic_decay_on_sweep() {
    Spherical model;
    model.set_parameter_values({10.0});
    double prev = std::numeric_limits<double>::infinity();
    for (double h = 0.1; h < 10.0; h += 0.1) {
        double rho = model.evaluate(h);
        CHECK_TRUE(rho < prev);
        prev = rho;
    }
}

void sc_spherical_negative_distance_throws() {
    Spherical model;
    CHECK_THROWS(model.evaluate(-1.0));
}

void sc_spherical_clone_independent_values() {
    Spherical original;
    original.set_parameter_values({25.0});
    auto clone = original.clone();
    original.set_parameter_values({1.0});
    CHECK_EQ(clone->parameters()[0].value(), 25.0);
}

// ------------------------------------------------------------------
// SpatialCorrelationTests.cs -- Cross-kernel invariants
// ------------------------------------------------------------------

void all_kernels_output_is_in_unit_interval_over_sweep() {
    std::vector<std::unique_ptr<ICorrelationModel>> kernels;
    kernels.push_back(std::make_unique<BasicExponential>());
    kernels.push_back(std::make_unique<PoweredExponential>());
    kernels.push_back(std::make_unique<Spherical>());
    for (auto& k : kernels) {
        for (double h = 0.0; h <= 50.0; h += 0.5) {
            double rho = k->evaluate(h);
            CHECK_TRUE(rho >= 0.0 && rho <= 1.0);
        }
    }
}

}  // namespace

int main() {
    // CorrelationFunctionTests.cs -- BasicExponential
    basic_exponential_constructor_initializes_correctly();
    basic_exponential_zero_distance_returns_one();
    basic_exponential_known_distance_returns_correct_value();
    basic_exponential_increasing_distance_decreasing_correlation();
    basic_exponential_all_distances_returns_values_between_zero_and_one();
    basic_exponential_negative_distance_throws();
    basic_exponential_set_parameter_values_updates_range();
    basic_exponential_set_parameter_values_wrong_count_throws();
    basic_exponential_clone_creates_independent_copy();
    basic_exponential_clone_preserves_bounds();
    basic_exponential_practical_range_is_approximately_3_times_range();
    // CorrelationFunctionTests.cs -- PoweredExponential
    powered_exponential_constructor_initializes_correctly();
    powered_exponential_zero_distance_returns_one();
    powered_exponential_smoothness1_equals_basic_exponential();
    powered_exponential_smoothness2_is_gaussian_correlation();
    powered_exponential_increasing_distance_decreasing_correlation();
    powered_exponential_all_distances_returns_values_between_zero_and_one();
    powered_exponential_negative_distance_throws();
    powered_exponential_set_parameter_values_updates_both_parameters();
    powered_exponential_set_parameter_values_wrong_count_throws();
    powered_exponential_clone_creates_independent_copy();
    powered_exponential_higher_smoothness_slower_near_origin_decay();
    // CorrelationFunctionTests.cs -- Spherical
    spherical_constructor_initializes_correctly();
    spherical_zero_distance_returns_one();
    spherical_at_range_returns_zero();
    spherical_beyond_range_returns_zero();
    spherical_known_distance_returns_correct_value();
    spherical_increasing_distance_within_range_decreasing_correlation();
    spherical_all_distances_returns_values_between_zero_and_one();
    spherical_negative_distance_throws();
    spherical_set_parameter_values_updates_range();
    spherical_clone_creates_independent_copy();
    spherical_midpoint_returns_correct_value();
    // CorrelationFunctionTests.cs -- interface
    all_correlation_functions_implement_icorrelation_model();
    all_correlation_functions_have_required_members();
    // CorrelationFunctionTests.cs -- comparison
    all_correlation_functions_zero_distance_return_one();
    all_correlation_functions_are_monotonically_decreasing();
    compare_support_spherical_compact_others_asymptotic();
    // CorrelationFunctionTests.cs -- edge cases
    all_correlation_functions_very_small_distance_returns_near_one();
    all_correlation_functions_very_large_distance_returns_near_zero();
    all_correlation_functions_very_small_range_fast_decay();
    all_correlation_functions_very_large_range_slow_decay();
    basic_exponential_zero_range_returns_zero_correlation();
    // SpatialCorrelationTests.cs -- BasicExponential
    sc_basic_exponential_defaults_one_parameter_correct_type();
    sc_basic_exponential_at_zero_distance_returns_one();
    sc_basic_exponential_at_range_distance_returns_exp_minus_one();
    sc_basic_exponential_large_distance_decays_toward_zero();
    sc_basic_exponential_evaluate_negative_distance_throws();
    sc_basic_exponential_evaluate_non_positive_range_returns_zero();
    sc_basic_exponential_set_parameter_values_wrong_count_throws();
    sc_basic_exponential_clone_independent_values();
    // SpatialCorrelationTests.cs -- PoweredExponential
    sc_powered_exponential_defaults_two_parameters_correct_type();
    sc_powered_exponential_at_zero_distance_returns_one();
    sc_powered_exponential_with_smoothness1_reduces_to_exponential();
    sc_powered_exponential_with_smoothness2_is_gaussian();
    sc_powered_exponential_non_positive_range_returns_zero();
    sc_powered_exponential_negative_distance_throws();
    sc_powered_exponential_clone_independent_values();
    // SpatialCorrelationTests.cs -- Spherical
    sc_spherical_defaults_one_parameter_correct_type();
    sc_spherical_at_zero_distance_returns_one();
    sc_spherical_at_or_beyond_range_returns_zero();
    sc_spherical_half_range_matches_closed_form();
    sc_spherical_non_positive_range_returns_zero();
    sc_spherical_monotonic_decay_on_sweep();
    sc_spherical_negative_distance_throws();
    sc_spherical_clone_independent_values();
    // SpatialCorrelationTests.cs -- cross-kernel invariants
    all_kernels_output_is_in_unit_interval_over_sweep();

    return bftest::summary("spatial_correlation");
}
