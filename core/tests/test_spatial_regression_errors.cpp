// S3 support ctest (C++-only): SpatialRegressionErrors -- the Gaussian-Process spatial-error
// model ε ~ MVN(0, Σ), Σ_ij = σ²·ρ(h_ij) (Renard BHM framework). This is internal support (not a
// public-API distribution), so hardcoded oracles transcribed from the upstream C# test file are
// correct here (public-API oracle values stay in fixtures/ only; SpatialGEV's public-API fit
// oracles arrive via the S4/P4 emitter path).
//
// Structural oracles transcribed VALUES-UNALTERED from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/SpatialExtremes/SpatialRegressionErrorsTests.cs @ fc28c0c
// (38 [TestMethod]s, zero XML/serialization methods). The two C# null-guard tests
//   - Constructor_NullCoordinates_ThrowsArgumentNullException
//   - SetParameterValues_Null_ThrowsArgumentNullException
// are VACUOUS in this port: the ported signatures take std::vector<...> by const-ref, which cannot
// be null (same treatment as S1/S2). They are documented as skipped in task-S3-report.md; the
// remaining 36 methods are transcribed below with matching tolerances (1e-10 on the transcribed
// analytic leaf checks).
//
// Instance-type C# assertions (IsInstanceOfType(CorrelationFunction, typeof(BasicExponential)))
// are mirrored via the ported ICorrelationModel::type() enum, the C++-idiomatic equivalent.
//
// The C# test file exercises GetKrigingPrediction / GetIDWPrediction only INDIRECTLY (no dedicated
// oracle method). A small block of ADDED analytic kriging/IDW checks -- verifiable by hand -- is
// clearly separated at the end (simple-kriging interpolation property at a data site, variance
// bounds 0 <= var <= σ², IDW weighted-average value and (Mean, Variance) field order). See
// task-S3-report.md for the kriging-oracle situation and why the Cholesky-null -> IDW fallback
// branch is not reachable through the public API with a valid PSD correlation matrix.
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "corehydro/models/spatial_extremes/copula_models/spatial_regression_errors.hpp"
#include "corehydro/models/spatial_extremes/spatial_correlation/correlation_function_type.hpp"
#include "check.hpp"

namespace {

using corehydro::models::spatial_extremes::CorrelationFunctionType;
using corehydro::models::spatial_extremes::SpatialRegressionErrors;

using Coords = std::vector<std::vector<double>>;

// UTF-8 byte sequences for the parameter names the C# test asserts on:
//   σ  = U+03C3 = CF 83 ;  ε = U+03B5 = CE B5 ;  ₁₂₃ = U+2081/2/3 = E2 82 81 / 82 / 83
const std::string kSigmaName = "Error Scale (\xCF\x83)";
const std::string kEps1 = "\xCE\xB5\xE2\x82\x81";
const std::string kEps2 = "\xCE\xB5\xE2\x82\x82";
const std::string kEps3 = "\xCE\xB5\xE2\x82\x83";
const std::string kEpsPrefix = "\xCE\xB5";

// ---- Test data helpers (mirror the C# private helpers) ----

Coords create_three_site_coordinates() { return Coords{{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}}; }

Coords create_five_site_coordinates() {
    return Coords{{0.0, 0.0}, {10.0, 0.0}, {20.0, 0.0}, {30.0, 0.0}, {40.0, 0.0}};
}

Coords create_minimal_coordinates() { return Coords{{0.0, 0.0}, {5.0, 0.0}}; }

// =====================================================================================
// Constructor Tests
// =====================================================================================

void constructor_valid_inputs_initializes_correctly() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_EQ(errors.sites(), 3);
    CHECK_TRUE(errors.number_of_parameters() > 0);
    CHECK_TRUE(errors.error_parameters().size() == 3);
}

void constructor_exponential_correlation_creates_correct_function() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_EQ(errors.correlation_function().type(), CorrelationFunctionType::Exponential);
}

void constructor_powered_exponential_correlation_creates_correct_function() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::PoweredExponential);
    CHECK_EQ(errors.correlation_function().type(), CorrelationFunctionType::PoweredExponential);
}

void constructor_spherical_correlation_creates_correct_function() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Spherical);
    CHECK_EQ(errors.correlation_function().type(), CorrelationFunctionType::Spherical);
}

void constructor_invalid_coordinate_dimensions_throws() {
    // 1D coordinates instead of 2D.
    Coords coords{{0.0}, {1.0}};
    CHECK_THROWS(SpatialRegressionErrors(coords, CorrelationFunctionType::Exponential));
}

void constructor_3d_coordinates_throws() {
    Coords coords{{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    CHECK_THROWS(SpatialRegressionErrors(coords, CorrelationFunctionType::Exponential));
}

void constructor_custom_max_error_uses_custom_bound() {
    auto coords = create_three_site_coordinates();
    double max_error = 25.0;
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential, max_error);
    CHECK_NEAR(errors.parameters()[0].upper_bound(), max_error, 1e-10);
}

// =====================================================================================
// Parameter Tests
// =====================================================================================

void parameters_structure_is_correct() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    // Exponential: 1 sigma + 1 range + 3 site errors = 5 total.
    CHECK_EQ(errors.number_of_parameters(), 5);
    CHECK_EQ(errors.parameters()[0].name(), kSigmaName);
    CHECK_EQ(errors.parameters()[1].name(), std::string("Range"));
    CHECK_EQ(errors.parameters()[2].name(), kEps1);
    CHECK_EQ(errors.parameters()[3].name(), kEps2);
    CHECK_EQ(errors.parameters()[4].name(), kEps3);
}

void parameters_powered_exponential_has_extra_parameter() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::PoweredExponential);
    // PoweredExponential: 1 sigma + 2 corr params + 3 site errors = 6 total.
    CHECK_EQ(errors.number_of_parameters(), 6);
}

void error_parameters_contains_only_errors() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    auto eps = errors.error_parameters();
    CHECK_EQ(static_cast<int>(eps.size()), 3);
    for (const auto* p : eps) {
        CHECK_TRUE(p->name().compare(0, kEpsPrefix.size(), kEpsPrefix) == 0);
    }
}

void default_parameters_errors_are_zero() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    for (const auto* p : errors.error_parameters()) {
        CHECK_NEAR(p->value(), 0.0, 1e-10);
    }
}

// =====================================================================================
// SetParameterValues Tests
// =====================================================================================

void set_parameter_values_valid_values_updates_all_parameters() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({2.0, 5.0, 0.5, -0.3, 0.1});
    CHECK_NEAR(errors.parameters()[0].value(), 2.0, 1e-10);
    CHECK_NEAR(errors.parameters()[1].value(), 5.0, 1e-10);
    CHECK_NEAR(errors.error_parameters()[0]->value(), 0.5, 1e-10);
    CHECK_NEAR(errors.error_parameters()[1]->value(), -0.3, 1e-10);
    CHECK_NEAR(errors.error_parameters()[2]->value(), 0.1, 1e-10);
}

void set_parameter_values_wrong_count_throws() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_THROWS(errors.set_parameter_values({1.0, 2.0}));  // Too few.
}

void set_parameter_values_powered_exponential_updates_all_parameters() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::PoweredExponential);
    errors.set_parameter_values({2.0, 5.0, 1.5, 0.5, -0.3, 0.1});
    CHECK_NEAR(errors.parameters()[0].value(), 2.0, 1e-10);
    CHECK_NEAR(errors.parameters()[1].value(), 5.0, 1e-10);
    CHECK_NEAR(errors.parameters()[2].value(), 1.5, 1e-10);
    CHECK_NEAR(errors.error_parameters()[0]->value(), 0.5, 1e-10);
}

// =====================================================================================
// GetError Tests
// =====================================================================================

void get_error_returns_correct_value() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({2.0, 5.0, 0.5, -0.3, 0.1});
    CHECK_NEAR(errors.get_error(0), 0.5, 1e-10);
    CHECK_NEAR(errors.get_error(1), -0.3, 1e-10);
    CHECK_NEAR(errors.get_error(2), 0.1, 1e-10);
}

void get_error_negative_index_throws() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_THROWS(errors.get_error(-1));
}

void get_error_index_beyond_range_throws() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_THROWS(errors.get_error(3));  // Only 0, 1, 2 valid.
}

// =====================================================================================
// PDF / LogPDF Tests
// =====================================================================================

void log_pdf_valid_parameters_returns_finite_value() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.0, 0.0, 0.0});
    double log_pdf = errors.log_pdf();
    CHECK_TRUE(!std::isnan(log_pdf));
    CHECK_TRUE(std::isfinite(log_pdf));
}

void log_pdf_zero_errors_has_higher_density() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.0, 0.0, 0.0});
    double log_pdf_zero = errors.log_pdf();
    errors.set_parameter_values({1.0, 2.0, 0.5, -0.3, 0.8});
    double log_pdf_nonzero = errors.log_pdf();
    CHECK_TRUE(log_pdf_zero > log_pdf_nonzero);
}

void pdf_valid_parameters_returns_positive_value() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.1, -0.1, 0.05});
    double pdf = errors.pdf();
    CHECK_TRUE(pdf > 0);
    CHECK_TRUE(!std::isnan(pdf));
}

void pdf_consistent_with_log_pdf() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.1, -0.1, 0.05});
    double pdf = errors.pdf();
    double log_pdf = errors.log_pdf();
    CHECK_NEAR(std::log(pdf), log_pdf, 1e-10);
}

void log_pdf_different_correlations_all_return_finite_values() {
    auto coords = create_three_site_coordinates();
    CorrelationFunctionType types[] = {CorrelationFunctionType::Exponential,
                                       CorrelationFunctionType::PoweredExponential,
                                       CorrelationFunctionType::Spherical};
    for (auto type : types) {
        SpatialRegressionErrors errors(coords, type);
        std::vector<double> values;
        for (const auto& p : errors.parameters()) values.push_back(p.value());
        errors.set_parameter_values(values);
        double log_pdf = errors.log_pdf();
        CHECK_TRUE(!std::isnan(log_pdf));
        CHECK_TRUE(std::isfinite(log_pdf));
    }
}

void log_pdf_larger_sigma_lower_density_at_mean() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors small_sigma(coords, CorrelationFunctionType::Exponential);
    small_sigma.set_parameter_values({0.5, 2.0, 0.0, 0.0, 0.0});
    SpatialRegressionErrors large_sigma(coords, CorrelationFunctionType::Exponential);
    large_sigma.set_parameter_values({5.0, 2.0, 0.0, 0.0, 0.0});
    CHECK_TRUE(small_sigma.log_pdf() > large_sigma.log_pdf());
}

// =====================================================================================
// Clone Tests
// =====================================================================================

void clone_creates_independent_copy() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors original(coords, CorrelationFunctionType::Exponential);
    original.set_parameter_values({2.0, 5.0, 0.5, -0.3, 0.1});
    SpatialRegressionErrors clone = original.clone();
    // Modify original.
    original.set_parameter_values({10.0, 10.0, 1.0, 1.0, 1.0});
    CHECK_NEAR(clone.parameters()[0].value(), 2.0, 1e-10);
    CHECK_NEAR(clone.get_error(0), 0.5, 1e-10);
}

void clone_preserves_site_count() {
    auto coords = create_five_site_coordinates();
    SpatialRegressionErrors original(coords, CorrelationFunctionType::Spherical);
    SpatialRegressionErrors clone = original.clone();
    CHECK_EQ(original.sites(), clone.sites());
    CHECK_EQ(original.number_of_parameters(), clone.number_of_parameters());
}

void clone_preserves_bounds() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors original(coords, CorrelationFunctionType::Exponential, 50.0);
    SpatialRegressionErrors clone = original.clone();
    CHECK_NEAR(original.parameters()[0].upper_bound(), clone.parameters()[0].upper_bound(), 1e-10);
    CHECK_NEAR(original.parameters()[0].lower_bound(), clone.parameters()[0].lower_bound(), 1e-10);
}

void clone_creates_valid_mvn() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors original(coords, CorrelationFunctionType::Exponential);
    original.set_parameter_values({1.5, 3.0, 0.2, -0.1, 0.3});
    SpatialRegressionErrors clone = original.clone();
    CHECK_NEAR(original.log_pdf(), clone.log_pdf(), 1e-10);
}

// =====================================================================================
// SetDefaultParameters Tests
// =====================================================================================

void set_default_parameters_creates_valid_state() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_default_parameters(10.0);
    CHECK_EQ(errors.number_of_parameters(), 5);
    CHECK_TRUE(errors.parameters()[0].value() > 0);
    for (const auto* p : errors.error_parameters()) {
        CHECK_NEAR(p->value(), 0.0, 1e-10);
    }
}

void set_default_parameters_respects_max_error_bound() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_default_parameters(20.0);
    CHECK_NEAR(errors.parameters()[0].upper_bound(), 20.0, 1e-10);
    for (const auto* p : errors.error_parameters()) {
        CHECK_NEAR(p->lower_bound(), -20.0, 1e-10);
        CHECK_NEAR(p->upper_bound(), 20.0, 1e-10);
    }
}

// =====================================================================================
// Distance Matrix Tests
// =====================================================================================

void distance_matrix_computed_correctly() {
    // Triangle: sites at (0,0), (3,0), (0,4).
    Coords coords{{0.0, 0.0}, {3.0, 0.0}, {0.0, 4.0}};
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 10.0, 0.5, 0.5, 0.5});
    double log_pdf1 = errors.log_pdf();

    Coords far_coords{{0.0, 0.0}, {100.0, 0.0}, {0.0, 100.0}};
    SpatialRegressionErrors far_errors(far_coords, CorrelationFunctionType::Exponential);
    far_errors.set_parameter_values({1.0, 10.0, 0.5, 0.5, 0.5});
    double log_pdf2 = far_errors.log_pdf();

    CHECK_TRUE(log_pdf1 != log_pdf2);
}

// =====================================================================================
// Edge Cases
// =====================================================================================

void minimal_sites_works_correctly() {
    auto coords = create_minimal_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_EQ(errors.sites(), 2);
    CHECK_EQ(errors.number_of_parameters(), 4);  // σ + range + 2 errors.
    errors.set_parameter_values({1.0, 5.0, 0.1, -0.1});
    CHECK_TRUE(std::isfinite(errors.log_pdf()));
}

void colocated_sites_handled_correctly() {
    Coords coords{{0.0, 0.0}, {0.0, 0.0}};  // Same location.
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 5.0, 0.0, 0.0});
    CHECK_TRUE(!std::isnan(errors.log_pdf()));
}

void many_sites_works_correctly() {
    Coords coords(10, std::vector<double>(2, 0.0));
    for (int i = 0; i < 10; ++i) coords[i][0] = i * 5.0;
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    CHECK_EQ(errors.sites(), 10);
    CHECK_EQ(errors.number_of_parameters(), 12);  // σ + range + 10 errors.
    std::vector<double> values;
    for (const auto& p : errors.parameters()) values.push_back(p.value());
    errors.set_parameter_values(values);
    CHECK_TRUE(std::isfinite(errors.log_pdf()));
}

void very_small_sigma_handled_correctly() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({0.001, 2.0, 0.0, 0.0, 0.0});
    CHECK_TRUE(std::isfinite(errors.log_pdf()));
}

void large_errors_produce_low_density() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors small_errors(coords, CorrelationFunctionType::Exponential);
    small_errors.set_parameter_values({1.0, 2.0, 0.1, 0.1, 0.1});
    SpatialRegressionErrors large_errors(coords, CorrelationFunctionType::Exponential);
    large_errors.set_parameter_values({1.0, 2.0, 5.0, 5.0, 5.0});
    CHECK_TRUE(small_errors.log_pdf() > large_errors.log_pdf());
}

// =====================================================================================
// Spatial Correlation Tests
// =====================================================================================

void closer_sites_higher_correlation() {
    Coords close_coords{{0.0, 0.0}, {0.1, 0.0}, {0.2, 0.0}};
    Coords far_coords{{0.0, 0.0}, {100.0, 0.0}, {200.0, 0.0}};
    SpatialRegressionErrors close_errors(close_coords, CorrelationFunctionType::Exponential);
    SpatialRegressionErrors far_errors(far_coords, CorrelationFunctionType::Exponential);
    std::vector<double> values{1.0, 1.0, 0.5, 0.5, 0.5};
    close_errors.set_parameter_values(values);
    far_errors.set_parameter_values(values);
    CHECK_TRUE(close_errors.log_pdf() > far_errors.log_pdf());
}

// =====================================================================================
// ADDED analytic kriging / IDW checks (no C# oracle method -- see file header).
// =====================================================================================

// Simple-kriging interpolation property: at an observed site j, ε*|ε recovers ε_j with zero
// prediction variance (k* equals column j of K, so K⁻¹k* = e_j; predMean = ε_j, predVar = 0).
void kriging_at_observed_site_recovers_error() {
    auto coords = create_three_site_coordinates();  // (0,0),(1,0),(0,1)
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.5, -0.3, 0.1});
    auto pred = errors.get_kriging_prediction({1.0, 0.0});  // exactly site 1
    CHECK_NEAR(pred.first, -0.3, 1e-8);   // Mean recovers ε_1.
    CHECK_NEAR(pred.second, 0.0, 1e-8);   // Variance collapses to 0.
}

// Predicted variance stays within [0, σ²] at an off-site location (quadForm >= 0 for PD K, and
// the C# clamps the lower end to 0).
void kriging_variance_within_bounds() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.5, -0.3, 0.1});
    auto pred = errors.get_kriging_prediction({0.5, 0.5});
    double sigma2 = 1.0 * 1.0;
    CHECK_TRUE(pred.second >= 0.0);
    CHECK_TRUE(pred.second <= sigma2 + 1e-12);
}

// IDW weighted-average value and (Mean, Variance) field order. Two sites at (0,0),(2,0) with
// errors 1.0, 3.0 and σ=1; new point (1,0) is equidistant so w0 = w1 = 1:
//   Mean = (1*1 + 1*3) / 2 = 2.0 ;  Variance = (1²·σ² + 1²·σ²) / 2² = 2 / 4 = 0.5.
void idw_weighted_average_and_field_order() {
    Coords coords{{0.0, 0.0}, {2.0, 0.0}};
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 5.0, 1.0, 3.0});
    auto pred = errors.get_idw_prediction({1.0, 0.0});
    CHECK_NEAR(pred.first, 2.0, 1e-10);    // Mean is the first field.
    CHECK_NEAR(pred.second, 0.5, 1e-10);   // Variance is the second field.
}

// IDW recovers the observed error when the new point is (numerically) colocated with a site: the
// 1e-10 distance floor makes that site's weight dominate.
void idw_colocated_recovers_error() {
    Coords coords{{0.0, 0.0}, {10.0, 0.0}};
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 5.0, 0.5, -0.3});
    auto pred = errors.get_idw_prediction({0.0, 0.0});  // exactly site 0
    CHECK_NEAR(pred.first, 0.5, 1e-6);
    CHECK_TRUE(pred.second >= 0.0);
}

void kriging_and_idw_bad_coordinate_dimension_throws() {
    auto coords = create_three_site_coordinates();
    SpatialRegressionErrors errors(coords, CorrelationFunctionType::Exponential);
    errors.set_parameter_values({1.0, 2.0, 0.5, -0.3, 0.1});
    CHECK_THROWS(errors.get_kriging_prediction({1.0, 0.0, 0.0}));
    CHECK_THROWS(errors.get_idw_prediction({1.0}));
}

}  // namespace

int main() {
    // Constructor
    constructor_valid_inputs_initializes_correctly();
    constructor_exponential_correlation_creates_correct_function();
    constructor_powered_exponential_correlation_creates_correct_function();
    constructor_spherical_correlation_creates_correct_function();
    constructor_invalid_coordinate_dimensions_throws();
    constructor_3d_coordinates_throws();
    constructor_custom_max_error_uses_custom_bound();
    // Parameter structure
    parameters_structure_is_correct();
    parameters_powered_exponential_has_extra_parameter();
    error_parameters_contains_only_errors();
    default_parameters_errors_are_zero();
    // SetParameterValues
    set_parameter_values_valid_values_updates_all_parameters();
    set_parameter_values_wrong_count_throws();
    set_parameter_values_powered_exponential_updates_all_parameters();
    // GetError
    get_error_returns_correct_value();
    get_error_negative_index_throws();
    get_error_index_beyond_range_throws();
    // PDF / LogPDF
    log_pdf_valid_parameters_returns_finite_value();
    log_pdf_zero_errors_has_higher_density();
    pdf_valid_parameters_returns_positive_value();
    pdf_consistent_with_log_pdf();
    log_pdf_different_correlations_all_return_finite_values();
    log_pdf_larger_sigma_lower_density_at_mean();
    // Clone
    clone_creates_independent_copy();
    clone_preserves_site_count();
    clone_preserves_bounds();
    clone_creates_valid_mvn();
    // SetDefaultParameters
    set_default_parameters_creates_valid_state();
    set_default_parameters_respects_max_error_bound();
    // Distance matrix
    distance_matrix_computed_correctly();
    // Edge cases
    minimal_sites_works_correctly();
    colocated_sites_handled_correctly();
    many_sites_works_correctly();
    very_small_sigma_handled_correctly();
    large_errors_produce_low_density();
    // Spatial correlation
    closer_sites_higher_correlation();
    // Added analytic kriging / IDW checks
    kriging_at_observed_site_recovers_error();
    kriging_variance_within_bounds();
    idw_weighted_average_and_field_order();
    idw_colocated_recovers_error();
    kriging_and_idw_bad_coordinate_dimension_throws();
    return chtest::summary("test_spatial_regression_errors");
}
