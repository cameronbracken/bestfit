// S4 support ctest (C++-only): SpatialGEV -- Renard's Bayesian hierarchical spatial GEV model
// (per-site GEV level, spatial-trend level, spatial-error + copula-dependence level, and a
// hyperparameter-prior level). SpatialGEV assembles the S1 correlation models, the S2
// GaussianCopula, and the S3 SpatialRegressionErrors on top of the ported GeneralLinearFunction
// trends and the GeneralizedExtremeValue leaf distribution. Full-fit likelihood / MLE / MAP /
// posterior oracles come from the P4 dotnet emitter, NOT S4; this file transcribes only the
// STRUCTURAL / DETERMINISM / TRIVIALLY-ANALYTIC assertions (this is internal-support test
// territory, so hardcoded oracles are correct here; public-API oracle values stay in fixtures/).
//
// Structural oracles transcribed (values unaltered where they exist) from
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/SpatialExtremes/SpatialGEVTests.cs @ fc28c0c
//
// The C# fixtures draw their at-site data from System.Random (a .NET LCG, not the ported
// Mersenne Twister), so the exact data VALUES are unreproducible in C++. This file regenerates
// equivalent homogeneous GEV data with the ported bit-exact MersenneTwister + GEV inverse-CDF,
// then applies the same STRUCTURAL assertions (counts, ordering, finiteness, bounds, trivially
// analytic closed forms). No data-value oracle is claimed for the generated fixtures.
//
// Skipped C# methods (see task-S4-report.md for the full list + reasons):
//   - All XML / ToXElement tests (ToXElement_*): XML serialization is a project-wide non-port.
//   - All SpatialGEVAnalysis_* tests: SpatialGEVAnalysis is a separate class (not ported in S4).
//   - The four Constructor_WithNull*_ThrowsArgumentNullException tests: VACUOUS in this port
//     (the ported ctor takes std::vector<...> / GeneralLinearFunction by value, none nullable).
//   - SetParameterValues_NullParameters_ThrowsArgumentNullException: VACUOUS (const-ref vector).
//   - Model_ImplementsISimulatable / Model_InheritsFromModelBase: mirrored as compile-time
//     static_assert (below) rather than a runtime IsInstanceOfType.
//
// The full-fit numeric quantities P4 will fill in are named in the // P4 pending block at the end.
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "bestfit/models/spatial_extremes/spatial_gev.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/simulatable.hpp"
#include "bestfit/models/trend_functions/general_linear_function.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

namespace {

using bestfit::models::spatial_extremes::CorrelationFunctionType;
using bestfit::models::spatial_extremes::GaussianCopula;
using bestfit::models::spatial_extremes::SpatialGEV;
using bestfit::models::spatial_extremes::SpatialRegressionErrors;
using bestfit::models::trend_functions::GeneralLinearFunction;
using bestfit::numerics::distributions::GeneralizedExtremeValue;
using bestfit::numerics::sampling::MersenneTwister;

using Grid = std::vector<std::vector<double>>;

// Compile-time mirror of Model_InheritsFromModelBase / Model_ImplementsISimulatable.
static_assert(std::is_base_of<bestfit::models::ModelBase, SpatialGEV>::value,
              "SpatialGEV must derive from ModelBase");
static_assert(
    std::is_base_of<bestfit::models::ISimulatable<std::vector<double>>, SpatialGEV>::value,
    "SpatialGEV must implement ISimulatable<std::vector<double>>");

// ---- Test data helpers (mirror the C# private fixtures; data regenerated with the ported RNG) ----

// 30 observations x 5 sites, homogeneous GEV(10000, 2500, 0). Row-major [year][site].
Grid create_test_at_site_data() {
    Grid data(30, std::vector<double>(5, 0.0));
    MersenneTwister rng(12345);
    GeneralizedExtremeValue gev(10000, 2500, 0.0);
    for (int site = 0; site < 5; ++site)
        for (int year = 0; year < 30; ++year)
            data[year][site] = gev.inverse_cdf(rng.next_double());
    return data;
}

Grid create_test_coordinates() {
    return Grid{{0.0, 0.0}, {10.0, 5.0}, {22.0, 8.0}, {35.0, 12.0}, {50.0, 15.0}};
}

std::pair<Grid, Grid> create_minimal_test_data() {
    Grid data(10, std::vector<double>(2, 0.0));
    MersenneTwister rng(54321);
    GeneralizedExtremeValue gev(6000, 1500, 0.0);
    for (int i = 0; i < 10; ++i) {
        data[i][0] = gev.inverse_cdf(rng.next_double());
        data[i][1] = gev.inverse_cdf(rng.next_double());
    }
    Grid coords{{0.0, 0.0}, {10.0, 0.0}};
    return {data, coords};
}

Grid create_data_with_missing_values() {
    Grid data = create_test_at_site_data();
    data[5][2] = std::numeric_limits<double>::quiet_NaN();
    data[10][0] = std::numeric_limits<double>::quiet_NaN();
    data[15][4] = std::numeric_limits<double>::quiet_NaN();
    data[20][1] = std::numeric_limits<double>::quiet_NaN();
    data[25][3] = std::numeric_limits<double>::quiet_NaN();
    return data;
}

SpatialGEV create_test_model() {
    return SpatialGEV(create_test_at_site_data(), create_test_coordinates(),
                      GeneralLinearFunction("Location"), GeneralLinearFunction("Scale"),
                      GeneralLinearFunction("Shape"));
}

SpatialGEV create_model_with_copula() {
    SpatialGEV model = create_test_model();
    model.set_spatial_dependence(
        GaussianCopula(model.coordinates(), CorrelationFunctionType::Exponential));
    model.set_use_copula_dependence(true);
    model.set_default_parameters();
    return model;
}

SpatialGEV create_model_with_spatial_errors() {
    SpatialGEV model = create_test_model();
    model.set_location_errors(
        SpatialRegressionErrors(model.coordinates(), CorrelationFunctionType::Exponential));
    model.set_scale_errors(
        SpatialRegressionErrors(model.coordinates(), CorrelationFunctionType::Exponential));
    model.set_use_location_errors(true);
    model.set_use_scale_errors(true);
    model.set_default_parameters();
    return model;
}

std::vector<double> current_values(const SpatialGEV& model) {
    std::vector<double> v;
    v.reserve(model.parameters().size());
    for (const auto& p : model.parameters()) v.push_back(p.value());
    return v;
}

// =====================================================================================
// Constructor Tests
// =====================================================================================

void constructor_with_valid_inputs_initializes_correctly() {
    SpatialGEV model = create_test_model();
    CHECK_EQ(model.sites(), 5);
    CHECK_EQ(model.observations(), 30);
}

void constructor_mismatched_dimensions_throws() {
    Grid data = create_test_at_site_data();  // 5 sites
    Grid coords{{0.0, 0.0}, {10.0, 0.0}, {20.0, 0.0}};  // 3 sites
    CHECK_THROWS(SpatialGEV(data, coords, GeneralLinearFunction("Location"),
                            GeneralLinearFunction("Scale"), GeneralLinearFunction("Shape")));
}

void constructor_default_options_are_correct() {
    SpatialGEV model = create_test_model();
    CHECK_TRUE(!model.use_copula_dependence());
    CHECK_TRUE(!model.use_location_errors());
    CHECK_TRUE(!model.use_scale_errors());
    CHECK_TRUE(!model.use_shape_errors());
    CHECK_TRUE(model.use_log_link_for_location());
    CHECK_TRUE(model.use_log_link_for_scale());
}

void constructor_site_weights_initialized_to_one() {
    SpatialGEV model = create_test_model();
    CHECK_EQ(static_cast<int>(model.site_weights().size()), 5);
    for (int i = 0; i < model.sites(); ++i) CHECK_NEAR(model.site_weights()[i], 1.0, 1e-10);
}

void constructor_minimal_sites_initializes_correctly() {
    auto [data, coords] = create_minimal_test_data();
    SpatialGEV model(data, coords, GeneralLinearFunction("Location"),
                     GeneralLinearFunction("Scale"), GeneralLinearFunction("Shape"));
    CHECK_EQ(model.sites(), 2);
    CHECK_EQ(model.observations(), 10);
}

// =====================================================================================
// Parameter Tests
// =====================================================================================

void parameters_basic_model_has_correct_count() {
    SpatialGEV model = create_test_model();
    CHECK_EQ(model.number_of_parameters(), 3);
}

void parameters_with_copula_has_additional_parameters() {
    SpatialGEV model = create_model_with_copula();
    CHECK_TRUE(model.number_of_parameters() > 3);
}

void parameters_with_spatial_errors_has_additional_parameters() {
    SpatialGEV model = create_model_with_spatial_errors();
    CHECK_TRUE(model.number_of_parameters() > 3);
}

void set_parameter_values_valid_parameters_updates_values() {
    SpatialGEV model = create_test_model();
    std::vector<double> values;
    for (const auto& p : model.parameters()) values.push_back(p.value() * 1.1);
    model.set_parameter_values(values);
    for (std::size_t i = 0; i < values.size(); ++i)
        CHECK_NEAR(model.parameters()[i].value(), values[i], 1e-10);
}

void set_parameter_values_wrong_count_throws() {
    SpatialGEV model = create_test_model();
    std::vector<double> values{1.0, 2.0};
    CHECK_THROWS(model.set_parameter_values(values));
}

void set_default_parameters_creates_valid_initial_values() {
    SpatialGEV model = create_test_model();
    model.set_default_parameters();
    CHECK_EQ(static_cast<int>(model.parameters().size()), 3);
    for (const auto& p : model.parameters()) {
        CHECK_TRUE(!std::isnan(p.value()) && !std::isinf(p.value()));
        CHECK_TRUE(p.value() >= p.lower_bound());
        CHECK_TRUE(p.value() <= p.upper_bound());
    }
}

// =====================================================================================
// GetGEVParameters Tests
// =====================================================================================

void get_gev_parameters_valid_site_returns_valid_parameters() {
    SpatialGEV model = create_test_model();
    for (int site = 0; site < model.sites(); ++site) {
        std::vector<double> gev = model.get_gev_parameters(site);
        CHECK_EQ(static_cast<int>(gev.size()), 3);
        CHECK_TRUE(!std::isnan(gev[0]) && gev[0] > 0);  // log-link location => positive
        CHECK_TRUE(!std::isnan(gev[1]) && gev[1] > 0);  // scale positive
        CHECK_TRUE(!std::isnan(gev[2]));
    }
}

void get_gev_parameters_negative_site_throws() {
    SpatialGEV model = create_test_model();
    CHECK_THROWS(model.get_gev_parameters(-1));
}

void get_gev_parameters_site_beyond_range_throws() {
    SpatialGEV model = create_test_model();
    CHECK_THROWS(model.get_gev_parameters(model.sites()));
}

void get_gev_parameters_identity_link_location_returns_valid_parameters() {
    SpatialGEV model = create_test_model();
    model.set_use_log_link_for_location(false);
    model.set_default_parameters();
    std::vector<double> gev = model.get_gev_parameters(0);
    CHECK_TRUE(!std::isnan(gev[0]));
}

void get_gev_parameters_identity_link_scale_returns_valid_parameters() {
    SpatialGEV model = create_test_model();
    model.set_use_log_link_for_scale(false);
    model.set_default_parameters();
    std::vector<double> gev = model.get_gev_parameters(0);
    CHECK_TRUE(!std::isnan(gev[1]) && gev[1] > 0);
}

void get_gev_parameters_with_location_errors_includes_error_contribution() {
    SpatialGEV model = create_model_with_spatial_errors();
    std::vector<double> p0 = model.get_gev_parameters(0);
    std::vector<double> p1 = model.get_gev_parameters(1);
    CHECK_TRUE(!std::isnan(p0[0]) && !std::isnan(p1[0]));
}

// =====================================================================================
// LogLikelihood Tests
// =====================================================================================

void log_likelihood_valid_parameters_returns_finite_value() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    double ll = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(ll));
    CHECK_TRUE(ll < 0);
}

void log_likelihood_invalid_parameter_count_returns_negative_infinity() {
    SpatialGEV model = create_test_model();
    std::vector<double> p{1.0};
    double ll = model.log_likelihood(p);
    CHECK_EQ(ll, -std::numeric_limits<double>::infinity());
}

void log_likelihood_better_fit_has_higher_likelihood() {
    SpatialGEV model = create_test_model();
    std::vector<double> good = current_values(model);
    double good_ll = model.log_likelihood(good);
    CHECK_TRUE(std::isfinite(good_ll));
    std::vector<double> bad = good;
    bad[bad.size() - 1] += 0.5;  // perturb shape
    double bad_ll = model.log_likelihood(bad);
    CHECK_TRUE(good_ll > bad_ll);
}

void log_likelihood_with_copula_returns_finite_value() {
    SpatialGEV model = create_model_with_copula();
    std::vector<double> p = current_values(model);
    double ll = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(ll));
}

void log_likelihood_with_spatial_errors_returns_finite_value() {
    SpatialGEV model = create_model_with_spatial_errors();
    std::vector<double> p = current_values(model);
    double ll = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(ll));
}

void log_likelihood_with_missing_data_handles_nan_correctly() {
    SpatialGEV model(create_data_with_missing_values(), create_test_coordinates(),
                     GeneralLinearFunction("Location"), GeneralLinearFunction("Scale"),
                     GeneralLinearFunction("Shape"));
    std::vector<double> p = current_values(model);
    double ll = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(ll));
}

void log_likelihood_site_weights_affect_result() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    double ll1 = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(ll1));
    model.site_weights()[0] = 0.5;
    model.site_weights()[1] = 1.5;
    double ll2 = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(ll2));
    CHECK_TRUE(std::fabs(ll1 - ll2) > 1e-10);
}

// =====================================================================================
// DataLogLikelihood / Pointwise Tests
// =====================================================================================

void data_log_likelihood_verify_composition() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    double data_ll = model.data_log_likelihood(p);
    double prior_ll = model.prior_log_likelihood(p);
    double total_ll = model.log_likelihood(p);
    CHECK_NEAR(data_ll + prior_ll, total_ll, 1e-10);
}

void data_log_likelihood_valid_parameters_returns_finite_value() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    CHECK_TRUE(std::isfinite(model.data_log_likelihood(p)));
}

void pointwise_data_log_likelihood_returns_correct_count() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    std::vector<double> pw = model.pointwise_data_log_likelihood(p);
    CHECK_EQ(static_cast<int>(pw.size()), model.observations());
}

void pointwise_data_log_likelihood_values_are_finite() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    std::vector<double> pw = model.pointwise_data_log_likelihood(p);
    for (double v : pw) CHECK_TRUE(!std::isnan(v));
}

void pointwise_data_log_likelihood_sums_to_data_log_likelihood() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    double data_ll = model.data_log_likelihood(p);
    CHECK_TRUE(std::isfinite(data_ll));
    std::vector<double> pw = model.pointwise_data_log_likelihood(p);
    double sum = 0.0;
    for (double v : pw) sum += v;
    CHECK_NEAR(data_ll, sum, 1e-6);
}

void pointwise_data_log_likelihood_components_returns_correct_count() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    auto comps = model.pointwise_data_log_likelihood_components(p);
    CHECK_EQ(static_cast<int>(comps.size()), model.observations());
}

void pointwise_data_log_likelihood_components_has_correct_properties() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    auto comps = model.pointwise_data_log_likelihood_components(p);
    CHECK_EQ(comps[0].index(), 0);
    CHECK_TRUE(!std::isnan(comps[0].log_likelihood()));
    CHECK_TRUE(comps[0].type() == bestfit::models::DataComponentType::Exact);
}

void pointwise_prior_log_likelihood_returns_prior_components() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    auto comps = model.pointwise_prior_log_likelihood(p);
    CHECK_TRUE(static_cast<int>(comps.size()) >= model.number_of_parameters());
}

void pointwise_prior_log_likelihood_components_have_valid_log_likelihoods() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    auto comps = model.pointwise_prior_log_likelihood(p);
    for (const auto& c : comps) CHECK_TRUE(!std::isnan(c.log_likelihood()));
}

// The deliberately-broken decomposition (see header of spatial_gev.hpp): with spatial errors on,
// PointwisePriorLogLikelihood emits SpatialError components in ADDITION to the parameter priors,
// so its count strictly exceeds NumberOfParameters and those extra components carry the
// PriorComponentType::SpatialError tag.
void pointwise_prior_log_likelihood_emits_spatial_error_components() {
    SpatialGEV model = create_model_with_spatial_errors();
    std::vector<double> p = current_values(model);
    auto comps = model.pointwise_prior_log_likelihood(p);
    CHECK_TRUE(static_cast<int>(comps.size()) > model.number_of_parameters());
    int spatial_error_count = 0;
    for (const auto& c : comps)
        if (c.type() == bestfit::models::PriorComponentType::SpatialError) ++spatial_error_count;
    CHECK_TRUE(spatial_error_count >= 2);  // location + scale errors enabled
}

// =====================================================================================
// PDF / CDF / InverseCDF Tests
// =====================================================================================

void pdf_valid_input_returns_positive_value() {
    SpatialGEV model = create_test_model();
    for (int site = 0; site < model.sites(); ++site) {
        std::vector<double> gev = model.get_gev_parameters(site);
        double pdf = model.pdf(gev[0], site);
        CHECK_TRUE(pdf > 0 && !std::isnan(pdf));
    }
}

void cdf_valid_input_returns_value_between_zero_and_one() {
    SpatialGEV model = create_test_model();
    for (int site = 0; site < model.sites(); ++site) {
        std::vector<double> gev = model.get_gev_parameters(site);
        double cdf = model.cdf(gev[0], site);
        CHECK_TRUE(cdf >= 0 && cdf <= 1 && !std::isnan(cdf));
    }
}

void inverse_cdf_consistent_with_cdf() {
    SpatialGEV model = create_test_model();
    double probs[] = {0.1, 0.25, 0.5, 0.75, 0.9};
    for (int site = 0; site < model.sites(); ++site) {
        for (double pr : probs) {
            double q = model.inverse_cdf(pr, site);
            CHECK_NEAR(model.cdf(q, site), pr, 1e-6);
        }
    }
}

void inverse_cdf_increasing_probabilities_returns_increasing_quantiles() {
    SpatialGEV model = create_test_model();
    double probs[] = {0.1, 0.5, 0.9};
    for (int site = 0; site < model.sites(); ++site) {
        double prev = -std::numeric_limits<double>::infinity();
        for (double pr : probs) {
            double q = model.inverse_cdf(pr, site);
            CHECK_TRUE(q > prev);
            prev = q;
        }
    }
}

// =====================================================================================
// GenerateRandomValues Tests
// =====================================================================================

void generate_random_values_returns_correct_count() {
    SpatialGEV model = create_test_model();
    int sample_size = 100;
    std::vector<double> samples = model.generate_random_values(sample_size, 12345);
    CHECK_EQ(static_cast<int>(samples.size()), sample_size * model.sites());
}

void generate_random_values_zero_sample_size_throws() {
    SpatialGEV model = create_test_model();
    CHECK_THROWS(model.generate_random_values(0));
}

void generate_random_values_negative_sample_size_throws() {
    SpatialGEV model = create_test_model();
    CHECK_THROWS(model.generate_random_values(-10));
}

void generate_random_values_different_seeds_produce_different_values() {
    SpatialGEV model = create_test_model();
    std::vector<double> s1 = model.generate_random_values(50, 111);
    std::vector<double> s2 = model.generate_random_values(50, 222);
    bool any_different = false;
    for (std::size_t i = 0; i < s1.size(); ++i)
        if (std::fabs(s1[i] - s2[i]) > 1e-10) {
            any_different = true;
            break;
        }
    CHECK_TRUE(any_different);
}

void generate_random_values_same_seed_produces_reproducible_values() {
    SpatialGEV model = create_test_model();
    std::vector<double> s1 = model.generate_random_values(50, 12345);
    std::vector<double> s2 = model.generate_random_values(50, 12345);
    CHECK_EQ(s1.size(), s2.size());
    for (std::size_t i = 0; i < s1.size(); ++i) CHECK_NEAR(s1[i], s2[i], 0.0);  // bit-for-bit
}

void generate_random_values_produces_positive_values() {
    SpatialGEV model = create_test_model();
    std::vector<double> samples = model.generate_random_values(1000, 12345);
    int positive = 0;
    for (double s : samples)
        if (s > 0) ++positive;
    double ratio = static_cast<double>(positive) / static_cast<double>(samples.size());
    CHECK_TRUE(ratio > 0.95);
}

// =====================================================================================
// Clone Tests
// =====================================================================================

void clone_creates_independent_copy() {
    SpatialGEV original = create_test_model();
    original.site_weights()[0] = 2.0;
    SpatialGEV clone = original.clone();
    original.site_weights()[0] = 5.0;
    CHECK_NEAR(clone.site_weights()[0], 2.0, 1e-10);
}

void clone_preserves_configuration() {
    SpatialGEV original = create_test_model();
    original.set_use_log_link_for_location(false);
    original.set_use_log_link_for_scale(false);
    SpatialGEV clone = original.clone();
    CHECK_EQ(clone.use_log_link_for_location(), original.use_log_link_for_location());
    CHECK_EQ(clone.use_log_link_for_scale(), original.use_log_link_for_scale());
    CHECK_EQ(clone.use_copula_dependence(), original.use_copula_dependence());
    CHECK_EQ(clone.use_location_errors(), original.use_location_errors());
}

void clone_preserves_dimensions() {
    SpatialGEV original = create_test_model();
    SpatialGEV clone = original.clone();
    CHECK_EQ(clone.sites(), original.sites());
    CHECK_EQ(clone.observations(), original.observations());
    CHECK_EQ(clone.number_of_parameters(), original.number_of_parameters());
}

void clone_with_copula_preserves_copula() {
    SpatialGEV original = create_model_with_copula();
    SpatialGEV clone = original.clone();
    CHECK_TRUE(clone.use_copula_dependence());
    CHECK_TRUE(clone.spatial_dependence() != nullptr);
}

void clone_with_spatial_errors_preserves_errors() {
    SpatialGEV original = create_model_with_spatial_errors();
    SpatialGEV clone = original.clone();
    CHECK_TRUE(clone.use_location_errors());
    CHECK_TRUE(clone.use_scale_errors());
    CHECK_TRUE(clone.location_errors() != nullptr);
    CHECK_TRUE(clone.scale_errors() != nullptr);
}

// =====================================================================================
// Validation Tests
// =====================================================================================

void validate_valid_model_returns_valid() {
    SpatialGEV model = create_test_model();
    CHECK_TRUE(model.validate().is_valid);
}

void validate_copula_enabled_but_null_returns_invalid() {
    SpatialGEV model = create_test_model();
    model.set_use_copula_dependence(true);
    auto result = model.validate();
    CHECK_TRUE(!result.is_valid);
    bool mentions_copula = false;
    for (const auto& m : result.validation_messages)
        if (m.find("Copula") != std::string::npos) mentions_copula = true;
    CHECK_TRUE(mentions_copula);
}

void validate_location_errors_enabled_but_null_returns_invalid() {
    SpatialGEV model = create_test_model();
    model.set_use_location_errors(true);
    CHECK_TRUE(!model.validate().is_valid);
}

void validate_scale_errors_enabled_but_null_returns_invalid() {
    SpatialGEV model = create_test_model();
    model.set_use_scale_errors(true);
    CHECK_TRUE(!model.validate().is_valid);
}

void validate_shape_errors_enabled_but_null_returns_invalid() {
    SpatialGEV model = create_test_model();
    model.set_use_shape_errors(true);
    CHECK_TRUE(!model.validate().is_valid);
}

void validate_with_copula_returns_valid() {
    SpatialGEV model = create_model_with_copula();
    CHECK_TRUE(model.validate().is_valid);
}

void validate_with_spatial_errors_returns_valid() {
    SpatialGEV model = create_model_with_spatial_errors();
    CHECK_TRUE(model.validate().is_valid);
}

// =====================================================================================
// Edge Cases
// =====================================================================================

void model_with_two_sites_works_correctly() {
    auto [data, coords] = create_minimal_test_data();
    SpatialGEV model(data, coords, GeneralLinearFunction("Location"),
                     GeneralLinearFunction("Scale"), GeneralLinearFunction("Shape"));
    CHECK_TRUE(model.validate().is_valid);
    std::vector<double> p = current_values(model);
    CHECK_TRUE(std::isfinite(model.log_likelihood(p)));
}

void model_zero_weight_excludes_site() {
    SpatialGEV model = create_test_model();
    std::vector<double> p = current_values(model);
    double all_sites = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(all_sites));
    model.site_weights()[0] = 0.0;
    double exclude_one = model.log_likelihood(p);
    CHECK_TRUE(std::isfinite(exclude_one));
    CHECK_TRUE(std::fabs(all_sites - exclude_one) > 1e-10);
}

void model_all_nan_at_one_site_handles_correctly() {
    Grid data = create_test_at_site_data();
    for (int i = 0; i < 30; ++i) data[i][2] = std::numeric_limits<double>::quiet_NaN();
    SpatialGEV model(data, create_test_coordinates(), GeneralLinearFunction("Location"),
                     GeneralLinearFunction("Scale"), GeneralLinearFunction("Shape"));
    std::vector<double> p = current_values(model);
    CHECK_TRUE(std::isfinite(model.log_likelihood(p)));
}

// =====================================================================================
// Link Function Tests
// =====================================================================================

void log_link_location_produces_positive_values() {
    SpatialGEV model = create_test_model();
    for (int site = 0; site < model.sites(); ++site)
        CHECK_TRUE(model.get_gev_parameters(site)[0] > 0);
}

void log_link_scale_produces_positive_values() {
    SpatialGEV model = create_test_model();
    for (int site = 0; site < model.sites(); ++site)
        CHECK_TRUE(model.get_gev_parameters(site)[1] > 0);
}

void identity_link_scale_enforces_positive_constraint() {
    SpatialGEV model = create_test_model();
    model.set_use_log_link_for_scale(false);
    model.set_default_parameters();
    CHECK_TRUE(model.get_gev_parameters(0)[1] > 0);
}

// =====================================================================================
// Uncertainty / Effective-Sample-Size Helpers
// =====================================================================================

void compute_intersite_correlation_returns_valid_matrix() {
    SpatialGEV model = create_test_model();
    Grid corr = model.compute_intersite_correlation();
    CHECK_EQ(static_cast<int>(corr.size()), model.sites());
    CHECK_EQ(static_cast<int>(corr[0].size()), model.sites());
    for (int i = 0; i < model.sites(); ++i) CHECK_NEAR(corr[i][i], 1.0, 1e-10);
    for (int i = 0; i < model.sites(); ++i)
        for (int j = i + 1; j < model.sites(); ++j) CHECK_NEAR(corr[i][j], corr[j][i], 1e-10);
    for (int i = 0; i < model.sites(); ++i)
        for (int j = 0; j < model.sites(); ++j)
            CHECK_TRUE(corr[i][j] >= -1 && corr[i][j] <= 1);
}

void compute_effective_sample_size_returns_positive_value() {
    SpatialGEV model = create_test_model();
    double n_eff = model.compute_effective_sample_size();
    CHECK_TRUE(n_eff > 0);
    CHECK_TRUE(n_eff <= model.observations() * model.sites());
}

void compute_effective_sample_size_with_custom_matrix_works_correctly() {
    SpatialGEV model = create_test_model();
    Grid corr(model.sites(), std::vector<double>(model.sites(), 0.0));
    for (int i = 0; i < model.sites(); ++i)
        for (int j = 0; j < model.sites(); ++j) corr[i][j] = (i == j) ? 1.0 : 0.5;
    double n_eff = model.compute_effective_sample_size(corr);
    // n_eff = 30*5 / (1 + 4*0.5) = 150 / 3 = 50
    double expected =
        (model.observations() * model.sites()) / (1.0 + (model.sites() - 1) * 0.5);
    CHECK_NEAR(n_eff, expected, 1e-6);
}

void compute_variance_inflation_factor_returns_valid_value() {
    SpatialGEV model = create_test_model();
    CHECK_TRUE(model.compute_variance_inflation_factor() >= 1.0);
}

void compute_variance_inflation_factor_with_known_correlation_returns_correct_value() {
    SpatialGEV model = create_test_model();
    Grid corr(model.sites(), std::vector<double>(model.sites(), 0.0));
    for (int i = 0; i < model.sites(); ++i)
        for (int j = 0; j < model.sites(); ++j) corr[i][j] = (i == j) ? 1.0 : 0.3;
    double vif = model.compute_variance_inflation_factor(corr);
    // VIF = 1 + (Sites-1)*0.3 = 1 + 4*0.3 = 2.2
    double expected = 1.0 + (model.sites() - 1) * 0.3;
    CHECK_NEAR(vif, expected, 1e-6);
}

void compute_effective_sample_size_weights_updates_site_weights() {
    SpatialGEV model = create_test_model();
    model.compute_effective_sample_size_weights();
    CHECK_EQ(static_cast<int>(model.site_weights().size()), model.sites());
    double sum = 0.0;
    for (int i = 0; i < model.sites(); ++i) {
        CHECK_TRUE(model.site_weights()[i] > 0);
        sum += model.site_weights()[i];
    }
    CHECK_NEAR(sum, static_cast<double>(model.sites()), 1e-6);
}

void compute_effective_sample_size_weights_with_custom_matrix_works_correctly() {
    SpatialGEV model = create_test_model();
    Grid corr(model.sites(), std::vector<double>(model.sites(), 0.0));
    for (int i = 0; i < model.sites(); ++i)
        for (int j = 0; j < model.sites(); ++j) corr[i][j] = (i == j) ? 1.0 : 0.0;
    model.compute_effective_sample_size_weights(corr);
    for (int i = 0; i < model.sites(); ++i) CHECK_NEAR(model.site_weights()[i], 1.0, 1e-6);
}

void compute_effective_sample_size_weights_mismatched_matrix_throws() {
    SpatialGEV model = create_test_model();
    Grid wrong(3, std::vector<double>(3, 0.0));
    CHECK_THROWS(model.compute_effective_sample_size_weights(wrong));
}

void configure_for_proper_coverage_enables_required_components() {
    SpatialGEV model = create_test_model();
    model.configure_for_proper_coverage();
    CHECK_TRUE(model.use_copula_dependence());
    CHECK_TRUE(model.spatial_dependence() != nullptr);
    CHECK_TRUE(model.use_location_errors());
    CHECK_TRUE(model.location_errors() != nullptr);
}

void configure_for_proper_coverage_with_optional_errors_enables_all_components() {
    SpatialGEV model = create_test_model();
    model.configure_for_proper_coverage(CorrelationFunctionType::Spherical, true, true);
    CHECK_TRUE(model.use_copula_dependence());
    CHECK_TRUE(model.use_location_errors());
    CHECK_TRUE(model.use_scale_errors());
    CHECK_TRUE(model.use_shape_errors());
    CHECK_TRUE(model.scale_errors() != nullptr);
    CHECK_TRUE(model.shape_errors() != nullptr);
}

void configure_for_proper_coverage_rebuilds_parameter_list() {
    SpatialGEV model = create_test_model();
    int original = model.number_of_parameters();
    model.configure_for_proper_coverage(CorrelationFunctionType::Exponential, true, false);
    CHECK_TRUE(model.number_of_parameters() > original);
}

// =====================================================================================
// PredictAtUngauged / GetGEVAtUngauged
// =====================================================================================

void predict_at_ungauged_returns_valid_gev_parameters() {
    SpatialGEV model = create_test_model();
    std::vector<double> new_coords{25.0, 10.0};
    auto [gev_params, err_var] = model.predict_at_ungauged(new_coords);
    CHECK_EQ(static_cast<int>(gev_params.size()), 3);
    CHECK_TRUE(!std::isnan(gev_params[0]) && gev_params[0] > 0);
    CHECK_TRUE(!std::isnan(gev_params[1]) && gev_params[1] > 0);
    CHECK_TRUE(!std::isnan(gev_params[2]));
}

void predict_at_ungauged_returns_error_variances() {
    SpatialGEV model = create_test_model();
    std::vector<double> new_coords{25.0, 10.0};
    auto [gev_params, err_var] = model.predict_at_ungauged(new_coords);
    CHECK_EQ(static_cast<int>(err_var.size()), 3);
    CHECK_NEAR(err_var[0], 0.0, 1e-10);
    CHECK_NEAR(err_var[1], 0.0, 1e-10);
    CHECK_NEAR(err_var[2], 0.0, 1e-10);
}

void predict_at_ungauged_with_spatial_errors_returns_non_negative_variances() {
    SpatialGEV model = create_model_with_spatial_errors();
    std::vector<double> new_coords{25.0, 10.0};
    auto [gev_params, err_var] = model.predict_at_ungauged(new_coords);
    CHECK_TRUE(err_var[0] >= 0);
    CHECK_TRUE(err_var[1] >= 0);
}

void predict_at_ungauged_invalid_coordinates_throws() {
    SpatialGEV model = create_test_model();
    std::vector<double> bad{1.0};
    CHECK_THROWS(model.predict_at_ungauged(bad));
}

void get_gev_at_ungauged_returns_valid_distribution() {
    SpatialGEV model = create_test_model();
    std::vector<double> new_coords{25.0, 10.0};
    GeneralizedExtremeValue gev = model.get_gev_at_ungauged(new_coords);
    double q = gev.inverse_cdf(0.99);
    CHECK_TRUE(!std::isnan(q) && q > 0);
    CHECK_NEAR(gev.cdf(q), 0.99, 1e-6);
}

void get_gev_at_ungauged_at_existing_site_matches_site_gev() {
    SpatialGEV model = create_test_model();
    Grid coords = create_test_coordinates();
    std::vector<double> site0{coords[0][0], coords[0][1]};
    GeneralizedExtremeValue predicted = model.get_gev_at_ungauged(site0);
    std::vector<double> site0_params = model.get_gev_parameters(0);
    CHECK_NEAR(predicted.xi(), site0_params[0], 1e-6);
    CHECK_NEAR(predicted.alpha(), site0_params[1], 1e-6);
    CHECK_NEAR(predicted.kappa(), site0_params[2], 1e-6);
}

// =====================================================================================
// P4 pending: full-fit numeric oracles (NOT part of S4). The P4 dotnet emitter fills in
// exact expected values for these quantities (transcribed against the real RMC.BestFit
// SpatialGEV) via the fixture path:
//   - DataLogLikelihood / LogLikelihood at a fixed hyperparameter vector on a fixed
//     (cross-language reproducible) at-site dataset.
//   - MLE / MAP hyperparameter estimates and their log-likelihood at the optimum.
//   - A seeded DEMCz/DEMCzs posterior-chain digest for SpatialGEV (short_exact), plus a
//     GenerateRandomValues cross-language stream digest.
//   - WAIC / LOO-CV computed from PointwiseDataLogLikelihoodComponents (which, by the
//     documented non-canonical decomposition, EXCLUDE the spatial-error term).
// S4 asserts only structure/determinism above; no full-fit numeric oracle is claimed here.
// =====================================================================================

}  // namespace

int main() {
    // Constructors
    constructor_with_valid_inputs_initializes_correctly();
    constructor_mismatched_dimensions_throws();
    constructor_default_options_are_correct();
    constructor_site_weights_initialized_to_one();
    constructor_minimal_sites_initializes_correctly();
    // Parameters
    parameters_basic_model_has_correct_count();
    parameters_with_copula_has_additional_parameters();
    parameters_with_spatial_errors_has_additional_parameters();
    set_parameter_values_valid_parameters_updates_values();
    set_parameter_values_wrong_count_throws();
    set_default_parameters_creates_valid_initial_values();
    // GetGEVParameters
    get_gev_parameters_valid_site_returns_valid_parameters();
    get_gev_parameters_negative_site_throws();
    get_gev_parameters_site_beyond_range_throws();
    get_gev_parameters_identity_link_location_returns_valid_parameters();
    get_gev_parameters_identity_link_scale_returns_valid_parameters();
    get_gev_parameters_with_location_errors_includes_error_contribution();
    // LogLikelihood
    log_likelihood_valid_parameters_returns_finite_value();
    log_likelihood_invalid_parameter_count_returns_negative_infinity();
    log_likelihood_better_fit_has_higher_likelihood();
    log_likelihood_with_copula_returns_finite_value();
    log_likelihood_with_spatial_errors_returns_finite_value();
    log_likelihood_with_missing_data_handles_nan_correctly();
    log_likelihood_site_weights_affect_result();
    // DataLogLikelihood / Pointwise
    data_log_likelihood_verify_composition();
    data_log_likelihood_valid_parameters_returns_finite_value();
    pointwise_data_log_likelihood_returns_correct_count();
    pointwise_data_log_likelihood_values_are_finite();
    pointwise_data_log_likelihood_sums_to_data_log_likelihood();
    pointwise_data_log_likelihood_components_returns_correct_count();
    pointwise_data_log_likelihood_components_has_correct_properties();
    pointwise_prior_log_likelihood_returns_prior_components();
    pointwise_prior_log_likelihood_components_have_valid_log_likelihoods();
    pointwise_prior_log_likelihood_emits_spatial_error_components();
    // PDF / CDF / InverseCDF
    pdf_valid_input_returns_positive_value();
    cdf_valid_input_returns_value_between_zero_and_one();
    inverse_cdf_consistent_with_cdf();
    inverse_cdf_increasing_probabilities_returns_increasing_quantiles();
    // GenerateRandomValues
    generate_random_values_returns_correct_count();
    generate_random_values_zero_sample_size_throws();
    generate_random_values_negative_sample_size_throws();
    generate_random_values_different_seeds_produce_different_values();
    generate_random_values_same_seed_produces_reproducible_values();
    generate_random_values_produces_positive_values();
    // Clone
    clone_creates_independent_copy();
    clone_preserves_configuration();
    clone_preserves_dimensions();
    clone_with_copula_preserves_copula();
    clone_with_spatial_errors_preserves_errors();
    // Validation
    validate_valid_model_returns_valid();
    validate_copula_enabled_but_null_returns_invalid();
    validate_location_errors_enabled_but_null_returns_invalid();
    validate_scale_errors_enabled_but_null_returns_invalid();
    validate_shape_errors_enabled_but_null_returns_invalid();
    validate_with_copula_returns_valid();
    validate_with_spatial_errors_returns_valid();
    // Edge cases
    model_with_two_sites_works_correctly();
    model_zero_weight_excludes_site();
    model_all_nan_at_one_site_handles_correctly();
    // Link functions
    log_link_location_produces_positive_values();
    log_link_scale_produces_positive_values();
    identity_link_scale_enforces_positive_constraint();
    // Uncertainty helpers
    compute_intersite_correlation_returns_valid_matrix();
    compute_effective_sample_size_returns_positive_value();
    compute_effective_sample_size_with_custom_matrix_works_correctly();
    compute_variance_inflation_factor_returns_valid_value();
    compute_variance_inflation_factor_with_known_correlation_returns_correct_value();
    compute_effective_sample_size_weights_updates_site_weights();
    compute_effective_sample_size_weights_with_custom_matrix_works_correctly();
    compute_effective_sample_size_weights_mismatched_matrix_throws();
    configure_for_proper_coverage_enables_required_components();
    configure_for_proper_coverage_with_optional_errors_enables_all_components();
    configure_for_proper_coverage_rebuilds_parameter_list();
    // Ungauged prediction
    predict_at_ungauged_returns_valid_gev_parameters();
    predict_at_ungauged_returns_error_variances();
    predict_at_ungauged_with_spatial_errors_returns_non_negative_variances();
    predict_at_ungauged_invalid_coordinates_throws();
    get_gev_at_ungauged_returns_valid_distribution();
    get_gev_at_ungauged_at_existing_site_matches_site_gev();
    return bftest::summary("test_spatial_gev");
}
