// Tests for GeneralLinearFunction (M7): the covariate linear model
// f(x) = beta_0 + beta_1*x_1 + ... + beta_p*x_p.
//
// Oracles transcribed from upstream/RMC-BestFit/src/RMC.BestFit.Tests/Univariate/TrendFunctions/
// GeneralLinearFunctionTests.cs @ fc28c0c -- every test method exercising ported surface, values
// unaltered.
//
// This is a structural/behavioral internal-support port (like test_trend_functions.cpp), so the
// oracle values are transcribed directly from the C# test file rather than routed through
// fixtures/ (no fixture kind exists for trends until M13).
//
// Skipped upstream methods (reasons; also in the task report):
//   Test_Constructor_XElement_RestoresModel, Test_Constructor_XElement_NullThrows,
//   Test_ToXElement_ContainsTypeAttribute, Test_ToXElement_ContainsCovariateInfo,
//   Test_RoundTrip_PreservesAllProperties, Test_RoundTrip_PreservesCovariateMatrix,
//   Test_RoundTrip_InterceptOnlyModel                 -- XML serialization not ported.
//   Test_PropertyChanged_OwnerName, Test_PropertyChanged_Covariates -- INPC not ported.
//   Test_Constructor_NullOwnerName_Throws, Test_SetParameterValues_NullThrows -- a null
//     std::string / std::vector is unrepresentable through the C++ value-type signatures.
#include <memory>
#include <string>
#include <vector>

#include "bestfit/models/trend_functions/general_linear_function.hpp"
#include "bestfit/models/trend_functions/support/i_trend_model.hpp"
#include "bestfit/models/trend_functions/support/trend_model_type.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "check.hpp"

using bestfit::models::trend_functions::GeneralLinearFunction;
using bestfit::models::trend_functions::TrendModelType;
using bestfit::numerics::distributions::Uniform;
using bestfit::numerics::math::linalg::Matrix;
using bestfit::numerics::math::linalg::Matrix2D;

namespace {

// UTF-8 byte escapes for the upstream parameter names (kept as explicit escapes so the
// literals survive any compiler source-charset handling; see the trend headers).
const std::string kBeta0 = "\xCE\xB2\xE2\x82\x80";  // "beta_0" with subscript zero
const std::string kBeta0Ascii = "\xCE\xB2"
                                "0";  // "beta0" (ASCII-zero fallback the C# test accepts)

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// =====================================================================================
// Constructor tests
// =====================================================================================

// Test_Constructor_EmptyConstructor_CreatesInterceptOnlyModel
void constructor_empty_creates_intercept_only_model() {
    GeneralLinearFunction model;

    CHECK_TRUE(model.type() == TrendModelType::GeneralLinear);
    CHECK_EQ(model.number_of_parameters(), 1);  // Just intercept
    CHECK_EQ(model.number_of_covariates(), 0);
}

// Test_Constructor_WithOwnerName_SetsOwnerName
void constructor_with_owner_name_sets_owner_name() {
    GeneralLinearFunction model("Location");

    CHECK_EQ(model.owner_name(), std::string("Location"));
    CHECK_EQ(model.number_of_parameters(), 1);
}

// Test_Constructor_WithCovariates_CreatesCorrectParameters
void constructor_with_covariates_creates_correct_parameters() {
    // 3 sites, 2 covariates
    Matrix covariates(Matrix2D{
        {100.0, 0.5},   // Site 1: elevation=100, latitude=0.5
        {200.0, 0.6},   // Site 2
        {150.0, 0.55},  // Site 3
    });

    GeneralLinearFunction model("Location", covariates);

    // intercept + 2 coefficients = 3 parameters
    CHECK_EQ(model.number_of_parameters(), 3);
    CHECK_EQ(model.number_of_covariates(), 2);
    CHECK_EQ(model.number_of_observations(), 3);
}

// =====================================================================================
// Type tests
// =====================================================================================

// Test_Type_ReturnsGeneralLinear
void type_returns_general_linear() {
    GeneralLinearFunction model;
    CHECK_TRUE(model.type() == TrendModelType::GeneralLinear);
}

// =====================================================================================
// SetDefaultParameters tests
// =====================================================================================

// Test_SetDefaultParameters_InterceptOnly
void set_default_parameters_intercept_only() {
    GeneralLinearFunction model;

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(1));
    CHECK_TRUE(contains(model.parameters()[0].name(), kBeta0) ||
               contains(model.parameters()[0].name(), kBeta0Ascii));
}

// Test_SetDefaultParameters_WithCovariates
void set_default_parameters_with_covariates() {
    Matrix covariates(Matrix2D{{1.0, 2.0, 3.0}});  // 3 covariates
    GeneralLinearFunction model("Scale", covariates);

    // 4 parameters: intercept + 3 coefficients
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(4));
}

// Test_SetDefaultParameters_CoefficientBounds
void set_default_parameters_coefficient_bounds() {
    Matrix covariates(Matrix2D{{1.0}});
    GeneralLinearFunction model("Location", covariates);

    // Coefficient (not intercept) should have [-1, 1] bounds
    CHECK_EQ(model.parameters()[1].lower_bound(), -1.0);
    CHECK_EQ(model.parameters()[1].upper_bound(), 1.0);
}

// Test_SetDefaultParameters_CoefficientPrior
void set_default_parameters_coefficient_prior() {
    Matrix covariates(Matrix2D{{1.0}});
    GeneralLinearFunction model("Location", covariates);

    const auto* prior = dynamic_cast<const Uniform*>(&model.parameters()[1].prior_distribution());
    CHECK_TRUE(prior != nullptr);
}

// =====================================================================================
// Covariates property tests
// =====================================================================================

// Test_Covariates_SetAndGet
void covariates_set_and_get() {
    GeneralLinearFunction model;
    Matrix covariates(Matrix2D{{1.0, 2.0}, {3.0, 4.0}});

    model.set_covariates(covariates);

    CHECK_EQ(model.number_of_covariates(), 2);
    CHECK_EQ(model.number_of_observations(), 2);
    // Parameters should be updated (intercept + 2 coefficients)
    CHECK_EQ(model.number_of_parameters(), 3);
}

// Test_Covariates_SetNull_ResetsToInterceptOnly
void covariates_set_null_resets_to_intercept_only() {
    Matrix covariates(Matrix2D{{1.0, 2.0}});
    GeneralLinearFunction model("Test", covariates);
    CHECK_EQ(model.number_of_covariates(), 2);

    model.set_covariates(std::nullopt);

    CHECK_EQ(model.number_of_covariates(), 0);
    CHECK_EQ(model.number_of_parameters(), 1);
}

// Test_NumberOfObservations_NoCovariates_ReturnsZero
void number_of_observations_no_covariates_returns_zero() {
    GeneralLinearFunction model;
    CHECK_EQ(model.number_of_observations(), 0);
}

// =====================================================================================
// Predict tests
// =====================================================================================

// Test_Predict_InterceptOnly_ReturnsIntercept
void predict_intercept_only_returns_intercept() {
    GeneralLinearFunction model;
    model.parameters()[0].set_value(50.0);

    CHECK_NEAR(model.predict(0), 50.0, 1e-10);
}

// Test_Predict_WithCovariates_ReturnsLinearCombination
void predict_with_covariates_returns_linear_combination() {
    // 2 sites, 2 covariates
    Matrix covariates(Matrix2D{
        {10.0, 5.0},    // Site 0
        {20.0, 10.0},   // Site 1
    });
    GeneralLinearFunction model("Location", covariates);
    model.parameters()[0].set_value(100.0);  // beta_0 (intercept)
    model.parameters()[1].set_value(0.5);    // beta_1
    model.parameters()[2].set_value(2.0);    // beta_2

    // y(0) = 100 + 0.5 * 10 + 2.0 * 5 = 100 + 5 + 10 = 115
    CHECK_NEAR(model.predict(0), 115.0, 1e-10);

    // y(1) = 100 + 0.5 * 20 + 2.0 * 10 = 100 + 10 + 20 = 130
    CHECK_NEAR(model.predict(1), 130.0, 1e-10);
}

// Test_Predict_IndexOutOfRange_Throws
void predict_index_out_of_range_throws() {
    Matrix covariates(Matrix2D{{1.0}, {2.0}});
    GeneralLinearFunction model("Test", covariates);

    CHECK_THROWS(model.predict(5));  // Index 5 doesn't exist
}

// Test_Predict_NegativeIndex_Throws
void predict_negative_index_throws() {
    Matrix covariates(Matrix2D{{1.0}});
    GeneralLinearFunction model("Test", covariates);

    CHECK_THROWS(model.predict(-1));
}

// =====================================================================================
// PredictWithCovariates tests
// =====================================================================================

// Test_PredictWithCovariates_NoCovariatesModel_ReturnsIntercept
// (the C# checks both null and Array.Empty<double>(); the null case collapses into the
// empty-vector case in C++)
void predict_with_covariates_no_covariates_model_returns_intercept() {
    GeneralLinearFunction model;
    model.parameters()[0].set_value(75.0);

    CHECK_NEAR(model.predict_with_covariates({}), 75.0, 1e-10);
    CHECK_NEAR(model.predict_with_covariates(std::vector<double>{}), 75.0, 1e-10);
}

// Test_PredictWithCovariates_CorrectLinearCombination
void predict_with_covariates_correct_linear_combination() {
    Matrix covariates(Matrix2D{{1.0, 2.0}});  // Define structure
    GeneralLinearFunction model("Location", covariates);
    model.parameters()[0].set_value(10.0);  // beta_0
    model.parameters()[1].set_value(2.0);   // beta_1
    model.parameters()[2].set_value(3.0);   // beta_2

    // Predict at ungauged site with covariates [5.0, 4.0]
    std::vector<double> new_covariates = {5.0, 4.0};
    double result = model.predict_with_covariates(new_covariates);

    // y = 10 + 2*5 + 3*4 = 10 + 10 + 12 = 32
    CHECK_NEAR(result, 32.0, 1e-10);
}

// Test_PredictWithCovariates_WrongLength_Throws
void predict_with_covariates_wrong_length_throws() {
    Matrix covariates(Matrix2D{{1.0, 2.0}});  // 2 covariates
    GeneralLinearFunction model("Location", covariates);

    // Pass only 1 covariate when 2 expected
    CHECK_THROWS(model.predict_with_covariates(std::vector<double>{5.0}));
}

// Test_PredictWithCovariates_NullOrEmpty_ReturnsIntercept
void predict_with_covariates_null_or_empty_returns_intercept() {
    Matrix covariates(Matrix2D{{1.0}});  // 1 covariate defined
    GeneralLinearFunction model("Location", covariates);
    model.parameters()[0].set_value(100.0);

    // Null or empty returns intercept (design decision to be lenient); null maps to empty
    double result = model.predict_with_covariates({});
    CHECK_NEAR(result, 100.0, 1e-10);
}

// Test_PredictWithCovariates_SpatialPrediction
void predict_with_covariates_spatial_prediction() {
    // Simulate spatial prediction scenario. Sites have elevation and latitude.
    Matrix gauge_sites(Matrix2D{
        {500.0, 35.0},    // Site 1
        {1000.0, 36.0},   // Site 2
        {750.0, 35.5},    // Site 3
    });

    GeneralLinearFunction model("Location", gauge_sites);
    model.parameters()[0].set_value(50.0);  // Intercept
    model.parameters()[1].set_value(0.01);  // Elevation effect
    model.parameters()[2].set_value(1.0);   // Latitude effect

    // Predict at ungauged site
    std::vector<double> ungauged_site = {800.0, 35.8};
    double prediction = model.predict_with_covariates(ungauged_site);

    // y = 50 + 0.01*800 + 1.0*35.8 = 50 + 8 + 35.8 = 93.8
    CHECK_NEAR(prediction, 93.8, 1e-10);
}

// =====================================================================================
// Clone tests
// =====================================================================================

// Test_Clone_CreatesIndependentCopy
void clone_creates_independent_copy() {
    Matrix covariates(Matrix2D{{1.0, 2.0}, {3.0, 4.0}});
    GeneralLinearFunction original("Location", covariates);
    original.parameters()[0].set_value(100.0);
    original.parameters()[1].set_value(0.5);
    original.parameters()[2].set_value(-0.3);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<GeneralLinearFunction*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    // Verify values match
    CHECK_EQ(original.owner_name(), clone->owner_name());
    CHECK_EQ(original.number_of_parameters(), clone->number_of_parameters());
    CHECK_EQ(original.parameters()[0].value(), clone->parameters()[0].value());

    // Verify independence
    original.parameters()[0].set_value(999.0);
    CHECK_EQ(clone->parameters()[0].value(), 100.0);
}

// Test_Clone_ClonesCovariateMatrix
// (the C# test mutates the caller's array, which the C# model aliases; the C++ ctor takes
// the matrix by value, so the deep-copy assertion mutates the ORIGINAL model's stored
// matrix instead and verifies the clone is unaffected)
void clone_clones_covariate_matrix() {
    Matrix covariates(Matrix2D{{10.0, 20.0}});
    GeneralLinearFunction original("Test", covariates);

    auto clone_ptr = original.clone();
    auto* clone = dynamic_cast<GeneralLinearFunction*>(clone_ptr.get());
    CHECK_TRUE(clone != nullptr);

    // Verify covariate matrix is cloned
    CHECK_EQ(original.number_of_covariates(), clone->number_of_covariates());
    CHECK_EQ(original.number_of_observations(), clone->number_of_observations());

    // Modifying the original's covariates shouldn't affect the clone
    (*original.covariates())(0, 0) = 999.0;
    CHECK_EQ((*clone->covariates())(0, 0), 10.0);
}

// Test_Clone_PreservesPrediction
void clone_preserves_prediction() {
    Matrix covariates(Matrix2D{{5.0, 3.0}});
    GeneralLinearFunction original("Location", covariates);
    original.parameters()[0].set_value(10.0);
    original.parameters()[1].set_value(2.0);
    original.parameters()[2].set_value(1.0);

    auto clone_ptr = original.clone();

    CHECK_NEAR(clone_ptr->predict(0), original.predict(0), 1e-10);
}

// =====================================================================================
// SetParameterValues tests
// =====================================================================================

// Test_SetParameterValues_UpdatesAllParameters
void set_parameter_values_updates_all_parameters() {
    Matrix covariates(Matrix2D{{1.0, 2.0}});
    GeneralLinearFunction model("Test", covariates);

    model.set_parameter_values({10.0, 20.0, 30.0});

    CHECK_EQ(model.parameters()[0].value(), 10.0);
    CHECK_EQ(model.parameters()[1].value(), 20.0);
    CHECK_EQ(model.parameters()[2].value(), 30.0);
}

// Test_SetParameterValues_WrongLengthThrows
void set_parameter_values_wrong_length_throws() {
    Matrix covariates(Matrix2D{{1.0}});
    GeneralLinearFunction model("Test", covariates);

    // Model has 2 parameters, but we're passing 1
    CHECK_THROWS(model.set_parameter_values({10.0}));
}

// =====================================================================================
// Edge cases
// =====================================================================================

// Test_SingleCovariate_SingleObservation
void single_covariate_single_observation() {
    Matrix covariates(Matrix2D{{5.0}});
    GeneralLinearFunction model("Location", covariates);
    model.parameters()[0].set_value(10.0);
    model.parameters()[1].set_value(2.0);

    // y = 10 + 2 * 5 = 20
    CHECK_NEAR(model.predict(0), 20.0, 1e-10);
}

// Test_ManyCovariates
void many_covariates() {
    // Test with 10 covariates
    Matrix covariates(2, 10);
    for (int j = 0; j < 10; j++) {
        covariates(0, j) = j + 1;
        covariates(1, j) = (j + 1) * 2;
    }

    GeneralLinearFunction model("Test", covariates);

    CHECK_EQ(model.number_of_parameters(), 11);  // intercept + 10
    CHECK_EQ(model.number_of_covariates(), 10);
}

// Test_NegativeCovariates
void negative_covariates() {
    Matrix covariates(Matrix2D{{-5.0, -3.0}});
    GeneralLinearFunction model("Test", covariates);
    model.parameters()[0].set_value(100.0);
    model.parameters()[1].set_value(1.0);
    model.parameters()[2].set_value(2.0);

    // y = 100 + 1*(-5) + 2*(-3) = 100 - 5 - 6 = 89
    CHECK_NEAR(model.predict(0), 89.0, 1e-10);
}

}  // namespace

int main() {
    // Constructors
    constructor_empty_creates_intercept_only_model();
    constructor_with_owner_name_sets_owner_name();
    constructor_with_covariates_creates_correct_parameters();

    // Type
    type_returns_general_linear();

    // SetDefaultParameters
    set_default_parameters_intercept_only();
    set_default_parameters_with_covariates();
    set_default_parameters_coefficient_bounds();
    set_default_parameters_coefficient_prior();

    // Covariates property
    covariates_set_and_get();
    covariates_set_null_resets_to_intercept_only();
    number_of_observations_no_covariates_returns_zero();

    // Predict
    predict_intercept_only_returns_intercept();
    predict_with_covariates_returns_linear_combination();
    predict_index_out_of_range_throws();
    predict_negative_index_throws();

    // PredictWithCovariates
    predict_with_covariates_no_covariates_model_returns_intercept();
    predict_with_covariates_correct_linear_combination();
    predict_with_covariates_wrong_length_throws();
    predict_with_covariates_null_or_empty_returns_intercept();
    predict_with_covariates_spatial_prediction();

    // Clone
    clone_creates_independent_copy();
    clone_clones_covariate_matrix();
    clone_preserves_prediction();

    // SetParameterValues
    set_parameter_values_updates_all_parameters();
    set_parameter_values_wrong_length_throws();

    // Edge cases
    single_covariate_single_observation();
    many_covariates();
    negative_covariates();

    return bftest::summary("general_linear_function");
}
