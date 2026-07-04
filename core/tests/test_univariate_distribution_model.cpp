// Standalone test for bestfit::models::UnivariateDistributionModel (Phase 4 T6 + Phase 5 M8).
//
// Oracle for behavior is the C# source itself:
//   - upstream/RMC-BestFit/src/RMC.BestFit/Models/UnivariateDistribution/
//     UnivariateDistribution.cs @ fc28c0c (stationary path; nonstationary is M9), and its
//     base UnivariateDistributionModelBase.cs @ fc28c0c;
//   - transcriptions of the upstream test classes (stationary methods only):
//     RMC.BestFit.Tests/Univariate/UnivariateDistributionTests.cs,
//     UnivariateDistributionExpandedTests.cs, ThresholdLikelihoodGuardTests.cs, and
//     Test_Numerics/Mathematics/Integration/Test_Integration.cs (Test_GaussLegendre20).
// The remaining checks are SELF-CONSISTENCY (log-likelihood vs closed-form Normal, pointwise
// sums, invalid-parameter -inf guards, default-parameter construction, censored-data
// consistency across the four series types). Hardcoded oracles are allowed here (internal
// support); public-API oracle values stay in fixtures/ only. The absolute match to the real
// C# model is validated by the T12 dotnet emitter against the fixture-driven runners.
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/support/quantile_prior.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/models/trend_functions/support/i_trend_model.hpp"
#include "bestfit/models/trend_functions/support/trend_model_type.hpp"
#include "bestfit/models/univariate_distribution/base/univariate_distribution_model_base.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/exponential.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/gumbel.hpp"
#include "bestfit/numerics/distributions/ln_normal.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/math/integration/integration.hpp"
#include "bestfit/numerics/tools.hpp"
#include "check.hpp"

using bestfit::models::DataComponent;
using bestfit::models::DataComponentType;
using bestfit::models::DataFrame;
using bestfit::models::ExactData;
using bestfit::models::ExactSeries;
using bestfit::models::IntervalData;
using bestfit::models::ThresholdData;
using bestfit::models::UncertainData;
using bestfit::models::UnivariateDistributionModel;
using bestfit::models::ValidationResult;
using bestfit::numerics::distributions::GeneralizedExtremeValue;
using bestfit::numerics::distributions::Gumbel;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::UnivariateDistributionType;
using bestfit::numerics::math::integration::Integration;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Phase-4 dataset (brief's canonical sample).
std::vector<double> sample_data() {
    return {12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600};
}

// ===========================================================================================
// Phase 4 (T6/T12) checks -- must keep passing unchanged after the M8 DataFrame rebase.
// ===========================================================================================

void test_data_log_likelihood_matches_closed_form_normal() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    double mu = 16000.0;
    double sigma = 4500.0;
    std::vector<double> p{mu, sigma};

    Normal expected_dist(mu, sigma);
    double expected = 0.0;
    for (double x : sample_data()) expected += expected_dist.log_pdf(x);

    CHECK_NEAR(model.data_log_likelihood(p), expected, 1e-9);
}

void test_data_log_likelihood_negative_infinity_on_invalid_parameters() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    std::vector<double> invalid_p{16000.0, -1.0};  // sigma <= 0 is invalid for Normal
    double result = model.data_log_likelihood(invalid_p);

    CHECK_TRUE(result == -std::numeric_limits<double>::infinity());
}

void test_pointwise_sums_to_data_log_likelihood_and_has_matching_length() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    std::vector<double> p{16000.0, 4500.0};
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(p);

    CHECK_EQ(pointwise.size(), sample_data().size());

    double sum = 0.0;
    for (double ll : pointwise) sum += ll;
    CHECK_NEAR(sum, model.data_log_likelihood(p), 1e-9);
}

void test_pointwise_invalid_parameters_returns_all_negative_infinity() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    std::vector<double> invalid_p{16000.0, -1.0};
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(invalid_p);

    CHECK_EQ(pointwise.size(), sample_data().size());
    for (double ll : pointwise) CHECK_TRUE(ll == -std::numeric_limits<double>::infinity());
}

void test_pointwise_components_one_exact_component_per_value_matching_ll() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    std::vector<double> p{16000.0, 4500.0};
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(p);
    std::vector<DataComponent> components = model.pointwise_data_log_likelihood_components(p);

    CHECK_EQ(components.size(), sample_data().size());
    for (std::size_t i = 0; i < components.size(); ++i) {
        CHECK_TRUE(components[i].type() == DataComponentType::Exact);
        CHECK_EQ(components[i].index(), static_cast<int>(i));
        CHECK_NEAR(components[i].value(), sample_data()[i], 1e-12);
        CHECK_NEAR(components[i].log_likelihood(), pointwise[i], 1e-12);
    }
}

void test_pointwise_components_negative_infinity_on_invalid_parameters() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    std::vector<double> invalid_p{16000.0, -1.0};
    std::vector<DataComponent> components = model.pointwise_data_log_likelihood_components(invalid_p);

    CHECK_EQ(components.size(), sample_data().size());
    for (const auto& c : components) {
        CHECK_TRUE(c.log_likelihood() == -std::numeric_limits<double>::infinity());
        CHECK_TRUE(c.type() == DataComponentType::Exact);
    }
}

void test_set_default_parameters_populates_bounds_and_uniform_priors() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    // Constructor already calls set_default_parameters(); call again explicitly too, to
    // confirm idempotency (mirrors the C# setter re-invoking SetDefaultParameters()).
    model.set_default_parameters();

    CHECK_EQ(model.number_of_parameters(), model.distribution().number_of_parameters());
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));

    for (const auto& param : model.parameters()) {
        CHECK_TRUE(param.lower_bound() <= param.value());
        CHECK_TRUE(param.value() <= param.upper_bound());

        // Prior must be a Uniform(lower_bound, upper_bound).
        const auto* uniform_prior =
            dynamic_cast<const bestfit::numerics::distributions::Uniform*>(&param.prior_distribution());
        CHECK_TRUE(uniform_prior != nullptr);
        CHECK_NEAR(uniform_prior->min(), param.lower_bound(), 1e-12);
        CHECK_NEAR(uniform_prior->max(), param.upper_bound(), 1e-12);

        // C# UnivariateDistribution.SetDefaultParameters (line 628):
        // IsPositive = lowers[i] == Tools.DoubleMachineEpsilon. For Normal that flags
        // sigma (scale, constraint lower bound == DoubleMachineEpsilon) positive and
        // leaves mu (location) unflagged.
        CHECK_EQ(param.is_positive(),
                 param.lower_bound() == bestfit::numerics::kDoubleMachineEpsilon);
    }
    CHECK_TRUE(!model.parameters()[0].is_positive());  // mu
    CHECK_TRUE(model.parameters()[1].is_positive());   // sigma

    // The distribution itself is left set to the initials (mirrors the C# setter behavior).
    std::vector<double> dist_params = model.distribution().get_parameters();
    for (std::size_t i = 0; i < dist_params.size(); ++i) {
        CHECK_NEAR(dist_params[i], model.parameters()[i].value(), 1e-12);
    }
}

void test_construct_from_owned_distribution_pointer() {
    auto normal = std::make_unique<Normal>();
    UnivariateDistributionModel model(std::move(normal), sample_data());

    CHECK_EQ(model.distribution_type(), UnivariateDistributionType::Normal);
    CHECK_EQ(model.number_of_parameters(), 2);
}

// T12's Jeffreys 1/scale prior override (prior_log_likelihood /
// pointwise_prior_log_likelihood): default-on, adds a `-log(scale)` term on top of the
// per-parameter Uniform priors for the scale parameter (Normal: index 1, sigma). This is the
// mechanism documented in fixtures/estimation/map_normal.json's and bayes_normal.json's source
// notes as the reason MAP/Bayesian posteriors diverge from the pure MLE.
void test_use_jeffreys_rule_for_scale_defaults_true() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    CHECK_TRUE(model.use_jeffreys_rule_for_scale());
}

void test_jeffreys_prior_adds_negative_log_scale_term() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    // Use the default-parameter initials (guaranteed inside the Uniform prior bounds, so the
    // per-parameter Uniform log_pdf terms are finite and identical between the two calls below).
    std::vector<double> p;
    for (const auto& param : model.parameters()) p.push_back(param.value());
    double sigma = p[1];

    CHECK_TRUE(model.use_jeffreys_rule_for_scale());  // still the default here
    double with_jeffreys = model.prior_log_likelihood(p);

    model.set_use_jeffreys_rule_for_scale(false);
    double without_jeffreys = model.prior_log_likelihood(p);

    // The two calls differ ONLY by the Jeffreys `-log(sigma)` term; the per-parameter Uniform
    // priors are unaffected by the toggle.
    CHECK_NEAR(with_jeffreys - without_jeffreys, -std::log(sigma), 1e-9);
}

void test_jeffreys_prior_nonpositive_scale_is_negative_infinity() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());
    CHECK_TRUE(model.use_jeffreys_rule_for_scale());

    std::vector<double> zero_scale{16000.0, 0.0};
    CHECK_TRUE(model.prior_log_likelihood(zero_scale) == -std::numeric_limits<double>::infinity());

    std::vector<double> negative_scale{16000.0, -1.0};
    CHECK_TRUE(model.prior_log_likelihood(negative_scale) ==
               -std::numeric_limits<double>::infinity());
}

void test_pointwise_prior_log_likelihood_appends_jeffreys_scale_component() {
    UnivariateDistributionModel model(UnivariateDistributionType::Normal, sample_data());

    std::vector<double> p;
    for (const auto& param : model.parameters()) p.push_back(param.value());
    double sigma = p[1];

    std::vector<bestfit::models::PriorComponent> components =
        model.pointwise_prior_log_likelihood(p);

    // One component per parameter, PLUS the trailing Jeffreys Scale component.
    CHECK_EQ(components.size(), model.parameters().size() + 1);

    const auto& jeffreys_component = components.back();
    CHECK_TRUE(jeffreys_component.type() == bestfit::models::PriorComponentType::JeffreysScalePrior);
    CHECK_NEAR(jeffreys_component.log_likelihood(), -std::log(sigma), 1e-12);

    // Self-consistency (per this method's header comment): the components must sum to
    // `prior_log_likelihood`.
    double sum = 0.0;
    for (const auto& c : components) sum += c.log_likelihood();
    CHECK_NEAR(sum, model.prior_log_likelihood(p), 1e-9);

    // Disabling the toggle drops the Jeffreys component, leaving one-per-parameter.
    model.set_use_jeffreys_rule_for_scale(false);
    std::vector<bestfit::models::PriorComponent> without_jeffreys =
        model.pointwise_prior_log_likelihood(p);
    CHECK_EQ(without_jeffreys.size(), model.parameters().size());
}

// ===========================================================================================
// M8: additive Numerics ports -- censored likelihood methods on UnivariateDistributionBase
// (UnivariateDistributionBase.cs:146-201 @ a2c4dbf) and Integration.GaussLegendre20
// (Integration.cs:62 @ a2c4dbf).
// ===========================================================================================

// Term-for-term contract of the four small censored-likelihood methods against the surface
// they are defined from (C# bodies are one-liners over LogPDF/LogCDF/LogCCDF/CDF).
void test_distribution_base_censored_log_likelihood_methods() {
    Normal n(100.0, 10.0);

    CHECK_NEAR(n.log_likelihood(105.0), n.log_pdf(105.0), 1e-15);
    CHECK_NEAR(n.log_likelihood_left_censored(90.0, 3), 3.0 * n.log_cdf(90.0), 1e-12);
    CHECK_NEAR(n.log_likelihood_right_censored(110.0, 2), 2.0 * n.log_ccdf(110.0), 1e-12);
    CHECK_NEAR(n.log_likelihood_intervals(95.0, 105.0),
               std::log(n.cdf(105.0) - n.cdf(95.0)), 1e-12);

    // Degenerate interval -> log(0) = -inf (the C# Math.Log(0) behavior).
    CHECK_TRUE(n.log_likelihood_intervals(105.0, 105.0) == -kInf);
}

// Transcribed: Test_Integration.Test_GaussLegendre20 (three oracle integrands).
void test_gauss_legendre20_oracles() {
    // x^3 over [0,1] = 0.25 (exact for degree <= 39)
    double e1 = Integration::gauss_legendre20([](double x) { return x * x * x; }, 0.0, 1.0);
    CHECK_NEAR(e1, 0.25, 1e-14);

    // Cosine over [0,1] = sin(1)
    double e2 = Integration::gauss_legendre20([](double x) { return std::cos(x); }, 0.0, 1.0);
    CHECK_NEAR(e2, std::sin(1.0), 1e-14);

    // log(x) over [1,2] = 2*ln(2) - 1
    double e3 = Integration::gauss_legendre20([](double x) { return std::log(x); }, 1.0, 2.0);
    CHECK_NEAR(e3, 2.0 * std::log(2.0) - 1.0, 1e-14);
}

// ===========================================================================================
// M8: transcriptions of UnivariateDistributionTests.cs (stationary surface).
// ===========================================================================================

// Deterministic Normal(100, 15) fixture (C# InlineNormalData: fixed RNG seed 12345; the
// bit-exact Mersenne Twister makes the C++ sample identical to the C# one).
const std::vector<double>& inline_normal_data() {
    static const std::vector<double> data = Normal(100.0, 15.0).generate_random_values(100, 12345);
    return data;
}

// Deterministic GEV(50, 15, 0.1) fixture (C# InlineGEVData, seed 23456).
const std::vector<double>& inline_gev_data() {
    static const std::vector<double> data =
        GeneralizedExtremeValue(50.0, 15.0, 0.1).generate_random_values(100, 23456);
    return data;
}

DataFrame make_data_frame(const std::vector<double>& data) {
    DataFrame df;
    df.set_exact_series(ExactSeries(data));
    return df;
}

const std::vector<double> kInlineNormalTrueParams{100.0, 15.0};

// C# Test_Constructor_CreatesValidModel.
void test_constructor_creates_valid_model() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    CHECK_TRUE(model.distribution_type() == UnivariateDistributionType::Normal);
    CHECK_TRUE(model.parameters().size() > 0);
}

// C# Test_Constructor_AllDistributionTypes: every supported type constructs with a non-null
// distribution and parameters (unsupported enum values throw in the C# ctor and are marked
// Inconclusive there; here the unsupported-type throw is asserted explicitly).
// GeneralizedNormal is on the C# whitelist but its distribution is not ported to the C++
// core yet, so it is left out of the construct loop (the C# test's catch-all Inconclusive
// gives the same tolerance); see the is_supported test below, where it still reports true.
void test_constructor_all_supported_distribution_types() {
    const UnivariateDistributionType supported[] = {
        UnivariateDistributionType::Exponential,
        UnivariateDistributionType::GammaDistribution,
        UnivariateDistributionType::GeneralizedExtremeValue,
        UnivariateDistributionType::GeneralizedLogistic,
        UnivariateDistributionType::GeneralizedPareto,
        UnivariateDistributionType::Gumbel,
        UnivariateDistributionType::KappaFour,
        UnivariateDistributionType::LnNormal,
        UnivariateDistributionType::Logistic,
        UnivariateDistributionType::LogNormal,
        UnivariateDistributionType::LogPearsonTypeIII,
        UnivariateDistributionType::Normal,
        UnivariateDistributionType::PearsonTypeIII,
        UnivariateDistributionType::Weibull,
    };
    for (UnivariateDistributionType t : supported) {
        UnivariateDistributionModel model(make_data_frame(inline_normal_data()), t);
        CHECK_TRUE(model.distribution().number_of_parameters() > 0);
        CHECK_TRUE(model.parameters().size() > 0);
    }

    // Unsupported type -> C# ArgumentOutOfRangeException -> std::out_of_range.
    bool threw = false;
    try {
        UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                          UnivariateDistributionType::TruncatedNormal);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

// C# Test_Constructor_NormalDistribution_HasTwoParameters.
void test_constructor_normal_has_two_parameters() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
}

// C# Test_Constructor_GEV_HasThreeParameters.
void test_constructor_gev_has_three_parameters() {
    UnivariateDistributionModel model(make_data_frame(inline_gev_data()),
                                      UnivariateDistributionType::GeneralizedExtremeValue);
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(3));
}

// C# Test_Parameters_HasCorrectCount.
void test_parameters_has_correct_count() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
}

// C# Test_SetParameterValues_UpdatesDistribution.
void test_set_parameter_values_updates_distribution() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    model.set_parameter_values({123.0, 45.0});

    std::vector<double> dist_params = model.distribution().get_parameters();
    CHECK_NEAR(dist_params[0], 123.0, 1e-10);
    CHECK_NEAR(dist_params[1], 45.0, 1e-10);
}

// C# Test_SetParameterValues_UpdatesModelParameters.
void test_set_parameter_values_updates_model_parameters() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    model.set_parameter_values({50.0, 10.0});

    CHECK_NEAR(model.parameters()[0].value(), 50.0, 1e-10);
    CHECK_NEAR(model.parameters()[1].value(), 10.0, 1e-10);
}

// C# Test_SetParameterValues_RoundTrip (Math.PI / Math.E).
void test_set_parameter_values_round_trip() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    std::vector<double> original{3.14159265358979312, 2.71828182845904509};
    model.set_parameter_values(original);

    for (std::size_t i = 0; i < original.size(); ++i) {
        CHECK_NEAR(model.parameters()[i].value(), original[i], 1e-15);
    }
}

// C# Test_LogLikelihood_ReturnsFiniteValue.
void test_log_likelihood_returns_finite_value() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    std::vector<double> params = kInlineNormalTrueParams;  // mutable lvalue (M14 signature)
    double ll = model.log_likelihood(params);

    CHECK_TRUE(!std::isnan(ll));
    CHECK_TRUE(!std::isinf(ll));
}

// C# Test_LogLikelihood_NegativeForContinuousDistributions.
void test_log_likelihood_negative_for_continuous_distributions() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    std::vector<double> params = kInlineNormalTrueParams;  // mutable lvalue (M14 signature)
    CHECK_TRUE(model.log_likelihood(params) < 0.0);
}

// C# Test_LogLikelihood_InvalidParameters_ReturnsNegativeInfinity.
void test_log_likelihood_invalid_parameters_returns_negative_infinity() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    std::vector<double> params{100.0, -10.0};  // mutable lvalue (M14 signature)
    CHECK_TRUE(model.log_likelihood(params) == -kInf);
}

// C# Test_LogPrior_WithUniformPrior_ReturnsFinite.
void test_log_prior_with_uniform_prior_returns_finite() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    model.parameters()[0].set_prior_distribution(
        std::make_unique<bestfit::numerics::distributions::Uniform>(-1e10, 1e10));
    model.parameters()[1].set_prior_distribution(
        std::make_unique<bestfit::numerics::distributions::Uniform>(0.001, 1e10));

    std::vector<double> params = kInlineNormalTrueParams;  // mutable lvalue (M14 signature)
    double log_prior = model.prior_log_likelihood(params);

    CHECK_TRUE(!std::isnan(log_prior));
    CHECK_TRUE(log_prior != -kInf);
}

// C# Test_LogPrior_WithInformativePrior_ReturnsFinite.
void test_log_prior_with_informative_prior_returns_finite() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    model.parameters()[0].set_prior_distribution(std::make_unique<Normal>(100.0, 10.0));
    model.parameters()[1].set_prior_distribution(
        std::make_unique<bestfit::numerics::distributions::Exponential>(0.001, 15.0));

    std::vector<double> params = kInlineNormalTrueParams;  // mutable lvalue (M14 signature)
    double log_prior = model.prior_log_likelihood(params);

    CHECK_TRUE(!std::isnan(log_prior));
    CHECK_TRUE(!std::isinf(log_prior));
}

// C# Test_LogPrior_OutsidePriorSupport_ReturnsNegativeInfinity.
void test_log_prior_outside_prior_support_returns_negative_infinity() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    model.parameters()[0].set_prior_distribution(
        std::make_unique<bestfit::numerics::distributions::Uniform>(0.0, 50.0));
    model.parameters()[1].set_prior_distribution(
        std::make_unique<bestfit::numerics::distributions::Uniform>(1.0, 100.0));

    // mu = 100 is outside [0, 50]
    std::vector<double> params{100.0, 15.0};  // mutable lvalue (M14 signature)
    CHECK_TRUE(model.prior_log_likelihood(params) == -kInf);
}

// C# Test_Validate_ValidModel_ReturnsTrue.
void test_validate_valid_model_returns_true() {
    UnivariateDistributionModel model(make_data_frame(inline_normal_data()),
                                      UnivariateDistributionType::Normal);

    ValidationResult result = model.validate();
    CHECK_TRUE(result.is_valid);
}

// C# Test_ChangeDistributionType_UpdatesParameters.
void test_change_distribution_type_updates_parameters() {
    UnivariateDistributionModel normal_model(make_data_frame(inline_normal_data()),
                                             UnivariateDistributionType::Normal);
    UnivariateDistributionModel gev_model(make_data_frame(inline_normal_data()),
                                          UnivariateDistributionType::GeneralizedExtremeValue);

    CHECK_EQ(normal_model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_EQ(gev_model.parameters().size(), static_cast<std::size_t>(3));
}

// ===========================================================================================
// M8: transcriptions of UnivariateDistributionExpandedTests.cs (stationary methods only).
// ===========================================================================================

// C# MakeNormalModel helper (deterministic, no RNG dependence).
UnivariateDistributionModel make_normal_model() {
    DataFrame df;
    df.set_exact_series(ExactSeries(
        {12500, 15300, 8900, 22100, 18700, 14200, 9800, 28500, 17400, 11600}));
    return UnivariateDistributionModel(std::move(df), UnivariateDistributionType::Normal);
}

// C# IsSupportedDistributionType_AllSupportedValues_ReturnTrue.
void test_is_supported_distribution_type_all_supported_values_return_true() {
    const UnivariateDistributionType supported[] = {
        UnivariateDistributionType::Exponential,
        UnivariateDistributionType::GammaDistribution,
        UnivariateDistributionType::GeneralizedExtremeValue,
        UnivariateDistributionType::GeneralizedLogistic,
        UnivariateDistributionType::GeneralizedNormal,
        UnivariateDistributionType::GeneralizedPareto,
        UnivariateDistributionType::Gumbel,
        UnivariateDistributionType::KappaFour,
        UnivariateDistributionType::LnNormal,
        UnivariateDistributionType::Logistic,
        UnivariateDistributionType::LogNormal,
        UnivariateDistributionType::LogPearsonTypeIII,
        UnivariateDistributionType::Normal,
        UnivariateDistributionType::PearsonTypeIII,
        UnivariateDistributionType::Weibull,
    };
    for (UnivariateDistributionType t : supported) {
        CHECK_TRUE(UnivariateDistributionModel::is_supported_distribution_type(t));
    }
    // And a not-in-whitelist type reports false.
    CHECK_TRUE(!UnivariateDistributionModel::is_supported_distribution_type(
        UnivariateDistributionType::TruncatedNormal));
}

// C# CreateDistribution_Normal_ReturnsNormalInstance.
void test_create_distribution_normal_returns_normal_instance() {
    auto dist = UnivariateDistributionModel::create_distribution(UnivariateDistributionType::Normal);
    CHECK_TRUE(dynamic_cast<Normal*>(dist.get()) != nullptr);
}

// C# CreateDistribution_Gumbel_ReturnsGumbelInstance.
void test_create_distribution_gumbel_returns_gumbel_instance() {
    auto dist = UnivariateDistributionModel::create_distribution(UnivariateDistributionType::Gumbel);
    CHECK_TRUE(dynamic_cast<Gumbel*>(dist.get()) != nullptr);
}

// C# CreateDistribution_GEV_ReturnsGEVInstance.
void test_create_distribution_gev_returns_gev_instance() {
    auto dist = UnivariateDistributionModel::create_distribution(
        UnivariateDistributionType::GeneralizedExtremeValue);
    CHECK_TRUE(dynamic_cast<GeneralizedExtremeValue*>(dist.get()) != nullptr);
}

// C# Validate_WellFormedModel_IsValid.
void test_validate_well_formed_model_is_valid() {
    UnivariateDistributionModel model = make_normal_model();
    CHECK_TRUE(model.validate().is_valid);
}

// C# Validate_NullDataFrame_IsInvalid. ADAPTED: the C# test sets `model.DataFrame = null!`;
// the C++ setter takes a DataFrame by value (no null), so the same validate branch is reached
// through a model constructed without a data frame (the C# parameterless ctor state).
void test_validate_no_data_frame_is_invalid() {
    UnivariateDistributionModel model;  // C# default ctor: LP3, no DataFrame
    model.set_distribution(std::make_unique<Normal>(100.0, 10.0));

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    bool mentions_data_frame = false;
    for (const auto& m : result.validation_messages) {
        if (m.find("DataFrame") != std::string::npos) mentions_data_frame = true;
    }
    CHECK_TRUE(mentions_data_frame);
}

// C# Validate_LogDistributionWithNegativeData_IsInvalid.
void test_validate_log_distribution_with_negative_data_is_invalid() {
    DataFrame df;
    df.set_exact_series(ExactSeries({-1.0, 5.0, 10.0, 20.0, 100.0}));
    UnivariateDistributionModel model(std::move(df), UnivariateDistributionType::LogNormal);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    bool mentions_log = false;
    for (const auto& m : result.validation_messages) {
        if (m.find("Log") != std::string::npos || m.find("non positive") != std::string::npos)
            mentions_log = true;
    }
    CHECK_TRUE(mentions_log);
}

// C# GetParameterValues_ReturnsParameterCount (stationary path: index is unused by the
// ConstantTrend-equivalent, see the model header).
void test_get_parameter_values_returns_parameter_count() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({100.0, 15.0});

    std::vector<double> values = model.get_parameter_values(0);
    CHECK_EQ(values.size(), static_cast<std::size_t>(2));
}

// C# GetParameterValues_ReturnsCurrentValues.
void test_get_parameter_values_returns_current_values() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({42.0, 7.5});

    std::vector<double> values = model.get_parameter_values(0);
    CHECK_NEAR(values[0], 42.0, 1e-12);
    CHECK_NEAR(values[1], 7.5, 1e-12);
}

// C# PointwiseDataLogLikelihood_LengthEqualsSeriesCount.
void test_pointwise_length_equals_series_count() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({16500.0, 6000.0});

    std::vector<double> pw = model.pointwise_data_log_likelihood({16500.0, 6000.0});
    CHECK_EQ(pw.size(), model.data_frame().exact_series().count());
}

// C# PointwiseDataLogLikelihood_SumEqualsDataLogLikelihood.
void test_pointwise_sum_equals_data_log_likelihood() {
    UnivariateDistributionModel model = make_normal_model();
    std::vector<double> p{16500.0, 6000.0};

    std::vector<double> pw = model.pointwise_data_log_likelihood(p);
    double total = model.data_log_likelihood(p);

    double sum = 0.0;
    for (double v : pw) sum += v;
    CHECK_NEAR(sum, total, 1e-6);
}

// C# LogLikelihood_EqualsDataPlusPrior.
void test_log_likelihood_equals_data_plus_prior() {
    UnivariateDistributionModel model = make_normal_model();
    std::vector<double> p{16500.0, 6000.0};

    double total = model.log_likelihood(p);
    double data_ll = model.data_log_likelihood(p);
    double prior_ll = model.prior_log_likelihood(p);

    CHECK_NEAR(total, data_ll + prior_ll, 1e-6);
}

// C# Distribution_SetToNewType_UpdatesDistributionType.
void test_distribution_set_to_new_type_updates_distribution_type() {
    UnivariateDistributionModel model = make_normal_model();

    model.set_distribution(
        UnivariateDistributionModel::create_distribution(UnivariateDistributionType::Gumbel));

    CHECK_TRUE(model.distribution_type() == UnivariateDistributionType::Gumbel);
}

// ===========================================================================================
// M8: transcriptions of ThresholdLikelihoodGuardTests.cs -- the ModelKind.Univariate /
// stationary rows only. Deferred with the sibling models: the Mixture rows (M10), the
// CompetingRisks rows (M11), the PointProcess rows + the PointProcess-only test (M12), and
// the nonstationary split-threshold likelihood test (M9).
// ===========================================================================================

// C# CreateThresholdDataFrame + CreateUnivariateModel, then SetThresholdCounts (reflection
// upstream; the C++ ThresholdData setters are public, so the counts are set directly --
// crucially AFTER the model boundary, so nothing reprocesses/overwrites the edge case).
UnivariateDistributionModel make_threshold_guard_model() {
    DataFrame df;
    df.threshold_series().add(ThresholdData(2000, 2002, 100.0));

    UnivariateDistributionModel model;
    model.set_use_default_flat_priors(false);
    model.set_distribution(std::make_unique<Normal>(100.0, 10.0));
    model.set_data_frame(std::move(df));
    return model;
}

// C# ThresholdLikelihood_OneZeroCountSide_RemainsFiniteAndPointwiseConsistent,
// DataRows (Univariate, 0, 2) and (Univariate, 2, 0).
void check_threshold_one_zero_count_side(int number_below, int number_above) {
    UnivariateDistributionModel model = make_threshold_guard_model();
    ThresholdData& threshold = model.data_frame().threshold_series()[0];
    threshold.set_number_below(number_below);
    threshold.set_number_above(number_above);

    std::vector<double> p{100.0, 10.0};
    double total = model.data_log_likelihood(p);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(p);
    std::vector<DataComponent> components = model.pointwise_data_log_likelihood_components(p);

    CHECK_TRUE(!std::isnan(total) && !std::isinf(total));
    CHECK_EQ(pointwise.size(), static_cast<std::size_t>(1));
    CHECK_EQ(components.size(), static_cast<std::size_t>(1));
    CHECK_TRUE(!std::isnan(pointwise[0]) && !std::isinf(pointwise[0]));
    double sum = 0.0;
    for (double v : pointwise) sum += v;
    CHECK_NEAR(total, sum, 1e-10);
    CHECK_NEAR(pointwise[0], components[0].log_likelihood(), 1e-10);
    CHECK_EQ(components[0].count(), number_below + number_above);
}

void test_threshold_likelihood_one_zero_count_side_remains_finite_and_pointwise_consistent() {
    check_threshold_one_zero_count_side(0, 2);
    check_threshold_one_zero_count_side(2, 0);
}

// C# ThresholdLikelihood_BothCountsZero_ContributesZero, DataRow(ModelKind.Univariate).
void test_threshold_likelihood_both_counts_zero_contributes_zero() {
    UnivariateDistributionModel model = make_threshold_guard_model();
    ThresholdData& threshold = model.data_frame().threshold_series()[0];
    threshold.set_number_below(0);
    threshold.set_number_above(0);

    std::vector<double> p{100.0, 10.0};
    double total = model.data_log_likelihood(p);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(p);
    std::vector<DataComponent> components = model.pointwise_data_log_likelihood_components(p);

    CHECK_NEAR(total, 0.0, 1e-12);
    CHECK_EQ(pointwise.size(), static_cast<std::size_t>(1));
    CHECK_NEAR(pointwise[0], 0.0, 1e-12);
    CHECK_EQ(components.size(), static_cast<std::size_t>(1));
    CHECK_NEAR(components[0].log_likelihood(), 0.0, 1e-12);
    CHECK_EQ(components[0].count(), 0);
}

// ===========================================================================================
// M8: internal-support consistency checks for the full stationary censored likelihood
// (exact / low-outlier / uncertain / interval / threshold). The C# source is the oracle for
// structure; these checks recompute each branch term-for-term from the same primitives.
// ===========================================================================================

// The M4->M8 contract: set_data_frame() must run process_threshold_series() at the model
// boundary (the effective C# once-per-mutation event cadence), and the likelihood entry
// points must NOT reprocess (the C# StationaryData_LogLikelihood never calls
// ProcessThresholdSeries -- see the base header's cadence note).
void test_set_data_frame_processes_threshold_series_at_model_boundary() {
    DataFrame df;
    ThresholdData threshold(2000, 2002, 100.0);
    threshold.set_number_above(1);
    df.threshold_series().add(threshold);

    UnivariateDistributionModel model;
    model.set_use_default_flat_priors(false);
    model.set_distribution(std::make_unique<Normal>(100.0, 10.0));
    model.set_data_frame(std::move(df));

    // Duration 3, NumberAbove 1, no explicit points -> NumberBelow = 2.
    CHECK_EQ(model.data_frame().threshold_series()[0].number_below(), 2);
    CHECK_EQ(model.data_frame().threshold_series()[0].number_above(), 1);

    // Zero the counts by hand; the likelihood entry points must not reprocess them.
    model.data_frame().threshold_series()[0].set_number_below(0);
    model.data_frame().threshold_series()[0].set_number_above(0);
    std::vector<double> params{100.0, 10.0};  // mutable lvalue (M14 signature)
    CHECK_NEAR(model.data_log_likelihood(params), 0.0, 1e-12);
    CHECK_EQ(model.data_frame().threshold_series()[0].number_below(), 0);
    CHECK_EQ(model.data_frame().threshold_series()[0].number_above(), 0);
}

// Mixed-type frame: one plain exact, one low-outlier exact, one uncertain, one interval, one
// threshold window. Every branch of the stationary type-switch is exercised and recomputed.
void test_mixed_censored_data_log_likelihood_matches_componentwise_recomputation() {
    DataFrame df;
    df.exact_series().add(ExactData(0, 105.0));
    df.exact_series().add(ExactData(1, 80.0, 0.0, /*is_low_outlier=*/true));
    df.uncertain_series().add(UncertainData(2, std::make_unique<Normal>(102.0, 5.0)));
    df.interval_series().add(IntervalData(3, 90.0, 95.0, 100.0));
    df.threshold_series().add(ThresholdData(2000, 2002, 100.0));
    df.set_low_outlier_threshold(95.0);

    UnivariateDistributionModel model;
    model.set_use_default_flat_priors(false);
    model.set_distribution(std::make_unique<Normal>(100.0, 10.0));
    model.set_data_frame(std::move(df));

    // Post-boundary: threshold window (duration 3) has NumberBelow = 3 (no explicit points
    // inside its 2000-2002 window; the other series sit at indexes 0-3).
    CHECK_EQ(model.data_frame().threshold_series()[0].number_below(), 3);
    // Adjust to a two-sided window for coverage of both censored sides.
    model.data_frame().threshold_series()[0].set_number_below(2);
    model.data_frame().threshold_series()[0].set_number_above(1);

    std::vector<double> p{100.0, 10.0};
    Normal fitted(100.0, 10.0);

    // Exact.
    double expected_exact = fitted.log_pdf(105.0);
    // Low outlier -> LogLikelihood_LeftCensored(LowOutlierThreshold, 1).
    double expected_low_outlier = fitted.log_cdf(95.0);
    // Uncertain -> log(GaussLegendre20(me.pdf * model.pdf) / mass) over the 1e-8 window.
    Normal me(102.0, 5.0);
    double a = me.inverse_cdf(1e-8);
    double b = me.inverse_cdf(1.0 - 1e-8);
    double mass = (1.0 - 1e-8) - 1e-8;
    double ep = Integration::gauss_legendre20(
                    [&](double q) { return me.pdf(q) * fitted.pdf(q); }, a, b) /
                mass;
    double expected_uncertain = std::log(ep);
    // Interval -> log(CDF(upper) - CDF(lower)).
    double expected_interval = std::log(fitted.cdf(100.0) - fitted.cdf(90.0));
    // Threshold -> below * LogCDF + above * LogCCDF.
    double expected_threshold = 2.0 * fitted.log_cdf(100.0) + 1.0 * fitted.log_ccdf(100.0);

    double expected_total = expected_exact + expected_low_outlier + expected_uncertain +
                            expected_interval + expected_threshold;

    CHECK_NEAR(model.data_log_likelihood(p), expected_total, 1e-9);

    // Pointwise: C# ordering is exact, uncertain, interval, threshold.
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(p);
    CHECK_EQ(pointwise.size(), static_cast<std::size_t>(5));
    CHECK_NEAR(pointwise[0], expected_exact, 1e-9);
    CHECK_NEAR(pointwise[1], expected_low_outlier, 1e-9);
    CHECK_NEAR(pointwise[2], expected_uncertain, 1e-9);
    CHECK_NEAR(pointwise[3], expected_interval, 1e-9);
    CHECK_NEAR(pointwise[4], expected_threshold, 1e-9);

    // Components: types, values, counts, and names per the C# component construction.
    std::vector<DataComponent> components = model.pointwise_data_log_likelihood_components(p);
    CHECK_EQ(components.size(), static_cast<std::size_t>(5));
    for (std::size_t i = 0; i < components.size(); ++i) {
        CHECK_EQ(components[i].index(), static_cast<int>(i));
        CHECK_NEAR(components[i].log_likelihood(), pointwise[i], 1e-12);
    }
    CHECK_TRUE(components[0].type() == DataComponentType::Exact);
    CHECK_NEAR(components[0].value(), 105.0, 1e-12);
    // Low outlier component reports the low-outlier threshold as its value (C# 1450).
    CHECK_TRUE(components[1].type() == DataComponentType::Exact);
    CHECK_NEAR(components[1].value(), 95.0, 1e-12);
    // Uncertain component reports the measurement-error distribution mean.
    CHECK_TRUE(components[2].type() == DataComponentType::Uncertain);
    CHECK_NEAR(components[2].value(), 102.0, 1e-12);
    // Interval component reports the midpoint.
    CHECK_TRUE(components[3].type() == DataComponentType::Interval);
    CHECK_NEAR(components[3].value(), 95.0, 1e-12);
    // Threshold component: LeftCensored with the combined count and "start-end" name.
    CHECK_TRUE(components[4].type() == DataComponentType::LeftCensored);
    CHECK_NEAR(components[4].value(), 100.0, 1e-12);
    CHECK_EQ(components[4].count(), 3);
    CHECK_TRUE(components[4].name().has_value());
    CHECK_TRUE(*components[4].name() == "2000-2002");
}

// Invalid parameters against the mixed frame: the pointwise array covers the total explicit
// record count and the component list carries the per-type placeholder values (C# 1410-1431).
void test_mixed_censored_data_invalid_parameters_placeholders() {
    DataFrame df;
    df.exact_series().add(ExactData(0, 105.0));
    df.uncertain_series().add(UncertainData(2, std::make_unique<Normal>(102.0, 5.0)));
    df.interval_series().add(IntervalData(3, 90.0, 95.0, 100.0));
    df.threshold_series().add(ThresholdData(2000, 2002, 100.0));

    UnivariateDistributionModel model;
    model.set_use_default_flat_priors(false);
    model.set_distribution(std::make_unique<Normal>(100.0, 10.0));
    model.set_data_frame(std::move(df));

    std::vector<double> invalid_p{100.0, -1.0};
    CHECK_TRUE(model.data_log_likelihood(invalid_p) == -kInf);

    std::vector<double> pointwise = model.pointwise_data_log_likelihood(invalid_p);
    CHECK_EQ(pointwise.size(), static_cast<std::size_t>(4));
    for (double v : pointwise) CHECK_TRUE(v == -kInf);

    std::vector<DataComponent> components =
        model.pointwise_data_log_likelihood_components(invalid_p);
    CHECK_EQ(components.size(), static_cast<std::size_t>(4));
    for (const auto& c : components) CHECK_TRUE(c.log_likelihood() == -kInf);
    CHECK_TRUE(components[0].type() == DataComponentType::Exact);
    CHECK_TRUE(components[1].type() == DataComponentType::Uncertain);
    CHECK_NEAR(components[1].value(), 102.0, 1e-12);  // ME distribution mean
    CHECK_TRUE(components[2].type() == DataComponentType::Interval);
    CHECK_NEAR(components[2].value(), 95.0, 1e-12);  // interval midpoint
    CHECK_TRUE(components[3].type() == DataComponentType::LeftCensored);
    CHECK_EQ(components[3].count(), 3);  // NumberBelow + NumberAbove after boundary processing
}

// ===========================================================================================
// M8: base-class (UnivariateDistributionModelBase) surface.
// ===========================================================================================

// ===========================================================================================
// M9: nonstationary state + trends, quantile priors, Clone, seeded simulation.
// Transcriptions: the 17 UnivariateDistributionExpandedTests.cs methods M8 deferred, every
// NonstationaryLogLikelihoodHotPathTests.cs method, and ThresholdLikelihoodGuardTests.cs'
// NonstationaryUnivariateThresholdLikelihood_SplitThresholdsRemainFinite. Internal-support
// checks (clearly marked) pin the StepFunction IsPositive ledger item, the quantile-prior
// defaults, the trend parameter plumbing, and the simulation delegation stream.
// ===========================================================================================

using bestfit::models::trend_functions::TrendModelType;

// C# IsNonstationary_SetTrue_PopulatesTrendModels.
void test_is_nonstationary_set_true_populates_trend_models() {
    UnivariateDistributionModel model = make_normal_model();
    CHECK_TRUE(!model.is_nonstationary());  // default should be stationary

    model.set_is_nonstationary(true);

    CHECK_TRUE(model.is_nonstationary());
    CHECK_TRUE(model.trend_models().size() >=
               static_cast<std::size_t>(model.distribution().number_of_parameters()));
}

// C# IsNonstationary_TrueThenFalse_TrendModelsAreConstant.
void test_is_nonstationary_true_then_false_trend_models_are_constant() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_is_nonstationary(true);

    model.set_is_nonstationary(false);

    CHECK_TRUE(!model.is_nonstationary());
    for (const auto& t : model.trend_models()) {
        CHECK_TRUE(t->type() == TrendModelType::Constant);
    }
}
// (C# IsNonstationary_SetToCurrentValue_DoesNotRaisePropertyChanged is SKIPPED: INPC.)

// C# Alpha_SetterRoundTrips (the PropertyChanged half is INPC, skipped).
void test_alpha_setter_round_trips() {
    UnivariateDistributionModel model = make_normal_model();
    CHECK_NEAR(model.alpha(), 0.5, 1e-15);  // C# _alpha default

    model.set_alpha(0.05);

    CHECK_NEAR(model.alpha(), 0.05, 1e-12);
}

// C# ParameterTimeIndex_SetterRoundTrips.
void test_parameter_time_index_setter_round_trips() {
    UnivariateDistributionModel model = make_normal_model();

    model.set_parameter_time_index(1995);

    CHECK_EQ(model.parameter_time_index(), 1995);
}

// C# Validate_NonstationaryWithZeroAlpha_IsInvalid.
void test_validate_nonstationary_with_zero_alpha_is_invalid() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_is_nonstationary(true);
    model.set_alpha(0.0);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    bool mentions_alpha = false;
    for (const auto& m : result.validation_messages) {
        if (m.find("Alpha") != std::string::npos) mentions_alpha = true;
    }
    CHECK_TRUE(mentions_alpha);
}

// C# Validate_NonstationaryWithAlphaOne_IsInvalid.
void test_validate_nonstationary_with_alpha_one_is_invalid() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_is_nonstationary(true);
    model.set_alpha(1.0);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    bool mentions_alpha = false;
    for (const auto& m : result.validation_messages) {
        if (m.find("Alpha") != std::string::npos) mentions_alpha = true;
    }
    CHECK_TRUE(mentions_alpha);
}

// Internal support: the C# Validate nonstationary ParameterTimeIndex range branch
// (UnivariateDistribution.cs 2276-2286) -- outside [firstIndex, lastIndex + 100] is invalid.
void test_validate_nonstationary_parameter_time_index_out_of_range_is_invalid() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_is_nonstationary(true);  // exact indexes 0..9 -> valid range [0, 109]
    model.set_parameter_time_index(-50);

    ValidationResult result = model.validate();

    CHECK_TRUE(!result.is_valid);
    bool mentions_pti = false;
    for (const auto& m : result.validation_messages) {
        if (m.find("ParameterTimeIndex") != std::string::npos) mentions_pti = true;
    }
    CHECK_TRUE(mentions_pti);
}

// C# GenerateRandomValues_RequestedSize_ReturnsArrayOfThatLength.
void test_generate_random_values_requested_size() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({100.0, 15.0});

    std::vector<double> samples = model.generate_random_values(50, /*seed=*/42);

    CHECK_EQ(samples.size(), static_cast<std::size_t>(50));
}

// C# GenerateRandomValues_SameSeed_IsReproducible.
void test_generate_random_values_same_seed_is_reproducible() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({100.0, 15.0});

    std::vector<double> a = model.generate_random_values(20, /*seed=*/7);
    std::vector<double> b = model.generate_random_values(20, /*seed=*/7);

    for (std::size_t i = 0; i < 20; ++i) CHECK_NEAR(a[i], b[i], 1e-12);
}

// C# GenerateRandomValues_DifferentSeeds_ProducesDifferentSamples.
void test_generate_random_values_different_seeds_produce_different_samples() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({100.0, 15.0});

    std::vector<double> a = model.generate_random_values(20, /*seed=*/1);
    std::vector<double> b = model.generate_random_values(20, /*seed=*/2);

    bool any_diff = false;
    for (std::size_t i = 0; i < 20; ++i) {
        if (std::abs(a[i] - b[i]) > 1e-9) {
            any_diff = true;
            break;
        }
    }
    CHECK_TRUE(any_diff);
}

// C# GenerateRandomValues_SampleSizeZero_Throws and _NegativeSampleSize_Throws
// (ArgumentOutOfRangeException -> std::out_of_range per this file's exception mapping).
void test_generate_random_values_non_positive_sample_size_throws() {
    UnivariateDistributionModel model = make_normal_model();

    bool threw = false;
    try {
        model.generate_random_values(0, /*seed=*/1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK_TRUE(threw);

    threw = false;
    try {
        model.generate_random_values(-5, /*seed=*/1);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

// Internal support (seeded-simulation determinism + route pin): the model delegates to
// Distribution.GenerateRandomValues (C# 2310) -- the Numerics inverse-CDF Mersenne Twister
// stream -- so the seeded model sample is bit-identical to the distribution's own.
void test_generate_random_values_matches_distribution_stream() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_parameter_values({100.0, 15.0});

    std::vector<double> expected = Normal(100.0, 15.0).generate_random_values(50, 42);
    std::vector<double> actual = model.generate_random_values(50, 42);

    CHECK_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < actual.size(); ++i) CHECK_TRUE(actual[i] == expected[i]);
}

// C# Clone_ProducesDistinctInstance (AreNotSame -> distinct objects by construction here).
void test_clone_produces_distinct_instance() {
    UnivariateDistributionModel original = make_normal_model();

    UnivariateDistributionModel clone = original.clone();

    CHECK_TRUE(&clone != &original);
    CHECK_TRUE(&clone.distribution() != &original.distribution());
}

// C# Clone_PreservesDistributionType.
void test_clone_preserves_distribution_type() {
    UnivariateDistributionModel original = make_normal_model();

    UnivariateDistributionModel clone = original.clone();

    CHECK_TRUE(clone.distribution_type() == original.distribution_type());
}

// C# Clone_PreservesParameterValues.
void test_clone_preserves_parameter_values() {
    UnivariateDistributionModel original = make_normal_model();
    original.set_parameter_values({105.0, 17.5});

    UnivariateDistributionModel clone = original.clone();

    CHECK_EQ(clone.parameters().size(), original.parameters().size());
    for (std::size_t i = 0; i < original.parameters().size(); ++i) {
        CHECK_NEAR(clone.parameters()[i].value(), original.parameters()[i].value(), 1e-12);
    }
}

// C# fidelity pin (M9 review): the C# clone routes through the (DataFrame, Distribution)
// ctor, whose `Distribution = distribution.Clone()` keeps the cloned Distribution's current
// parameters (the C# SetDefaultParameters never calls Distribution.SetParameters). The C++
// ctor's retained Phase 4 deviation syncs the distribution to constraint initials, so
// clone() must re-sync the clone's distribution to the original's current parameters or a
// seeded simulation from the clone silently diverges from the original's.
void test_clone_syncs_distribution_parameters() {
    UnivariateDistributionModel original = make_normal_model();
    original.set_parameter_values({105.0, 17.5});

    UnivariateDistributionModel clone = original.clone();

    const std::vector<double> expected = original.distribution().get_parameters();
    const std::vector<double> actual = clone.distribution().get_parameters();
    CHECK_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) CHECK_TRUE(actual[i] == expected[i]);

    // Seeded ISimulatable streams bit-match between original and clone (the C# end state).
    const std::vector<double> from_original = original.generate_random_values(3, 42);
    const std::vector<double> from_clone = clone.generate_random_values(3, 42);
    CHECK_EQ(from_clone.size(), from_original.size());
    for (std::size_t i = 0; i < from_original.size(); ++i)
        CHECK_TRUE(from_clone[i] == from_original[i]);
}

// C# Clone_PreservesIsNonstationary.
void test_clone_preserves_is_nonstationary() {
    UnivariateDistributionModel original = make_normal_model();
    original.set_is_nonstationary(true);

    UnivariateDistributionModel clone = original.clone();

    CHECK_TRUE(clone.is_nonstationary() == original.is_nonstationary());
    CHECK_EQ(clone.trend_models().size(), original.trend_models().size());
}

// C# Clone_PreservesAlpha.
void test_clone_preserves_alpha() {
    UnivariateDistributionModel original = make_normal_model();
    original.set_alpha(0.25);

    UnivariateDistributionModel clone = original.clone();

    CHECK_NEAR(clone.alpha(), 0.25, 1e-12);
}

// C# Clone_ParametersAreIndependent. ADAPTED (the M9 decision the M8 report flagged): the
// C# Clone ALIASES the DataFrame (the same frame object is shared), which the value-typed
// move-only C++ frame cannot do -- clone() DEEP-COPIES the frame instead (divergence
// documented in the model header). The parameter-independence assertion below is the C# test
// unchanged; the frame independence is additionally pinned as the C++-only consequence.
void test_clone_parameters_are_independent() {
    UnivariateDistributionModel original = make_normal_model();
    original.set_parameter_values({100.0, 15.0});

    UnivariateDistributionModel clone = original.clone();
    clone.parameters()[0].set_value(999.0);

    CHECK_NEAR(original.parameters()[0].value(), 100.0, 1e-12);

    // C++-only pin: the deep-copied frame is independent too (in C# it would be shared).
    clone.data_frame().set_low_outlier_threshold(123.0);
    CHECK_NEAR(original.data_frame().low_outlier_threshold(), 0.0, 1e-12);
}

// ===========================================================================================
// M9: NonstationaryLogLikelihoodHotPathTests.cs (all five methods, values unaltered).
// ===========================================================================================

constexpr int kHotPathFixtureSize = 60;

// C# InlineFloodData = new Normal(15000, 5000).GenerateRandomValues(60, 42) (bit-exact via
// the shared Mersenne Twister).
const std::vector<double>& inline_flood_data() {
    static const std::vector<double> data =
        Normal(15000.0, 5000.0).generate_random_values(kHotPathFixtureSize, 42);
    return data;
}

// C# CreateDataFrame: ExactData(1960 + i, value).
DataFrame create_hot_path_data_frame() {
    DataFrame df;
    for (int i = 0; i < kHotPathFixtureSize; ++i) {
        df.exact_series().add(ExactData(1960 + i, inline_flood_data()[static_cast<std::size_t>(i)]));
    }
    return df;
}

// C# CreateNormalLinearMu: Normal NS distribution with a Linear trend on the location
// parameter; FullTimeSeries pre-built like UnivariateAnalysis.RunAsync does before MCMC.
UnivariateDistributionModel create_normal_linear_mu() {
    UnivariateDistributionModel dist(create_hot_path_data_frame(),
                                     UnivariateDistributionType::Normal);
    dist.set_is_nonstationary(true);
    dist.set_trend_model(0, TrendModelType::Linear);  // Linear on mu
    dist.data_frame().create_full_time_series();
    return dist;
}

// C# CallNonstationaryDataLogLikelihood: the same path BayesianAnalysis uses.
double call_nonstationary_data_log_likelihood(const UnivariateDistributionModel& dist,
                                              const std::vector<double>& parameters) {
    std::unique_ptr<bestfit::numerics::distributions::UnivariateDistributionBase> working =
        dist.distribution().clone();
    return dist.nonstationary_data_log_likelihood(*working, parameters);
}

// C# NonstationaryDataLogLikelihood_MatchesSumOfPointwise.
void test_nonstationary_data_log_likelihood_matches_sum_of_pointwise() {
    UnivariateDistributionModel dist = create_normal_linear_mu();
    // [mu_intercept, mu_slope, sigma]
    std::vector<double> parameters{15000.0, 50.0, 5000.0};

    double total = call_nonstationary_data_log_likelihood(dist, parameters);
    std::vector<double> pointwise = dist.pointwise_data_log_likelihood(parameters);

    CHECK_TRUE(std::isfinite(total));
    CHECK_EQ(pointwise.size(), static_cast<std::size_t>(kHotPathFixtureSize));
    double sum = 0.0;
    for (double v : pointwise) sum += v;
    CHECK_NEAR(total, sum, 1e-6);
}

// C# NonstationaryDataLogLikelihood_RepeatedCallsReturnSameValue (bit-for-bit).
void test_nonstationary_data_log_likelihood_repeated_calls_return_same_value() {
    UnivariateDistributionModel dist = create_normal_linear_mu();
    std::vector<double> parameters{15000.0, 50.0, 5000.0};

    double a = call_nonstationary_data_log_likelihood(dist, parameters);
    double b = call_nonstationary_data_log_likelihood(dist, parameters);
    double c = call_nonstationary_data_log_likelihood(dist, parameters);

    CHECK_TRUE(a == b);
    CHECK_TRUE(b == c);
}

// C# NonstationaryDataLogLikelihood_DifferentParametersAreIndependent.
void test_nonstationary_data_log_likelihood_different_parameters_are_independent() {
    UnivariateDistributionModel dist = create_normal_linear_mu();
    std::vector<double> params_a{15000.0, 50.0, 5000.0};
    std::vector<double> params_b{14000.0, 80.0, 6000.0};

    double a1 = call_nonstationary_data_log_likelihood(dist, params_a);
    double b = call_nonstationary_data_log_likelihood(dist, params_b);
    double a2 = call_nonstationary_data_log_likelihood(dist, params_a);

    CHECK_TRUE(a1 == a2);
    CHECK_TRUE(a1 != b);
}

// C# NonstationaryDataLogLikelihood_HotLoopSmokeTest (10,000 calls x 60 points < 10 s).
void test_nonstationary_data_log_likelihood_hot_loop_smoke_test() {
    UnivariateDistributionModel dist = create_normal_linear_mu();
    std::vector<double> parameters{15000.0, 50.0, 5000.0};
    call_nonstationary_data_log_likelihood(dist, parameters);  // warm-up (C# JIT parity)

    constexpr int kIterations = 10000;
    auto start = std::chrono::steady_clock::now();
    double total_sink = 0.0;
    for (int i = 0; i < kIterations; ++i) {
        parameters[0] = 15000.0 + (i % 7);
        total_sink += call_nonstationary_data_log_likelihood(dist, parameters);
    }
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;

    CHECK_TRUE(std::isfinite(total_sink));
    CHECK_TRUE(elapsed.count() < 10.0);
}

// C# NonstationaryPointwise_LengthAndSum_AreCorrect.
void test_nonstationary_pointwise_length_and_sum_are_correct() {
    UnivariateDistributionModel dist = create_normal_linear_mu();
    std::vector<double> parameters{15000.0, 50.0, 5000.0};

    std::vector<double> pointwise = dist.pointwise_data_log_likelihood(parameters);
    double sum = 0.0;
    for (double v : pointwise) sum += v;

    std::unique_ptr<bestfit::numerics::distributions::UnivariateDistributionBase> working =
        dist.distribution().clone();
    double total = dist.nonstationary_data_log_likelihood(*working, parameters);

    CHECK_EQ(pointwise.size(), dist.data_frame().full_time_series().size());
    CHECK_NEAR(total, sum, 1e-6);
}

// ===========================================================================================
// M9: ThresholdLikelihoodGuardTests.cs -- the nonstationary split-threshold likelihood half
// (the DataFrame-state half was transcribed in M4's test_data_frame.cpp).
// ===========================================================================================

// C# NonstationaryUnivariateThresholdLikelihood_SplitThresholdsRemainFinite.
void test_nonstationary_split_threshold_likelihood_remains_finite() {
    DataFrame df;
    ThresholdData threshold(2000, 2002, 100.0);
    threshold.set_number_above(1);
    df.threshold_series().add(threshold);

    UnivariateDistributionModel model;
    model.set_use_default_flat_priors(false);
    model.set_distribution(std::make_unique<Normal>(100.0, 10.0));
    model.set_data_frame(std::move(df));
    model.set_is_nonstationary(true);

    std::vector<double> parameters{100.0, 10.0};
    double total = model.data_log_likelihood(parameters);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(parameters);
    std::vector<DataComponent> components =
        model.pointwise_data_log_likelihood_components(parameters);

    CHECK_TRUE(!std::isnan(total) && !std::isinf(total));
    CHECK_EQ(pointwise.size(), static_cast<std::size_t>(3));
    double sum = 0.0;
    for (double v : pointwise) sum += v;
    CHECK_NEAR(total, sum, 1e-10);
    CHECK_EQ(components.size(), static_cast<std::size_t>(3));
    for (const auto& component : components) CHECK_EQ(component.count(), 1);
    for (double v : pointwise) CHECK_TRUE(!std::isnan(v) && !std::isinf(v));
}

// ===========================================================================================
// M9: internal-support checks -- trend plumbing, the StepFunction IsPositive ledger pin,
// GetNonstationaryReturnLevel, and the quantile-prior default/likelihood surface.
// ===========================================================================================

// SetTrendModel replaces the indexed trend, rebuilds the Parameters list from all trends,
// throws std::out_of_range for a bad index (C# ArgumentOutOfRangeException), and the
// IsNonstationary set-false branch resets every trend to Constant (C# 408-417).
void test_set_trend_model_replaces_trend_and_rebuilds_parameters() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_is_nonstationary(true);
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));

    model.set_trend_model(0, TrendModelType::Linear);

    CHECK_TRUE(model.trend_models()[0]->type() == TrendModelType::Linear);
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(3));  // linear mu + const sigma

    // Linear slope defaults (C# 669-675): value 0, symmetric +-tdelta1 bounds.
    const bestfit::models::ModelParameter& slope = model.trend_models()[0]->parameters()[1];
    CHECK_NEAR(slope.value(), 0.0, 1e-12);
    CHECK_NEAR(slope.lower_bound(), -slope.upper_bound(), 1e-12);

    bool threw = false;
    try {
        model.set_trend_model(5, TrendModelType::Linear);
    } catch (const std::out_of_range&) {
        threw = true;
    }
    CHECK_TRUE(threw);

    model.set_is_nonstationary(false);
    for (const auto& t : model.trend_models()) CHECK_TRUE(t->type() == TrendModelType::Constant);
    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
}

// >>> LEDGER MUST (M8 review hand-off): the C# StepFunction trend branches assign
// Parameters[1].IsPositive = lowers[i] == Tools.DoubleMachineEpsilon (SetTrendModel line
// ~986 and SetDefaultParameters line ~728). For Normal, sigma's constraint lower bound IS
// DoubleMachineEpsilon and mu's is not, so the step level (mu_2) is flagged positive only on
// the sigma trend.
void test_step_function_trend_sets_is_positive_from_constraint_lower_bound() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_is_nonstationary(true);

    model.set_trend_model(1, TrendModelType::StepFunction);  // sigma (scale)
    const auto& sigma_trend = *model.trend_models()[1];
    CHECK_TRUE(sigma_trend.type() == TrendModelType::StepFunction);
    CHECK_TRUE(sigma_trend.parameters()[0].is_positive());
    CHECK_TRUE(sigma_trend.parameters()[1].is_positive());  // the ledger MUST (C# ~986)

    model.set_trend_model(0, TrendModelType::StepFunction);  // mu (location)
    const auto& mu_trend = *model.trend_models()[0];
    CHECK_TRUE(!mu_trend.parameters()[0].is_positive());
    CHECK_TRUE(!mu_trend.parameters()[1].is_positive());
    // Change point defaults: value tmid, bounds [tmin, tmax] (indexes 0..9 -> 4.5, [0, 9]).
    CHECK_NEAR(mu_trend.parameters()[2].value(), 4.5, 1e-12);
    CHECK_NEAR(mu_trend.parameters()[2].lower_bound(), 0.0, 1e-12);
    CHECK_NEAR(mu_trend.parameters()[2].upper_bound(), 9.0, 1e-12);

    // Re-running SetDefaultParameters exercises the C# ~728 StepFunction branch on the
    // already-installed trends; the IsPositive assignment must survive it.
    model.set_default_parameters();
    CHECK_TRUE(model.trend_models()[1]->parameters()[1].is_positive());
    CHECK_TRUE(!model.trend_models()[0]->parameters()[1].is_positive());
}

// Nonstationary parameter plumbing: GetParameterValues predicts through the trends,
// SetParameterValues syncs the distribution at ParameterTimeIndex (C# 1998/2006/2020), and
// the ParameterTimeIndex setter re-syncs (C# 456-463).
void test_nonstationary_parameter_plumbing_syncs_distribution_at_time_index() {
    UnivariateDistributionModel model = create_normal_linear_mu();
    // IsNonstationary set-true: PTI = ceil((1960 + 2019) / 2) = 1990 (C# 425).
    CHECK_EQ(model.parameter_time_index(), 1990);

    model.set_parameter_values({15000.0, 50.0, 5000.0});

    std::vector<double> v0 = model.get_parameter_values(1960);
    CHECK_NEAR(v0[0], 15000.0, 1e-9);
    CHECK_NEAR(v0[1], 5000.0, 1e-9);
    std::vector<double> v1 = model.get_parameter_values(2000);
    CHECK_NEAR(v1[0], 17000.0, 1e-9);  // 15000 + 50 * (2000 - 1960)

    std::vector<double> dp = model.distribution().get_parameters();
    CHECK_NEAR(dp[0], 16500.0, 1e-9);  // predicted at PTI = 1990
    CHECK_NEAR(dp[1], 5000.0, 1e-9);

    model.set_parameter_time_index(2000);
    dp = model.distribution().get_parameters();
    CHECK_NEAR(dp[0], 17000.0, 1e-9);
}

// GetNonstationaryReturnLevel (C# 2035): stationary -> null (empty here); nonstationary ->
// InverseCDF(1 - Alpha) at each index from the first to max(ParameterTimeIndex, last).
void test_get_nonstationary_return_level() {
    UnivariateDistributionModel stationary = make_normal_model();
    CHECK_TRUE(stationary.get_nonstationary_return_level().empty());

    UnivariateDistributionModel model = create_normal_linear_mu();
    model.set_parameter_values({15000.0, 50.0, 5000.0});

    // Alpha default 0.5 -> Normal InverseCDF(0.5) == mu, so the return level is the mu trend.
    std::vector<double> rl = model.get_nonstationary_return_level();
    CHECK_EQ(rl.size(), static_cast<std::size_t>(kHotPathFixtureSize));  // 1960..2019
    CHECK_NEAR(rl[0], 15000.0, 1e-9);
    CHECK_NEAR(rl[30], 16500.0, 1e-9);
    CHECK_NEAR(rl[59], 17950.0, 1e-9);

    // A ParameterTimeIndex beyond the last index extends the sequence (C# Math.Max).
    model.set_parameter_time_index(2030);
    CHECK_EQ(model.get_nonstationary_return_level().size(), static_cast<std::size_t>(71));
}

// C# Math.Round(x, 2) equivalent (MidpointRounding.ToEven under the default FE_TONEAREST).
double round_half_even_2(double value) { return std::nearbyint(value * 100.0) / 100.0; }

// SetDefaultQuantilePriors (C# 1024), single-quantile: one LnNormal prior at alpha = 0.1 with
// mu = Round(InverseCDF(0.9), 2) and sigma = Round(0.15 * mu, 2).
void test_set_default_quantile_priors_single_quantile_builds_one_lnnormal() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_use_single_quantile(true);
    model.set_enable_quantile_priors(true);  // triggers set_default_quantile_priors (enabled)

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(1));
    CHECK_NEAR(model.quantile_priors()[0].alpha(), 0.1, 1e-15);
    const auto* ln = dynamic_cast<const bestfit::numerics::distributions::LnNormal*>(
        &model.quantile_priors()[0].distribution());
    CHECK_TRUE(ln != nullptr);

    double mu = round_half_even_2(model.distribution().inverse_cdf(1.0 - 0.1));
    double sigma = round_half_even_2(mu * 0.15);
    bestfit::numerics::distributions::LnNormal expected;
    expected.set_parameters({mu, sigma});
    std::vector<double> actual_params = model.quantile_priors()[0].distribution().get_parameters();
    std::vector<double> expected_params = expected.get_parameters();
    CHECK_EQ(actual_params.size(), expected_params.size());
    for (std::size_t i = 0; i < actual_params.size(); ++i) {
        CHECK_NEAR(actual_params[i], expected_params[i], 1e-12);
    }
}

// SetDefaultQuantilePriors, multi: one prior per distribution parameter at alpha = 10^-i,
// plus the shrink (count > qCount) and extend (count < qCount, alpha = last/10) paths.
void test_set_default_quantile_priors_multi_builds_one_per_parameter() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_enable_quantile_priors(true);  // use_single_quantile() false (default) -> 2

    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(2));
    CHECK_NEAR(model.quantile_priors()[0].alpha(), 0.1, 1e-15);
    CHECK_NEAR(model.quantile_priors()[1].alpha(), 0.01, 1e-15);

    model.set_use_single_quantile(true);  // shrink path
    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(1));
    CHECK_NEAR(model.quantile_priors()[0].alpha(), 0.1, 1e-15);

    model.set_use_single_quantile(false);  // extend path: new alpha = last / 10
    CHECK_EQ(model.quantile_priors().size(), static_cast<std::size_t>(2));
    CHECK_NEAR(model.quantile_priors()[1].alpha(), 0.01, 1e-15);
}

// The single-quantile prior term of Prior_LogLikelihood (C# 1854-1857) and its
// PointwisePriorLogLikelihood component (C# 1941-1946).
void test_single_quantile_prior_term_in_prior_log_likelihood() {
    UnivariateDistributionModel model = make_normal_model();
    std::vector<double> p{16500.0, 6000.0};

    double base = model.prior_log_likelihood(p);

    model.set_use_single_quantile(true);
    model.set_enable_quantile_priors(true);
    // The C# SetDefaultQuantilePriors does NOT process; _quantilePriorsTrue stays empty and
    // the quantile term stays inert until ProcessQuantilePriors runs (setter/analysis/Clone).
    CHECK_NEAR(model.prior_log_likelihood(p), base, 1e-12);

    model.process_quantile_priors();

    double with_quantile = model.prior_log_likelihood(p);
    double alpha = model.quantile_priors()[0].alpha();
    double expected_term = model.quantile_priors()[0].distribution().log_pdf(
        Normal(16500.0, 6000.0).inverse_cdf(1.0 - alpha));
    CHECK_NEAR(with_quantile - base, expected_term, 1e-9);

    std::vector<bestfit::models::PriorComponent> components =
        model.pointwise_prior_log_likelihood(p);
    // parameters + Jeffreys + the quantile component.
    CHECK_EQ(components.size(), model.parameters().size() + 2);
    CHECK_TRUE(components.back().type() == bestfit::models::PriorComponentType::QuantilePrior);
    CHECK_NEAR(components.back().log_likelihood(), expected_term, 1e-9);
}

// DEFERRAL PIN: the multi-quantile branch (C# 1858-1875 / 1947-1971) requires
// IStandardError::QuantileJacobian, which is not on the ported distribution base. Rather
// than silently omitting the Jacobian term (a silent misfit), the branch throws.
void test_multi_quantile_prior_branch_throws_logic_error() {
    UnivariateDistributionModel model = make_normal_model();
    model.set_enable_quantile_priors(true);  // multi (use_single_quantile false)
    model.process_quantile_priors();

    std::vector<double> p{16500.0, 6000.0};
    bool threw = false;
    try {
        model.prior_log_likelihood(p);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK_TRUE(threw);

    threw = false;
    try {
        model.pointwise_prior_log_likelihood(p);
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK_TRUE(threw);
}

void test_base_class_quantile_prior_state_defaults() {
    UnivariateDistributionModel model = make_normal_model();
    bestfit::models::UnivariateDistributionModelBase& base = model;

    // Defaults per the C# backing fields (lines 60-70).
    CHECK_TRUE(base.use_jeffreys_rule_for_scale());
    CHECK_TRUE(!base.enable_quantile_priors());
    CHECK_TRUE(!base.use_single_quantile());
    CHECK_TRUE(base.quantile_priors().empty());

    // The QuantilePriors setter replaces the list (and reprocesses; observable here via the
    // stored list -- _quantilePriorsTrue is protected in both languages).
    std::vector<bestfit::models::QuantilePrior> priors;
    priors.emplace_back(0.01, std::make_unique<Normal>(75000.0, 15000.0));
    base.set_quantile_priors(std::move(priors));
    CHECK_EQ(base.quantile_priors().size(), static_cast<std::size_t>(1));
    CHECK_NEAR(base.quantile_priors()[0].alpha(), 0.01, 1e-15);
}

}  // namespace

int main() {
    // Phase 4 checks (unchanged).
    test_data_log_likelihood_matches_closed_form_normal();
    test_data_log_likelihood_negative_infinity_on_invalid_parameters();
    test_pointwise_sums_to_data_log_likelihood_and_has_matching_length();
    test_pointwise_invalid_parameters_returns_all_negative_infinity();
    test_pointwise_components_one_exact_component_per_value_matching_ll();
    test_pointwise_components_negative_infinity_on_invalid_parameters();
    test_set_default_parameters_populates_bounds_and_uniform_priors();
    test_construct_from_owned_distribution_pointer();
    test_use_jeffreys_rule_for_scale_defaults_true();
    test_jeffreys_prior_adds_negative_log_scale_term();
    test_jeffreys_prior_nonpositive_scale_is_negative_infinity();
    test_pointwise_prior_log_likelihood_appends_jeffreys_scale_component();

    // M8: additive Numerics ports.
    test_distribution_base_censored_log_likelihood_methods();
    test_gauss_legendre20_oracles();

    // M8: UnivariateDistributionTests.cs transcriptions.
    test_constructor_creates_valid_model();
    test_constructor_all_supported_distribution_types();
    test_constructor_normal_has_two_parameters();
    test_constructor_gev_has_three_parameters();
    test_parameters_has_correct_count();
    test_set_parameter_values_updates_distribution();
    test_set_parameter_values_updates_model_parameters();
    test_set_parameter_values_round_trip();
    test_log_likelihood_returns_finite_value();
    test_log_likelihood_negative_for_continuous_distributions();
    test_log_likelihood_invalid_parameters_returns_negative_infinity();
    test_log_prior_with_uniform_prior_returns_finite();
    test_log_prior_with_informative_prior_returns_finite();
    test_log_prior_outside_prior_support_returns_negative_infinity();
    test_validate_valid_model_returns_true();
    test_change_distribution_type_updates_parameters();

    // M8: UnivariateDistributionExpandedTests.cs transcriptions (stationary).
    test_is_supported_distribution_type_all_supported_values_return_true();
    test_create_distribution_normal_returns_normal_instance();
    test_create_distribution_gumbel_returns_gumbel_instance();
    test_create_distribution_gev_returns_gev_instance();
    test_validate_well_formed_model_is_valid();
    test_validate_no_data_frame_is_invalid();
    test_validate_log_distribution_with_negative_data_is_invalid();
    test_get_parameter_values_returns_parameter_count();
    test_get_parameter_values_returns_current_values();
    test_pointwise_length_equals_series_count();
    test_pointwise_sum_equals_data_log_likelihood();
    test_log_likelihood_equals_data_plus_prior();
    test_distribution_set_to_new_type_updates_distribution_type();

    // M8: ThresholdLikelihoodGuardTests.cs transcriptions (Univariate rows).
    test_threshold_likelihood_one_zero_count_side_remains_finite_and_pointwise_consistent();
    test_threshold_likelihood_both_counts_zero_contributes_zero();

    // M8: internal-support consistency for the censored surface.
    test_set_data_frame_processes_threshold_series_at_model_boundary();
    test_mixed_censored_data_log_likelihood_matches_componentwise_recomputation();
    test_mixed_censored_data_invalid_parameters_placeholders();

    // M8: base-class surface.
    test_base_class_quantile_prior_state_defaults();

    // M9: UnivariateDistributionExpandedTests.cs transcriptions (the deferred 17).
    test_is_nonstationary_set_true_populates_trend_models();
    test_is_nonstationary_true_then_false_trend_models_are_constant();
    test_alpha_setter_round_trips();
    test_parameter_time_index_setter_round_trips();
    test_validate_nonstationary_with_zero_alpha_is_invalid();
    test_validate_nonstationary_with_alpha_one_is_invalid();
    test_validate_nonstationary_parameter_time_index_out_of_range_is_invalid();
    test_generate_random_values_requested_size();
    test_generate_random_values_same_seed_is_reproducible();
    test_generate_random_values_different_seeds_produce_different_samples();
    test_generate_random_values_non_positive_sample_size_throws();
    test_generate_random_values_matches_distribution_stream();
    test_clone_produces_distinct_instance();
    test_clone_preserves_distribution_type();
    test_clone_preserves_parameter_values();
    test_clone_syncs_distribution_parameters();
    test_clone_preserves_is_nonstationary();
    test_clone_preserves_alpha();
    test_clone_parameters_are_independent();

    // M9: NonstationaryLogLikelihoodHotPathTests.cs transcriptions.
    test_nonstationary_data_log_likelihood_matches_sum_of_pointwise();
    test_nonstationary_data_log_likelihood_repeated_calls_return_same_value();
    test_nonstationary_data_log_likelihood_different_parameters_are_independent();
    test_nonstationary_data_log_likelihood_hot_loop_smoke_test();
    test_nonstationary_pointwise_length_and_sum_are_correct();

    // M9: ThresholdLikelihoodGuardTests.cs -- the nonstationary split-threshold half.
    test_nonstationary_split_threshold_likelihood_remains_finite();

    // M9: internal-support checks (trend plumbing, ledger pin, quantile priors).
    test_set_trend_model_replaces_trend_and_rebuilds_parameters();
    test_step_function_trend_sets_is_positive_from_constraint_lower_bound();
    test_nonstationary_parameter_plumbing_syncs_distribution_at_time_index();
    test_get_nonstationary_return_level();
    test_set_default_quantile_priors_single_quantile_builds_one_lnnormal();
    test_set_default_quantile_priors_multi_builds_one_per_parameter();
    test_single_quantile_prior_term_in_prior_log_likelihood();
    test_multi_quantile_prior_branch_throws_logic_error();

    return bftest::summary("univariate_distribution_model");
}
