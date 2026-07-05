// Transcribed from: upstream/RMC-BestFit/src/RMC.BestFit.Tests/Univariate/
// Bulletin17CDistributionTests.cs (@ fc28c0c) for the Phase 6 Task B9 construction slice.
// B10 extends this file with the moment-machinery tests (the four upstream methods below
// plus the self-consistency supplements).
//
// Transcribed structural methods (values/tolerances unaltered):
//   Constructor_Default_UsesLogPearsonTypeIII, Constructor_WithDataFrameAndType_BuildsParameters,
//   Constructor_WithDistributionInstance_ClonesDistribution (adapted -- see below),
//   Constructor_NullDistribution_Throws, IsSupportedDistributionType_AcceptsB17CFamily,
//   IsSupportedDistributionType_RejectsNonB17CFamily,
//   CreateDistribution_AllSupportedTypes_ReturnsCorrectType, CreateDistribution_UnsupportedType_Throws,
//   IsNonstationary_AlwaysFalse, SetParameterValues_WrongLength_Throws,
//   SetParameterValues_RoundTrip_UpdatesParametersAndDistribution, SampleSize_NullDataFrame_IsZero,
//   Validate_GoodFixture_IsValid, Validate_NullDataFrame_IsInvalid,
//   Validate_LogDistribution_NonPositiveData_IsInvalid, Clone_ReturnsSeparateInstance_WithMatchingState
//   (extended with a clone-independence mutation check per the task brief),
//   GenerateRandomValues_FixedSeed_IsDeterministic, GenerateRandomValues_NonPositiveSampleSize_Throws,
//   GenerateRandomValues_RequiresDistribution,
//   PointwiseMomentConditions_ColumnMeans_MatchMomentConditionsG_LP3 / _Normal (B10, the
//   CON-23 invariant), WeightedErrorDirectionScore_NaturalParameters_IgnoresCurrentLinkController
//   (B10), WeightedErrorDirectionScoreFromLinked_InverseLinksBeforeScoring (B10).
//
// SKIPPED upstream methods (each with reason):
//   - ToXElement_FromXElement_RoundTripsCoreState, FromXElement_NoDataFrame_ConstructsWithoutThrowing,
//     FromXElement_NullXml_Throws: XML serialization is a project-wide deliberate skip.
//   - Constructor_NullDataFrame_Throws: the C++ DataFrame is a move-only VALUE type (M4
//     decision); a null frame is structurally unrepresentable, so the guard has no analog.
//     The distribution-null half of the C# null-guard pair IS transcribed below.
//
// Adaptations:
//   - Constructor_WithDistributionInstance_ClonesDistribution: the C# ctor CLONES a shared
//     reference; the C++ ctor takes a unique_ptr and the caller transfers sole ownership
//     (the UnivariateDistributionModel precedent), so aliasing is impossible by construction.
//     The test keeps the type assertion and checks the model owns a DIFFERENT object than the
//     prototype the clone was taken from.
//   - Assert.AreSame(df, model.DataFrame): reference identity has no analog for the value-typed
//     frame; the test asserts the moved-in frame's observable state (record count) instead.
//   - The C# DataFrame recomputes plotting positions automatically through INPC events; the C++
//     port's explicit-invalidation contract (M4/M5) means tests that consume plotting positions
//     (the nonparametric-moments supplement) call calculate_plotting_positions() themselves.
//
// SUPPLEMENTS (no upstream analog; internal C++-only checks, hardcoded values allowed):
//   - the GMM IGMMModel-constructor wiring (upstream's only IGMMModel-ctor test is
//     Test_Constructor_NullModel_Throws, unrepresentable for a reference parameter);
//   - DataFrame::GetNonparametricMoments / GetNonparametricMomentsROS deterministic checks
//     (grep-verified: NO upstream test exists for either method -- see the B9 report);
//   - SetPenaltyFunction / SetRandomPenaltyFunction closure wiring and guards;
//   - SetInitialParameters restores constraint initials;
//   - (B10) exact-data-only MomentConditions against hand-computed sample moments, the
//     mixed-frame CON-23 invariant + CensoringAsymmetryScore bounds, QuantileVariance vs
//     an independent finite-difference delta method, and a GeneralizedMethodOfMoments
//     end-to-end smoke run (finiteness only; exact fit oracles land at B12).
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/estimation/generalized_method_of_moments.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/data_frame/data_types/interval_data.hpp"
#include "bestfit/models/data_frame/data_types/threshold_data.hpp"
#include "bestfit/models/data_frame/data_types/uncertain_data.hpp"
#include "bestfit/models/link_functions/asinh_link.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "bestfit/numerics/distributions/log_pearson_type_iii.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/pearson_type_iii.hpp"
#include "bestfit/numerics/functions/i_link_function.hpp"
#include "bestfit/numerics/functions/link_controller.hpp"
#include "bestfit/numerics/functions/log_link.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

using bestfit::estimation::GeneralizedMethodOfMoments;
using bestfit::models::Bulletin17CDistribution;
using bestfit::models::DataFrame;
using bestfit::models::ExactData;
using bestfit::models::IntervalData;
using bestfit::models::ThresholdData;
using bestfit::models::UncertainData;
using bestfit::models::link_functions::ASinHLink;
using bestfit::numerics::distributions::LogNormal;
using bestfit::numerics::distributions::LogPearsonTypeIII;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::PearsonTypeIII;
using bestfit::numerics::distributions::UnivariateDistributionType;
using bestfit::numerics::functions::ILinkFunction;
using bestfit::numerics::functions::LinkController;
using bestfit::numerics::functions::LogLink;
using bestfit::numerics::sampling::MersenneTwister;

namespace {

constexpr int kFixtureSize = 50;

// Deterministic Log-Pearson-III-like flood fixture (C# InlineFloodData: fixed RNG seed so
// the file does not depend on the Verification project's TestData).
std::vector<double> inline_flood_data() {
    return LogNormal(8.0, 0.4).generate_random_values(kFixtureSize, 12345);
}

DataFrame create_flood_data_frame() {
    DataFrame df;
    std::vector<double> data = inline_flood_data();
    for (std::size_t i = 0; i < data.size(); ++i) {
        // Add as exact systematic record (simulated water years).
        df.exact_series().add(ExactData(1970 + static_cast<int>(i), data[i]));
    }
    return df;
}

// ---------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------

// C# Constructor_Default_UsesLogPearsonTypeIII.
void test_constructor_default_uses_log_pearson_type_iii() {
    Bulletin17CDistribution model;

    CHECK_TRUE(model.distribution_type() == UnivariateDistributionType::LogPearsonTypeIII);
    CHECK_TRUE(model.distribution() != nullptr);
    CHECK_TRUE(!model.has_data_frame());  // "Default constructor leaves DataFrame null."
    // "SetUpQuantilePenalties seeds at least one quantile penalty row."
    CHECK_TRUE(model.quantile_penalties().size() >= 1);
}

// C# Constructor_WithDataFrameAndType_BuildsParameters.
void test_constructor_with_data_frame_and_type_builds_parameters() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);

    CHECK_TRUE(model.distribution_type() == UnivariateDistributionType::LogPearsonTypeIII);
    // AreSame(df, model.DataFrame) adapted: the moved-in frame's observable state.
    CHECK_TRUE(model.has_data_frame());
    CHECK_EQ(model.data_frame().total_record_length(), kFixtureSize);
    // "LP3 has three parameters (xi, alpha, kappa) so the parameter list should match."
    CHECK_EQ(model.number_of_parameters(), 3);
}

// C# Constructor_WithDistributionInstance_ClonesDistribution (adapted -- see file header).
void test_constructor_with_distribution_instance() {
    LogPearsonTypeIII prototype;
    Bulletin17CDistribution model(create_flood_data_frame(), prototype.clone());

    CHECK_TRUE(model.distribution() !=
               static_cast<const bestfit::numerics::distributions::UnivariateDistributionBase*>(
                   &prototype));
    CHECK_TRUE(model.distribution()->type() == prototype.type());
}

// C# Constructor_NullDistribution_Throws.
void test_constructor_null_distribution_throws() {
    CHECK_THROWS(Bulletin17CDistribution(
        create_flood_data_frame(),
        std::unique_ptr<bestfit::numerics::distributions::UnivariateDistributionBase>{}));
}

// ---------------------------------------------------------------------------------------
// IsSupportedDistributionType gate
// ---------------------------------------------------------------------------------------

// C# IsSupportedDistributionType_AcceptsB17CFamily.
void test_is_supported_distribution_type_accepts_b17c_family() {
    using T = UnivariateDistributionType;
    CHECK_TRUE(Bulletin17CDistribution::is_supported_distribution_type(T::Exponential));
    CHECK_TRUE(Bulletin17CDistribution::is_supported_distribution_type(T::GammaDistribution));
    CHECK_TRUE(Bulletin17CDistribution::is_supported_distribution_type(T::LogNormal));
    CHECK_TRUE(Bulletin17CDistribution::is_supported_distribution_type(T::LogPearsonTypeIII));
    CHECK_TRUE(Bulletin17CDistribution::is_supported_distribution_type(T::Normal));
    CHECK_TRUE(Bulletin17CDistribution::is_supported_distribution_type(T::PearsonTypeIII));
}

// C# IsSupportedDistributionType_RejectsNonB17CFamily.
void test_is_supported_distribution_type_rejects_non_b17c_family() {
    using T = UnivariateDistributionType;
    CHECK_TRUE(!Bulletin17CDistribution::is_supported_distribution_type(T::GeneralizedExtremeValue));
    CHECK_TRUE(!Bulletin17CDistribution::is_supported_distribution_type(T::Weibull));
    CHECK_TRUE(!Bulletin17CDistribution::is_supported_distribution_type(T::KappaFour));
    CHECK_TRUE(!Bulletin17CDistribution::is_supported_distribution_type(T::GeneralizedPareto));
}

// ---------------------------------------------------------------------------------------
// CreateDistribution factory
// ---------------------------------------------------------------------------------------

// C# CreateDistribution_AllSupportedTypes_ReturnsCorrectType (IsInstanceOfType asserted via
// the polymorphic type() tag, the port's runtime type discriminator).
void test_create_distribution_all_supported_types() {
    using T = UnivariateDistributionType;
    CHECK_TRUE(Bulletin17CDistribution::create_distribution(T::LogPearsonTypeIII)->type() ==
               T::LogPearsonTypeIII);
    CHECK_TRUE(Bulletin17CDistribution::create_distribution(T::LogNormal)->type() == T::LogNormal);
    CHECK_TRUE(Bulletin17CDistribution::create_distribution(T::Normal)->type() == T::Normal);
    CHECK_TRUE(Bulletin17CDistribution::create_distribution(T::PearsonTypeIII)->type() ==
               T::PearsonTypeIII);
    CHECK_TRUE(Bulletin17CDistribution::create_distribution(T::GammaDistribution)->type() ==
               T::GammaDistribution);
    CHECK_TRUE(Bulletin17CDistribution::create_distribution(T::Exponential)->type() ==
               T::Exponential);
}

// C# CreateDistribution_UnsupportedType_Throws.
void test_create_distribution_unsupported_type_throws() {
    CHECK_THROWS(Bulletin17CDistribution::create_distribution(
        UnivariateDistributionType::GeneralizedExtremeValue));
}

// ---------------------------------------------------------------------------------------
// Properties / parameter management
// ---------------------------------------------------------------------------------------

// C# IsNonstationary_AlwaysFalse.
void test_is_nonstationary_always_false() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);
    CHECK_TRUE(!model.is_nonstationary());
}

// C# SetParameterValues_WrongLength_Throws.
void test_set_parameter_values_wrong_length_throws() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    CHECK_THROWS(model.set_parameter_values({1.0, 2.0, 3.0, 4.0}));
}

// C# SetParameterValues_RoundTrip_UpdatesParametersAndDistribution.
void test_set_parameter_values_round_trip() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);

    model.set_parameter_values({123.0, 45.0});

    CHECK_NEAR(model.parameters()[0].value(), 123.0, 1e-12);
    CHECK_NEAR(model.parameters()[1].value(), 45.0, 1e-12);
    const auto* dist = dynamic_cast<const Normal*>(model.distribution());
    CHECK_TRUE(dist != nullptr);
    CHECK_NEAR(dist->mu(), 123.0, 1e-12);
    CHECK_NEAR(dist->sigma(), 45.0, 1e-12);
}

// C# SampleSize_NullDataFrame_IsZero.
void test_sample_size_null_data_frame_is_zero() {
    Bulletin17CDistribution model;
    CHECK_EQ(model.sample_size(), 0);
}

// ---------------------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------------------

// C# Validate_GoodFixture_IsValid.
void test_validate_good_fixture_is_valid() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);
    CHECK_TRUE(model.validate().is_valid);
}

// C# Validate_NullDataFrame_IsInvalid.
void test_validate_null_data_frame_is_invalid() {
    Bulletin17CDistribution model;  // Default constructor does not set a DataFrame.
    auto result = model.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(result.validation_messages.size() > 0);
}

// C# Validate_LogDistribution_NonPositiveData_IsInvalid.
void test_validate_log_distribution_non_positive_data_is_invalid() {
    DataFrame df;
    df.exact_series().add(ExactData(1990, 1000.0));
    df.exact_series().add(ExactData(1991, 0.0));   // disallowed for log distributions
    df.exact_series().add(ExactData(1992, -5.0));  // disallowed
    df.exact_series().add(ExactData(1993, 2000.0));

    Bulletin17CDistribution model(std::move(df), UnivariateDistributionType::LogNormal);
    auto result = model.validate();
    CHECK_TRUE(!result.is_valid);

    bool found = false;
    for (const std::string& m : result.validation_messages) {
        std::string lower = m;
        for (char& c : lower) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (lower.find("log-based") != std::string::npos ||
            lower.find("non-positive") != std::string::npos ||
            lower.find("log") != std::string::npos) {
            found = true;
            break;
        }
    }
    CHECK_TRUE(found);
}

// ---------------------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------------------

// C# Clone_ReturnsSeparateInstance_WithMatchingState, extended with the clone-independence
// mutation check the task brief requires (mutate the clone, original unchanged).
void test_clone_returns_separate_instance_with_matching_state() {
    Bulletin17CDistribution original(create_flood_data_frame(),
                                     UnivariateDistributionType::LogPearsonTypeIII);

    std::unique_ptr<bestfit::models::IGMMModel> clone = original.clone();
    CHECK_TRUE(clone != nullptr);
    CHECK_TRUE(clone.get() != static_cast<bestfit::models::IGMMModel*>(&original));

    auto* cloned = dynamic_cast<Bulletin17CDistribution*>(clone.get());
    CHECK_TRUE(cloned != nullptr);
    CHECK_TRUE(cloned->distribution_type() == original.distribution_type());
    CHECK_EQ(cloned->number_of_parameters(), original.number_of_parameters());
    // Matching state: parameter values carried over (the C# XElement round trip preserves
    // them; the direct deep clone must too).
    for (int i = 0; i < original.number_of_parameters(); ++i)
        CHECK_NEAR(cloned->parameters()[static_cast<std::size_t>(i)].value(),
                   original.parameters()[static_cast<std::size_t>(i)].value(), 1e-12);
    CHECK_EQ(static_cast<int>(cloned->quantile_penalties().size()),
             static_cast<int>(original.quantile_penalties().size()));

    // Clone independence: mutate the clone, the original is unchanged.
    double before = original.parameters()[0].value();
    cloned->parameters()[0].set_value(before + 123.0);
    cloned->quantile_penalties()[0].set_mean(42.0);
    CHECK_NEAR(original.parameters()[0].value(), before, 0.0);
    CHECK_TRUE(std::isnan(original.quantile_penalties()[0].mean()));  // default Mean is NaN
}

// ---------------------------------------------------------------------------------------
// GenerateRandomValues
// ---------------------------------------------------------------------------------------

// C# GenerateRandomValues_FixedSeed_IsDeterministic.
void test_generate_random_values_fixed_seed_is_deterministic() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});

    std::vector<double> s1 = model.generate_random_values(20, 42);
    std::vector<double> s2 = model.generate_random_values(20, 42);

    CHECK_EQ(static_cast<int>(s1.size()), 20);
    CHECK_TRUE(s1 == s2);
}

// C# GenerateRandomValues_NonPositiveSampleSize_Throws.
void test_generate_random_values_non_positive_sample_size_throws() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});
    CHECK_THROWS(model.generate_random_values(0, 1));
    CHECK_THROWS(model.generate_random_values(-5, 1));
}

// C# GenerateRandomValues_RequiresDistribution (the C# method exercises only the normal
// sampling path; the null-distribution state is unreachable through the public API).
void test_generate_random_values_requires_distribution() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});
    std::vector<double> values = model.generate_random_values(5, 1);
    CHECK_EQ(static_cast<int>(values.size()), 5);
}

// ---------------------------------------------------------------------------------------
// SUPPLEMENT: GMM IGMMModel-constructor wiring (B9 fills the B8 ctor slot)
// ---------------------------------------------------------------------------------------

void test_gmm_model_constructor_wiring() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});

    GeneralizedMethodOfMoments gmm(model);

    CHECK_EQ(gmm.number_of_parameters(), 2);
    CHECK_EQ(gmm.number_of_moment_conditions(), 2);
    CHECK_EQ(gmm.sample_size(), kFixtureSize);
    CHECK_TRUE(gmm.model() == static_cast<bestfit::models::IGMMModel*>(&model));
    CHECK_TRUE(gmm.identification_status() ==
               GeneralizedMethodOfMoments::GMMIdentificationStatus::JustIdentified);
    // Initial values / bounds come from the model's parameters.
    CHECK_NEAR(gmm.initial_values()[0], model.parameters()[0].value(), 0.0);
    CHECK_NEAR(gmm.initial_values()[1], model.parameters()[1].value(), 0.0);
    CHECK_NEAR(gmm.lower_bounds()[0], model.parameters()[0].lower_bound(), 0.0);
    CHECK_NEAR(gmm.upper_bounds()[1], model.parameters()[1].upper_bound(), 0.0);
    // B10: the delegate now runs the real MomentConditions and returns finite results.
    bestfit::estimation::MomentConditionResult mc = gmm.moment_condition_function()({100.0, 15.0});
    CHECK_EQ(mc.G.length(), 2);
    CHECK_EQ(mc.S.number_of_rows(), 2);
    CHECK_EQ(mc.S.number_of_columns(), 2);
    for (int i = 0; i < 2; ++i) {
        CHECK_TRUE(std::isfinite(mc.G[i]));
        for (int j = 0; j < 2; ++j) CHECK_TRUE(std::isfinite(mc.S(i, j)));
    }
}

// ---------------------------------------------------------------------------------------
// SUPPLEMENT: penalty-function wiring (SetPenaltyFunction / SetRandomPenaltyFunction)
// ---------------------------------------------------------------------------------------

void test_set_penalty_function_deterministic() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});

    // Zero enabled penalties -> the stored PenaltyFunction is cleared (null).
    model.set_penalty_function();
    CHECK_TRUE(!model.penalty_function());

    // Enable one quantile penalty (real space) and one parameter penalty; the deterministic
    // closure must equal the hand-composed sum of the two ported penalty Function() calls.
    const int nt = kFixtureSize;
    model.quantile_penalties()[0].set_enabled(true);
    model.quantile_penalties()[0].set_use_log10(false);
    model.quantile_penalties()[0].set_aep(0.01);
    model.quantile_penalties()[0].set_mean(140.0);
    model.quantile_penalties()[0].set_mse(4.0);
    model.parameter_penalties()[0].set_enabled(true);
    model.parameter_penalties()[0].set_mean(98.0);
    model.parameter_penalties()[0].set_mse(2.0);

    model.set_penalty_function();
    CHECK_TRUE(static_cast<bool>(model.penalty_function()));

    std::vector<double> theta = {100.0, 15.0};
    double actual = model.penalty_function()(theta);

    double q99 = Normal(100.0, 15.0).inverse_cdf(1.0 - 0.01);
    double expected = model.parameter_penalties()[0].function(100.0, nt) +
                      model.quantile_penalties()[0].function(q99, nt);
    CHECK_NEAR(actual, expected, 1e-12);
}

void test_set_penalty_function_random_is_seed_deterministic() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});
    model.quantile_penalties()[0].set_enabled(true);
    model.quantile_penalties()[0].set_use_log10(false);
    model.quantile_penalties()[0].set_aep(0.01);
    model.quantile_penalties()[0].set_mean(140.0);
    model.quantile_penalties()[0].set_mse(4.0);
    model.parameter_penalties()[0].set_enabled(true);
    model.parameter_penalties()[0].set_mean(98.0);
    model.parameter_penalties()[0].set_mse(2.0);

    std::vector<double> theta = {100.0, 15.0};

    model.set_penalty_function();
    double deterministic = model.penalty_function()(theta);

    MersenneTwister prng1(42);
    model.set_penalty_function(&prng1);
    double random1 = model.penalty_function()(theta);

    MersenneTwister prng2(42);
    model.set_penalty_function(&prng2);
    double random2 = model.penalty_function()(theta);

    CHECK_NEAR(random1, random2, 0.0);           // same seed -> same perturbed means
    CHECK_TRUE(random1 != deterministic);        // means were perturbed
}

void test_set_random_penalty_function_guards_and_value() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});
    model.parameter_penalties()[0].set_enabled(true);
    model.parameter_penalties()[0].set_mean(98.0);
    model.parameter_penalties()[0].set_mse(2.0);

    MersenneTwister prng(7);
    CHECK_THROWS(model.set_random_penalty_function({1.0}, &prng));         // wrong length
    CHECK_THROWS(model.set_random_penalty_function({1.0, 2.0}, nullptr));  // null prng

    // Recentring at the parent parameters: with seed 7 the perturbed mean is
    // parent[0] + sqrt(MSE) * StandardZ(u1); reproduce the draw with a twin PRNG.
    MersenneTwister twin(7);
    double z = Normal::standard_z(twin.next_double());
    model.set_random_penalty_function({100.0, 15.0}, &prng);
    CHECK_TRUE(static_cast<bool>(model.penalty_function()));
    double actual = model.penalty_function()({100.0, 15.0});
    double perturbed_mean = 100.0 + std::sqrt(2.0) * z;
    double expected = 0.5 * (100.0 - perturbed_mean) * (100.0 - perturbed_mean) /
                      (2.0 * kFixtureSize);
    CHECK_NEAR(actual, expected, 1e-12);
}

// ---------------------------------------------------------------------------------------
// SUPPLEMENT: SetInitialParameters restores constraint initials
// ---------------------------------------------------------------------------------------

void test_set_initial_parameters_restores_initials() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    std::vector<double> initials;
    for (const auto& p : model.parameters()) initials.push_back(p.value());

    model.set_parameter_values({1.0, 2.0});
    model.set_initial_parameters();

    for (std::size_t i = 0; i < initials.size(); ++i)
        CHECK_NEAR(model.parameters()[i].value(), initials[i], 1e-12);
}

// ---------------------------------------------------------------------------------------
// SUPPLEMENT: DataFrame nonparametric moments (plain + ROS)
// ---------------------------------------------------------------------------------------

// Fewer than 4 exact points -> null (std::nullopt) on both methods.
void test_nonparametric_moments_insufficient_data_returns_null() {
    DataFrame df;
    df.exact_series().add(ExactData(2000, 10.0));
    df.exact_series().add(ExactData(2001, 20.0));
    df.exact_series().add(ExactData(2002, 30.0));
    df.calculate_plotting_positions();
    CHECK_TRUE(!df.get_nonparametric_moments().has_value());
    CHECK_TRUE(!df.get_nonparametric_moments_ros().has_value());
}

// No low outliers -> ROS falls back to the plain method (identical output).
void test_nonparametric_moments_ros_fallback_no_outliers() {
    DataFrame df;
    for (int i = 0; i < 10; ++i)
        df.exact_series().add(ExactData(2000 + i, 100.0 + 10.0 * i));
    df.calculate_plotting_positions();

    auto plain = df.get_nonparametric_moments();
    auto ros = df.get_nonparametric_moments_ros();
    CHECK_TRUE(plain.has_value());
    CHECK_TRUE(ros.has_value());
    for (int k = 0; k < 4; ++k) CHECK_NEAR((*ros)[static_cast<std::size_t>(k)],
                                           (*plain)[static_cast<std::size_t>(k)], 0.0);
}

// Hand-verifiable moments: a symmetric 5-point sample. The Hirsch-Stedinger positions for a
// pure exact series are symmetric Weibull positions, so the empirical distribution is
// symmetric about 100: mean = 100 and skewness = 0 up to integration roundoff.
void test_nonparametric_moments_symmetric_sample() {
    DataFrame df;
    const double values[5] = {90.0, 95.0, 100.0, 105.0, 110.0};
    for (int i = 0; i < 5; ++i) df.exact_series().add(ExactData(2000 + i, values[i]));
    df.calculate_plotting_positions();

    auto moments = df.get_nonparametric_moments();
    CHECK_TRUE(moments.has_value());
    CHECK_NEAR((*moments)[0], 100.0, 1e-6);   // mean, by symmetry
    CHECK_NEAR((*moments)[2], 0.0, 1e-9);     // skewness, by symmetry
    CHECK_TRUE((*moments)[1] > 0.0 && (*moments)[1] < 20.0);  // sd within the data range
    CHECK_TRUE(std::isfinite((*moments)[3]));
}

// ROS imputation pinned without circularity: build a frame whose UNCENSORED points are
// exactly collinear in (z, value) with intercept 100 / slope 20 (z from the frame's own
// plotting positions), then corrupt the three smallest values and flag them low outliers.
// The ROS regression through the collinear uncensored points recovers the line exactly, so
// the imputed values equal the clean values and the ROS moments equal the CLEAN frame's
// plain moments; the corrupted frame's plain moments must differ.
void test_nonparametric_moments_ros_imputes_from_regression() {
    // Pass 1: placeholder ascending values to obtain the rank-based plotting positions.
    DataFrame proto;
    const int n = 10;
    for (int i = 0; i < n; ++i) proto.exact_series().add(ExactData(2000 + i, 10.0 * (i + 1)));
    proto.calculate_plotting_positions();
    std::vector<double> z(n);
    Normal std_normal(0.0, 1.0);
    for (int i = 0; i < n; ++i)
        z[static_cast<std::size_t>(i)] =
            std_normal.inverse_cdf(proto.exact_series()[static_cast<std::size_t>(i)]
                                       .plotting_position_complement());

    // Collinear clean values w_i = 100 + 20 z_i (monotone in rank, so the recomputed
    // plotting positions are identical).
    auto make_frame = [&](bool corrupt) {
        DataFrame df;
        for (int i = 0; i < n; ++i) {
            double w = 100.0 + 20.0 * z[static_cast<std::size_t>(i)];
            df.exact_series().add(ExactData(2000 + i, w));
        }
        if (corrupt) {
            // Replace the three smallest values, preserving ranks.
            std::vector<std::size_t> order(static_cast<std::size_t>(n));
            for (std::size_t k = 0; k < order.size(); ++k) order[k] = k;
            std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                return df.exact_series()[a].value() < df.exact_series()[b].value();
            });
            const double tiny[3] = {0.5, 1.0, 2.0};
            for (int k = 0; k < 3; ++k)
                df.exact_series()[order[static_cast<std::size_t>(k)]].set_value(tiny[k]);
        }
        df.calculate_plotting_positions();
        return df;
    };

    DataFrame clean = make_frame(false);
    DataFrame censored = make_frame(true);
    // Flag the three corrupted points as low outliers via the threshold path.
    censored.set_low_outlier_threshold(3.0);
    censored.set_low_outliers_from_threshold();
    censored.calculate_plotting_positions();
    CHECK_EQ(censored.number_of_low_outliers(), 3);

    auto clean_moments = clean.get_nonparametric_moments();
    auto ros_moments = censored.get_nonparametric_moments_ros();
    auto plain_corrupted = censored.get_nonparametric_moments();
    CHECK_TRUE(clean_moments.has_value());
    CHECK_TRUE(ros_moments.has_value());
    CHECK_TRUE(plain_corrupted.has_value());

    for (int k = 0; k < 4; ++k)
        CHECK_NEAR((*ros_moments)[static_cast<std::size_t>(k)],
                   (*clean_moments)[static_cast<std::size_t>(k)], 1e-9);
    // The corrupted values must actually change the plain moments (proves the ROS branch
    // did the imputation rather than falling back).
    CHECK_TRUE(std::fabs((*plain_corrupted)[0] - (*clean_moments)[0]) > 1e-3);
}

// ---------------------------------------------------------------------------------------
// B10: Pointwise vs Scalar Moment-Condition Row-Ordering Invariant (CON-23)
// ---------------------------------------------------------------------------------------

// C# PointwiseMomentConditions_ColumnMeans_MatchMomentConditionsG_LP3 (line 476): the
// column-wise mean of PointwiseMomentConditions equals the G vector from MomentConditions,
// i.e. (1/n) sum_i result[i, j] = G[j].
void test_pointwise_column_means_match_g_lp3() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);
    model.set_default_parameters();

    std::vector<double> p;
    for (const auto& mp : model.parameters()) p.push_back(mp.value());

    bestfit::estimation::MomentConditionResult mc = model.moment_conditions(p);
    // "PointwiseMomentConditions delegate must be exposed by Bulletin17CDistribution."
    CHECK_TRUE(static_cast<bool>(model.pointwise_moment_conditions()));
    auto pointwise = model.pointwise_moment_conditions()(p);

    int n = static_cast<int>(pointwise.size());
    int q = n > 0 ? static_cast<int>(pointwise[0].size()) : 0;
    // "Pointwise matrix must have one column per parameter (q)."
    CHECK_EQ(model.number_of_parameters(), q);
    // "Scalar G vector length must match parameter count."
    CHECK_EQ(mc.G.length(), q);

    for (int j = 0; j < q; ++j) {
        double col_mean = 0.0;
        for (int i = 0; i < n; ++i)
            col_mean += pointwise[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        col_mean /= n;
        CHECK_NEAR(mc.G[j], col_mean, 1e-9);
    }
}

// C# PointwiseMomentConditions_ColumnMeans_MatchMomentConditionsG_Normal (line 511): the
// same invariant against a Normal distribution (a different supported-distribution branch
// in the moment-condition setup).
void test_pointwise_column_means_match_g_normal() {
    Bulletin17CDistribution model(create_flood_data_frame(), UnivariateDistributionType::Normal);
    model.set_default_parameters();

    std::vector<double> p;
    for (const auto& mp : model.parameters()) p.push_back(mp.value());

    bestfit::estimation::MomentConditionResult mc = model.moment_conditions(p);
    CHECK_TRUE(static_cast<bool>(model.pointwise_moment_conditions()));
    auto pointwise = model.pointwise_moment_conditions()(p);

    int n = static_cast<int>(pointwise.size());
    int q = n > 0 ? static_cast<int>(pointwise[0].size()) : 0;
    for (int j = 0; j < q; ++j) {
        double col_mean = 0.0;
        for (int i = 0; i < n; ++i)
            col_mean += pointwise[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        col_mean /= n;
        CHECK_NEAR(mc.G[j], col_mean, 1e-9);
    }
}

// ---------------------------------------------------------------------------------------
// B10: Weighted Error Direction Score
// ---------------------------------------------------------------------------------------

// The upstream tests' LinkController fixture: ASinHLink(theta0, 0.5, 0.25), LogLink,
// ASinHLink(theta2, 0.5, -0.25).
LinkController make_weds_link_controller(const std::vector<double>& theta) {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(std::make_unique<ASinHLink>(theta[0], /*scale=*/0.5, /*epsilon=*/0.25));
    links.push_back(std::make_unique<LogLink>());
    links.push_back(std::make_unique<ASinHLink>(theta[2], /*scale=*/0.5, /*epsilon=*/-0.25));
    return LinkController(std::move(links));
}

// C# WeightedErrorDirectionScore_NaturalParameters_IgnoresCurrentLinkController (line 544):
// WEDS is a natural-parameter diagnostic; installing a non-identity LinkController must not
// change the score when the caller supplies theta-space parameters.
void test_weds_natural_parameters_ignores_current_link_controller() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);
    model.set_default_parameters();
    std::vector<double> theta;
    for (const auto& mp : model.parameters()) theta.push_back(mp.value());

    std::vector<double> baseline = model.weighted_error_direction_score(theta);

    model.set_link_controller(make_weds_link_controller(theta));

    std::vector<double> with_temporary_links = model.weighted_error_direction_score(theta);

    CHECK_EQ(static_cast<int>(baseline.size()), static_cast<int>(with_temporary_links.size()));
    for (std::size_t i = 0; i < baseline.size(); ++i) {
        // "Baseline WEDS[i] should be finite for the inline fixture."
        CHECK_TRUE(std::isfinite(baseline[i]));
        // "WEDS[i] changed after installing temporary uncertainty links."
        CHECK_NEAR(baseline[i], with_temporary_links[i], 1e-12);
    }
}

// C# WeightedErrorDirectionScoreFromLinked_InverseLinksBeforeScoring (line 577): the
// explicit link-space helper matches the natural-parameter method after inverse-linking
// through the current controller.
void test_weds_from_linked_inverse_links_before_scoring() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);
    model.set_default_parameters();
    std::vector<double> theta;
    for (const auto& mp : model.parameters()) theta.push_back(mp.value());

    model.set_link_controller(make_weds_link_controller(theta));

    std::vector<double> linked = model.link_controller().link(theta);
    std::vector<double> natural_score = model.weighted_error_direction_score(theta);
    std::vector<double> linked_score = model.weighted_error_direction_score_from_linked(linked);

    CHECK_EQ(static_cast<int>(natural_score.size()), static_cast<int>(linked_score.size()));
    for (std::size_t i = 0; i < natural_score.size(); ++i)
        CHECK_NEAR(natural_score[i], linked_score[i], 1e-12);
}

// ---------------------------------------------------------------------------------------
// B10 SUPPLEMENT: exact-data-only MomentConditions against hand-computed sample moments
// ---------------------------------------------------------------------------------------

// Frame: y = {8, 10, 12, 14} (n = Ns = 4), Normal model evaluated at (mu, sigma) = (11, 2),
// so q = 2 and the moment conditions are g = [y - mu, c2 (y - mu)^2 - sigma^2].
//
// Bessel corrections: c2 = Ns/(Ns-1) = 4/3 (c3 unused, q = 2).
// Unconditional moments: mu = 11, sigma^2 = 4.
// Row contributions [eg1, eg2]:
//   y = 8:  [-3, (4/3)*9 - 4 =  8   ]     y = 10: [-1, (4/3)*1 - 4 = -8/3]
//   y = 12: [ 1, (4/3)*1 - 4 = -8/3 ]     y = 14: [ 3, (4/3)*9 - 4 =  8  ]
// G = column means = [0, (8 - 8/3 - 8/3 + 8)/4] = [0, (32/3)/4] = [0, 8/3].
//
// Covariance: no low outliers and a Normal family, so every exact row takes the MODEL-BASED
// path with mu2 = sigma^2 = 4, mu4 = 3 sigma^4 = 48:
//   M11 = mu2 = 4, M12 = model mu3 = gamma sigma^3 = 0, M22 = mu4 - mu2^2 = 48 - 16 = 32.
// Sum over 4 rows / n = [[4, 0], [0, 32]]; final S = E[gg'] - G G' =
//   [[4, 0 - 0*(8/3)], [0, 32 - (8/3)^2]] = [[4, 0], [0, 32 - 64/9]].
void test_moment_conditions_exact_data_hand_computed() {
    DataFrame df;
    df.exact_series().add(ExactData(2000, 8.0));
    df.exact_series().add(ExactData(2001, 10.0));
    df.exact_series().add(ExactData(2002, 12.0));
    df.exact_series().add(ExactData(2003, 14.0));

    Bulletin17CDistribution model(std::move(df), UnivariateDistributionType::Normal);
    bestfit::estimation::MomentConditionResult mc = model.moment_conditions({11.0, 2.0});

    CHECK_EQ(mc.G.length(), 2);
    CHECK_NEAR(mc.G[0], 0.0, 1e-12);
    CHECK_NEAR(mc.G[1], 8.0 / 3.0, 1e-12);
    CHECK_NEAR(mc.S(0, 0), 4.0, 1e-12);
    CHECK_NEAR(mc.S(0, 1), 0.0, 1e-12);
    CHECK_NEAR(mc.S(1, 0), 0.0, 1e-12);
    CHECK_NEAR(mc.S(1, 1), 32.0 - 64.0 / 9.0, 1e-12);
}

// ---------------------------------------------------------------------------------------
// B10 SUPPLEMENT: mixed-data frame -- CensoringAsymmetryScore bounds + CON-23 invariant
// ---------------------------------------------------------------------------------------

// A frame exercising every accumulation branch: 20 exact points (2 flagged low outliers via
// the threshold path), one interval record, one uncertain record, and one threshold record
// (window 1900-1950, duration 51, NumberAbove = 2 -> NumberBelow = 49 after
// process_threshold_series, since no explicit data falls inside the window).
DataFrame create_mixed_data_frame() {
    DataFrame df;
    df.exact_series().add(ExactData(1970, 5.0));
    df.exact_series().add(ExactData(1971, 8.0));
    for (int i = 2; i < 20; ++i)
        df.exact_series().add(ExactData(1970 + i, 100.0 + 10.0 * i));
    df.interval_series().add(IntervalData(1955, 60.0, 80.0, 100.0));
    df.uncertain_series().add(UncertainData(1960, std::make_unique<Normal>(150.0, 20.0)));
    ThresholdData threshold(1900, 1950, 200.0);
    threshold.set_number_above(2);
    df.threshold_series().add(std::move(threshold));
    df.set_low_outlier_threshold(10.0);
    df.set_low_outliers_from_threshold();
    df.calculate_plotting_positions();
    return df;
}

void test_censoring_asymmetry_score_bounded_mixed_frame() {
    Bulletin17CDistribution model(create_mixed_data_frame(), UnivariateDistributionType::Normal);
    CHECK_EQ(model.number_of_parameters(), 2);
    CHECK_EQ(model.data_frame().number_of_low_outliers(), 2);

    std::vector<double> theta;
    for (const auto& mp : model.parameters()) theta.push_back(mp.value());

    std::vector<double> score = model.censoring_asymmetry_score(theta);
    CHECK_EQ(static_cast<int>(score.size()), 2);
    for (double s : score) {
        CHECK_TRUE(std::isfinite(s));
        CHECK_TRUE(s >= -1.0 && s <= 1.0);
    }
}

// The CON-23 invariant must also hold when every data type (low outliers, exact, uncertain,
// interval, threshold left+right blocks) contributes weighted rows.
void test_pointwise_column_means_match_g_mixed_frame() {
    Bulletin17CDistribution model(create_mixed_data_frame(), UnivariateDistributionType::Normal);
    std::vector<double> p;
    for (const auto& mp : model.parameters()) p.push_back(mp.value());

    bestfit::estimation::MomentConditionResult mc = model.moment_conditions(p);
    auto pointwise = model.pointwise_moment_conditions()(p);

    // Row count: 20 exact + 1 interval + 1 uncertain + 49 below + 2 above = 73.
    int n = static_cast<int>(pointwise.size());
    CHECK_EQ(n, model.data_frame().total_record_length());
    CHECK_EQ(n, 73);

    int q = static_cast<int>(pointwise[0].size());
    for (int j = 0; j < q; ++j) {
        double col_mean = 0.0;
        for (int i = 0; i < n; ++i)
            col_mean += pointwise[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
        col_mean /= n;
        CHECK_NEAR(mc.G[j], col_mean, 1e-9);
    }
}

// ---------------------------------------------------------------------------------------
// B10 SUPPLEMENT: QuantileVariance vs an independent finite-difference delta method
// ---------------------------------------------------------------------------------------

// Independently form g_fd' Sigma g_fd with a central-difference quantile gradient computed
// straight from the base distribution's inverse CDF (never calling QuantileGradient), then
// compare against quantile_variance. Tolerance sits in the rel 1e-5..1e-8
// optimizer/covariance band (finite-difference truncation is the limiting term).
void test_quantile_variance_matches_finite_difference() {
    auto fd_quadratic_form = [](const std::function<double(const std::vector<double>&)>& quantile,
                                const std::vector<double>& theta,
                                const std::vector<std::vector<double>>& sigma) {
        std::size_t p = theta.size();
        std::vector<double> grad(p);
        for (std::size_t i = 0; i < p; ++i) {
            double h = 1e-6 * std::max(1.0, std::fabs(theta[i]));
            std::vector<double> up = theta, down = theta;
            up[i] += h;
            down[i] -= h;
            grad[i] = (quantile(up) - quantile(down)) / (2.0 * h);
        }
        double result = 0.0;
        for (std::size_t i = 0; i < p; ++i)
            for (std::size_t j = 0; j < p; ++j) result += grad[i] * grad[j] * sigma[i][j];
        return result;
    };

    // Normal-backed B17C (the IStandardError QuantileGradient branch).
    {
        Bulletin17CDistribution model(create_flood_data_frame(),
                                      UnivariateDistributionType::Normal);
        std::vector<double> theta = {100.0, 15.0};
        std::vector<std::vector<double>> sigma = {{2.0, 0.3}, {0.3, 0.5}};
        double expected = fd_quadratic_form(
            [](const std::vector<double>& t) { return Normal(t[0], t[1]).inverse_cdf(0.99); },
            theta, sigma);
        double actual = model.quantile_variance(0.99, theta, sigma);
        CHECK_TRUE(std::fabs(actual - expected) <= 1e-6 * std::fabs(expected));
    }

    // LP3-backed B17C (the PearsonTypeIII QuantileGradientForMoments branch; the gradient
    // is computed on the PT3 base in log space, so the finite difference runs on PT3 too).
    {
        Bulletin17CDistribution model(create_flood_data_frame(),
                                      UnivariateDistributionType::LogPearsonTypeIII);
        std::vector<double> theta = {3.0, 0.25, 0.1};
        std::vector<std::vector<double>> sigma = {
            {2e-3, 1e-4, 5e-5}, {1e-4, 8e-4, 2e-5}, {5e-5, 2e-5, 6e-4}};
        double expected = fd_quadratic_form(
            [](const std::vector<double>& t) {
                return PearsonTypeIII(t[0], t[1], t[2]).inverse_cdf(0.99);
            },
            theta, sigma);
        double actual = model.quantile_variance(0.99, theta, sigma);
        CHECK_TRUE(std::fabs(actual - expected) <= 1e-5 * std::fabs(expected));
    }

    // Guards: probability must be in (0, 1); parameter/covariance arity must match.
    {
        Bulletin17CDistribution model(create_flood_data_frame(),
                                      UnivariateDistributionType::Normal);
        CHECK_THROWS(model.quantile_gradient(0.0, {100.0, 15.0}));
        CHECK_THROWS(model.quantile_gradient(1.0, {100.0, 15.0}));
        CHECK_THROWS(model.quantile_gradient(0.5, {100.0}));
        CHECK_THROWS(model.quantile_variance(0.5, {100.0, 15.0}, {{1.0}}));
    }
}

// ---------------------------------------------------------------------------------------
// B10 SUPPLEMENT: GeneralizedMethodOfMoments end-to-end smoke (finiteness only; the exact
// fit oracles land at B12)
// ---------------------------------------------------------------------------------------

void test_gmm_estimate_end_to_end_smoke() {
    Bulletin17CDistribution model(create_flood_data_frame(),
                                  UnivariateDistributionType::LogPearsonTypeIII);
    GeneralizedMethodOfMoments gmm(model);

    CHECK_TRUE(gmm.estimate());

    const std::vector<double>& values = gmm.best_parameter_set().values;
    CHECK_EQ(static_cast<int>(values.size()), 3);
    for (double v : values) CHECK_TRUE(std::isfinite(v));

    auto covariance = gmm.get_covariance_matrix();
    CHECK_EQ(covariance.number_of_rows(), 3);
    for (int i = 0; i < 3; ++i) {
        CHECK_TRUE(std::isfinite(covariance(i, i)));
        CHECK_TRUE(covariance(i, i) >= 0.0);
        for (int j = 0; j < 3; ++j) CHECK_TRUE(std::isfinite(covariance(i, j)));
    }
}

}  // namespace

int main() {
    test_constructor_default_uses_log_pearson_type_iii();
    test_constructor_with_data_frame_and_type_builds_parameters();
    test_constructor_with_distribution_instance();
    test_constructor_null_distribution_throws();
    test_is_supported_distribution_type_accepts_b17c_family();
    test_is_supported_distribution_type_rejects_non_b17c_family();
    test_create_distribution_all_supported_types();
    test_create_distribution_unsupported_type_throws();
    test_is_nonstationary_always_false();
    test_set_parameter_values_wrong_length_throws();
    test_set_parameter_values_round_trip();
    test_sample_size_null_data_frame_is_zero();
    test_validate_good_fixture_is_valid();
    test_validate_null_data_frame_is_invalid();
    test_validate_log_distribution_non_positive_data_is_invalid();
    test_clone_returns_separate_instance_with_matching_state();
    test_generate_random_values_fixed_seed_is_deterministic();
    test_generate_random_values_non_positive_sample_size_throws();
    test_generate_random_values_requires_distribution();

    test_gmm_model_constructor_wiring();
    test_set_penalty_function_deterministic();
    test_set_penalty_function_random_is_seed_deterministic();
    test_set_random_penalty_function_guards_and_value();
    test_set_initial_parameters_restores_initials();

    test_nonparametric_moments_insufficient_data_returns_null();
    test_nonparametric_moments_ros_fallback_no_outliers();
    test_nonparametric_moments_symmetric_sample();
    test_nonparametric_moments_ros_imputes_from_regression();

    test_pointwise_column_means_match_g_lp3();
    test_pointwise_column_means_match_g_normal();
    test_weds_natural_parameters_ignores_current_link_controller();
    test_weds_from_linked_inverse_links_before_scoring();
    test_moment_conditions_exact_data_hand_computed();
    test_censoring_asymmetry_score_bounded_mixed_frame();
    test_pointwise_column_means_match_g_mixed_frame();
    test_quantile_variance_matches_finite_difference();
    test_gmm_estimate_end_to_end_smoke();

    return bftest::summary("bulletin17c");
}
