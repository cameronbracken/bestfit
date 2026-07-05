// Transcribed from: upstream/RMC-BestFit/src/RMC.BestFit.Tests/Univariate/
// Bulletin17CDistributionTests.cs (@ fc28c0c) for the Phase 6 Task B9 construction slice.
// B10 extends this file with the moment-machinery tests.
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
//   GenerateRandomValues_RequiresDistribution.
//
// SKIPPED upstream methods (each with reason):
//   - ToXElement_FromXElement_RoundTripsCoreState, FromXElement_NoDataFrame_ConstructsWithoutThrowing,
//     FromXElement_NullXml_Throws: XML serialization is a project-wide deliberate skip.
//   - PointwiseMomentConditions_ColumnMeans_MatchMomentConditionsG_LP3 / _Normal,
//     WeightedErrorDirectionScore_NaturalParameters_IgnoresCurrentLinkController,
//     WeightedErrorDirectionScoreFromLinked_InverseLinksBeforeScoring: the moment machinery
//     (MomentConditions, PointwiseMomentConditions, WEDS) is Task B10's slice.
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
//   - SetInitialParameters restores constraint initials.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/estimation/generalized_method_of_moments.hpp"
#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "bestfit/numerics/distributions/log_pearson_type_iii.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

using bestfit::estimation::GeneralizedMethodOfMoments;
using bestfit::models::Bulletin17CDistribution;
using bestfit::models::DataFrame;
using bestfit::models::ExactData;
using bestfit::numerics::distributions::LogNormal;
using bestfit::numerics::distributions::LogPearsonTypeIII;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::UnivariateDistributionType;
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
    // The moment-condition machinery is B10; the B9 accessor returns a callable stub that
    // throws std::logic_error when invoked.
    CHECK_THROWS(gmm.moment_condition_function()({100.0, 15.0}));
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

    return bftest::summary("bulletin17c");
}
