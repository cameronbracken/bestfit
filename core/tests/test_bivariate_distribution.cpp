// B1 support ctest (C++-only): the BivariateDistribution copula-coupled bivariate model
// (two IUnivariateModel marginals + a BivariateCopula). Internal-support surface, so
// hardcoded oracles are correct here (public-API oracle values otherwise live in fixtures/).
//
// Oracles transcribed UNALTERED from three upstream test files under
// upstream/RMC-BestFit/src/RMC.BestFit.Tests/Bivariate/ @ fc28c0c:
//   - BivariateDistributionTests.cs            (structural per-copula parameter counts, LL
//                                               finiteness, pointwise sum invariant)
//   - BivariateDistributionMarginalInterfaceTests.cs (the P1 IUnivariateModel guard, the
//                                               explicit-interface Distribution accessors,
//                                               the IsNonstationary guards, and the
//                                               mixed-marginal construction/swap acceptance)
//   - BivariateDistributionStudentTTests.cs    (the StudentT 2-parameter copula path)
//
// The 200-sample fixtures in BivariateDistributionTests.cs are `new Normal(100,15)
// .GenerateRandomValues(200, 12345)` / `new Gumbel(50,15).GenerateRandomValues(200, 67890)`;
// the bit-exact Mersenne Twister reproduces them identically in C++ here.
//
// P4 landed (see test_bivariate_p4_fixed_param_oracle below + fixtures/estimation/*):
//   - DataLogLikelihood at fixed copula parameters (exact value) under IFM, dumped from the real
//     RMC.BestFit via the dotnet emitter and asserted here (route b, C++-only, 1e-9 abs);
//   - the exact MLE-recovered copula parameter(s) under IFM (Normal 1-param + StudentT 2-param)
//     and the seeded ISimulatable<Matrix2D> draw are oracle-verified cross-language against the
//     real C# in bivariate_smoke.json / bivariate_sim.json.
// PseudoLikelihood DEFERRED (upstream finding, see the P4 report): the real C# model-level MLE
//   returns Estimate()=false under PseudoLikelihood because the marginal plotting positions are
//   never calculated on the shared builder path (the C++ port instead returns a degenerate
//   ~0.5 without validating) -- reconciling that lifecycle divergence is a P5 follow-up.
//
// SKIPPED C# test methods (with reason):
//   - Test_ToXElement_ContainsRequiredAttributes / Test_FromXElement_RestoresModel
//     (BivariateDistributionTests.cs) and XElementRoundTrip_StudentT_PreservesBothParameters
//     (StudentT): XML serialization (ToXElement / XElement ctor) is a project-wide non-port.
//   - Test_Constructor_DifferentCopulaTypes / Test_LogLikelihood_DifferentCopulas: the C#
//     `Assert.Inconclusive`-on-throw pattern reduces to construct-doesn't-throw + finite-LL,
//     already covered by the Normal/Frank/Gumbel/Clayton construction checks below.
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "corehydro/models/bivariate_distribution/bivariate_distribution.hpp"
#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "corehydro/models/univariate_distribution/mixture_model.hpp"
#include "corehydro/models/univariate_distribution/point_process_model.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/copulas/base/copula_type.hpp"
#include "corehydro/numerics/distributions/copulas/normal_copula.hpp"
#include "corehydro/numerics/distributions/copulas/student_t_copula.hpp"
#include "corehydro/numerics/distributions/gumbel.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::models::BivariateDistribution;
using corehydro::models::Bulletin17CDistribution;
using corehydro::models::DataFrame;
using corehydro::models::ExactSeries;
using corehydro::models::IUnivariateModel;
using corehydro::models::MixtureModel;
using corehydro::models::PointProcessModel;
using corehydro::models::UnivariateDistributionModel;
using corehydro::numerics::distributions::Gumbel;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::distributions::copulas::CopulaType;
using corehydro::numerics::distributions::copulas::NormalCopula;
using corehydro::numerics::distributions::copulas::StudentTCopula;

namespace {

constexpr double kNegInf = -std::numeric_limits<double>::infinity();
constexpr int kFixtureSize = 200;

// ---------------------------------------------------------------------------------------
// The P1 INTERFACE GUARD (BivariateDistributionMarginalInterfaceTests.cs
// TargetModelClasses_ImplementIUnivariateModel, lines 48-53). In C# this is
// `typeof(IUnivariateModel).IsAssignableFrom(typeof(T))` at runtime; the faithful C++
// transcription is a compile-time convertibility check per model type. THIS is what P1's
// resolution on UnivariateDistributionModel makes compile.
static_assert(std::is_convertible_v<UnivariateDistributionModel*, IUnivariateModel*>,
              "UnivariateDistribution must implement IUnivariateModel");
static_assert(std::is_convertible_v<Bulletin17CDistribution*, IUnivariateModel*>,
              "Bulletin17CDistribution must implement IUnivariateModel");
static_assert(std::is_convertible_v<PointProcessModel*, IUnivariateModel*>,
              "PointProcessModel must implement IUnivariateModel");
static_assert(std::is_convertible_v<MixtureModel*, IUnivariateModel*>,
              "MixtureModel must implement IUnivariateModel");

// BivariateDistribution IS a ModelBase and IS an ISimulatable (structural type check).
static_assert(std::is_convertible_v<BivariateDistribution*, corehydro::models::ModelBase*>,
              "BivariateDistribution must derive ModelBase");

// C# SampleX / SampleY (interface + StudentT tests, 20 values each).
const std::vector<double> kSampleX = {98.1,  102.7, 115.3, 88.4,  104.9, 92.0,  110.5,
                                      99.2,  107.6, 101.3, 95.8,  108.1, 103.4, 97.6,
                                      112.9, 89.5,  106.2, 100.0, 93.7,  109.4};
const std::vector<double> kSampleY = {75.2, 82.1, 93.6, 68.7, 84.3, 72.5, 90.4,
                                      78.9, 88.5, 81.0, 74.1, 87.6, 82.8, 76.3,
                                      91.5, 69.2, 86.0, 80.4, 73.5, 89.1};

// Build a pre-fit Normal UnivariateDistribution marginal (C# BuildNormalMarginal /
// CreateFittedMarginals): parameters seeded so Validate() passes without an MLE call.
UnivariateDistributionModel build_normal_marginal(double mu, double sigma,
                                                  const std::vector<double>& data) {
    DataFrame df;
    df.set_exact_series(ExactSeries(data));
    UnivariateDistributionModel model(std::move(df), UnivariateDistributionType::Normal);
    model.set_parameter_values({mu, sigma});
    return model;
}

// Build a seeded 200-sample Normal(100,15) marginal (BivariateDistributionTests.cs inline
// fixture); parameters explicitly set to the generating values (C# CreateConfiguredModel).
UnivariateDistributionModel build_configured_normal_marginal(int n) {
    std::vector<double> data = Normal(100.0, 15.0).generate_random_values(kFixtureSize, 12345);
    data.resize(static_cast<std::size_t>(n));
    DataFrame df;
    df.set_exact_series(ExactSeries(data));
    UnivariateDistributionModel model(std::move(df), UnivariateDistributionType::Normal);
    model.set_parameter_values({100.0, 15.0});
    return model;
}

UnivariateDistributionModel build_configured_gumbel_marginal(int n) {
    std::vector<double> data = Gumbel(50.0, 15.0).generate_random_values(kFixtureSize, 67890);
    data.resize(static_cast<std::size_t>(n));
    DataFrame df;
    df.set_exact_series(ExactSeries(data));
    UnivariateDistributionModel model(std::move(df), UnivariateDistributionType::Gumbel);
    model.set_parameter_values({50.0, 15.0});
    return model;
}

std::vector<double> current_parameters(const BivariateDistribution& model) {
    std::vector<double> p;
    for (const auto& mp : model.parameters()) p.push_back(mp.value());
    return p;
}

// ======================= BivariateDistributionTests.cs =================================

// Test_Constructor_EmptyConstructor.
void test_constructor_empty() {
    BivariateDistribution model;
    CHECK_TRUE(dynamic_cast<NormalCopula*>(&model.copula()) != nullptr);
    CHECK_TRUE(model.copula_type() == CopulaType::Normal);
}

// Test_Constructor_WithMarginals.
void test_constructor_with_marginals() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::Normal);

    CHECK_TRUE(model.marginal_x() != nullptr);
    CHECK_TRUE(model.marginal_y() != nullptr);
    CHECK_TRUE(dynamic_cast<NormalCopula*>(&model.copula()) != nullptr);
}

// Test_Constructor_DifferentCopulaTypes (Normal/Gumbel/Clayton/Frank all construct).
void test_constructor_different_copula_types() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);

    for (CopulaType t : {CopulaType::Normal, CopulaType::Gumbel, CopulaType::Clayton,
                         CopulaType::Frank}) {
        BivariateDistribution model(mx, my, t);
        CHECK_TRUE(model.copula_type() == t);
    }
}

// Test_Parameters_NormalCopula_HasOneCopulaParameter.
void test_parameters_normal_copula_one_parameter() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::Normal);

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(1));
    CHECK_TRUE(model.parameters()[0].name() == "Dependency (θ)");
}

// Test_Parameters_StudentTCopula_HasTwoCopulaParameters.
void test_parameters_studentt_copula_two_parameters() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::StudentT);

    CHECK_EQ(model.parameters().size(), static_cast<std::size_t>(2));
    CHECK_TRUE(model.parameters()[1].name() == "DegreesOfFreedom");
}

// Test_LogLikelihood_ReturnsFiniteValue.
void test_log_likelihood_finite() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::Normal);

    std::vector<double> p = current_parameters(model);
    double ll = model.log_likelihood(p);
    CHECK_TRUE(!std::isnan(ll));
    CHECK_TRUE(!std::isinf(ll));
}

// Test_LogLikelihood_DifferentCopulas (Normal + Frank produce non-NaN LL).
void test_log_likelihood_different_copulas() {
    for (CopulaType t : {CopulaType::Normal, CopulaType::Frank}) {
        UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
        UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
        BivariateDistribution model(mx, my, t);
        std::vector<double> p = current_parameters(model);
        double ll = model.log_likelihood(p);
        CHECK_TRUE(!std::isnan(ll));
    }
}

// Test_PointwiseDataLogLikelihood_SumsToTotal_AtKnownParameters.
void test_pointwise_sums_to_total() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::Normal);

    std::vector<double> p = current_parameters(model);
    std::vector<double> pointwise = model.pointwise_data_log_likelihood(p);
    double sum = 0.0;
    for (double v : pointwise) sum += v;
    double total = model.data_log_likelihood(p);
    CHECK_NEAR(sum, total, 1e-6);
}

// Test_PointwiseDataLogLikelihoodComponents_ReturnsCorrectCount.
void test_pointwise_components_count() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::Normal);

    std::vector<double> p = current_parameters(model);
    auto components = model.pointwise_data_log_likelihood_components(p);
    CHECK_EQ(components.size(), static_cast<std::size_t>(kFixtureSize));
}

// Test_Validate_ValidModel_ReturnsTrue.
void test_validate_valid_model() {
    UnivariateDistributionModel mx = build_configured_normal_marginal(kFixtureSize);
    UnivariateDistributionModel my = build_configured_gumbel_marginal(kFixtureSize);
    BivariateDistribution model(mx, my, CopulaType::Normal);

    CHECK_TRUE(model.validate().is_valid);
}

// ================= BivariateDistributionMarginalInterfaceTests.cs ======================

// MixtureModel_ExplicitInterfaceDistribution_ReturnsMixtureProperty: the interface accessor
// routes to the same underlying Mixture object the concrete mixture() accessor exposes.
void test_mixture_interface_distribution_routes() {
    MixtureModel model;  // default: two-Normal mixture
    IUnivariateModel& iface = model;
    CHECK_TRUE(iface.distribution() == model.mixture());
}

// PointProcessModel_ExplicitInterfaceDistribution_ReturnsSameObject: the interface accessor
// routes to the same underlying CompetingRisks object.
void test_point_process_interface_distribution_routes() {
    PointProcessModel model;  // default ctor builds a CompetingRisks
    IUnivariateModel& iface = model;
    CHECK_TRUE(iface.distribution() == model.distribution());
}

// NonTrendModels_ReportStationary.
void test_non_trend_models_report_stationary() {
    Bulletin17CDistribution b17c;
    PointProcessModel pot;
    MixtureModel mix;

    CHECK_TRUE(static_cast<IUnivariateModel&>(b17c).is_nonstationary() == false);
    CHECK_TRUE(static_cast<IUnivariateModel&>(pot).is_nonstationary() == false);
    CHECK_TRUE(static_cast<IUnivariateModel&>(mix).is_nonstationary() == false);
}

// Construct_WithMixedMarginalTypes_DoesNotThrow: the ctor accepts any IUnivariateModel pair
// (Normal marginal for X, Mixture-backed marginal for Y) and preserves marginal identity.
void test_construct_with_mixed_marginal_types() {
    UnivariateDistributionModel normal_x = build_normal_marginal(101.0, 7.5, kSampleX);

    DataFrame df_y;
    df_y.set_exact_series(ExactSeries(kSampleY));
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components;
    components.push_back(std::make_unique<Normal>(77.0, 5.0));
    components.push_back(std::make_unique<Normal>(88.0, 5.0));
    MixtureModel mix_y(std::move(df_y), std::move(components));

    IUnivariateModel& ix = normal_x;
    IUnivariateModel& iy = mix_y;
    BivariateDistribution bivariate(ix, iy, CopulaType::Normal);

    CHECK_TRUE(bivariate.marginal_x() == &ix);
    CHECK_TRUE(bivariate.marginal_y() == &iy);
}

// SwapMarginal_AcrossInterfaceImplementations_Succeeds: swapping X from a Univariate to a
// Mixture-backed marginal via the setter preserves identity and does not throw.
void test_swap_marginal_across_implementations() {
    UnivariateDistributionModel orig_x = build_normal_marginal(101.0, 7.5, kSampleX);
    UnivariateDistributionModel orig_y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution bivariate(orig_x, orig_y, CopulaType::Normal);

    DataFrame df_x2;
    df_x2.set_exact_series(ExactSeries(kSampleX));
    std::vector<std::unique_ptr<UnivariateDistributionBase>> components;
    components.push_back(std::make_unique<Normal>(95.0, 10.0));
    components.push_back(std::make_unique<Normal>(110.0, 10.0));
    MixtureModel new_x(std::move(df_x2), std::move(components));

    IUnivariateModel& inew_x = new_x;
    bivariate.set_marginal_x(&inew_x);
    CHECK_TRUE(bivariate.marginal_x() == &inew_x);
}

// ==================== BivariateDistributionStudentTTests.cs =============================

// CreateCopula_StudentT_ReturnsStudentTCopula.
void test_create_copula_studentt() {
    auto copula = BivariateDistribution::create_copula(CopulaType::StudentT);
    CHECK_TRUE(copula != nullptr);
    CHECK_TRUE(dynamic_cast<StudentTCopula*>(copula.get()) != nullptr);
    CHECK_TRUE(copula->type() == CopulaType::StudentT);
    CHECK_EQ(copula->number_of_copula_parameters(), 2);
}

// CopulaType_SwitchToStudentT_ReplacesCopula.
void test_copula_type_switch_to_studentt() {
    BivariateDistribution dist;  // defaults to Normal
    CHECK_TRUE(dynamic_cast<NormalCopula*>(&dist.copula()) != nullptr);

    dist.set_copula_type(CopulaType::StudentT);
    CHECK_TRUE(dynamic_cast<StudentTCopula*>(&dist.copula()) != nullptr);
    CHECK_TRUE(dist.copula_type() == CopulaType::StudentT);
}

// CopulaType_SwitchNormalToStudentT_RebuildsParametersTwoEntries.
void test_copula_type_switch_normal_to_studentt_rebuilds() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::Normal);

    CHECK_EQ(dist.parameters().size(), static_cast<std::size_t>(1));
    CHECK_TRUE(dist.parameters()[0].name() == "Dependency (θ)");

    dist.set_copula_type(CopulaType::StudentT);
    CHECK_EQ(dist.parameters().size(), static_cast<std::size_t>(2));
    CHECK_TRUE(dist.parameters()[0].name() == "Dependency (θ)");
    CHECK_TRUE(dist.parameters()[1].name() == "DegreesOfFreedom");
}

// CopulaType_SwitchStudentTToNormal_RebuildsParametersOneEntry.
void test_copula_type_switch_studentt_to_normal_rebuilds() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    CHECK_EQ(dist.parameters().size(), static_cast<std::size_t>(2));
    dist.set_copula_type(CopulaType::Normal);
    CHECK_EQ(dist.parameters().size(), static_cast<std::size_t>(1));
    CHECK_TRUE(dist.parameters()[0].name() == "Dependency (θ)");
}

// SetDefaultParameters_StudentT_ProducesTwoParametersWithCorrectBoundsAndInitialValues.
void test_set_default_parameters_studentt_bounds_and_initials() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    CHECK_EQ(dist.parameters().size(), static_cast<std::size_t>(2));
    const auto& theta = dist.parameters()[0];
    const auto& df = dist.parameters()[1];

    CHECK_TRUE(theta.name() == "Dependency (θ)");
    CHECK_TRUE(df.name() == "DegreesOfFreedom");

    CHECK_TRUE(theta.lower_bound() > -1.0 && theta.lower_bound() < 0.0);
    CHECK_TRUE(theta.upper_bound() < 1.0 && theta.upper_bound() > 0.0);
    CHECK_TRUE(df.lower_bound() > 2.0 && df.lower_bound() < 3.0);
    CHECK_NEAR(df.upper_bound(), 30.0, 1e-9);

    CHECK_TRUE(theta.value() >= theta.lower_bound() && theta.value() <= theta.upper_bound());
    CHECK_NEAR(df.value(), 5.0, 1e-9);

    CHECK_TRUE(theta.prior_distribution().parameters_valid());
    CHECK_TRUE(df.prior_distribution().parameters_valid());
}

// SetParameterValues_StudentT_UpdatesBothThetaAndDegreesOfFreedom.
void test_set_parameter_values_studentt_updates_both() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    dist.set_parameter_values({0.6, 8.0});
    CHECK_NEAR(dist.parameters()[0].value(), 0.6, 1e-9);
    CHECK_NEAR(dist.parameters()[1].value(), 8.0, 1e-9);
    CHECK_NEAR(dist.copula().theta(), 0.6, 1e-9);
    CHECK_NEAR(dynamic_cast<StudentTCopula&>(dist.copula()).degrees_of_freedom(), 8.0, 1e-9);
}

// SetParameterValues_StudentT_NonIntegerDf_IsPreserved.
void test_set_parameter_values_studentt_non_integer_df() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    dist.set_parameter_values({0.5, 7.25});
    CHECK_NEAR(dist.parameters()[1].value(), 7.25, 1e-12);
    CHECK_NEAR(dynamic_cast<StudentTCopula&>(dist.copula()).degrees_of_freedom(), 7.25, 1e-12);
}

// DataLogLikelihood_StudentT_WrongParameterCount_ReturnsNegativeInfinity.
void test_data_log_likelihood_studentt_wrong_count() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    std::vector<double> wrong = {0.5};  // too few -- StudentT needs 2
    double ll = dist.data_log_likelihood(wrong);
    CHECK_TRUE(ll == kNegInf);
}

// DataLogLikelihood_StudentT_CorrectParameterCount_ReturnsFinite.
void test_data_log_likelihood_studentt_correct_count() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    std::vector<double> p = {0.5, 5.0};
    double ll = dist.data_log_likelihood(p);
    CHECK_TRUE(!std::isnan(ll));
    CHECK_TRUE(!std::isinf(ll));
}

// Validate_StudentT_WellConfigured_IsValid.
void test_validate_studentt_well_configured() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    CHECK_TRUE(dist.validate().is_valid);
}

// Validate_StudentT_WrongParameterCount_IsInvalid: remove the 2nd parameter, leaving a stale
// 1-parameter config against a 2-parameter StudentT copula.
void test_validate_studentt_wrong_parameter_count() {
    UnivariateDistributionModel x = build_normal_marginal(101.8, 7.5, kSampleX);
    UnivariateDistributionModel y = build_normal_marginal(80.8, 7.5, kSampleY);
    BivariateDistribution dist(x, y, CopulaType::StudentT);

    dist.parameters().pop_back();  // simulate a stale 1-parameter configuration
    auto result = dist.validate();
    CHECK_TRUE(!result.is_valid);
    bool found = false;
    for (const auto& m : result.validation_messages)
        if (m.find("unexpected size") != std::string::npos) found = true;
    CHECK_TRUE(found);
}

// P4 oracle (route b): exact fixed-parameter DataLogLikelihood under InferenceFromMargins, dumped
// from the REAL RMC.BestFit BivariateDistribution via tools/oracle_emitter (Numerics @ a2c4dbf,
// RMC-BestFit @ fc28c0c) on the shared bivariate fixture (Normal copula, two fixed Normal
// marginals). Deterministic (no fit) -> hardcoded C++-only oracle at 1e-9 abs. The exact IFM +
// StudentT MLE fits and the seeded ISimulatable<Matrix2D> draw are oracle-verified cross-language
// in fixtures/estimation/bivariate_smoke.json + bivariate_sim.json.
void test_bivariate_p4_fixed_param_oracle() {
    UnivariateDistributionModel mx = build_normal_marginal(
        11.66, 2.0, {10, 12, 9, 14, 11, 13, 8, 15, 10.5, 12.5, 11.2, 13.3, 9.4, 14.1, 12.2});
    UnivariateDistributionModel my = build_normal_marginal(
        22.7, 3.0, {21, 20, 22, 25, 19, 27, 18, 24, 23, 21, 26, 22, 20, 28, 25});
    BivariateDistribution model(mx, my, CopulaType::Normal);  // default estimation = IFM.
    std::vector<double> p{0.6};  // fixed copula dependency parameter.
    CHECK_NEAR(model.data_log_likelihood(p), 3.8175438947131464, 1e-9);
}

}  // namespace

int main() {
    test_constructor_empty();
    test_constructor_with_marginals();
    test_constructor_different_copula_types();
    test_parameters_normal_copula_one_parameter();
    test_parameters_studentt_copula_two_parameters();
    test_log_likelihood_finite();
    test_log_likelihood_different_copulas();
    test_pointwise_sums_to_total();
    test_pointwise_components_count();
    test_validate_valid_model();

    test_mixture_interface_distribution_routes();
    test_point_process_interface_distribution_routes();
    test_non_trend_models_report_stationary();
    test_construct_with_mixed_marginal_types();
    test_swap_marginal_across_implementations();

    test_create_copula_studentt();
    test_copula_type_switch_to_studentt();
    test_copula_type_switch_normal_to_studentt_rebuilds();
    test_copula_type_switch_studentt_to_normal_rebuilds();
    test_set_default_parameters_studentt_bounds_and_initials();
    test_set_parameter_values_studentt_updates_both();
    test_set_parameter_values_studentt_non_integer_df();
    test_data_log_likelihood_studentt_wrong_count();
    test_data_log_likelihood_studentt_correct_count();
    test_validate_studentt_well_configured();
    test_validate_studentt_wrong_parameter_count();

    test_bivariate_p4_fixed_param_oracle();

    return chtest::summary("bivariate_distribution");
}
