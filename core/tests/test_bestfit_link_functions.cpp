// Transcribed C# oracle tests for the BestFit link-function layer (Task B2):
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/ASinHLinkTests.cs        @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/SESLinkTests.cs          @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/LogSESLinkTests.cs       @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/LogASinHLinkTests.cs     @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/CenteredLinkTests.cs     @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/YeoJohnsonLinkTests.cs   @ fc28c0c
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/LinkFunctions/LinkFunctionSerializationTests.cs
//     @ fc28c0c (non-XML factory assertions only)
// plus brief-required supplements (SES/LogSES Newton-convergence diagnostics).
//
// Skipped upstream test methods (exhaustive; all are XML-serialization surface unless
// noted -- the port drops ToXElement / the XElement constructors / CreateFromXElement
// per the repo-wide rule):
//   ASinHLinkTests.cs: Test_Serialization_RoundTrip_Symmetric,
//     Test_Serialization_RoundTrip_Asymmetric, Test_ToXElement_ElementName,
//     Test_Factory_CreatesASinHLink (factory is XElement-based upstream; the enum
//     factory is covered below).
//   SESLinkTests.cs: none skipped (no XML tests in the file).
//   LogSESLinkTests.cs: none skipped (no XML tests in the file).
//   LogASinHLinkTests.cs: Serialization_RoundTrip_PreservesProperties,
//     BestFitFactory_CreatesLogASinHLink (XElement-based; enum factory covered below).
//   CenteredLinkTests.cs: none skipped. Test_Constructor_NullInner_Throws is
//     transcribed (nullptr unique_ptr maps the C# null reference).
//   YeoJohnsonLinkTests.cs: Constructor_Values_Null_ThrowsArgumentNullException (a C++
//     std::vector cannot be null -- no ArgumentNullException analog),
//     Constructor_XElement_Null_ThrowsArgumentNullException, XmlRoundTrip_PreservesLambda,
//     ToXElement_ElementNameIsYeoJohnsonLink, XmlRoundTrip_ThenLinkInverse_RecoverX (XML).
//   LinkFunctionSerializationTests.cs: SESLink_RoundTrip, SESLink_ToXElement_ElementName,
//     LogSESLink_RoundTrip, LogSESLink_ToXElement_ElementName, LogASinHLink_RoundTrip,
//     LogASinHLink_ToXElement_ElementName, CenteredLink_RoundTrip_WithSESInner,
//     CenteredLink_RoundTrip_WithIdentityInner, CenteredLink_ToXElement_ElementName,
//     LinkController_RoundTrip_WithBestFitLinks, LinkController_RoundTrip_NullSlots (all
//     XML round-trips). The non-XML type-dispatch assertions of
//     BestFitLinkFunctionFactory_AllTypes and BestFitLinkFunctionFactory_UnknownType_Throws
//     are transcribed against the enum factory below.
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/models/link_functions/asinh_link.hpp"
#include "bestfit/models/link_functions/best_fit_link_function_factory.hpp"
#include "bestfit/models/link_functions/centered_link.hpp"
#include "bestfit/models/link_functions/log_asinh_link.hpp"
#include "bestfit/models/link_functions/log_ses_link.hpp"
#include "bestfit/models/link_functions/ses_link.hpp"
#include "bestfit/models/link_functions/yeo_johnson_link.hpp"
#include "bestfit/numerics/functions/identity_link.hpp"
#include "bestfit/numerics/functions/link_function_type.hpp"
#include "bestfit/numerics/functions/log_link.hpp"
#include "bestfit/numerics/functions/logit_link.hpp"
#include "bestfit/numerics/functions/probit_link.hpp"
#include "bestfit/numerics/functions/complementary_log_log_link.hpp"
#include "check.hpp"

using bestfit::models::link_functions::ASinHLink;
using bestfit::models::link_functions::BestFitLinkFunctionFactory;
using bestfit::models::link_functions::BestFitLinkFunctionType;
using bestfit::models::link_functions::CenteredLink;
using bestfit::models::link_functions::LogASinHLink;
using bestfit::models::link_functions::LogSESLink;
using bestfit::models::link_functions::SESLink;
using bestfit::numerics::functions::IdentityLink;
using bestfit::numerics::functions::ILinkFunction;
using bestfit::numerics::functions::LinkFunctionType;
using bestfit::numerics::functions::LogLink;
// The two YeoJohnsonLink classes collide by simple name; keep both fully qualified via
// namespace aliases (the B2 brief calls out this coexistence explicitly).
namespace bf_models = bestfit::models::link_functions;
namespace bf_numfun = bestfit::numerics::functions;

namespace {

// Constants transcribed from the upstream test files.
const double kDeltaH = 1e-7;         // finite-difference step (all files)
const double kRoundTripTol = 1e-10;  // ASinH / SES / LogASinH / Centered round-trip tol
const double kLogSesRoundTripTol = 1e-8;  // LogSESLinkTests.RoundTripTol
const double kYjRoundTripTol = 1e-9;      // YeoJohnsonLinkTests.RoundTripTol
const double kDerivativeTol = 1e-4;       // all files

// ══════════════════════════════════════════════
//  ASinHLinkTests.cs
// ══════════════════════════════════════════════

void test_asinh_constructor_default_sets_default_values() {
    ASinHLink link;
    CHECK_EQ(link.gamma0(), 0.0);
    CHECK_EQ(link.scale(), 1.0);
    CHECK_EQ(link.epsilon(), 0.0);
    CHECK_EQ(link.delta(), 1.0);
    CHECK_TRUE(!link.use_adaptive_epsilon());
    CHECK_EQ(link.parent_indicator(), 0.0);
    CHECK_EQ(link.epsilon_max(), 0.5);
    CHECK_EQ(link.epsilon_slope(), 1.0);
}

void test_asinh_constructor_two_param_symmetric_defaults() {
    ASinHLink link(/*gamma0=*/0.5, /*scale=*/0.3);
    CHECK_EQ(link.gamma0(), 0.5);
    CHECK_EQ(link.scale(), 0.3);
    CHECK_EQ(link.epsilon(), 0.0);
    CHECK_EQ(link.delta(), 1.0);
}

void test_asinh_constructor_four_param_sets_values() {
    ASinHLink link(0.5, 0.3, /*epsilon=*/0.2, /*delta=*/1.5);
    CHECK_EQ(link.gamma0(), 0.5);
    CHECK_EQ(link.scale(), 0.3);
    CHECK_EQ(link.epsilon(), 0.2);
    CHECK_EQ(link.delta(), 1.5);
}

void test_asinh_constructor_small_scale_floors_at_eps() {
    ASinHLink link(0.0, -1.0);
    CHECK_TRUE(link.scale() > 0.0);
}

void test_asinh_constructor_small_delta_floors_at_eps() {
    ASinHLink link(0.0, 1.0, 0.0, -1.0);
    CHECK_TRUE(link.delta() > 0.0);
}

void test_asinh_link_symmetric_at_center_returns_zero() {
    double gamma0s[] = {0.0, 0.5, -0.5, 1.2, -2.0};
    for (double gamma0 : gamma0s) {
        ASinHLink link(gamma0, 0.3);
        CHECK_NEAR(link.link(gamma0), 0.0, 1e-15);
    }
}

void test_asinh_inverse_link_symmetric_at_zero_returns_gamma0() {
    double gamma0s[] = {0.0, 0.5, -0.5, 1.2, -2.0};
    for (double gamma0 : gamma0s) {
        ASinHLink link(gamma0, 0.3);
        CHECK_NEAR(link.inverse_link(0.0), gamma0, 1e-15);
    }
}

void test_asinh_round_trip_symmetric_default_params() {
    ASinHLink link;
    double gammas[] = {-100.0, -10.0, -1.0, -0.01, 0.0, 0.01, 1.0, 10.0, 100.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_round_trip_symmetric_typical_lp3() {
    ASinHLink link(0.5, 0.3);
    double gammas[] = {-2.0, -0.5, 0.0, 0.5, 2.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_round_trip_positive_epsilon() {
    ASinHLink link(0.5, 0.3, 0.3);
    double gammas[] = {-5.0, -1.0, 0.0, 0.5, 1.0, 5.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_round_trip_negative_epsilon() {
    ASinHLink link(-0.3, 0.5, -0.4);
    double gammas[] = {-5.0, -1.0, 0.0, 0.5, 1.0, 5.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_round_trip_heavy_tails() {
    ASinHLink link(0.0, 1.0, 0.2, 1.5);
    double gammas[] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_round_trip_light_tails() {
    ASinHLink link(0.0, 1.0, 0.0, 0.5);
    double gammas[] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_d_link_symmetric_matches_finite_difference() {
    ASinHLink link;
    double gammas[] = {0.0, 0.5, -0.5, 2.0, -3.0, 10.0, -10.0};
    for (double gamma : gammas) {
        double analytical = link.d_link(gamma);
        double numerical =
            (link.link(gamma + kDeltaH) - link.link(gamma - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(analytical, numerical, kDerivativeTol);
    }
}

void test_asinh_d_link_asymmetric_matches_finite_difference() {
    ASinHLink link(0.3, 0.5, 0.3, 1.2);
    double gammas[] = {-2.0, -0.5, 0.0, 0.5, 2.0};
    for (double gamma : gammas) {
        double analytical = link.d_link(gamma);
        double numerical =
            (link.link(gamma + kDeltaH) - link.link(gamma - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(analytical, numerical, kDerivativeTol);
    }
}

void test_asinh_d_link_lp3_params_matches_finite_difference() {
    ASinHLink link(0.5, 0.3, 0.15);
    double gammas[] = {-1.0, 0.0, 0.5, 1.0, 3.0};
    for (double gamma : gammas) {
        double analytical = link.d_link(gamma);
        double numerical =
            (link.link(gamma + kDeltaH) - link.link(gamma - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(analytical, numerical, kDerivativeTol);
    }
}

void test_asinh_d_link_symmetric_always_positive() {
    ASinHLink link(0.5, 0.3);
    double test_points[] = {-100, -10, -1, -0.01, 0, 0.01, 0.5, 1, 10, 100};
    for (double gamma : test_points) CHECK_TRUE(link.d_link(gamma) > 0);
}

void test_asinh_d_link_asymmetric_always_positive() {
    ASinHLink link(0.0, 1.0, 0.4, 1.3);
    double test_points[] = {-100, -10, -1, -0.01, 0, 0.01, 0.5, 1, 10, 100};
    for (double gamma : test_points) CHECK_TRUE(link.d_link(gamma) > 0);
}

void test_asinh_link_symmetric_strictly_increasing() {
    ASinHLink link(0.0, 1.0);
    double sorted[] = {-100, -10, -1, 0, 1, 10, 100};
    for (int i = 0; i < 6; ++i) CHECK_TRUE(link.link(sorted[i + 1]) > link.link(sorted[i]));
}

void test_asinh_link_asymmetric_strictly_increasing() {
    ASinHLink link(0.0, 1.0, 0.3, 1.5);
    double sorted[] = {-100, -10, -1, 0, 1, 10, 100};
    for (int i = 0; i < 6; ++i) CHECK_TRUE(link.link(sorted[i + 1]) > link.link(sorted[i]));
}

void test_asinh_positive_epsilon_inflates_positive_tail() {
    ASinHLink symmetric(0.0, 1.0);
    ASinHLink skewed(0.0, 1.0, 0.4);
    double eta_test = 2.0;
    CHECK_TRUE(skewed.inverse_link(eta_test) > symmetric.inverse_link(eta_test));
    CHECK_TRUE(skewed.inverse_link(-eta_test) > symmetric.inverse_link(-eta_test));
}

void test_asinh_symmetric_case_reduces_to_simple_asinh() {
    double gamma0 = 0.5, scale = 0.3;
    ASinHLink link(gamma0, scale, 0.0, 1.0);
    CHECK_NEAR(link.link(gamma0), 0.0, 1e-15);
    CHECK_NEAR(link.inverse_link(0.0), gamma0, 1e-15);
    CHECK_NEAR(link.d_link(gamma0), 1.0 / scale, 1e-12);
}

void test_asinh_delta_greater_than_one_inverse_link_compresses() {
    ASinHLink standard(0.0, 1.0, 0.0, 1.0);
    ASinHLink heavy(0.0, 1.0, 0.0, 1.5);
    double eta = 3.0;
    CHECK_TRUE(heavy.inverse_link(eta) < standard.inverse_link(eta));
    CHECK_TRUE(heavy.inverse_link(-eta) > standard.inverse_link(-eta));
}

void test_asinh_adaptive_epsilon_zero_parent_symmetric() {
    ASinHLink adaptive(0.0, 1.0);
    adaptive.set_use_adaptive_epsilon(true);
    adaptive.set_parent_indicator(0.0);
    adaptive.set_epsilon_max(0.5);
    ASinHLink symmetric(0.0, 1.0);
    double test_points[] = {-5, -1, 0, 1, 5};
    for (double gamma : test_points)
        CHECK_NEAR(adaptive.link(gamma), symmetric.link(gamma), 1e-15);
}

void test_asinh_adaptive_epsilon_positive_parent_right_inflation() {
    ASinHLink adaptive(0.5, 0.3);
    adaptive.set_use_adaptive_epsilon(true);
    adaptive.set_parent_indicator(1.0);
    adaptive.set_epsilon_max(0.5);
    adaptive.set_epsilon_slope(1.0);
    ASinHLink symmetric(0.5, 0.3);
    double eta = 2.0;
    CHECK_TRUE(adaptive.inverse_link(eta) > symmetric.inverse_link(eta));
}

void test_asinh_adaptive_epsilon_saturation() {
    double eps_max = 0.5;
    ASinHLink link_large_pos(0.0, 1.0);
    link_large_pos.set_use_adaptive_epsilon(true);
    link_large_pos.set_parent_indicator(100.0);
    link_large_pos.set_epsilon_max(eps_max);
    link_large_pos.set_epsilon_slope(1.0);
    ASinHLink link_large_neg(0.0, 1.0);
    link_large_neg.set_use_adaptive_epsilon(true);
    link_large_neg.set_parent_indicator(-100.0);
    link_large_neg.set_epsilon_max(eps_max);
    link_large_neg.set_epsilon_slope(1.0);

    ASinHLink fixed_pos(0.0, 1.0, eps_max);
    ASinHLink fixed_neg(0.0, 1.0, -eps_max);

    CHECK_NEAR(link_large_pos.link(2.0), fixed_pos.link(2.0), 1e-6);
    CHECK_NEAR(link_large_neg.link(2.0), fixed_neg.link(2.0), 1e-6);
}

void test_asinh_round_trip_adaptive_epsilon() {
    ASinHLink link(0.5, 0.3);
    link.set_use_adaptive_epsilon(true);
    link.set_parent_indicator(0.5);
    link.set_epsilon_max(0.4);
    link.set_epsilon_slope(1.0);
    double gammas[] = {-3.0, -0.5, 0.0, 0.5, 3.0};
    for (double gamma : gammas)
        CHECK_NEAR(link.inverse_link(link.link(gamma)), gamma, kRoundTripTol);
}

void test_asinh_known_values_standard() {
    ASinHLink link(0.0, 1.0);
    CHECK_NEAR(link.link(0.0), 0.0, 1e-15);
    CHECK_NEAR(link.link(std::sinh(1.0)), std::sinh(1.0), 1e-12);
    CHECK_NEAR(link.inverse_link(0.0), 0.0, 1e-15);
    CHECK_NEAR(link.d_link(0.0), 1.0, 1e-15);
}

void test_asinh_near_linear_behavior_symmetric() {
    double gamma0 = 0.5, scale = 0.3;
    ASinHLink link(gamma0, scale);
    double small_delta = 0.001;
    double eta = link.link(gamma0 + small_delta);
    double linear = small_delta / scale;
    CHECK_NEAR(eta, linear, 1e-8);
}

void test_asinh_large_gamma_round_trip_stable_domain() {
    double gammas[] = {1e2, -1e2, 1e3, -1e3};
    for (double gamma : gammas) {
        ASinHLink link(0.0, 1.0);
        double eta = link.link(gamma);
        CHECK_TRUE(!std::isnan(eta));
        CHECK_TRUE(!std::isinf(eta));
        CHECK_NEAR(link.inverse_link(eta), gamma, std::fabs(gamma) * 1e-10);
    }
}

void test_asinh_large_gamma_asymmetric_round_trip_stable_domain() {
    double gammas[] = {1e2, -1e2, 1e3, -1e3};
    for (double gamma : gammas) {
        ASinHLink link(0.0, 1.0, 0.3, 1.2);
        double eta = link.link(gamma);
        CHECK_TRUE(!std::isnan(eta));
        CHECK_TRUE(!std::isinf(eta));
        CHECK_NEAR(link.inverse_link(eta), gamma, std::fabs(gamma) * 1e-8);
    }
}

void test_asinh_very_small_scale_still_monotone() {
    ASinHLink link(0.0, 1e-10);
    CHECK_TRUE(link.d_link(0.0) > 0);
    CHECK_TRUE(link.d_link(5.0) > 0);
    double eta = link.link(1.0);
    CHECK_TRUE(!std::isnan(eta));
}

// ══════════════════════════════════════════════
//  SESLinkTests.cs
// ══════════════════════════════════════════════

void test_ses_constructor_default_sets_default_values() {
    SESLink link;
    CHECK_EQ(link.a(), 1.0);
    CHECK_EQ(link.lambda(), 0.4);
    CHECK_TRUE(link.use_adaptive_lambda());
    CHECK_EQ(link.parent_indicator(), 0.0);
    CHECK_EQ(link.lambda_max(), 0.8);
    CHECK_EQ(link.lambda_slope(), 1.0);
    CHECK_EQ(link.max_iterations(), 20);
    CHECK_EQ(link.tolerance(), 1e-12);
}

void test_ses_constructor_custom_sets_values() {
    SESLink link(/*a=*/2.0, /*use_adaptive_lambda=*/false, /*parent_indicator=*/1.5,
                 /*lambda_max=*/0.6, /*lambda_slope=*/2.0, /*max_iterations=*/50,
                 /*tolerance=*/1e-14);
    CHECK_EQ(link.a(), 2.0);
    CHECK_TRUE(!link.use_adaptive_lambda());
    CHECK_EQ(link.parent_indicator(), 1.5);
    CHECK_EQ(link.lambda_max(), 0.6);
    CHECK_EQ(link.lambda_slope(), 2.0);
    CHECK_EQ(link.max_iterations(), 50);
    CHECK_EQ(link.tolerance(), 1e-14);
}

void test_ses_round_trip_symmetric_default_params() {
    // Adaptive lambda with ParentIndicator=0 => lambda_eff ~ 0 (symmetric)
    SESLink link;
    double values[] = {-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0};
    for (double x : values) {
        double recovered = link.inverse_link(link.link(x));
        CHECK_NEAR(recovered, x, std::max(kRoundTripTol, std::fabs(x) * 1e-8));
    }
}

void test_ses_round_trip_fixed_lambda() {
    SESLink link(1.0, false);
    link.set_lambda(0.4);
    double values[] = {-50.0, -5.0, -0.1, 0.0, 0.1, 5.0, 50.0};
    for (double x : values) {
        double recovered = link.inverse_link(link.link(x));
        CHECK_NEAR(recovered, x, std::max(kRoundTripTol, std::fabs(x) * 1e-8));
    }
}

void test_ses_round_trip_negative_lambda() {
    SESLink link(1.0, false);
    link.set_lambda(-0.3);
    double values[] = {-20.0, -1.0, 0.0, 1.0, 20.0};
    for (double x : values) {
        double recovered = link.inverse_link(link.link(x));
        CHECK_NEAR(recovered, x, std::max(kRoundTripTol, std::fabs(x) * 1e-8));
    }
}

void test_ses_round_trip_large_a() {
    SESLink link(3.0, false);
    link.set_lambda(0.0);
    double values[] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (double x : values) {
        double recovered = link.inverse_link(link.link(x));
        CHECK_NEAR(recovered, x, std::max(kRoundTripTol, std::fabs(x) * 1e-6));
    }
}

void test_ses_inverse_link_at_zero_returns_zero() {
    SESLink link;
    CHECK_NEAR(link.inverse_link(0.0), 0.0, 1e-15);
}

void test_ses_inverse_link_symmetric_when_lambda_zero() {
    SESLink link(1.0, false);
    link.set_lambda(0.0);
    double etas[] = {0.5, 1.0, 2.0};
    for (double eta : etas) {
        double pos = link.inverse_link(eta);
        double neg = link.inverse_link(-eta);
        CHECK_NEAR(pos, -neg, 1e-12);
    }
}

void test_ses_inverse_link_monotone() {
    SESLink link(1.0, false);
    link.set_lambda(0.4);
    double prev = link.inverse_link(-10.0);
    for (double eta = -9.0; eta <= 10.0; eta += 0.5) {
        double current = link.inverse_link(eta);
        CHECK_TRUE(current > prev);
        prev = current;
    }
}

void test_ses_d_link_finite_difference_symmetric_case() {
    SESLink link(1.0, false);
    link.set_lambda(0.0);
    double test_points[] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

void test_ses_d_link_finite_difference_asymmetric_case() {
    SESLink link(1.0, false);
    link.set_lambda(0.5);
    double test_points[] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

void test_ses_d_link_at_zero_approximately_one() {
    SESLink link(1.0, false);
    link.set_lambda(0.0);
    CHECK_NEAR(link.d_link(0.0), 1.0, 1e-6);
}

void test_ses_d_link_always_positive() {
    SESLink link(1.0, false);
    link.set_lambda(0.4);
    double test_points[] = {-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0};
    for (double x : test_points) CHECK_TRUE(link.d_link(x) > 0);
}

void test_ses_adaptive_lambda_zero_indicator_symmetric_result() {
    SESLink link;
    link.set_parent_indicator(0.0);
    double pos = link.inverse_link(2.0);
    double neg = link.inverse_link(-2.0);
    CHECK_NEAR(pos, -neg, 1e-10);
}

void test_ses_adaptive_lambda_positive_indicator_positive_tail_stronger() {
    SESLink link;
    link.set_parent_indicator(2.0);
    double pos = std::fabs(link.inverse_link(3.0));
    double neg = std::fabs(link.inverse_link(-3.0));
    CHECK_TRUE(pos > neg);
}

void test_ses_adaptive_lambda_negative_indicator_negative_tail_stronger() {
    SESLink link;
    link.set_parent_indicator(-2.0);
    double pos = std::fabs(link.inverse_link(3.0));
    double neg = std::fabs(link.inverse_link(-3.0));
    CHECK_TRUE(neg > pos);
}

void test_ses_link_small_a() {
    SESLink link(0.1, false);
    link.set_lambda(0.0);
    double x = 5.0;
    double recovered = link.inverse_link(link.link(x));
    CHECK_NEAR(recovered, x, 1e-6);
}

void test_ses_inverse_link_extreme_tails() {
    SESLink link(1.0, false);
    link.set_lambda(0.0);
    double large_pos = link.inverse_link(10.0);
    CHECK_TRUE(large_pos > 1000.0);
    CHECK_TRUE(std::isfinite(large_pos));
    double large_neg = link.inverse_link(-10.0);
    CHECK_TRUE(large_neg < -1000.0);
    CHECK_TRUE(std::isfinite(large_neg));
}

// Supplement (brief): the Newton-convergence diagnostics the C# exposes after each Link
// call. Defaults true / NaN; a normal solve converges; an iteration-starved solve does
// not and records its final |delta eta|.
void test_ses_last_inverse_converged_diagnostics() {
    SESLink link;
    CHECK_TRUE(link.last_inverse_converged());
    CHECK_TRUE(std::isnan(link.last_inverse_residual()));

    (void)link.link(5.0);
    CHECK_TRUE(link.last_inverse_converged());
    CHECK_TRUE(link.last_inverse_residual() < link.tolerance());

    SESLink starved(1.0, false, 0.0, 0.4, 1.0, /*max_iterations=*/1, /*tolerance=*/1e-15);
    starved.set_lambda(0.4);
    (void)starved.link(1e6);
    CHECK_TRUE(!starved.last_inverse_converged());
    CHECK_TRUE(std::isfinite(starved.last_inverse_residual()));
}

// ══════════════════════════════════════════════
//  LogSESLinkTests.cs
// ══════════════════════════════════════════════

void test_log_ses_constructor_default_sets_default_values() {
    LogSESLink link;
    CHECK_EQ(link.sigma0(), 1.0);
    CHECK_EQ(link.a(), 1.0);
    CHECK_EQ(link.lambda(), 0.2);
    CHECK_EQ(link.max_iterations(), 20);
    CHECK_EQ(link.tolerance(), 1e-12);
    // Supplement: the remaining C# property initializers.
    CHECK_TRUE(!link.use_adaptive_lambda());
    CHECK_EQ(link.parent_indicator(), 0.0);
    CHECK_EQ(link.lambda_max(), 0.4);
    CHECK_EQ(link.lambda_slope(), 1.0);
    CHECK_EQ(link.eps(), 1e-12);
    CHECK_TRUE(link.last_inverse_converged());
    CHECK_TRUE(std::isnan(link.last_inverse_residual()));
}

void test_log_ses_constructor_custom_sets_values() {
    LogSESLink link(/*sigma0=*/10.0, /*a=*/2.0, /*lambda=*/0.5);
    CHECK_EQ(link.sigma0(), 10.0);
    CHECK_EQ(link.a(), 2.0);
    CHECK_EQ(link.lambda(), 0.5);
}

void test_log_ses_constructor_clamps_sigma0() {
    LogSESLink link(-5.0);
    CHECK_TRUE(link.sigma0() > 0);
}

void test_log_ses_constructor_clamps_lambda() {
    LogSESLink link(1.0, 1.0, 2.0);
    CHECK_TRUE(link.lambda() < 1.0);
}

void test_log_ses_round_trip_default_params() {
    LogSESLink link(10.0);
    double sigmas[] = {0.1, 1.0, 5.0, 10.0, 20.0, 100.0};
    for (double sigma : sigmas) {
        double recovered = link.inverse_link(link.link(sigma));
        CHECK_NEAR(recovered, sigma, std::max(kLogSesRoundTripTol, sigma * 1e-8));
    }
}

void test_log_ses_round_trip_various_sigma0() {
    double sigma0_values[] = {0.5, 1.0, 10.0, 100.0};
    for (double sigma0 : sigma0_values) {
        LogSESLink link(sigma0, 1.0, 0.2);
        double test_sigmas[] = {sigma0 * 0.1, sigma0 * 0.5, sigma0, sigma0 * 2.0, sigma0 * 10.0};
        for (double sigma : test_sigmas) {
            double recovered = link.inverse_link(link.link(sigma));
            CHECK_NEAR(recovered, sigma, std::max(kLogSesRoundTripTol, sigma * 1e-8));
        }
    }
}

void test_log_ses_round_trip_symmetric_lambda() {
    LogSESLink link(5.0, 1.0, 0.0);
    double sigmas[] = {0.5, 1.0, 5.0, 25.0};
    for (double sigma : sigmas) {
        double recovered = link.inverse_link(link.link(sigma));
        CHECK_NEAR(recovered, sigma, std::max(kLogSesRoundTripTol, sigma * 1e-8));
    }
}

void test_log_ses_inverse_link_at_zero_returns_sigma0() {
    LogSESLink link(7.5);
    CHECK_NEAR(link.inverse_link(0.0), 7.5, 1e-10);
}

void test_log_ses_inverse_link_always_positive() {
    LogSESLink link(5.0);
    double etas[] = {-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0};
    for (double eta : etas) CHECK_TRUE(link.inverse_link(eta) > 0);
}

void test_log_ses_inverse_link_monotone() {
    LogSESLink link(5.0);
    double prev = link.inverse_link(-5.0);
    for (double eta = -4.5; eta <= 5.0; eta += 0.5) {
        double current = link.inverse_link(eta);
        CHECK_TRUE(current > prev);
        prev = current;
    }
}

void test_log_ses_d_link_finite_difference() {
    LogSESLink link(10.0, 1.0, 0.2);
    double test_sigmas[] = {1.0, 5.0, 10.0, 20.0, 50.0};
    for (double sigma : test_sigmas) {
        double finite_diff =
            (link.link(sigma + kDeltaH) - link.link(sigma - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(sigma), finite_diff, kDerivativeTol);
    }
}

void test_log_ses_d_link_always_positive() {
    LogSESLink link(5.0);
    double test_sigmas[] = {0.01, 0.5, 1.0, 5.0, 25.0, 100.0};
    for (double sigma : test_sigmas) CHECK_TRUE(link.d_link(sigma) > 0);
}

void test_log_ses_link_very_small_sigma() {
    LogSESLink link(10.0);
    double eta = link.link(1e-10);
    double recovered = link.inverse_link(eta);
    CHECK_TRUE(recovered > 0);
    CHECK_TRUE(std::isfinite(recovered));
}

void test_log_ses_link_very_large_sigma() {
    LogSESLink link(10.0);
    double eta = link.link(1e6);
    double recovered = link.inverse_link(eta);
    CHECK_NEAR(recovered, 1e6, 1e6 * 1e-6);
    CHECK_TRUE(std::isfinite(eta));
}

// ══════════════════════════════════════════════
//  LogASinHLinkTests.cs
// ══════════════════════════════════════════════

void test_log_asinh_constructor_default_sets_default_values() {
    LogASinHLink link;
    CHECK_EQ(link.sigma0(), 1.0);
    CHECK_EQ(link.log_scale(), 1.0);
    CHECK_EQ(link.epsilon(), 0.0);
    CHECK_EQ(link.delta(), 1.0);
    CHECK_TRUE(!link.use_adaptive_epsilon());
    CHECK_EQ(link.parent_indicator(), 0.0);
    CHECK_EQ(link.epsilon_max(), 0.5);
    CHECK_EQ(link.epsilon_slope(), 1.0);
    CHECK_EQ(link.eps(), 1e-12);
}

void test_log_asinh_constructor_custom_sets_values() {
    LogASinHLink link(/*sigma0=*/10.0, /*log_scale=*/0.25, /*epsilon=*/0.3, /*delta=*/0.9);
    CHECK_EQ(link.sigma0(), 10.0);
    CHECK_EQ(link.log_scale(), 0.25);
    CHECK_EQ(link.epsilon(), 0.3);
    CHECK_EQ(link.delta(), 0.9);
}

void test_log_asinh_constructor_invalid_positive_settings_are_floored() {
    LogASinHLink link(-10.0, -0.25, 0.0, -1.0);
    CHECK_TRUE(link.sigma0() > 0.0);
    CHECK_TRUE(link.log_scale() > 0.0);
    CHECK_TRUE(link.delta() > 0.0);
}

void test_log_asinh_symmetric_baseline_reduces_to_centered_log_link() {
    LogASinHLink link(10.0, 0.25);
    double sigma = 15.0;
    double expected_eta = std::log(sigma / 10.0) / 0.25;
    CHECK_NEAR(link.link(10.0), 0.0, 1e-15);
    CHECK_NEAR(link.inverse_link(0.0), 10.0, 1e-12);
    CHECK_NEAR(link.link(sigma), expected_eta, 1e-12);
}

void test_log_asinh_round_trip_practical_scale_values() {
    LogASinHLink link(10.0, 0.30, 0.25, 0.85);
    double sigmas[] = {0.1, 1.0, 5.0, 10.0, 20.0, 100.0};
    for (double sigma : sigmas) {
        double recovered = link.inverse_link(link.link(sigma));
        CHECK_NEAR(recovered, sigma, std::max(kRoundTripTol, sigma * 1e-10));
    }
}

void test_log_asinh_inverse_link_always_positive() {
    LogASinHLink link(10.0, 0.25, 0.4, 0.9);
    for (double eta = -6.0; eta <= 6.0; eta += 0.5) {
        double sigma = link.inverse_link(eta);
        CHECK_TRUE(sigma > 0.0);
        CHECK_TRUE(std::isfinite(sigma));
    }
}

void test_log_asinh_inverse_link_is_monotone_increasing() {
    LogASinHLink link(10.0, 0.25, 0.4, 0.9);
    double previous = link.inverse_link(-6.0);
    for (double eta = -5.5; eta <= 6.0; eta += 0.5) {
        double current = link.inverse_link(eta);
        CHECK_TRUE(current > previous);
        previous = current;
    }
}

void test_log_asinh_positive_epsilon_inflates_upper_scale_tail() {
    LogASinHLink link(10.0, 0.20, 0.4);
    double eta_center = link.link(10.0);
    double upper_factor = link.inverse_link(eta_center + 2.0) / 10.0;
    double lower_factor = 10.0 / link.inverse_link(eta_center - 2.0);
    CHECK_TRUE(upper_factor > lower_factor);
}

void test_log_asinh_delta_below_one_thickens_symmetric_multiplicative_tails() {
    LogASinHLink baseline(10.0, 0.20, 0.0, 1.0);
    LogASinHLink heavy_tail(10.0, 0.20, 0.0, 0.75);
    CHECK_TRUE(heavy_tail.inverse_link(2.0) > baseline.inverse_link(2.0));
    CHECK_TRUE(heavy_tail.inverse_link(-2.0) < baseline.inverse_link(-2.0));
}

void test_log_asinh_adaptive_epsilon_uses_parent_indicator() {
    LogASinHLink weak(10.0, 0.20);
    weak.set_use_adaptive_epsilon(true);
    weak.set_parent_indicator(0.1);
    weak.set_epsilon_max(0.7);
    weak.set_epsilon_slope(1.2);
    LogASinHLink strong(10.0, 0.20);
    strong.set_use_adaptive_epsilon(true);
    strong.set_parent_indicator(0.9);
    strong.set_epsilon_max(0.7);
    strong.set_epsilon_slope(1.2);

    double weak_eta_center = weak.link(10.0);
    double strong_eta_center = strong.link(10.0);
    double weak_upper = weak.inverse_link(weak_eta_center + 2.0);
    double strong_upper = strong.inverse_link(strong_eta_center + 2.0);
    CHECK_TRUE(strong_upper > weak_upper);
}

void test_log_asinh_d_link_matches_finite_difference() {
    LogASinHLink link(10.0, 0.25, 0.2, 0.9);
    double sigmas[] = {1.0, 5.0, 10.0, 20.0, 50.0};
    for (double sigma : sigmas) {
        double h = std::max(1e-6, sigma * 1e-6);
        double finite_difference = (link.link(sigma + h) - link.link(sigma - h)) / (2.0 * h);
        CHECK_NEAR(link.d_link(sigma), finite_difference, kDerivativeTol);
        CHECK_TRUE(link.d_link(sigma) > 0.0);
    }
}

// ══════════════════════════════════════════════
//  CenteredLinkTests.cs
// ══════════════════════════════════════════════

void test_centered_constructor_stores_properties() {
    auto inner = std::make_unique<IdentityLink>();
    const ILinkFunction* raw = inner.get();
    CenteredLink link(std::move(inner), /*mu0=*/100.0, /*scale=*/5.0);
    CHECK_TRUE(link.inner() == raw);  // Assert.AreSame
    CHECK_EQ(link.mu0(), 100.0);
    CHECK_EQ(link.scale(), 5.0);
}

void test_centered_constructor_null_inner_throws() {
    CHECK_THROWS(CenteredLink(nullptr, 0.0));
}

void test_centered_constructor_negative_scale_clamped_to_minimum() {
    CenteredLink link(std::make_unique<IdentityLink>(), 0.0, -5.0);
    CHECK_TRUE(link.scale() > 0);
}

void test_centered_constructor_zero_scale_clamped_to_minimum() {
    CenteredLink link(std::make_unique<IdentityLink>(), 0.0, 0.0);
    CHECK_TRUE(link.scale() > 0);
}

void test_centered_constructor_default_scale_is_one() {
    CenteredLink link(std::make_unique<IdentityLink>(), 50.0);
    CHECK_EQ(link.scale(), 1.0);
}

void test_centered_round_trip_identity_inner() {
    CenteredLink link(std::make_unique<IdentityLink>(), 100.0, 20.0);
    double values[] = {50.0, 80.0, 100.0, 120.0, 200.0};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kRoundTripTol);
}

void test_centered_link_identity_inner_known_values() {
    CenteredLink link(std::make_unique<IdentityLink>(), 100.0, 20.0);
    CHECK_NEAR(link.link(100.0), 0.0, 1e-12);
    CHECK_NEAR(link.link(120.0), 1.0, 1e-12);
    CHECK_NEAR(link.link(60.0), -2.0, 1e-12);
}

void test_centered_inverse_link_identity_inner_known_values() {
    CenteredLink link(std::make_unique<IdentityLink>(), 100.0, 20.0);
    CHECK_NEAR(link.inverse_link(0.0), 100.0, 1e-12);
    CHECK_NEAR(link.inverse_link(1.0), 120.0, 1e-12);
    CHECK_NEAR(link.inverse_link(-2.0), 60.0, 1e-12);
}

void test_centered_round_trip_ses_inner() {
    auto inner = std::make_unique<SESLink>(1.0, false);
    inner->set_lambda(0.3);
    CenteredLink link(std::move(inner), 500.0, 50.0);
    double values[] = {300.0, 450.0, 500.0, 550.0, 700.0};
    for (double x : values)
        CHECK_NEAR(link.inverse_link(link.link(x)), x,
                   std::max(kRoundTripTol, std::fabs(x) * 1e-8));
}

void test_centered_round_trip_log_link_inner() {
    CenteredLink link(std::make_unique<LogLink>(), 0.0, 1.0);
    double values[] = {0.1, 1.0, 5.0, 100.0};
    for (double x : values)
        CHECK_NEAR(link.inverse_link(link.link(x)), x, std::max(kRoundTripTol, x * 1e-10));
}

void test_centered_d_link_identity_inner_known_value() {
    CenteredLink link(std::make_unique<IdentityLink>(), 100.0, 20.0);
    CHECK_NEAR(link.d_link(100.0), 1.0 / 20.0, 1e-12);
    CHECK_NEAR(link.d_link(200.0), 1.0 / 20.0, 1e-12);
}

void test_centered_d_link_finite_difference_identity_inner() {
    CenteredLink link(std::make_unique<IdentityLink>(), 100.0, 20.0);
    double test_points[] = {50.0, 100.0, 150.0};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

void test_centered_d_link_finite_difference_ses_inner() {
    auto inner = std::make_unique<SESLink>(1.0, false);
    inner->set_lambda(0.0);
    CenteredLink link(std::move(inner), 50.0, 10.0);
    double test_points[] = {20.0, 50.0, 80.0};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

void test_centered_d_link_chain_rule_log_inner() {
    CenteredLink link(std::make_unique<LogLink>(), 0.0, 2.0);
    double x = 6.0;
    double z = (x - 0.0) / 2.0;
    double expected = (1.0 / z) / 2.0;
    CHECK_NEAR(link.d_link(x), expected, 1e-12);
}

void test_centered_d_link_always_positive() {
    auto inner = std::make_unique<SESLink>(1.0, false);
    inner->set_lambda(0.0);
    CenteredLink link(std::move(inner), 50.0, 10.0);
    double test_points[] = {0.0, 25.0, 50.0, 75.0, 100.0};
    for (double x : test_points) CHECK_TRUE(link.d_link(x) > 0);
}

void test_centered_link_at_mu0_maps_to_inner_link_of_zero() {
    SESLink reference(1.0, false);
    reference.set_lambda(0.0);
    double expected = reference.link(0.0);
    auto inner = std::make_unique<SESLink>(1.0, false);
    inner->set_lambda(0.0);
    CenteredLink link(std::move(inner), 200.0, 30.0);
    CHECK_NEAR(link.link(200.0), expected, 1e-12);
}

void test_centered_inverse_link_at_zero_maps_through_inner() {
    auto inner = std::make_unique<SESLink>(1.0, false);
    inner->set_lambda(0.0);
    CenteredLink link(std::move(inner), 200.0, 30.0);
    CHECK_NEAR(link.inverse_link(0.0), 200.0, 1e-10);
}

void test_centered_scale_affects_spread() {
    CenteredLink link_narrow(std::make_unique<IdentityLink>(), 100.0, 1.0);
    CenteredLink link_wide(std::make_unique<IdentityLink>(), 100.0, 10.0);
    CHECK_NEAR(link_narrow.inverse_link(2.0), 102.0, 1e-12);
    CHECK_NEAR(link_wide.inverse_link(2.0), 120.0, 1e-12);
}

// ══════════════════════════════════════════════
//  YeoJohnsonLinkTests.cs (the models-layer YeoJohnsonLink)
// ══════════════════════════════════════════════

void test_yj_constructor_default_lambda_is_one() {
    bf_models::YeoJohnsonLink link;
    CHECK_NEAR(link.lambda(), 1.0, 1e-10);
}

void test_yj_constructor_lambda_stores_value() {
    bf_models::YeoJohnsonLink link(0.5);
    CHECK_NEAR(link.lambda(), 0.5, 1e-10);
}

void test_yj_constructor_lambda_two_stores_value() {
    bf_models::YeoJohnsonLink link(2.0);
    CHECK_NEAR(link.lambda(), 2.0, 1e-10);
}

void test_yj_constructor_values_single_element_throws() {
    CHECK_THROWS(bf_models::YeoJohnsonLink(std::vector<double>{1.0}));
}

void test_yj_constructor_values_two_elements_produces_finite_lambda() {
    bf_models::YeoJohnsonLink link(std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0});
    CHECK_TRUE(std::isfinite(link.lambda()));
}

void test_yj_link_lambda_one_positive_x_is_identity() {
    bf_models::YeoJohnsonLink link(1.0);
    double positive_values[] = {0.5, 1.0, 2.0, 5.0, 10.0};
    for (double x : positive_values) CHECK_NEAR(link.link(x), x, 1e-10);
}

void test_yj_link_lambda_one_at_zero_is_zero() {
    bf_models::YeoJohnsonLink link(1.0);
    CHECK_NEAR(link.link(0.0), 0.0, 1e-10);
}

void test_yj_d_link_lambda_one_is_one() {
    bf_models::YeoJohnsonLink link(1.0);
    double test_points[] = {-5.0, -1.0, 0.0, 1.0, 5.0};
    for (double x : test_points) CHECK_NEAR(link.d_link(x), 1.0, 1e-10);
}

void test_yj_round_trip_positive_x_lambda_half() {
    bf_models::YeoJohnsonLink link(0.5);
    double values[] = {0.1, 0.5, 1.0, 2.0, 5.0, 10.0};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kYjRoundTripTol);
}

void test_yj_round_trip_negative_x_lambda_half() {
    bf_models::YeoJohnsonLink link(0.5);
    double values[] = {-10.0, -5.0, -2.0, -1.0, -0.5, -0.1};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kYjRoundTripTol);
}

void test_yj_round_trip_lambda_two_positive_x() {
    bf_models::YeoJohnsonLink link(2.0);
    double values[] = {0.5, 1.0, 2.0, 5.0};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kYjRoundTripTol);
}

void test_yj_round_trip_lambda_zero_positive_x() {
    bf_models::YeoJohnsonLink link(0.0);
    double values[] = {0.5, 1.0, 2.0, 5.0, 10.0};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kYjRoundTripTol);
}

void test_yj_d_link_lambda_half_positive_x_finite_difference() {
    bf_models::YeoJohnsonLink link(0.5);
    double test_points[] = {0.5, 1.0, 2.0, 5.0};
    for (double x : test_points) {
        double analytic = link.d_link(x);
        double fd = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(analytic, fd, kDerivativeTol);
    }
}

void test_yj_d_link_lambda_half_negative_x_finite_difference() {
    bf_models::YeoJohnsonLink link(0.5);
    double test_points[] = {-5.0, -2.0, -1.0, -0.5};
    for (double x : test_points) {
        double analytic = link.d_link(x);
        double fd = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(analytic, fd, kDerivativeTol);
    }
}

void test_yj_d_link_at_zero_lambda_one_is_one() {
    bf_models::YeoJohnsonLink link(1.0);
    CHECK_NEAR(link.d_link(0.0), 1.0, 1e-10);
}

void test_yj_d_link_positive_x_always_positive() {
    bf_models::YeoJohnsonLink link(0.5);
    double test_points[] = {0.1, 0.5, 1.0, 2.0, 10.0};
    for (double x : test_points) CHECK_TRUE(link.d_link(x) > 0);
}

// ══════════════════════════════════════════════
//  BestFitLinkFunctionFactory (non-XML dispatch assertions transcribed from
//  LinkFunctionSerializationTests.cs BestFitLinkFunctionFactory_AllTypes /
//  BestFitLinkFunctionFactory_UnknownType_Throws)
// ══════════════════════════════════════════════

void test_factory_all_types() {
    // Numerics types (the C# default-case fall-through to LinkFunctionFactory)
    CHECK_TRUE(dynamic_cast<IdentityLink*>(
                   BestFitLinkFunctionFactory::create(LinkFunctionType::Identity).get()) !=
               nullptr);
    CHECK_TRUE(dynamic_cast<LogLink*>(
                   BestFitLinkFunctionFactory::create(LinkFunctionType::Log).get()) != nullptr);
    CHECK_TRUE(dynamic_cast<bf_numfun::LogitLink*>(
                   BestFitLinkFunctionFactory::create(LinkFunctionType::Logit).get()) !=
               nullptr);
    CHECK_TRUE(dynamic_cast<bf_numfun::ProbitLink*>(
                   BestFitLinkFunctionFactory::create(LinkFunctionType::Probit).get()) !=
               nullptr);
    CHECK_TRUE(
        dynamic_cast<bf_numfun::ComplementaryLogLogLink*>(
            BestFitLinkFunctionFactory::create(LinkFunctionType::ComplementaryLogLog).get()) !=
        nullptr);

    // BestFit types
    CHECK_TRUE(dynamic_cast<SESLink*>(
                   BestFitLinkFunctionFactory::create(BestFitLinkFunctionType::SES).get()) !=
               nullptr);
    CHECK_TRUE(dynamic_cast<LogSESLink*>(
                   BestFitLinkFunctionFactory::create(BestFitLinkFunctionType::LogSES).get()) !=
               nullptr);
    CHECK_TRUE(
        dynamic_cast<LogASinHLink*>(
            BestFitLinkFunctionFactory::create(BestFitLinkFunctionType::LogASinH).get()) !=
        nullptr);
    // Supplement: the remaining BestFit types the upstream method doesn't list.
    CHECK_TRUE(dynamic_cast<ASinHLink*>(
                   BestFitLinkFunctionFactory::create(BestFitLinkFunctionType::ASinH).get()) !=
               nullptr);
    CHECK_TRUE(
        dynamic_cast<bf_models::YeoJohnsonLink*>(
            BestFitLinkFunctionFactory::create(BestFitLinkFunctionType::YeoJohnson).get()) !=
        nullptr);

    // CenteredLink defaults to an IdentityLink inner (the C# no-child-element fallback).
    auto centered =
        BestFitLinkFunctionFactory::create(BestFitLinkFunctionType::Centered);
    auto* centered_ptr = dynamic_cast<CenteredLink*>(centered.get());
    CHECK_TRUE(centered_ptr != nullptr);
    CHECK_TRUE(dynamic_cast<const IdentityLink*>(centered_ptr->inner()) != nullptr);
    CHECK_EQ(centered_ptr->mu0(), 0.0);
    CHECK_EQ(centered_ptr->scale(), 1.0);
}

void test_factory_unknown_type_throws() {
    CHECK_THROWS(BestFitLinkFunctionFactory::create(static_cast<BestFitLinkFunctionType>(999)));
}

}  // namespace

int main() {
    // ASinHLinkTests.cs
    test_asinh_constructor_default_sets_default_values();
    test_asinh_constructor_two_param_symmetric_defaults();
    test_asinh_constructor_four_param_sets_values();
    test_asinh_constructor_small_scale_floors_at_eps();
    test_asinh_constructor_small_delta_floors_at_eps();
    test_asinh_link_symmetric_at_center_returns_zero();
    test_asinh_inverse_link_symmetric_at_zero_returns_gamma0();
    test_asinh_round_trip_symmetric_default_params();
    test_asinh_round_trip_symmetric_typical_lp3();
    test_asinh_round_trip_positive_epsilon();
    test_asinh_round_trip_negative_epsilon();
    test_asinh_round_trip_heavy_tails();
    test_asinh_round_trip_light_tails();
    test_asinh_d_link_symmetric_matches_finite_difference();
    test_asinh_d_link_asymmetric_matches_finite_difference();
    test_asinh_d_link_lp3_params_matches_finite_difference();
    test_asinh_d_link_symmetric_always_positive();
    test_asinh_d_link_asymmetric_always_positive();
    test_asinh_link_symmetric_strictly_increasing();
    test_asinh_link_asymmetric_strictly_increasing();
    test_asinh_positive_epsilon_inflates_positive_tail();
    test_asinh_symmetric_case_reduces_to_simple_asinh();
    test_asinh_delta_greater_than_one_inverse_link_compresses();
    test_asinh_adaptive_epsilon_zero_parent_symmetric();
    test_asinh_adaptive_epsilon_positive_parent_right_inflation();
    test_asinh_adaptive_epsilon_saturation();
    test_asinh_round_trip_adaptive_epsilon();
    test_asinh_known_values_standard();
    test_asinh_near_linear_behavior_symmetric();
    test_asinh_large_gamma_round_trip_stable_domain();
    test_asinh_large_gamma_asymmetric_round_trip_stable_domain();
    test_asinh_very_small_scale_still_monotone();
    // SESLinkTests.cs
    test_ses_constructor_default_sets_default_values();
    test_ses_constructor_custom_sets_values();
    test_ses_round_trip_symmetric_default_params();
    test_ses_round_trip_fixed_lambda();
    test_ses_round_trip_negative_lambda();
    test_ses_round_trip_large_a();
    test_ses_inverse_link_at_zero_returns_zero();
    test_ses_inverse_link_symmetric_when_lambda_zero();
    test_ses_inverse_link_monotone();
    test_ses_d_link_finite_difference_symmetric_case();
    test_ses_d_link_finite_difference_asymmetric_case();
    test_ses_d_link_at_zero_approximately_one();
    test_ses_d_link_always_positive();
    test_ses_adaptive_lambda_zero_indicator_symmetric_result();
    test_ses_adaptive_lambda_positive_indicator_positive_tail_stronger();
    test_ses_adaptive_lambda_negative_indicator_negative_tail_stronger();
    test_ses_link_small_a();
    test_ses_inverse_link_extreme_tails();
    test_ses_last_inverse_converged_diagnostics();
    // LogSESLinkTests.cs
    test_log_ses_constructor_default_sets_default_values();
    test_log_ses_constructor_custom_sets_values();
    test_log_ses_constructor_clamps_sigma0();
    test_log_ses_constructor_clamps_lambda();
    test_log_ses_round_trip_default_params();
    test_log_ses_round_trip_various_sigma0();
    test_log_ses_round_trip_symmetric_lambda();
    test_log_ses_inverse_link_at_zero_returns_sigma0();
    test_log_ses_inverse_link_always_positive();
    test_log_ses_inverse_link_monotone();
    test_log_ses_d_link_finite_difference();
    test_log_ses_d_link_always_positive();
    test_log_ses_link_very_small_sigma();
    test_log_ses_link_very_large_sigma();
    // LogASinHLinkTests.cs
    test_log_asinh_constructor_default_sets_default_values();
    test_log_asinh_constructor_custom_sets_values();
    test_log_asinh_constructor_invalid_positive_settings_are_floored();
    test_log_asinh_symmetric_baseline_reduces_to_centered_log_link();
    test_log_asinh_round_trip_practical_scale_values();
    test_log_asinh_inverse_link_always_positive();
    test_log_asinh_inverse_link_is_monotone_increasing();
    test_log_asinh_positive_epsilon_inflates_upper_scale_tail();
    test_log_asinh_delta_below_one_thickens_symmetric_multiplicative_tails();
    test_log_asinh_adaptive_epsilon_uses_parent_indicator();
    test_log_asinh_d_link_matches_finite_difference();
    // CenteredLinkTests.cs
    test_centered_constructor_stores_properties();
    test_centered_constructor_null_inner_throws();
    test_centered_constructor_negative_scale_clamped_to_minimum();
    test_centered_constructor_zero_scale_clamped_to_minimum();
    test_centered_constructor_default_scale_is_one();
    test_centered_round_trip_identity_inner();
    test_centered_link_identity_inner_known_values();
    test_centered_inverse_link_identity_inner_known_values();
    test_centered_round_trip_ses_inner();
    test_centered_round_trip_log_link_inner();
    test_centered_d_link_identity_inner_known_value();
    test_centered_d_link_finite_difference_identity_inner();
    test_centered_d_link_finite_difference_ses_inner();
    test_centered_d_link_chain_rule_log_inner();
    test_centered_d_link_always_positive();
    test_centered_link_at_mu0_maps_to_inner_link_of_zero();
    test_centered_inverse_link_at_zero_maps_through_inner();
    test_centered_scale_affects_spread();
    // YeoJohnsonLinkTests.cs
    test_yj_constructor_default_lambda_is_one();
    test_yj_constructor_lambda_stores_value();
    test_yj_constructor_lambda_two_stores_value();
    test_yj_constructor_values_single_element_throws();
    test_yj_constructor_values_two_elements_produces_finite_lambda();
    test_yj_link_lambda_one_positive_x_is_identity();
    test_yj_link_lambda_one_at_zero_is_zero();
    test_yj_d_link_lambda_one_is_one();
    test_yj_round_trip_positive_x_lambda_half();
    test_yj_round_trip_negative_x_lambda_half();
    test_yj_round_trip_lambda_two_positive_x();
    test_yj_round_trip_lambda_zero_positive_x();
    test_yj_d_link_lambda_half_positive_x_finite_difference();
    test_yj_d_link_lambda_half_negative_x_finite_difference();
    test_yj_d_link_at_zero_lambda_one_is_one();
    test_yj_d_link_positive_x_always_positive();
    // LinkFunctionSerializationTests.cs (non-XML factory assertions)
    test_factory_all_types();
    test_factory_unknown_type_throws();
    return bftest::summary("test_bestfit_link_functions");
}
