// Transcribed C# oracle tests for the Numerics link-function layer (Task B1):
//   upstream/Numerics/Test_Numerics/Functions/Test_LinkFunctions.cs   @ a2c4dbf
//   upstream/Numerics/Test_Numerics/Functions/Test_FisherZLink.cs     @ a2c4dbf
//   upstream/Numerics/Test_Numerics/Functions/Test_YeoJohnsonLink.cs  @ a2c4dbf
// plus direct tests of the YeoJohnson transform class (no upstream test file exists under
// Test_Numerics/Data/Statistics for it; only Test_BoxCox.cs is there) and the brief-required
// supplements (round-trip identity + central-difference derivative checks for every link).
//
// Skipped upstream test methods (all XML-serialization-only; the port drops
// ToXElement/CreateFromXElement per the repo-wide rule):
//   Test_LinkFunctions.cs: Test_IdentityLink_RoundTrip_ToXElement,
//     Test_LogLink_RoundTrip_ToXElement, Test_LogitLink_RoundTrip_ToXElement,
//     Test_ProbitLink_RoundTrip_ToXElement, Test_ComplementaryLogLogLink_RoundTrip_ToXElement,
//     Test_LinkFunctionFactory_CreateFromXElement_UnknownType_Throws,
//     Test_LinkController_RoundTrip_Empty, Test_LinkController_RoundTrip_AllLinks,
//     Test_LinkController_RoundTrip_NullSlots.
//   Test_FisherZLink.cs: the CreateFromXElement half of Factory_CreatesFisherZLink.
//   Test_YeoJohnsonLink.cs: Constructor_XElement_Null_Throws,
//     Constructor_XElement_InvalidLambda_Throws, XmlRoundTrip_PreservesLambda, and the
//     CreateFromXElement half of Factory_CreatesYeoJohnsonLink. Constructor_Values_Null_Throws
//     is also skipped: a C++ std::vector cannot be null (no ArgumentNullException analog).
#include <cmath>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "bestfit/numerics/data/yeo_johnson.hpp"
#include "bestfit/numerics/functions/complementary_log_log_link.hpp"
#include "bestfit/numerics/functions/fisher_z_link.hpp"
#include "bestfit/numerics/functions/i_link_function.hpp"
#include "bestfit/numerics/functions/identity_link.hpp"
#include "bestfit/numerics/functions/link_controller.hpp"
#include "bestfit/numerics/functions/link_function_factory.hpp"
#include "bestfit/numerics/functions/link_function_type.hpp"
#include "bestfit/numerics/functions/log_link.hpp"
#include "bestfit/numerics/functions/logit_link.hpp"
#include "bestfit/numerics/functions/probit_link.hpp"
#include "bestfit/numerics/functions/yeo_johnson_link.hpp"
#include "bestfit/numerics/tools.hpp"
#include "check.hpp"

using bestfit::numerics::data::YeoJohnson;
using bestfit::numerics::functions::ComplementaryLogLogLink;
using bestfit::numerics::functions::FisherZLink;
using bestfit::numerics::functions::IdentityLink;
using bestfit::numerics::functions::ILinkFunction;
using bestfit::numerics::functions::LinkController;
using bestfit::numerics::functions::LinkFunctionFactory;
using bestfit::numerics::functions::LinkFunctionType;
using bestfit::numerics::functions::LogitLink;
using bestfit::numerics::functions::LogLink;
using bestfit::numerics::functions::ProbitLink;
using bestfit::numerics::functions::YeoJohnsonLink;

namespace {

// Constants transcribed from Test_LinkFunctions.cs.
const double kDeltaH = 1e-7;
const double kRoundTripTol = 1e-10;
const double kDerivativeTol = 1e-5;

const double kNaN = std::numeric_limits<double>::quiet_NaN();
const double kInf = std::numeric_limits<double>::infinity();

// ──────────────────────────────────────────────
//  IdentityLink (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_identity_link_link() {
    IdentityLink link;
    double values[] = {-100.0, -1.5, 0.0, 1.5, 100.0, std::numeric_limits<double>::max(),
                       std::numeric_limits<double>::lowest()};
    for (double x : values) CHECK_EQ(link.link(x), x);
}

void test_identity_link_inverse_link() {
    IdentityLink link;
    double values[] = {-100.0, -1.5, 0.0, 1.5, 100.0};
    for (double eta : values) CHECK_EQ(link.inverse_link(eta), eta);
}

void test_identity_link_d_link() {
    IdentityLink link;
    double values[] = {-100.0, 0.0, 100.0, 42.0};
    for (double x : values) CHECK_EQ(link.d_link(x), 1.0);
}

void test_identity_link_round_trip() {
    IdentityLink link;
    double values[] = {-1000.0, -1.0, 0.0, 1.0, 1000.0};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kRoundTripTol);
}

// ──────────────────────────────────────────────
//  LogLink (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_log_link_link_known_values() {
    LogLink link;
    CHECK_NEAR(link.link(1.0), 0.0, 1e-12);
    CHECK_NEAR(link.link(2.0), std::log(2.0), 1e-12);
    CHECK_NEAR(link.link(10.0), std::log(10.0), 1e-12);
    CHECK_NEAR(link.link(0.5), std::log(0.5), 1e-12);
}

void test_log_link_inverse_link_known_values() {
    LogLink link;
    CHECK_NEAR(link.inverse_link(0.0), 1.0, 1e-12);
    CHECK_NEAR(link.inverse_link(1.0), bestfit::numerics::kE, 1e-12);
    CHECK_NEAR(link.inverse_link(2.0), std::exp(2.0), 1e-12);
    CHECK_NEAR(link.inverse_link(-1.0), std::exp(-1.0), 1e-12);
}

void test_log_link_d_link_known_values() {
    LogLink link;
    CHECK_NEAR(link.d_link(1.0), 1.0, 1e-12);
    CHECK_NEAR(link.d_link(2.0), 0.5, 1e-12);
    CHECK_NEAR(link.d_link(10.0), 0.1, 1e-12);
    CHECK_NEAR(link.d_link(0.01), 100.0, 1e-6);
}

void test_log_link_round_trip() {
    LogLink link;
    double values[] = {0.001, 0.1, 1.0, 10.0, 100.0, 1e6};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, x * 1e-10);
}

void test_log_link_link_throws_for_zero() { CHECK_THROWS(LogLink().link(0.0)); }

void test_log_link_link_throws_for_negative() { CHECK_THROWS(LogLink().link(-1.0)); }

void test_log_link_d_link_throws_for_zero() { CHECK_THROWS(LogLink().d_link(0.0)); }

void test_log_link_derivative_consistency() {
    LogLink link;
    double test_points[] = {0.01, 0.5, 1.0, 5.0, 100.0};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

// ──────────────────────────────────────────────
//  LogitLink (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_logit_link_link_known_values() {
    LogitLink link;
    CHECK_NEAR(link.link(0.5), 0.0, 1e-12);
    // logit(0.731) = log(0.731/0.269)
    CHECK_NEAR(link.link(0.731), std::log(0.731 / 0.269), 1e-6);
    // logit(0.1) = log(1/9)
    CHECK_NEAR(link.link(0.1), std::log(1.0 / 9.0), 1e-10);
}

void test_logit_link_inverse_link_known_values() {
    LogitLink link;
    CHECK_NEAR(link.inverse_link(0.0), 0.5, 1e-12);
    CHECK_NEAR(link.inverse_link(100.0), 1.0, 1e-10);
    CHECK_NEAR(link.inverse_link(-100.0), 0.0, 1e-10);
    CHECK_NEAR(link.inverse_link(1.0), 1.0 / (1.0 + std::exp(-1.0)), 1e-10);
}

void test_logit_link_d_link_known_values() {
    LogitLink link;
    CHECK_NEAR(link.d_link(0.5), 4.0, 1e-12);
    CHECK_NEAR(link.d_link(0.1), 1.0 / (0.1 * 0.9), 1e-10);
}

void test_logit_link_round_trip() {
    LogitLink link;
    double values[] = {0.001, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kRoundTripTol);
}

void test_logit_link_link_throws_for_zero() { CHECK_THROWS(LogitLink().link(0.0)); }

void test_logit_link_link_throws_for_one() { CHECK_THROWS(LogitLink().link(1.0)); }

void test_logit_link_link_throws_for_negative() { CHECK_THROWS(LogitLink().link(-0.5)); }

void test_logit_link_inverse_link_extreme_values() {
    LogitLink link;
    // Very large positive eta: sigmoid should be very close to 1 without overflow.
    double high = link.inverse_link(710.0);
    CHECK_TRUE(high > 0.999 && high <= 1.0);
    // Very large negative eta: sigmoid should be very close to 0 without underflow.
    double low = link.inverse_link(-710.0);
    CHECK_TRUE(low >= 0.0 && low < 0.001);
}

void test_logit_link_derivative_consistency() {
    LogitLink link;
    double test_points[] = {0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

// ──────────────────────────────────────────────
//  ProbitLink (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_probit_link_link_known_values() {
    ProbitLink link;
    CHECK_NEAR(link.link(0.5), 0.0, 1e-10);
    CHECK_NEAR(link.link(0.975), 1.95996, 1e-4);
    CHECK_NEAR(link.link(0.025), -1.95996, 1e-4);
    CHECK_NEAR(link.link(0.8413), 1.0, 1e-3);
}

void test_probit_link_inverse_link_known_values() {
    ProbitLink link;
    CHECK_NEAR(link.inverse_link(0.0), 0.5, 1e-10);
    CHECK_NEAR(link.inverse_link(1.96), 0.975, 1e-3);
    CHECK_NEAR(link.inverse_link(-1.96), 0.025, 1e-3);
}

void test_probit_link_round_trip() {
    ProbitLink link;
    double values[] = {0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.999};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kRoundTripTol);
}

void test_probit_link_link_throws_for_zero() { CHECK_THROWS(ProbitLink().link(0.0)); }

void test_probit_link_link_throws_for_one() { CHECK_THROWS(ProbitLink().link(1.0)); }

void test_probit_link_derivative_consistency() {
    ProbitLink link;
    double test_points[] = {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

// ──────────────────────────────────────────────
//  ComplementaryLogLogLink (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_cloglog_link_known_values() {
    ComplementaryLogLogLink link;
    // h(1-1/e) = log(-log(1/e)) = log(1) = 0
    double x0 = 1.0 - 1.0 / bestfit::numerics::kE;
    CHECK_NEAR(link.link(x0), 0.0, 1e-10);
    // h(0.5) = log(log(2))
    CHECK_NEAR(link.link(0.5), std::log(std::log(2.0)), 1e-10);
}

void test_cloglog_inverse_link_known_values() {
    ComplementaryLogLogLink link;
    CHECK_NEAR(link.inverse_link(0.0), 1.0 - 1.0 / bestfit::numerics::kE, 1e-10);
    CHECK_NEAR(link.inverse_link(10.0), 1.0, 1e-4);
    CHECK_NEAR(link.inverse_link(-10.0), 0.0, 1e-4);
}

void test_cloglog_round_trip() {
    ComplementaryLogLogLink link;
    double values[] = {0.001, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.999};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kRoundTripTol);
}

void test_cloglog_link_throws_for_zero() { CHECK_THROWS(ComplementaryLogLogLink().link(0.0)); }

void test_cloglog_link_throws_for_one() { CHECK_THROWS(ComplementaryLogLogLink().link(1.0)); }

void test_cloglog_derivative_consistency() {
    ComplementaryLogLogLink link;
    double test_points[] = {0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95};
    for (double x : test_points) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kDerivativeTol);
    }
}

// ──────────────────────────────────────────────
//  LinkFunctionFactory (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_factory_creates_correct_types() {
    CHECK_TRUE(dynamic_cast<IdentityLink*>(LinkFunctionFactory::create(LinkFunctionType::Identity).get()) != nullptr);
    CHECK_TRUE(dynamic_cast<LogLink*>(LinkFunctionFactory::create(LinkFunctionType::Log).get()) != nullptr);
    CHECK_TRUE(dynamic_cast<LogitLink*>(LinkFunctionFactory::create(LinkFunctionType::Logit).get()) != nullptr);
    CHECK_TRUE(dynamic_cast<ProbitLink*>(LinkFunctionFactory::create(LinkFunctionType::Probit).get()) != nullptr);
    CHECK_TRUE(dynamic_cast<ComplementaryLogLogLink*>(
                   LinkFunctionFactory::create(LinkFunctionType::ComplementaryLogLog).get()) != nullptr);
}

void test_factory_throws_for_invalid_type() {
    CHECK_THROWS(LinkFunctionFactory::create(static_cast<LinkFunctionType>(999)));
}

void test_factory_round_trip_via_factory() {
    auto identity = LinkFunctionFactory::create(LinkFunctionType::Identity);
    CHECK_NEAR(identity->inverse_link(identity->link(5.0)), 5.0, kRoundTripTol);

    auto log_link = LinkFunctionFactory::create(LinkFunctionType::Log);
    CHECK_NEAR(log_link->inverse_link(log_link->link(2.5)), 2.5, kRoundTripTol);

    auto logit = LinkFunctionFactory::create(LinkFunctionType::Logit);
    CHECK_NEAR(logit->inverse_link(logit->link(0.3)), 0.3, kRoundTripTol);

    auto probit = LinkFunctionFactory::create(LinkFunctionType::Probit);
    CHECK_NEAR(probit->inverse_link(probit->link(0.7)), 0.7, kRoundTripTol);

    auto cloglog = LinkFunctionFactory::create(LinkFunctionType::ComplementaryLogLog);
    CHECK_NEAR(cloglog->inverse_link(cloglog->link(0.4)), 0.4, kRoundTripTol);
}

// ──────────────────────────────────────────────
//  LinkController (Test_LinkFunctions.cs)
// ──────────────────────────────────────────────

void test_link_controller_empty_acts_as_identity() {
    LinkController ctrl;
    CHECK_EQ(ctrl.count(), 0);

    std::vector<double> x = {100.0, 0.5, -0.1};
    std::vector<double> eta = ctrl.link(x);
    CHECK_NEAR(eta[0], 100.0, 1e-12);
    CHECK_NEAR(eta[1], 0.5, 1e-12);
    CHECK_NEAR(eta[2], -0.1, 1e-12);

    std::vector<double> x_back = ctrl.inverse_link(eta);
    CHECK_NEAR(x_back[0], 100.0, 1e-12);
    CHECK_NEAR(x_back[1], 0.5, 1e-12);
    CHECK_NEAR(x_back[2], -0.1, 1e-12);
}

void test_link_controller_single_link() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(std::make_unique<LogLink>());
    LinkController ctrl(std::move(links));
    CHECK_EQ(ctrl.count(), 1);

    std::vector<double> x = {2.0, 5.0, -3.0};
    std::vector<double> eta = ctrl.link(x);
    // First element transformed by LogLink.
    CHECK_NEAR(eta[0], std::log(2.0), 1e-12);
    // Remaining elements pass through (no link registered).
    CHECK_NEAR(eta[1], 5.0, 1e-12);
    CHECK_NEAR(eta[2], -3.0, 1e-12);
}

void test_link_controller_mixed_links() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(nullptr);
    links.push_back(std::make_unique<LogLink>());
    links.push_back(nullptr);
    LinkController ctrl(std::move(links));
    CHECK_EQ(ctrl.count(), 3);

    std::vector<double> x = {42.0, 10.0, -7.0};
    std::vector<double> eta = ctrl.link(x);
    CHECK_NEAR(eta[0], 42.0, 1e-12);             // null = identity
    CHECK_NEAR(eta[1], std::log(10.0), 1e-12);   // LogLink
    CHECK_NEAR(eta[2], -7.0, 1e-12);             // null = identity

    std::vector<double> x_back = ctrl.inverse_link(eta);
    CHECK_NEAR(x_back[0], 42.0, 1e-12);
    CHECK_NEAR(x_back[1], 10.0, 1e-10);
    CHECK_NEAR(x_back[2], -7.0, 1e-12);
}

void test_link_controller_round_trip() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(nullptr);
    links.push_back(std::make_unique<LogLink>());
    links.push_back(std::make_unique<LogitLink>());
    LinkController ctrl(std::move(links));

    std::vector<double> x = {100.0, 5.0, 0.3};
    std::vector<double> eta = ctrl.link(x);
    std::vector<double> x_back = ctrl.inverse_link(eta);
    CHECK_NEAR(x_back[0], x[0], kRoundTripTol);
    CHECK_NEAR(x_back[1], x[1], kRoundTripTol);
    CHECK_NEAR(x_back[2], x[2], kRoundTripTol);
}

void test_link_controller_array_longer_than_link_count() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(std::make_unique<LogLink>());
    LinkController ctrl(std::move(links));

    std::vector<double> x = {3.0, 7.0, 11.0, 13.0};
    std::vector<double> eta = ctrl.link(x);
    CHECK_NEAR(eta[0], std::log(3.0), 1e-12);
    CHECK_NEAR(eta[1], 7.0, 1e-12);
    CHECK_NEAR(eta[2], 11.0, 1e-12);
    CHECK_NEAR(eta[3], 13.0, 1e-12);
}

void test_link_controller_indexer() {
    auto log_link = std::make_unique<LogLink>();
    const ILinkFunction* raw = log_link.get();
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(nullptr);
    links.push_back(std::move(log_link));
    LinkController ctrl(std::move(links));

    CHECK_TRUE(ctrl[0] == nullptr);
    CHECK_TRUE(ctrl[1] == raw);       // Assert.AreSame
    CHECK_TRUE(ctrl[2] == nullptr);   // out of range
    CHECK_TRUE(ctrl[-1] == nullptr);  // negative index
}

void test_link_controller_for_location_scale_shape() {
    auto log_link = std::make_unique<LogLink>();
    const ILinkFunction* raw = log_link.get();
    LinkController ctrl =
        LinkController::for_location_scale_shape(nullptr, std::move(log_link), nullptr);
    CHECK_EQ(ctrl.count(), 3);
    CHECK_TRUE(ctrl[0] == nullptr);
    CHECK_TRUE(ctrl[1] == raw);
    CHECK_TRUE(ctrl[2] == nullptr);
}

void test_link_controller_link_jacobian() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(nullptr);
    links.push_back(std::make_unique<LogLink>());
    LinkController ctrl(std::move(links));

    std::vector<double> x = {42.0, 5.0};
    auto J = ctrl.link_jacobian(x);
    // First element: identity => diagonal = 1.0
    CHECK_NEAR(J(0, 0), 1.0, 1e-12);
    // Second element: LogLink => h'(5) = 1/5 = 0.2
    CHECK_NEAR(J(1, 1), 0.2, 1e-12);
    // Off-diagonal = 0
    CHECK_NEAR(J(0, 1), 0.0, 1e-12);
    CHECK_NEAR(J(1, 0), 0.0, 1e-12);
}

void test_link_controller_log_det_jacobian() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(std::make_unique<LogLink>());
    LinkController ctrl(std::move(links));
    // phi = [log(5)] in link-space.
    std::vector<double> phi = {std::log(5.0)};
    // InverseLink(log5) = 5; DLink(5) = 1/5; log|det J^{-1}| = -log|0.2| = log(5).
    CHECK_NEAR(ctrl.log_det_jacobian(phi), std::log(5.0), 1e-10);
}

void test_link_controller_log_det_jacobian_identity() {
    LinkController ctrl;
    std::vector<double> phi = {1.0, 2.0, 3.0};
    CHECK_NEAR(ctrl.log_det_jacobian(phi), 0.0, 1e-12);
}

void test_link_controller_log_det_jacobian_multiple_links() {
    std::vector<std::unique_ptr<ILinkFunction>> links;
    links.push_back(nullptr);
    links.push_back(std::make_unique<LogLink>());
    links.push_back(std::make_unique<LogLink>());
    LinkController ctrl(std::move(links));
    // phi = [100, log(3), log(7)]
    std::vector<double> phi = {100.0, std::log(3.0), std::log(7.0)};
    // index 0: null => 0; index 1: -log(1/3) = log(3); index 2: -log(1/7) = log(7).
    CHECK_NEAR(ctrl.log_det_jacobian(phi), std::log(3.0) + std::log(7.0), 1e-10);
}

// ──────────────────────────────────────────────
//  FisherZLink (Test_FisherZLink.cs)
// ──────────────────────────────────────────────

void test_fisherz_link_known_values() {
    FisherZLink link;
    CHECK_NEAR(link.link(0.0), 0.0, 1e-12);
    CHECK_NEAR(link.link(0.5), 0.5 * std::log(3.0), 1e-12);
    CHECK_NEAR(link.link(-0.5), -0.5 * std::log(3.0), 1e-12);
}

void test_fisherz_inverse_link_known_values() {
    FisherZLink link;
    CHECK_NEAR(link.inverse_link(0.0), 0.0, 1e-12);
    CHECK_NEAR(link.inverse_link(0.5 * std::log(3.0)), 0.5, 1e-12);
    CHECK_NEAR(link.inverse_link(-0.5 * std::log(3.0)), -0.5, 1e-12);
}

void test_fisherz_inverse_link_large_eta_approaches_bounds() {
    FisherZLink link;
    double upper = link.inverse_link(10.0);
    double lower = link.inverse_link(-10.0);
    CHECK_TRUE(bestfit::numerics::is_finite(upper));
    CHECK_TRUE(bestfit::numerics::is_finite(lower));
    CHECK_TRUE(upper < 1.0);
    CHECK_TRUE(upper > 0.99999999);
    CHECK_TRUE(lower > -1.0);
    CHECK_TRUE(lower < -0.99999999);
}

void test_fisherz_round_trip_recovers_input() {
    FisherZLink link;
    double values[] = {-0.99, -0.75, -0.25, 0.0, 0.25, 0.75, 0.99};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, 1e-12);
}

void test_fisherz_d_link_finite_difference_consistency() {
    FisherZLink link;
    double values[] = {-0.8, -0.25, 0.0, 0.25, 0.8};
    for (double x : values) {
        double expected = 1.0 / (1.0 - x * x);
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(link.d_link(x), expected, 1e-12);
        CHECK_NEAR(link.d_link(x), finite_diff, 1e-6);
    }
}

void test_fisherz_link_outside_open_interval_throws() {
    FisherZLink link;
    CHECK_THROWS(link.link(-1.0));
    CHECK_THROWS(link.link(1.0));
    CHECK_THROWS(link.link(kNaN));
    CHECK_THROWS(link.d_link(-1.0));
    CHECK_THROWS(link.d_link(1.0));
    CHECK_THROWS(link.d_link(kInf));
}

void test_fisherz_factory_creates_fisher_z_link() {
    CHECK_TRUE(dynamic_cast<FisherZLink*>(LinkFunctionFactory::create(LinkFunctionType::FisherZ).get()) != nullptr);
}

// ──────────────────────────────────────────────
//  YeoJohnsonLink (Test_YeoJohnsonLink.cs)
// ──────────────────────────────────────────────

// Test_YeoJohnsonLink.cs uses its own constants.
const double kYJRoundTripTol = 1e-9;
const double kYJDerivativeTol = 1e-4;

void test_yj_constructor_default_lambda_is_one() {
    YeoJohnsonLink link;
    CHECK_NEAR(link.lambda(), 1.0, 1e-12);
}

void test_yj_constructor_lambda_stores_value() {
    YeoJohnsonLink link(0.5);
    CHECK_NEAR(link.lambda(), 0.5, 1e-12);
}

void test_yj_constructor_lambda_invalid_values_throws() {
    CHECK_THROWS(YeoJohnsonLink(kNaN));
    CHECK_THROWS(YeoJohnsonLink(kInf));
    CHECK_THROWS(YeoJohnsonLink(-5.1));
    CHECK_THROWS(YeoJohnsonLink(5.1));
}

void test_yj_constructor_values_single_element_throws() {
    CHECK_THROWS(YeoJohnsonLink(std::vector<double>{1.0}));
}

void test_yj_constructor_values_non_finite_value_throws() {
    CHECK_THROWS(YeoJohnsonLink(std::vector<double>{1.0, kNaN}));
    CHECK_THROWS(YeoJohnsonLink(std::vector<double>{1.0, -kInf}));
}

void test_yj_constructor_values_produces_finite_lambda() {
    YeoJohnsonLink link(std::vector<double>{-2.0, -1.0, -0.25, 0.0, 0.5, 1.0, 3.0});
    CHECK_TRUE(bestfit::numerics::is_finite(link.lambda()));
}

void test_yj_link_lambda_one_is_identity() {
    YeoJohnsonLink link(1.0);
    double values[] = {-5.0, -1.0, 0.0, 0.5, 1.0, 5.0};
    for (double x : values) {
        CHECK_NEAR(link.link(x), x, 1e-10);
        CHECK_NEAR(link.d_link(x), 1.0, 1e-10);
    }
}

void test_yj_link_lambda_zero_uses_positive_log_branch() {
    YeoJohnsonLink link(0.0);
    CHECK_NEAR(link.link(2.0), std::log(3.0), 1e-12);
    CHECK_NEAR(link.inverse_link(std::log(3.0)), 2.0, 1e-12);
    CHECK_NEAR(link.d_link(2.0), 1.0 / 3.0, 1e-12);
}

void test_yj_link_lambda_two_uses_negative_log_branch() {
    YeoJohnsonLink link(2.0);
    CHECK_NEAR(link.link(-2.0), -std::log(3.0), 1e-12);
    CHECK_NEAR(link.inverse_link(-std::log(3.0)), -2.0, 1e-12);
    CHECK_NEAR(link.d_link(-2.0), 1.0 / 3.0, 1e-12);
}

void test_yj_round_trip_positive_and_negative_values() {
    YeoJohnsonLink link(0.5);
    double values[] = {-10.0, -5.0, -1.0, -0.1, 0.0, 0.1, 1.0, 5.0, 10.0};
    for (double x : values) CHECK_NEAR(link.inverse_link(link.link(x)), x, kYJRoundTripTol);
}

void test_yj_d_link_finite_difference_consistency() {
    YeoJohnsonLink link(0.5);
    double values[] = {-5.0, -2.0, -0.5, 0.5, 1.0, 5.0};
    for (double x : values) {
        double finite_diff = (link.link(x + kDeltaH) - link.link(x - kDeltaH)) / (2.0 * kDeltaH);
        CHECK_NEAR(link.d_link(x), finite_diff, kYJDerivativeTol);
    }
}

void test_yj_factory_creates_yeo_johnson_link() {
    auto created = LinkFunctionFactory::create(LinkFunctionType::YeoJohnson);
    auto* yj = dynamic_cast<YeoJohnsonLink*>(created.get());
    CHECK_TRUE(yj != nullptr);
    // Supplement (XML half of the upstream test is skipped): the factory default has lambda 1.
    CHECK_NEAR(yj->lambda(), 1.0, 1e-12);
}

// ──────────────────────────────────────────────
//  YeoJohnson transform class (no upstream test file; C# source-derived oracles)
// ──────────────────────────────────────────────

void test_yeo_johnson_transform_branches() {
    // Positive branch, |lambda| >= 1e-8: ((x+1)^lambda - 1)/lambda.
    CHECK_NEAR(YeoJohnson::transform(2.0, 0.5), 2.0 * (std::sqrt(3.0) - 1.0), 1e-12);
    // Positive branch, lambda ~ 0: log(x+1).
    CHECK_NEAR(YeoJohnson::transform(2.0, 0.0), std::log(3.0), 1e-12);
    // Negative branch, lambda != 2: -(((-x+1)^(2-lambda) - 1)/(2-lambda)).
    CHECK_NEAR(YeoJohnson::transform(-2.0, 0.5), -(std::pow(3.0, 1.5) - 1.0) / 1.5, 1e-12);
    // Negative branch, lambda == 2: -log(-x+1).
    CHECK_NEAR(YeoJohnson::transform(-2.0, 2.0), -std::log(3.0), 1e-12);
    // |lambda| > 5 -> NaN.
    CHECK_TRUE(std::isnan(YeoJohnson::transform(1.0, 5.5)));
    CHECK_TRUE(std::isnan(YeoJohnson::transform(1.0, -5.5)));
}

void test_yeo_johnson_derivative() {
    // x >= 0: (x+1)^(lambda-1).
    CHECK_NEAR(YeoJohnson::derivative(2.0, 0.5), std::pow(3.0, -0.5), 1e-12);
    CHECK_NEAR(YeoJohnson::derivative(0.0, 3.0), 1.0, 1e-12);
    // x < 0: (-x+1)^(1-lambda).
    CHECK_NEAR(YeoJohnson::derivative(-2.0, 0.5), std::pow(3.0, 0.5), 1e-12);
    // |lambda| > 5 -> NaN.
    CHECK_TRUE(std::isnan(YeoJohnson::derivative(1.0, 5.5)));
}

void test_yeo_johnson_inverse_transform_branches() {
    CHECK_NEAR(YeoJohnson::inverse_transform(2.0 * (std::sqrt(3.0) - 1.0), 0.5), 2.0, 1e-12);
    CHECK_NEAR(YeoJohnson::inverse_transform(std::log(3.0), 0.0), 2.0, 1e-12);
    CHECK_NEAR(YeoJohnson::inverse_transform(-(std::pow(3.0, 1.5) - 1.0) / 1.5, 0.5), -2.0, 1e-12);
    CHECK_NEAR(YeoJohnson::inverse_transform(-std::log(3.0), 2.0), -2.0, 1e-12);
    CHECK_TRUE(std::isnan(YeoJohnson::inverse_transform(1.0, 5.5)));
}

void test_yeo_johnson_vector_overloads() {
    std::vector<double> values = {-2.0, 0.0, 2.0};
    std::vector<double> transformed = YeoJohnson::transform(values, 0.5);
    CHECK_EQ(static_cast<int>(transformed.size()), 3);
    for (std::size_t i = 0; i < values.size(); ++i)
        CHECK_NEAR(transformed[i], YeoJohnson::transform(values[i], 0.5), 0.0);
    std::vector<double> back = YeoJohnson::inverse_transform(transformed, 0.5);
    for (std::size_t i = 0; i < values.size(); ++i) CHECK_NEAR(back[i], values[i], 1e-12);
}

void test_yeo_johnson_log_jacobian() {
    // Sum of log|dT/dy|: x=2 -> log(3^-0.5) = -0.5 log 3; x=-3 -> log(4^0.5) = log 2.
    std::vector<double> values = {2.0, -3.0};
    double expected = -0.5 * std::log(3.0) + std::log(2.0);
    CHECK_NEAR(YeoJohnson::log_jacobian(values, 0.5), expected, 1e-12);
}

void test_yeo_johnson_log_likelihood_lambda_one() {
    // At lambda = 1 the transform is the identity and the log-Jacobian is 0, so the C#
    // formula reduces to -n/2*LogSqrt2PI - n/2*log(sigmaSq) - sse/(2*sigmaSq).
    std::vector<double> values = {1.0, 2.0, 3.0, 4.0};
    // mu = 2.5, sse = 5, sigmaSq = 1.25.
    double expected = -2.0 * bestfit::numerics::kLogSqrt2PI - 2.0 * std::log(1.25) - 2.0;
    CHECK_NEAR(YeoJohnson::log_likelihood(values, 1.0), expected, 1e-12);
}

void test_yeo_johnson_log_likelihood_degenerate_is_neg_infinity() {
    // Constant data: sigmaSq == 0 -> NegativeInfinity.
    std::vector<double> values = {1.0, 1.0, 1.0};
    CHECK_TRUE(std::isinf(YeoJohnson::log_likelihood(values, 1.0)));
    CHECK_TRUE(YeoJohnson::log_likelihood(values, 1.0) < 0.0);
}

void test_yeo_johnson_fit_lambda_throws_for_degenerate_input() {
    CHECK_THROWS(YeoJohnson::fit_lambda(std::vector<double>{1.0}));
}

void test_yeo_johnson_fit_lambda_maximizes_log_likelihood() {
    std::vector<double> values = {-2.0, -1.0, -0.25, 0.0, 0.5, 1.0, 3.0};
    double lambda = YeoJohnson::fit_lambda(values);
    CHECK_TRUE(bestfit::numerics::is_finite(lambda));
    CHECK_TRUE(lambda >= -5.0 && lambda <= 5.0);
    // Local-maximum property of the MLE.
    double ll = YeoJohnson::log_likelihood(values, lambda);
    CHECK_TRUE(ll >= YeoJohnson::log_likelihood(values, lambda - 0.05));
    CHECK_TRUE(ll >= YeoJohnson::log_likelihood(values, lambda + 0.05));
}

// ──────────────────────────────────────────────
//  Brief-required supplements: round-trip identity to 1e-10 and central-difference
//  derivative checks for EVERY link over representative domain ranges.
// ──────────────────────────────────────────────

void check_round_trip_and_derivative(const ILinkFunction& link, const std::vector<double>& xs,
                                     double round_trip_tol = 1e-10) {
    for (double x : xs) {
        CHECK_NEAR(link.inverse_link(link.link(x)), x, round_trip_tol);
        double h = kDeltaH * std::max(1.0, std::fabs(x));
        double finite_diff = (link.link(x + h) - link.link(x - h)) / (2.0 * h);
        double d = link.d_link(x);
        // ~1e-6 relative tolerance for central differences.
        CHECK_NEAR(d, finite_diff, std::max(1e-6, 1e-6 * std::fabs(d)));
    }
}

void test_supplement_all_links_round_trip_and_derivative() {
    check_round_trip_and_derivative(IdentityLink(), {-1000.0, -1.0, -0.001, 0.0, 0.001, 1.0, 1000.0});
    check_round_trip_and_derivative(LogLink(), {0.001, 0.1, 0.5, 1.0, 10.0, 1000.0});
    check_round_trip_and_derivative(LogitLink(), {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99});
    check_round_trip_and_derivative(ProbitLink(), {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99});
    check_round_trip_and_derivative(ComplementaryLogLogLink(),
                                    {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99});
    check_round_trip_and_derivative(FisherZLink(), {-0.95, -0.5, -0.1, 0.0, 0.1, 0.5, 0.95});
    // YeoJohnson over several lambdas including the special cases 0 and 2 (and the
    // near-boundary +/-5). Round trip to 1e-9 (upstream Test_YeoJohnsonLink tolerance:
    // pow/root round trips lose a digit vs the closed-form links).
    double lambdas[] = {-5.0, -2.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.5, 5.0};
    for (double lambda : lambdas) {
        YeoJohnsonLink link(lambda);
        check_round_trip_and_derivative(link, {-3.0, -1.0, -0.1, 0.0, 0.1, 1.0, 3.0}, 1e-9);
    }
}

}  // namespace

int main() {
    // Test_LinkFunctions.cs
    test_identity_link_link();
    test_identity_link_inverse_link();
    test_identity_link_d_link();
    test_identity_link_round_trip();
    test_log_link_link_known_values();
    test_log_link_inverse_link_known_values();
    test_log_link_d_link_known_values();
    test_log_link_round_trip();
    test_log_link_link_throws_for_zero();
    test_log_link_link_throws_for_negative();
    test_log_link_d_link_throws_for_zero();
    test_log_link_derivative_consistency();
    test_logit_link_link_known_values();
    test_logit_link_inverse_link_known_values();
    test_logit_link_d_link_known_values();
    test_logit_link_round_trip();
    test_logit_link_link_throws_for_zero();
    test_logit_link_link_throws_for_one();
    test_logit_link_link_throws_for_negative();
    test_logit_link_inverse_link_extreme_values();
    test_logit_link_derivative_consistency();
    test_probit_link_link_known_values();
    test_probit_link_inverse_link_known_values();
    test_probit_link_round_trip();
    test_probit_link_link_throws_for_zero();
    test_probit_link_link_throws_for_one();
    test_probit_link_derivative_consistency();
    test_cloglog_link_known_values();
    test_cloglog_inverse_link_known_values();
    test_cloglog_round_trip();
    test_cloglog_link_throws_for_zero();
    test_cloglog_link_throws_for_one();
    test_cloglog_derivative_consistency();
    test_factory_creates_correct_types();
    test_factory_throws_for_invalid_type();
    test_factory_round_trip_via_factory();
    test_link_controller_empty_acts_as_identity();
    test_link_controller_single_link();
    test_link_controller_mixed_links();
    test_link_controller_round_trip();
    test_link_controller_array_longer_than_link_count();
    test_link_controller_indexer();
    test_link_controller_for_location_scale_shape();
    test_link_controller_link_jacobian();
    test_link_controller_log_det_jacobian();
    test_link_controller_log_det_jacobian_identity();
    test_link_controller_log_det_jacobian_multiple_links();
    // Test_FisherZLink.cs
    test_fisherz_link_known_values();
    test_fisherz_inverse_link_known_values();
    test_fisherz_inverse_link_large_eta_approaches_bounds();
    test_fisherz_round_trip_recovers_input();
    test_fisherz_d_link_finite_difference_consistency();
    test_fisherz_link_outside_open_interval_throws();
    test_fisherz_factory_creates_fisher_z_link();
    // Test_YeoJohnsonLink.cs
    test_yj_constructor_default_lambda_is_one();
    test_yj_constructor_lambda_stores_value();
    test_yj_constructor_lambda_invalid_values_throws();
    test_yj_constructor_values_single_element_throws();
    test_yj_constructor_values_non_finite_value_throws();
    test_yj_constructor_values_produces_finite_lambda();
    test_yj_link_lambda_one_is_identity();
    test_yj_link_lambda_zero_uses_positive_log_branch();
    test_yj_link_lambda_two_uses_negative_log_branch();
    test_yj_round_trip_positive_and_negative_values();
    test_yj_d_link_finite_difference_consistency();
    test_yj_factory_creates_yeo_johnson_link();
    // YeoJohnson transform class
    test_yeo_johnson_transform_branches();
    test_yeo_johnson_derivative();
    test_yeo_johnson_inverse_transform_branches();
    test_yeo_johnson_vector_overloads();
    test_yeo_johnson_log_jacobian();
    test_yeo_johnson_log_likelihood_lambda_one();
    test_yeo_johnson_log_likelihood_degenerate_is_neg_infinity();
    test_yeo_johnson_fit_lambda_throws_for_degenerate_input();
    test_yeo_johnson_fit_lambda_maximizes_log_likelihood();
    // Supplements
    test_supplement_all_links_round_trip_and_derivative();
    return bftest::summary("test_link_functions");
}
