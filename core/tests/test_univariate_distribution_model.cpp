// Standalone test for bestfit::models::UnivariateDistributionModel (Phase 4, Task T6).
//
// Oracle for behavior is the C# source itself (upstream/RMC-BestFit/src/RMC.BestFit/Models/
// UnivariateDistribution/UnivariateDistribution.cs @ fc28c0c), STATIONARY EXACT-DATA path
// only -- see the header of univariate_distribution_model.hpp for the exact method/line
// mapping. This ctest asserts SELF-CONSISTENCY (log-likelihood vs closed-form Normal,
// pointwise sum, invalid-parameter -inf guards, default-parameter construction). The
// absolute match to the real C# model's bounds/initials/priors is validated later by the
// T12 emitter oracle against the fixture-driven runners; there is no fixtures/ entry for
// this file (same rationale as test_model_base.cpp).
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "check.hpp"

using bestfit::models::DataComponent;
using bestfit::models::DataComponentType;
using bestfit::models::UnivariateDistributionModel;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::UnivariateDistributionType;

namespace {

// Phase-4 dataset (brief's canonical sample).
std::vector<double> sample_data() {
    return {12500, 15300, 9870, 21000, 18400, 11200, 26800, 14100, 19500, 11600};
}

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
    }

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

}  // namespace

int main() {
    test_data_log_likelihood_matches_closed_form_normal();
    test_data_log_likelihood_negative_infinity_on_invalid_parameters();
    test_pointwise_sums_to_data_log_likelihood_and_has_matching_length();
    test_pointwise_invalid_parameters_returns_all_negative_infinity();
    test_pointwise_components_one_exact_component_per_value_matching_ll();
    test_pointwise_components_negative_infinity_on_invalid_parameters();
    test_set_default_parameters_populates_bounds_and_uniform_priors();
    test_construct_from_owned_distribution_pointer();

    return bftest::summary("univariate_distribution_model");
}
