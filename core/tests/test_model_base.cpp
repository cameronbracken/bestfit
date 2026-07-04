// Standalone test for bestfit::models::ModelBase (Phase 4, Task T5).
//
// Oracle for behavior is the C# source itself (upstream/RMC-BestFit/src/RMC.BestFit/Models/
// Support/{IModel,ModelBase}.cs @ fc28c0c) -- this is a structural/behavioral port of the
// compute defaults (LogLikelihood/PriorLogLikelihood/PointwisePriorLogLikelihood/
// SetParameterValues), not numeric-fixture-driven, so there is no fixtures/ entry for this
// file. A test-local trivial derived model (StubNormalModel) exercises the abstract base
// through its virtual compute surface.
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "bestfit/models/support/data_component.hpp"
#include "bestfit/models/support/model_base.hpp"
#include "bestfit/models/support/model_parameter.hpp"
#include "bestfit/models/support/prior_component.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "check.hpp"

using bestfit::models::DataComponent;
using bestfit::models::ModelBase;
using bestfit::models::ModelParameter;
using bestfit::models::PriorComponent;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::Uniform;

namespace {

// A trivial derived model: two parameters (mu, sigma) with Normal(0,10) priors, evaluated
// against a small fixed dataset. data_log_likelihood returns NaN when sigma <= 0 (an easy way
// to force a non-finite data likelihood without any distribution machinery beyond Normal).
class StubNormalModel : public ModelBase {
   public:
    StubNormalModel() {
        parameters().push_back(ModelParameter("StubNormalModel", "mu", 0.0, -100.0, 100.0,
                                               std::make_unique<Normal>(0.0, 10.0)));
        parameters().push_back(ModelParameter("StubNormalModel", "sigma", 1.0, 1e-8, 100.0,
                                               std::make_unique<Normal>(0.0, 10.0)));
    }

    void set_default_parameters() override {
        // No-op stub (T5 scope): SetDefaultParameters is exercised in the full Models phase.
    }

    bestfit::models::ValidationResult validate() const override {
        // Trivially-valid stub (validate() became pure virtual on ModelBase in M8).
        return {};
    }

    double data_log_likelihood(std::vector<double>& p) const override {
        double mu = p[0];
        double sigma = p[1];
        if (sigma <= 0.0) return std::numeric_limits<double>::quiet_NaN();
        Normal dist(mu, sigma);
        double ll = 0.0;
        for (double x : data_) ll += dist.log_pdf(x);
        return ll;
    }

    std::vector<double> pointwise_data_log_likelihood(const std::vector<double>& p) const override {
        double mu = p[0];
        double sigma = p[1];
        Normal dist(mu, sigma);
        std::vector<double> result;
        result.reserve(data_.size());
        for (double x : data_) result.push_back(dist.log_pdf(x));
        return result;
    }

    std::vector<DataComponent> pointwise_data_log_likelihood_components(
        const std::vector<double>& p) const override {
        std::vector<double> pointwise = pointwise_data_log_likelihood(p);
        std::vector<DataComponent> result;
        result.reserve(pointwise.size());
        for (std::size_t i = 0; i < pointwise.size(); ++i) {
            result.emplace_back(static_cast<int>(i), pointwise[i], data_[i]);
        }
        return result;
    }

   private:
    std::vector<double> data_{1.0, 2.0, 3.0, 4.0, 5.0};
};

void test_log_likelihood_equals_data_plus_prior() {
    StubNormalModel model;
    std::vector<double> p{2.5, 1.5};

    double expected_data = model.data_log_likelihood(p);
    double expected_prior = model.prior_log_likelihood(p);
    double actual = model.log_likelihood(p);

    CHECK_NEAR(actual, expected_data + expected_prior, 1e-12);
}

void test_prior_log_likelihood_hand_summed() {
    StubNormalModel model;
    std::vector<double> p{2.5, 1.5};

    Normal prior(0.0, 10.0);
    double expected = prior.log_pdf(p[0]) + prior.log_pdf(p[1]);

    CHECK_NEAR(model.prior_log_likelihood(p), expected, 1e-12);
}

void test_prior_log_likelihood_wrong_length_is_negative_infinity() {
    StubNormalModel model;
    std::vector<double> p{2.5};  // wrong length: model has 2 parameters

    double result = model.prior_log_likelihood(p);
    CHECK_TRUE(result == -std::numeric_limits<double>::infinity());
}

void test_prior_log_likelihood_negative_infinity_on_nonfinite_prior_sum() {
    // Drives prior_log_likelihood's OWN post-sum finite-guard (model_base.hpp:88), as opposed
    // to the length-mismatch early return (model_base.hpp:82) or non-finiteness surfacing via
    // data_log_likelihood (covered by test_log_likelihood_negative_infinity_on_nonfinite_data_ll
    // below). Swap the sigma parameter's prior for a bounded Uniform(-1, 1): evaluating it at
    // an in-support-length, out-of-bounds point (5.0) makes Uniform::pdf return 0.0, so
    // log_pdf returns -inf (base class default: log_pdf returns -inf whenever pdf <= 0). The
    // mu parameter's prior stays Normal(0, 10) and is evaluated at a finite, in-support point,
    // so the -inf can only be coming from the prior sum itself, not any length check or the
    // data likelihood (prior_log_likelihood never calls data_log_likelihood).
    StubNormalModel model;
    model.parameters()[1].set_prior_distribution(std::make_unique<Uniform>(-1.0, 1.0));
    std::vector<double> p{0.0, 5.0};  // correct length: mu finite/in-bounds, sigma out-of-bounds

    double result = model.prior_log_likelihood(p);
    CHECK_TRUE(result == -std::numeric_limits<double>::infinity());
}

void test_log_likelihood_negative_infinity_on_nonfinite_data_ll() {
    StubNormalModel model;
    std::vector<double> p{2.5, -1.0};  // sigma <= 0 -> data_log_likelihood returns NaN

    double result = model.log_likelihood(p);
    CHECK_TRUE(result == -std::numeric_limits<double>::infinity());
}

void test_set_parameter_values_right_length_updates_values() {
    StubNormalModel model;
    std::vector<double> p{7.0, 3.0};

    model.set_parameter_values(p);

    CHECK_NEAR(model.parameters()[0].value(), 7.0, 1e-12);
    CHECK_NEAR(model.parameters()[1].value(), 3.0, 1e-12);

    // const overload is reachable too.
    const ModelBase& const_model = model;
    CHECK_NEAR(const_model.parameters()[0].value(), 7.0, 1e-12);
}

void test_set_parameter_values_wrong_length_throws() {
    StubNormalModel model;
    std::vector<double> p{1.0};  // wrong length: model has 2 parameters

    CHECK_THROWS(model.set_parameter_values(p));
}

void test_number_of_parameters() {
    StubNormalModel model;
    CHECK_EQ(model.number_of_parameters(), 2);
}

void test_pointwise_prior_log_likelihood_names_and_count() {
    StubNormalModel model;
    std::vector<double> p{2.5, 1.5};

    std::vector<PriorComponent> components = model.pointwise_prior_log_likelihood(p);
    CHECK_EQ(components.size(), static_cast<std::size_t>(2));

    Normal prior(0.0, 10.0);
    CHECK_NEAR(components[0].log_likelihood(), prior.log_pdf(p[0]), 1e-12);
    CHECK_NEAR(components[1].log_likelihood(), prior.log_pdf(p[1]), 1e-12);

    // Names should reference the owner name ("StubNormalModel") since ownerName is non-empty.
    CHECK_TRUE(components[0].name().find("StubNormalModel") != std::string::npos);
    CHECK_TRUE(components[1].name().find("StubNormalModel") != std::string::npos);
    CHECK_TRUE(components[0].name().find("Parameter Prior") != std::string::npos);
}

void test_pointwise_prior_log_likelihood_wrong_length_is_empty() {
    StubNormalModel model;
    std::vector<double> p{1.0};  // wrong length

    std::vector<PriorComponent> components = model.pointwise_prior_log_likelihood(p);
    CHECK_TRUE(components.empty());
}

void test_use_default_flat_priors_property() {
    StubNormalModel model;
    CHECK_TRUE(model.use_default_flat_priors());  // default matches C# (true)

    model.set_use_default_flat_priors(false);
    CHECK_TRUE(!model.use_default_flat_priors());
}

}  // namespace

int main() {
    test_log_likelihood_equals_data_plus_prior();
    test_prior_log_likelihood_hand_summed();
    test_prior_log_likelihood_wrong_length_is_negative_infinity();
    test_prior_log_likelihood_negative_infinity_on_nonfinite_prior_sum();
    test_log_likelihood_negative_infinity_on_nonfinite_data_ll();
    test_set_parameter_values_right_length_updates_values();
    test_set_parameter_values_wrong_length_throws();
    test_number_of_parameters();
    test_pointwise_prior_log_likelihood_names_and_count();
    test_pointwise_prior_log_likelihood_wrong_length_is_empty();
    test_use_default_flat_priors_property();

    return bftest::summary("model_base");
}
