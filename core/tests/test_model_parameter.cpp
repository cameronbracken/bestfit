// Standalone tests for corehydro::models::DataComponent, PriorComponent, and ModelParameter.
//
// Oracle for behavior is the C# source itself (upstream/RMC-BestFit/src/RMC.BestFit/Models/
// Support/{DataComponent,PriorComponent,ModelParameter}.cs @ fc28c0c) -- these are structural/
// behavioral ports (DisplayName branches, censoring/threshold predicates, clone independence),
// not numeric-fixture-driven, so there is no fixtures/ entry for this file.
#include <limits>
#include <memory>
#include <string>

#include "corehydro/models/support/data_component.hpp"
#include "corehydro/models/support/model_parameter.hpp"
#include "corehydro/models/support/prior_component.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "check.hpp"

using corehydro::models::DataComponent;
using corehydro::models::DataComponentType;
using corehydro::models::ModelParameter;
using corehydro::models::PriorComponent;
using corehydro::models::PriorComponentType;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

void test_data_component_exact_ctor() {
    DataComponent dc(2, -1.5, 3.14, "obs");
    CHECK_EQ(dc.index(), 2);
    CHECK_NEAR(dc.log_likelihood(), -1.5, 1e-12);
    CHECK_NEAR(dc.value(), 3.14, 1e-12);
    CHECK_TRUE(dc.type() == DataComponentType::Exact);
    CHECK_EQ(dc.count(), 1);
    CHECK_TRUE(dc.name().has_value());
    CHECK_EQ(*dc.name(), std::string("obs"));

    DataComponent unnamed(0, 0.0, 1.0);
    CHECK_TRUE(!unnamed.name().has_value());
}

void test_data_component_full_ctor() {
    DataComponent dc(5, -2.0, 10.0, DataComponentType::LeftCensored, 3, "threshold");
    CHECK_EQ(dc.index(), 5);
    CHECK_NEAR(dc.log_likelihood(), -2.0, 1e-12);
    CHECK_NEAR(dc.value(), 10.0, 1e-12);
    CHECK_TRUE(dc.type() == DataComponentType::LeftCensored);
    CHECK_EQ(dc.count(), 3);
    CHECK_EQ(*dc.name(), std::string("threshold"));

    // default count == 1, default name == nullopt
    DataComponent dc2(0, -1.0, 5.0, DataComponentType::Interval);
    CHECK_EQ(dc2.count(), 1);
    CHECK_TRUE(!dc2.name().has_value());
}

void test_data_component_is_censored_is_threshold() {
    DataComponent exact(0, 0.0, 1.0, DataComponentType::Exact);
    CHECK_TRUE(!exact.is_censored());
    CHECK_TRUE(!exact.is_threshold());

    DataComponent uncertain(0, 0.0, 1.0, DataComponentType::Uncertain);
    CHECK_TRUE(!uncertain.is_censored());

    DataComponent interval(0, 0.0, 1.0, DataComponentType::Interval);
    CHECK_TRUE(!interval.is_censored());

    DataComponent left_count1(0, 0.0, 1.0, DataComponentType::LeftCensored, 1);
    CHECK_TRUE(left_count1.is_censored());
    CHECK_TRUE(!left_count1.is_threshold());

    DataComponent left_count3(0, 0.0, 1.0, DataComponentType::LeftCensored, 3);
    CHECK_TRUE(left_count3.is_censored());
    CHECK_TRUE(left_count3.is_threshold());

    DataComponent right_count1(0, 0.0, 1.0, DataComponentType::RightCensored, 1);
    CHECK_TRUE(right_count1.is_censored());
    CHECK_TRUE(!right_count1.is_threshold());

    DataComponent right_count5(0, 0.0, 1.0, DataComponentType::RightCensored, 5);
    CHECK_TRUE(right_count5.is_censored());
    CHECK_TRUE(right_count5.is_threshold());
}

void test_data_component_to_string() {
    DataComponent exact(0, -1.2345, 3.14159, "x1");
    std::string s = exact.to_string();
    CHECK_TRUE(s.find("x1") != std::string::npos);
    CHECK_TRUE(s.find('~') == std::string::npos);

    DataComponent uncertain(1, -0.5, 2.0, DataComponentType::Uncertain);
    std::string su = uncertain.to_string();
    CHECK_TRUE(su.find('~') != std::string::npos);
    CHECK_TRUE(su.find("[1]") != std::string::npos);  // unnamed -> "[Index]"

    DataComponent interval(2, -0.5, 2.0, DataComponentType::Interval);
    std::string si = interval.to_string();
    CHECK_TRUE(si.find('[') != std::string::npos);
    CHECK_TRUE(si.find(']') != std::string::npos);

    DataComponent left(3, -0.5, 2.0, DataComponentType::LeftCensored, 4, "period");
    std::string sl = left.to_string();
    CHECK_TRUE(sl.find('<') != std::string::npos);
    CHECK_TRUE(sl.find("n=4") != std::string::npos);

    DataComponent right(4, -0.5, 2.0, DataComponentType::RightCensored, 7);
    std::string sr = right.to_string();
    CHECK_TRUE(sr.find('>') != std::string::npos);
    CHECK_TRUE(sr.find("n=7") != std::string::npos);
}

void test_prior_component() {
    PriorComponent pc("Location prior", -0.5);
    CHECK_EQ(pc.name(), std::string("Location prior"));
    CHECK_NEAR(pc.log_likelihood(), -0.5, 1e-12);
    CHECK_TRUE(pc.type() == PriorComponentType::ParameterPrior);

    std::string s = pc.to_string();
    CHECK_TRUE(s.find("Location prior") != std::string::npos);

    PriorComponent pc2("Jacobian term", -2.0, PriorComponentType::Jacobian);
    CHECK_TRUE(pc2.type() == PriorComponentType::Jacobian);
}

void test_model_parameter_defaults() {
    ModelParameter mp;
    CHECK_EQ(mp.owner_name(), std::string(""));
    CHECK_EQ(mp.name(), std::string("Parameter"));
    CHECK_NEAR(mp.value(), 0.0, 1e-12);
    CHECK_TRUE(mp.lower_bound() == std::numeric_limits<double>::lowest());
    CHECK_TRUE(mp.upper_bound() == std::numeric_limits<double>::max());
    CHECK_TRUE(!mp.is_positive());
    CHECK_TRUE(!mp.is_fixed());
    CHECK_TRUE(mp.prior_distribution().type() == UnivariateDistributionType::Uniform);
}

void test_model_parameter_display_name() {
    ModelParameter mp;
    mp.set_owner_name("");
    mp.set_name("X");
    CHECK_EQ(mp.display_name(), std::string("X"));

    mp.set_owner_name("Owner");
    mp.set_name("");
    CHECK_EQ(mp.display_name(), std::string("Owner"));

    mp.set_owner_name("Owner");
    mp.set_name("X");
    CHECK_EQ(mp.display_name(), std::string("Owner X"));
}

void test_model_parameter_full_ctor_and_prior() {
    auto normal = std::make_unique<Normal>(0.0, 1.0);
    ModelParameter mp("Owner", "Param", 1.5, -10.0, 10.0, std::move(normal), true, false);

    CHECK_EQ(mp.owner_name(), std::string("Owner"));
    CHECK_EQ(mp.name(), std::string("Param"));
    CHECK_NEAR(mp.value(), 1.5, 1e-12);
    CHECK_NEAR(mp.lower_bound(), -10.0, 1e-12);
    CHECK_NEAR(mp.upper_bound(), 10.0, 1e-12);
    CHECK_TRUE(mp.is_positive());
    CHECK_TRUE(!mp.is_fixed());

    Normal raw(0.0, 1.0);
    CHECK_NEAR(mp.prior_distribution().log_pdf(0.7), raw.log_pdf(0.7), 1e-12);
}

void test_model_parameter_getters_setters_round_trip() {
    ModelParameter mp;
    mp.set_value(3.3);
    mp.set_lower_bound(-1.0);
    mp.set_upper_bound(5.0);
    mp.set_is_positive(true);
    mp.set_is_fixed(true);
    mp.set_owner_name("Comp");
    mp.set_name("Loc");

    CHECK_NEAR(mp.value(), 3.3, 1e-12);
    CHECK_NEAR(mp.lower_bound(), -1.0, 1e-12);
    CHECK_NEAR(mp.upper_bound(), 5.0, 1e-12);
    CHECK_TRUE(mp.is_positive());
    CHECK_TRUE(mp.is_fixed());
    CHECK_EQ(mp.owner_name(), std::string("Comp"));
    CHECK_EQ(mp.name(), std::string("Loc"));

    mp.set_prior_distribution(std::make_unique<Normal>(2.0, 3.0));
    CHECK_NEAR(mp.prior_distribution().mean(), 2.0, 1e-12);
}

void test_model_parameter_clone_independence() {
    auto normal = std::make_unique<Normal>(0.0, 1.0);
    ModelParameter original("Owner", "Param", 1.5, -10.0, 10.0, std::move(normal));

    ModelParameter clone = original.clone();
    clone.prior_distribution().set_parameters({5.0, 2.0});

    auto original_params = original.prior_distribution().get_parameters();
    auto clone_params = clone.prior_distribution().get_parameters();

    CHECK_NEAR(original_params[0], 0.0, 1e-12);
    CHECK_NEAR(original_params[1], 1.0, 1e-12);
    CHECK_NEAR(clone_params[0], 5.0, 1e-12);
    CHECK_NEAR(clone_params[1], 2.0, 1e-12);

    // Mutating clone's other fields must not affect the original either.
    clone.set_value(99.0);
    CHECK_NEAR(original.value(), 1.5, 1e-12);
}

void test_model_parameter_copy_ctor_independence() {
    auto normal = std::make_unique<Normal>(0.0, 1.0);
    ModelParameter original("Owner", "Param", 1.5, -10.0, 10.0, std::move(normal));

    ModelParameter copy(original);
    copy.prior_distribution().set_parameters({7.0, 4.0});

    auto original_params = original.prior_distribution().get_parameters();
    CHECK_NEAR(original_params[0], 0.0, 1e-12);
    CHECK_NEAR(original_params[1], 1.0, 1e-12);
}

}  // namespace

int main() {
    test_data_component_exact_ctor();
    test_data_component_full_ctor();
    test_data_component_is_censored_is_threshold();
    test_data_component_to_string();
    test_prior_component();
    test_model_parameter_defaults();
    test_model_parameter_display_name();
    test_model_parameter_full_ctor_and_prior();
    test_model_parameter_getters_setters_round_trip();
    test_model_parameter_clone_independence();
    test_model_parameter_copy_ctor_independence();

    return chtest::summary("model_parameter");
}
