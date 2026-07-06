// Standalone contract test for bestfit::analyses::AnalysisBase and the four analysis interface
// mixins (IAnalysis, IBayesianAnalysis, IUnivariateAnalysis, IProbabilityOrdinates).
//
// There is NO upstream test file for these types (they are exercised indirectly by the analysis
// tests that arrive in A5-A9) and NO public-API fixture (interface support gets a C++-only ctest
// per the Phase-8 oracle policy). So this proves the *contract* the vtable + IsEstimated state
// declare -- not transcribed numeric oracles -- via a minimal derived stub:
//   * is_estimated() starts false; a derived run() reaches the protected setter and flips it true.
//   * validate() round-trips a fixed ValidationResult (is_valid + messages).
//   * every interface base pointer (IAnalysis*, IBayesianAnalysis*, IUnivariateAnalysis*,
//     IProbabilityOrdinates*) dispatches virtual calls to the derived overrides, including the
//     nullable-sentinel accessors.
#include <string>
#include <vector>

#include "bestfit/analyses/support/analysis_base.hpp"
#include "bestfit/analyses/support/i_analysis.hpp"
#include "bestfit/analyses/support/i_bayesian_analysis.hpp"
#include "bestfit/analyses/support/i_probability_ordinates.hpp"
#include "bestfit/analyses/support/i_univariate_analysis.hpp"
#include "bestfit/estimation/bayesian_analysis.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "check.hpp"

using bestfit::analyses::AnalysisBase;
using bestfit::analyses::IAnalysis;
using bestfit::analyses::IBayesianAnalysis;
using bestfit::analyses::IProbabilityOrdinates;
using bestfit::analyses::IUnivariateAnalysis;
using bestfit::estimation::BayesianAnalysis;
using bestfit::estimation::PointEstimateType;
using bestfit::estimation::SamplerType;
using bestfit::models::UnivariateDistributionModel;
using bestfit::models::ValidationResult;
using bestfit::numerics::data::ProbabilityOrdinates;
using bestfit::numerics::distributions::Normal;
using bestfit::numerics::distributions::UncertaintyAnalysisResults;
using bestfit::numerics::distributions::UnivariateDistributionBase;
using bestfit::numerics::distributions::UnivariateDistributionType;

namespace {

// A minimal concrete analysis deriving through the full interface chain. It implements the two
// pure-virtual compute members (run/validate) plus every interface accessor with simple
// sentinels, so the test can assert the contract and the vtable dispatch of each base pointer.
struct StubAnalysis : public AnalysisBase, public IUnivariateAnalysis {
    // Members that back the accessor sentinels. model_ is declared before bayes_ so it is
    // constructed first (BayesianAnalysis stores a reference to the model).
    Normal dist_{0.0, 1.0};
    ProbabilityOrdinates ordinates_{};
    UncertaintyAnalysisResults results_{};
    UnivariateDistributionModel model_;
    BayesianAnalysis bayes_;
    bool run_called_ = false;

    StubAnalysis()
        : model_(UnivariateDistributionType::Normal, {1.0, 2.0, 3.0, 4.0, 5.0}),
          bayes_(model_, SamplerType::DEMCzs) {}

    // Compute contract: run() reaches the protected setter (proving it is callable from a derived
    // run) and records that it fired; validate() returns a fixed result.
    void run() override {
        run_called_ = true;
        set_is_estimated(true);
    }

    ValidationResult validate() const override {
        ValidationResult v;
        v.is_valid = false;
        v.validation_messages = {"stub message"};
        return v;
    }

    // IBayesianAnalysis
    BayesianAnalysis& bayesian_analysis() override { return bayes_; }
    const UncertaintyAnalysisResults* analysis_results() const override {
        return is_estimated() ? &results_ : nullptr;
    }

    // IProbabilityOrdinates
    ProbabilityOrdinates& probability_ordinates() override { return ordinates_; }

    // IUnivariateAnalysis: nullable sentinels distinguishing branches.
    UnivariateDistributionBase* get_distribution(int index) override {
        return index == 1 ? &dist_ : nullptr;
    }
    UnivariateDistributionBase* get_point_estimate_distribution() override { return &dist_; }
    UnivariateDistributionBase* get_point_estimate_distribution(
        PointEstimateType point_estimator) override {
        return point_estimator == PointEstimateType::PosteriorMean ? &dist_ : nullptr;
    }
};

// ---- is_estimated() state + protected setter reachable from a derived run() ----
void test_is_estimated_starts_false_and_run_flips_it() {
    StubAnalysis stub;
    CHECK_TRUE(!stub.is_estimated());
    CHECK_TRUE(!stub.run_called_);
    stub.run();
    CHECK_TRUE(stub.run_called_);
    CHECK_TRUE(stub.is_estimated());
}

// ---- validate() round-trips the fixed ValidationResult ----
void test_validate_round_trips() {
    StubAnalysis stub;
    ValidationResult v = stub.validate();
    CHECK_TRUE(!v.is_valid);
    CHECK_EQ(v.validation_messages.size(), static_cast<std::size_t>(1));
    CHECK_TRUE(v.validation_messages[0] == "stub message");
}

// ---- IAnalysis* dispatch: run()/validate()/is_estimated() reach the overrides ----
void test_ianalysis_pointer_dispatch() {
    StubAnalysis stub;
    IAnalysis* p = &stub;
    CHECK_TRUE(!p->is_estimated());
    p->run();
    CHECK_TRUE(p->is_estimated());
    CHECK_TRUE(!p->validate().is_valid);
}

// ---- IBayesianAnalysis* dispatch: bayesian_analysis()/analysis_results() ----
void test_ibayesian_pointer_dispatch() {
    StubAnalysis stub;
    IBayesianAnalysis* p = &stub;
    // Same object reached through the base pointer.
    CHECK_TRUE(&p->bayesian_analysis() == &stub.bayes_);
    // Nullable results: null before estimation, valid pointer after.
    CHECK_TRUE(p->analysis_results() == nullptr);
    p->run();
    CHECK_TRUE(p->analysis_results() == &stub.results_);
}

// ---- IProbabilityOrdinates* dispatch: probability_ordinates() ----
void test_iprobability_ordinates_pointer_dispatch() {
    StubAnalysis stub;
    IProbabilityOrdinates* p = &stub;
    CHECK_TRUE(&p->probability_ordinates() == &stub.ordinates_);
}

// ---- IUnivariateAnalysis* dispatch: full accessor set through the diamond ----
void test_iunivariate_pointer_dispatch() {
    StubAnalysis stub;
    IUnivariateAnalysis* p = &stub;
    // Inherited from IAnalysis / IBayesianAnalysis / IProbabilityOrdinates through the diamond.
    CHECK_TRUE(!p->is_estimated());
    CHECK_TRUE(&p->bayesian_analysis() == &stub.bayes_);
    CHECK_TRUE(&p->probability_ordinates() == &stub.ordinates_);
    // Own accessors: nullable sentinels.
    CHECK_TRUE(p->get_distribution(0) == nullptr);
    CHECK_TRUE(p->get_distribution(1) == &stub.dist_);
    CHECK_TRUE(p->get_point_estimate_distribution() == &stub.dist_);
    CHECK_TRUE(p->get_point_estimate_distribution(PointEstimateType::PosteriorMean) == &stub.dist_);
    CHECK_TRUE(p->get_point_estimate_distribution(PointEstimateType::PosteriorMode) == nullptr);
}

// ---- The single shared virtual IAnalysis base: no ambiguity upcasting from the diamond ----
void test_shared_virtual_ianalysis_base_is_unambiguous() {
    StubAnalysis stub;
    // Reachable through both AnalysisBase and IUnivariateAnalysis -> IBayesianAnalysis -> IAnalysis;
    // a single virtual base subobject makes this upcast unambiguous.
    IAnalysis* via_base = static_cast<AnalysisBase*>(&stub);
    IAnalysis* via_iface = static_cast<IUnivariateAnalysis*>(&stub);
    CHECK_TRUE(via_base == via_iface);
}

}  // namespace

int main() {
    test_is_estimated_starts_false_and_run_flips_it();
    test_validate_round_trips();
    test_ianalysis_pointer_dispatch();
    test_ibayesian_pointer_dispatch();
    test_iprobability_ordinates_pointer_dispatch();
    test_iunivariate_pointer_dispatch();
    test_shared_virtual_ianalysis_base_is_unambiguous();

    return bftest::summary("analysis_base");
}
