// Structural / deterministic-aggregation tests for corehydro::analyses::CompositeAnalysis and its
// child-wrapper DTO corehydro::analyses::WeightedUnivariateAnalysis (X5).
//
// The composite has NO MCMC of its own: given fixed child posteriors the model-weight estimation
// and the CompetingRisks/Mixture aggregation are fully deterministic. These tests transcribe the
// ported surface from CompositeAnalysis.cs / WeightedUnivariateAnalysis.cs (RMC-BestFit @ fc28c0c),
// hand-deriving small-grid oracles where the upstream test file does not cover a ported member.
//
// Child analyses are supplied by a lightweight test double (FakeChild) that implements
// IUnivariateAnalysis with fully controllable is_estimated / analysis_results (AIC/BIC/RMSE) and a
// real injected BayesianAnalysis (DIC/WAIC/LOOIC computed from a synthetic posterior). The child
// distributions are LnNormal (strictly positive support) so the composite's XTransform=Logarithmic
// empirical-CDF path is exercised on valid (positive) x-values, exactly as a flood-frequency
// composite would be.
#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "corehydro/analyses/support/analysis_base.hpp"
#include "corehydro/analyses/support/i_univariate_analysis.hpp"
#include "corehydro/analyses/support/weighted_univariate_analysis.hpp"
#include "corehydro/analyses/univariate/composite_analysis.hpp"
#include "corehydro/estimation/bayesian_analysis.hpp"
#include "corehydro/models/univariate_distribution/univariate_distribution_model.hpp"
#include "corehydro/numerics/data/goodness_of_fit.hpp"
#include "corehydro/numerics/data/probability.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/mixture.hpp"
#include "corehydro/numerics/math/optimization/support/parameter_set.hpp"
#include "corehydro/numerics/sampling/mcmc/support/mcmc_results.hpp"
#include "check.hpp"

using corehydro::analyses::AnalysisBase;
using corehydro::analyses::AverageMethod;
using corehydro::analyses::CompositeAnalysis;
using corehydro::analyses::CompositeType;
using corehydro::analyses::IUnivariateAnalysis;
using corehydro::analyses::WeightedUnivariateAnalysis;
using corehydro::estimation::BayesianAnalysis;
using corehydro::estimation::PointEstimateType;
using corehydro::models::UnivariateDistributionModel;
using corehydro::models::ValidationResult;
using corehydro::numerics::data::GoodnessOfFit;
using corehydro::numerics::distributions::Mixture;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::math::optimization::ParameterSet;
using corehydro::numerics::sampling::mcmc::MCMCResults;
namespace prob = corehydro::numerics::data::probability;

namespace {

// Positive flow record (LnNormal-friendly), same shape as the A5 structural data.
const std::vector<double>& flow_data() {
    static const std::vector<double> v = {12500, 15300, 8900,  22100, 18700,
                                          14200, 9800,  28500, 17400, 11600};
    return v;
}

// A synthetic LnNormal(mean, sd) posterior with a deterministic spread (no RNG). MAP = (mean, sd).
MCMCResults build_synthetic_results(double mean, double sd, int sample_size) {
    std::vector<ParameterSet> output;
    output.reserve(static_cast<std::size_t>(sample_size));
    for (int i = 0; i < sample_size; ++i) {
        double t = sample_size == 1 ? 0.5
                                    : static_cast<double>(i) / static_cast<double>(sample_size - 1);
        double m = mean + 0.05 * mean * (2.0 * t - 1.0);  // mean +/- 5%
        double s = sd * (0.9 + 0.2 * t);                  // [0.9, 1.1] * sd, strictly positive
        output.emplace_back(std::vector<double>{m, s}, 0.0);
    }
    ParameterSet map(std::vector<double>{mean, sd}, 0.0);
    return MCMCResults(map, std::move(output), 0.10);
}

// Controllable IUnivariateAnalysis test double. Owns an LnNormal model + BayesianAnalysis with an
// injected posterior (real DIC/WAIC/LOOIC), plus a settable analysis-level IsEstimated and a
// settable AnalysisResults (AIC/BIC/RMSE). GetDistribution / GetPointEstimateDistribution return
// LnNormal draws from the injected posterior.
class FakeChild : public AnalysisBase, public IUnivariateAnalysis {
   public:
    using UAR = corehydro::numerics::distributions::UncertaintyAnalysisResults;

    FakeChild(double mean, double sd, int output_length)
        : model_(std::make_unique<UnivariateDistributionModel>(UnivariateDistributionType::LnNormal,
                                                               flow_data())),
          bayesian_(*model_) {
        model_->set_parameter_values({mean, sd});
        bayesian_.set_output_length(output_length);
        bayesian_.set_custom_mcmc_results(build_synthetic_results(mean, sd, output_length),
                                          /*skip_information_criteria=*/false);
    }

    // IBayesianAnalysis
    BayesianAnalysis& bayesian_analysis() override { return bayesian_; }
    const BayesianAnalysis& bayesian_analysis() const { return bayesian_; }
    const UAR* analysis_results() const override { return results_ ? &*results_ : nullptr; }

    // IProbabilityOrdinates
    corehydro::numerics::data::ProbabilityOrdinates& probability_ordinates() override {
        return ordinates_;
    }

    // IUnivariateAnalysis
    UnivariateDistributionBase* get_distribution(int index) override {
        if (!bayesian_.is_estimated() || !bayesian_.results()) return nullptr;
        auto d = model_->distribution().clone();
        d->set_parameters(bayesian_.results()->output[static_cast<std::size_t>(index)].values);
        cache_.push_back(std::move(d));
        return cache_.back().get();
    }
    UnivariateDistributionBase* get_point_estimate_distribution() override {
        return get_point_estimate_distribution(bayesian_.point_estimator());
    }
    UnivariateDistributionBase* get_point_estimate_distribution(PointEstimateType est) override {
        if (!bayesian_.is_estimated() || !bayesian_.results()) return nullptr;
        const std::vector<double>& parms = est == PointEstimateType::PosteriorMean
                                               ? bayesian_.results()->posterior_mean.values
                                               : bayesian_.results()->map.values;
        auto d = model_->distribution().clone();
        d->set_parameters(parms);
        cache_.push_back(std::move(d));
        return cache_.back().get();
    }

    // IAnalysis
    void run() override {}
    ValidationResult validate() const override { return ValidationResult{}; }

    // Test controls.
    void set_estimated(bool v) { set_is_estimated(v); }
    void set_analysis_results(double aic, double bic, double rmse) {
        UAR r;
        r.aic = aic;
        r.bic = bic;
        r.rmse = rmse;
        results_ = std::move(r);
    }

   private:
    std::unique_ptr<UnivariateDistributionModel> model_;
    BayesianAnalysis bayesian_;
    corehydro::numerics::data::ProbabilityOrdinates ordinates_;
    std::optional<UAR> results_;
    std::vector<std::unique_ptr<UnivariateDistributionBase>> cache_;
};

std::unique_ptr<FakeChild> make_child(double mean, double sd, int outlen = 50) {
    auto c = std::make_unique<FakeChild>(mean, sd, outlen);
    c->set_estimated(true);
    c->set_analysis_results(/*aic=*/100.0, /*bic=*/110.0, /*rmse=*/5.0);
    return c;
}

// ---------------------------------------------------------------------------------------------
// WeightedUnivariateAnalysis
// ---------------------------------------------------------------------------------------------

void test_wua_ctor_stores_analysis_and_weight() {
    auto child = make_child(15000, 5000);
    WeightedUnivariateAnalysis w(child.get(), 0.42);
    CHECK_TRUE(w.univariate_analysis() == child.get());
    CHECK_NEAR(w.weight(), 0.42, 1e-15);
}

void test_wua_setter_rejects_composite() {
    CompositeAnalysis composite;
    WeightedUnivariateAnalysis w;
    // Setter rejects a CompositeAnalysis child (C# ArgumentException).
    CHECK_THROWS(w.set_univariate_analysis(&composite));
    // Same rejection via the two-arg ctor.
    CHECK_THROWS(WeightedUnivariateAnalysis(&composite, 0.5));
}

void test_wua_validate() {
    // Valid: an estimated child with a valid inner Validate().
    auto child = make_child(15000, 5000);
    WeightedUnivariateAnalysis w(child.get(), 0.5);
    auto [ok, msg] = w.validate();
    CHECK_TRUE(ok);
    CHECK_TRUE(msg.empty());

    // Invalid: no analysis set.
    WeightedUnivariateAnalysis empty;
    auto [ok2, msg2] = empty.validate();
    CHECK_TRUE(!ok2);
    CHECK_TRUE(!msg2.empty());

    // Invalid: analysis present but not estimated.
    auto unfit = make_child(15000, 5000);
    unfit->set_estimated(false);
    WeightedUnivariateAnalysis w3(unfit.get(), 0.5);
    auto [ok3, msg3] = w3.validate();
    CHECK_TRUE(!ok3);
}

// ---------------------------------------------------------------------------------------------
// CompositeAnalysis: enum defaults
// ---------------------------------------------------------------------------------------------

void test_composite_defaults() {
    CompositeAnalysis c;
    CHECK_TRUE(c.composite_distribution_type() == CompositeType::CompetingRisks);
    CHECK_TRUE(c.model_average_method() == AverageMethod::DIC);
    CHECK_TRUE(c.dependency() == prob::DependencyType::Independent);
    CHECK_TRUE(c.is_maximum());
    CHECK_TRUE(c.probability_ordinates().count() > 0);
    CHECK_TRUE(c.analysis_results() == nullptr);
    CHECK_TRUE(!c.is_estimated());
    // Composite point-estimate distribution query is null when there are no children.
    CHECK_TRUE(c.get_point_estimate_distribution() == nullptr);
    // GetDistribution always returns null (composite can't yield a single component).
    CHECK_TRUE(c.get_distribution(0) == nullptr);
}

// ---------------------------------------------------------------------------------------------
// CompositeAnalysis: EstimateModelWeights
// ---------------------------------------------------------------------------------------------

void test_estimate_weights_equal() {
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(16000, 4000);
    auto c2 = make_child(14000, 6000);
    CompositeAnalysis c;
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.0));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.0));
    c.analyses().push_back(WeightedUnivariateAnalysis(c2.get(), 0.0));
    c.set_model_average_method(AverageMethod::Equal);
    c.set_composite_distribution_type(CompositeType::ModelAverage);

    double sum = 0.0;
    for (auto& wua : c.analyses()) {
        CHECK_NEAR(wua.weight(), 1.0 / 3.0, 1e-12);
        sum += wua.weight();
    }
    CHECK_NEAR(sum, 1.0, 1e-12);
}

void test_estimate_weights_aic_bic_rmse() {
    // Two fit children with distinct AIC/BIC/RMSE; a third unfit child must get weight 0.
    auto c0 = make_child(15000, 5000);
    c0->set_analysis_results(/*aic=*/100.0, /*bic=*/120.0, /*rmse=*/5.0);
    auto c1 = make_child(16000, 4000);
    c1->set_analysis_results(/*aic=*/104.0, /*bic=*/126.0, /*rmse=*/8.0);
    auto c2 = make_child(14000, 6000);
    c2->set_estimated(false);  // unfit -> weight 0

    CompositeAnalysis c;
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.0));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.0));
    c.analyses().push_back(WeightedUnivariateAnalysis(c2.get(), 0.0));

    // AIC.
    c.set_model_average_method(AverageMethod::AIC);
    c.set_composite_distribution_type(CompositeType::ModelAverage);
    {
        std::vector<double> exp = GoodnessOfFit::aic_weights({100.0, 104.0});
        CHECK_NEAR(c.analyses()[0].weight(), exp[0], 1e-9);
        CHECK_NEAR(c.analyses()[1].weight(), exp[1], 1e-9);
        CHECK_NEAR(c.analyses()[2].weight(), 0.0, 1e-12);
        CHECK_NEAR(c.analyses()[0].weight() + c.analyses()[1].weight(), 1.0, 1e-9);
    }

    // BIC (also routed through AIC weights per C#).
    c.set_model_average_method(AverageMethod::BIC);
    {
        std::vector<double> exp = GoodnessOfFit::aic_weights({120.0, 126.0});
        CHECK_NEAR(c.analyses()[0].weight(), exp[0], 1e-9);
        CHECK_NEAR(c.analyses()[1].weight(), exp[1], 1e-9);
        CHECK_NEAR(c.analyses()[2].weight(), 0.0, 1e-12);
    }

    // RMSE (inverse-MSE weights).
    c.set_model_average_method(AverageMethod::RMSE);
    {
        std::vector<double> exp = GoodnessOfFit::rmse_weights({5.0, 8.0});
        CHECK_NEAR(c.analyses()[0].weight(), exp[0], 1e-9);
        CHECK_NEAR(c.analyses()[1].weight(), exp[1], 1e-9);
        CHECK_NEAR(c.analyses()[2].weight(), 0.0, 1e-12);
    }
}

void test_estimate_weights_dic() {
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(16000, 4000);
    CompositeAnalysis c;
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.0));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.0));
    c.set_model_average_method(AverageMethod::DIC);
    c.set_composite_distribution_type(CompositeType::ModelAverage);

    double d0 = c0->bayesian_analysis().dic();
    double d1 = c1->bayesian_analysis().dic();
    CHECK_TRUE(std::isfinite(d0) && std::isfinite(d1));
    std::vector<double> exp = GoodnessOfFit::aic_weights({d0, d1});
    CHECK_NEAR(c.analyses()[0].weight(), exp[0], 1e-9);
    CHECK_NEAR(c.analyses()[1].weight(), exp[1], 1e-9);
    CHECK_NEAR(c.analyses()[0].weight() + c.analyses()[1].weight(), 1.0, 1e-9);
}

// ---------------------------------------------------------------------------------------------
// CompositeAnalysis: aggregation
// ---------------------------------------------------------------------------------------------

void test_competing_risks_aggregation() {
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(20000, 6000);
    CompositeAnalysis c;
    c.set_composite_distribution_type(CompositeType::CompetingRisks);
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.5));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.5));

    c.create_frequency_analysis_results();
    const auto* r = c.analysis_results();
    CHECK_TRUE(r != nullptr);
    std::size_t n = c.probability_ordinates().count();
    CHECK_EQ(r->confidence_intervals.size(), n);
    CHECK_EQ(r->mode_curve.size(), n);
    CHECK_TRUE(!r->mean_curve.empty());
    bool all_finite = true;
    for (std::size_t i = 0; i < n; ++i) {
        if (!std::isfinite(r->mode_curve[i]) || r->mode_curve[i] <= 0.0) all_finite = false;
        if (r->confidence_intervals[i][0] > r->confidence_intervals[i][1] + 1e-6) all_finite = false;
    }
    CHECK_TRUE(all_finite);

    // Determinism: a second aggregation reproduces the mode curve bit-for-bit (no RNG).
    double first0 = r->mode_curve[0];
    c.create_frequency_analysis_results();
    CHECK_TRUE(c.analysis_results()->mode_curve[0] == first0);
}

void test_mixture_aggregation() {
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(20000, 6000);
    CompositeAnalysis c;
    c.set_composite_distribution_type(CompositeType::Mixture);
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.5));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.5));

    c.create_frequency_analysis_results();
    const auto* r = c.analysis_results();
    CHECK_TRUE(r != nullptr);
    std::size_t n = c.probability_ordinates().count();
    CHECK_EQ(r->confidence_intervals.size(), n);
    CHECK_EQ(r->mode_curve.size(), n);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(r->mode_curve[i]) || r->mode_curve[i] <= 0.0) ok = false;
    CHECK_TRUE(ok);
}

void test_mixture_zero_inflation() {
    // Child weights summing to < 1 -> the composite point-estimate Mixture is zero-inflated.
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(20000, 6000);
    CompositeAnalysis c;
    c.set_composite_distribution_type(CompositeType::Mixture);
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.3));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.3));

    UnivariateDistributionBase* pe = c.get_point_estimate_distribution();
    CHECK_TRUE(pe != nullptr);
    auto* mix = dynamic_cast<Mixture*>(pe);
    CHECK_TRUE(mix != nullptr);
    CHECK_TRUE(mix->is_zero_inflated());
    CHECK_NEAR(mix->zero_weight(), 0.4, 1e-12);
}

// ---------------------------------------------------------------------------------------------
// CompositeAnalysis: run() guard + Validate
// ---------------------------------------------------------------------------------------------

void test_run_guard_throws_when_child_unfit() {
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(20000, 6000);
    c1->set_estimated(false);  // one child not estimated
    CompositeAnalysis c;
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.5));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.5));
    CHECK_THROWS(c.run());
}

void test_validate_empty_children() {
    CompositeAnalysis c;
    ValidationResult v = c.validate();
    CHECK_TRUE(!v.is_valid);
    CHECK_TRUE(!v.validation_messages.empty());
}

void test_run_populates_results() {
    auto c0 = make_child(15000, 5000);
    auto c1 = make_child(20000, 6000);
    CompositeAnalysis c;
    c.set_composite_distribution_type(CompositeType::CompetingRisks);
    // Mixture-weight validation only applies for the Mixture type; CompetingRisks accepts any.
    c.analyses().push_back(WeightedUnivariateAnalysis(c0.get(), 0.5));
    c.analyses().push_back(WeightedUnivariateAnalysis(c1.get(), 0.5));

    c.run();
    CHECK_TRUE(c.is_estimated());
    CHECK_TRUE(c.analysis_results() != nullptr);
}

}  // namespace

int main() {
    test_wua_ctor_stores_analysis_and_weight();
    test_wua_setter_rejects_composite();
    test_wua_validate();

    test_composite_defaults();

    test_estimate_weights_equal();
    test_estimate_weights_aic_bic_rmse();
    test_estimate_weights_dic();

    test_competing_risks_aggregation();
    test_mixture_aggregation();
    test_mixture_zero_inflation();

    test_run_guard_throws_when_child_unfit();
    test_validate_empty_children();
    test_run_populates_results();

    return chtest::summary("composite_analysis");
}
