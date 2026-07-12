// Standalone tests for the Models/Support contracts (ISimulatable, IUnivariateModel,
// IQuantilePriors) and the QuantilePrior class.
//
// Oracle for QuantilePrior behavior is the upstream C# test class
// upstream/RMC-BestFit/src/RMC.BestFit.Tests/ModelEstimation/QuantilePriorTests.cs @ fc28c0c,
// transcribed method-for-method below (same section order). Deliberately NOT transcribed
// (project-wide deferrals, per the ported headers):
//   - Test_Constructor_XElement_RestoresAllProperties + the "Serialization Tests" region
//     (ToXElement / XElement round-trips -- XML serialization is not ported)
//   - the "PropertyChanged Tests" region (INotifyPropertyChanged is not ported)
// These are structural/behavioral ports validated against the C# source itself, so there is
// no fixtures/ entry for this file (fixtures/ is for the public estimation API only).
//
// The three interfaces have no upstream tests; they are exercised here through minimal stub
// implementations that verify the contracts compile and dispatch polymorphically.
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "corehydro/models/support/i_quantile_priors.hpp"
#include "corehydro/models/support/i_univariate_model.hpp"
#include "corehydro/models/support/quantile_prior.hpp"
#include "corehydro/models/support/simulatable.hpp"
#include "corehydro/numerics/distributions/base/univariate_distribution_type.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/distributions/uniform.hpp"
#include "check.hpp"

using corehydro::models::IQuantilePriors;
using corehydro::models::ISimulatable;
using corehydro::models::IUnivariateModel;
using corehydro::models::QuantilePrior;
using corehydro::models::ValidationResult;
using corehydro::numerics::distributions::LogNormal;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::Uniform;
using corehydro::numerics::distributions::UnivariateDistributionBase;
using corehydro::numerics::distributions::UnivariateDistributionType;

namespace {

// Returns true iff any message mentions "alpha" or "exceedance" (mirrors the C# LINQ
// `messages.Any(m => m.Contains("alpha") || m.Contains("exceedance"))`).
bool mentions_alpha(const std::vector<std::string>& messages) {
    for (const auto& m : messages) {
        if (m.find("alpha") != std::string::npos || m.find("exceedance") != std::string::npos)
            return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// QuantilePrior: Constructor Tests
// ---------------------------------------------------------------------------

// C# Test_Constructor_EmptyConstructor_CreatesDefaultInstance
void test_constructor_empty_creates_default_instance() {
    QuantilePrior prior;
    // C# asserts Distribution is not null; the C++ default ctor owns a
    // Uniform(double.MinValue, double.MaxValue) prior (QuantilePrior.cs line 28).
    CHECK_TRUE(prior.distribution().type() == UnivariateDistributionType::Uniform);
    CHECK_NEAR(prior.alpha(), 0.0, 1e-15);
}

// C# Test_Constructor_WithParameters_SetsValues
void test_constructor_with_parameters_sets_values() {
    double alpha = 0.01;  // 1% exceedance (100-year flood)
    QuantilePrior prior(alpha, std::make_unique<Normal>(75000.0, 10000.0));

    CHECK_NEAR(prior.alpha(), 0.01, 1e-10);
    CHECK_TRUE(prior.distribution().type() == UnivariateDistributionType::Normal);
}

// ---------------------------------------------------------------------------
// QuantilePrior: Property Tests
// ---------------------------------------------------------------------------

// C# Test_Alpha_SetAndGet
void test_alpha_set_and_get() {
    QuantilePrior prior;
    prior.set_alpha(0.02);
    CHECK_NEAR(prior.alpha(), 0.02, 1e-10);
}

// C# Test_Distribution_SetAndGet (Assert.AreSame -> pointer identity of the owned object)
void test_distribution_set_and_get() {
    QuantilePrior prior;
    auto dist = std::make_unique<LogNormal>(10.0, 0.5);
    const UnivariateDistributionBase* raw = dist.get();
    prior.set_distribution(std::move(dist));
    CHECK_TRUE(&prior.distribution() == raw);
}

// C# Test_UpperValue_Returns95thPercentile
void test_upper_value_returns_95th_percentile() {
    Normal dist(100.0, 10.0);
    QuantilePrior prior(0.01, std::make_unique<Normal>(100.0, 10.0));

    double expected = dist.inverse_cdf(0.95);
    CHECK_NEAR(prior.upper_value(), expected, 1e-10);
    // Extra sanity (not in C#): the analytic N(100,10) 95th percentile.
    CHECK_NEAR(prior.upper_value(), 116.44853626951472, 1e-6);
}

// C# Test_LowerValue_Returns5thPercentile
void test_lower_value_returns_5th_percentile() {
    Normal dist(100.0, 10.0);
    QuantilePrior prior(0.01, std::make_unique<Normal>(100.0, 10.0));

    double expected = dist.inverse_cdf(0.05);
    CHECK_NEAR(prior.lower_value(), expected, 1e-10);
    // Extra sanity (not in C#): the analytic N(100,10) 5th percentile.
    CHECK_NEAR(prior.lower_value(), 83.55146373048528, 1e-6);
}

// C# Test_MeanValue_ReturnsMean
void test_mean_value_returns_mean() {
    QuantilePrior prior(0.01, std::make_unique<Normal>(100.0, 10.0));
    CHECK_NEAR(prior.mean_value(), 100.0, 1e-10);
}

// ---------------------------------------------------------------------------
// QuantilePrior: Validation Tests
// ---------------------------------------------------------------------------

// C# Test_Validate_ValidPrior_ReturnsTrue
void test_validate_valid_prior_returns_true() {
    QuantilePrior prior(0.01, std::make_unique<Normal>(75000.0, 10000.0));
    ValidationResult result = prior.validate();
    CHECK_TRUE(result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{0});
}

// C# Test_Validate_AlphaZero_ReturnsFalse
void test_validate_alpha_zero_returns_false() {
    QuantilePrior prior(0.0, std::make_unique<Normal>(75000.0, 10000.0));
    ValidationResult result = prior.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_TRUE(mentions_alpha(result.validation_messages));
}

// C# Test_Validate_AlphaOne_ReturnsFalse
void test_validate_alpha_one_returns_false() {
    QuantilePrior prior(1.0, std::make_unique<Normal>(75000.0, 10000.0));
    CHECK_TRUE(!prior.validate().is_valid);
}

// C# Test_Validate_AlphaNegative_ReturnsFalse
void test_validate_alpha_negative_returns_false() {
    QuantilePrior prior(-0.1, std::make_unique<Normal>(75000.0, 10000.0));
    CHECK_TRUE(!prior.validate().is_valid);
}

// C# Test_Validate_AlphaGreaterThanOne_ReturnsFalse
void test_validate_alpha_greater_than_one_returns_false() {
    QuantilePrior prior(1.5, std::make_unique<Normal>(75000.0, 10000.0));
    CHECK_TRUE(!prior.validate().is_valid);
}

// C# Test_Validate_AlphaNaN_ReturnsFalse
void test_validate_alpha_nan_returns_false() {
    QuantilePrior prior(std::numeric_limits<double>::quiet_NaN(),
                        std::make_unique<Normal>(75000.0, 10000.0));
    CHECK_TRUE(!prior.validate().is_valid);
}

// C# Test_Validate_AlphaInfinity_ReturnsFalse
void test_validate_alpha_infinity_returns_false() {
    QuantilePrior prior(std::numeric_limits<double>::infinity(),
                        std::make_unique<Normal>(75000.0, 10000.0));
    CHECK_TRUE(!prior.validate().is_valid);
}

// Not in C# (a C# NRE-by-construction case): an invalid prior distribution (Normal with
// sigma <= 0) must fail the ValidateParameters branch of Validate() (QuantilePrior.cs 149).
void test_validate_invalid_distribution_returns_false() {
    QuantilePrior prior(0.01, std::make_unique<Normal>(0.0, -1.0));
    ValidationResult result = prior.validate();
    CHECK_TRUE(!result.is_valid);
    CHECK_EQ(result.validation_messages.size(), std::size_t{1});
}

// ---------------------------------------------------------------------------
// QuantilePrior: Clone Tests
// ---------------------------------------------------------------------------

// C# Test_Clone_CreatesIndependentCopy
void test_clone_creates_independent_copy() {
    QuantilePrior original(0.01, std::make_unique<Normal>(75000.0, 10000.0));

    QuantilePrior clone = original.clone();

    // Verify values match
    CHECK_NEAR(clone.alpha(), original.alpha(), 0.0);
    CHECK_NEAR(clone.mean_value(), original.mean_value(), 1e-10);

    // Verify independence
    original.set_alpha(0.99);
    CHECK_NEAR(clone.alpha(), 0.01, 0.0);
}

// C# Test_Clone_PreservesDistributionType
void test_clone_preserves_distribution_type() {
    QuantilePrior original(0.01, std::make_unique<LogNormal>(10.0, 0.5));
    QuantilePrior clone = original.clone();
    CHECK_TRUE(clone.distribution().type() == UnivariateDistributionType::LogNormal);
}

// C# Test_Clone_PreservesDistributionParameters
void test_clone_preserves_distribution_parameters() {
    QuantilePrior original(0.01, std::make_unique<Normal>(100.0, 15.0));
    QuantilePrior clone = original.clone();

    CHECK_NEAR(clone.lower_value(), original.lower_value(), 1e-10);
    CHECK_NEAR(clone.upper_value(), original.upper_value(), 1e-10);
    CHECK_NEAR(clone.mean_value(), original.mean_value(), 1e-10);
}

// Not in C# (deep-copy semantics are implicit in C# Clone via Distribution.Clone()): the
// clone's distribution must be an independent object -- mutating it must not affect the
// original (mirrors the ModelParameter ownership tests).
void test_clone_distribution_is_deep_copy() {
    QuantilePrior original(0.01, std::make_unique<Normal>(100.0, 15.0));
    QuantilePrior clone = original.clone();

    CHECK_TRUE(&clone.distribution() != &original.distribution());
    clone.distribution().set_parameters({5.0, 2.0});

    auto original_params = original.distribution().get_parameters();
    CHECK_NEAR(original_params[0], 100.0, 1e-12);
    CHECK_NEAR(original_params[1], 15.0, 1e-12);

    // Copy construction/assignment deep-copy the same way.
    QuantilePrior copy(original);
    copy.distribution().set_parameters({7.0, 4.0});
    CHECK_NEAR(original.distribution().get_parameters()[0], 100.0, 1e-12);

    QuantilePrior assigned;
    assigned = original;
    assigned.distribution().set_parameters({9.0, 3.0});
    CHECK_NEAR(original.distribution().get_parameters()[0], 100.0, 1e-12);
    CHECK_NEAR(assigned.alpha(), 0.01, 0.0);
}

// ---------------------------------------------------------------------------
// QuantilePrior: Engineering Application Tests
// ---------------------------------------------------------------------------

// C# Test_QuantilePrior_100YearFlood
void test_quantile_prior_100_year_flood() {
    // Expert judgment: 100-year flood is most likely ~75,000 cfs
    // with 90% confidence interval [50,000, 100,000] cfs
    QuantilePrior prior(0.01, std::make_unique<Normal>(75000.0, 15000.0));

    double lower = prior.lower_value();  // 5th percentile
    double upper = prior.upper_value();  // 95th percentile

    CHECK_TRUE(lower > 40000.0 && lower < 60000.0);
    CHECK_TRUE(upper > 90000.0 && upper < 120000.0);
}

// C# Test_QuantilePrior_PMF
void test_quantile_prior_pmf() {
    // Probable Maximum Flood (PMF) - very rare event: 0.01% AEP = 10,000-year event
    QuantilePrior prior(0.0001, std::make_unique<LogNormal>(12.5, 0.4));

    CHECK_TRUE(prior.validate().is_valid);
    CHECK_NEAR(prior.alpha(), 0.0001, 0.0);
}

// C# Test_QuantilePrior_HistoricalFlood
void test_quantile_prior_historical_flood() {
    // Historical flood from 1889: estimated ~350,000 cfs
    QuantilePrior prior(0.001, std::make_unique<Normal>(350000.0, 50000.0));

    CHECK_TRUE(prior.mean_value() > 300000.0);
    CHECK_TRUE(prior.lower_value() > 200000.0);
    CHECK_TRUE(prior.upper_value() < 500000.0);
}

// C# Test_QuantilePrior_RegionalInformation
void test_quantile_prior_regional_information() {
    // Regional regression: 10-year flood ~ 25,000 cfs
    QuantilePrior prior(0.1, std::make_unique<Normal>(25000.0, 5000.0));

    CHECK_TRUE(prior.validate().is_valid);
    CHECK_NEAR(prior.alpha(), 0.1, 0.0);
}

// ---------------------------------------------------------------------------
// QuantilePrior: Edge Cases
// ---------------------------------------------------------------------------

// C# Test_QuantilePrior_VerySmallAlpha
void test_quantile_prior_very_small_alpha() {
    QuantilePrior prior(1e-6, std::make_unique<Normal>(1000000.0, 100000.0));
    CHECK_TRUE(prior.validate().is_valid);
}

// C# Test_QuantilePrior_AlphaNearOne
void test_quantile_prior_alpha_near_one() {
    QuantilePrior prior(0.99, std::make_unique<Normal>(1000.0, 100.0));
    CHECK_TRUE(prior.validate().is_valid);
}

// C# Test_QuantilePrior_WidePriorDistribution
void test_quantile_prior_wide_prior_distribution() {
    QuantilePrior prior(0.01, std::make_unique<Uniform>(10000.0, 1000000.0));
    CHECK_TRUE(prior.validate().is_valid);
    CHECK_TRUE(prior.upper_value() - prior.lower_value() > 800000.0);
}

// C# Test_QuantilePrior_NarrowPriorDistribution
void test_quantile_prior_narrow_prior_distribution() {
    QuantilePrior prior(0.01, std::make_unique<Normal>(75000.0, 500.0));
    double ci_width = prior.upper_value() - prior.lower_value();
    CHECK_TRUE(ci_width < 2000.0);
}

// ---------------------------------------------------------------------------
// QuantilePrior: Multiple Quantile Priors Tests
// ---------------------------------------------------------------------------

// C# Test_MultipleQuantilePriors_DifferentReturnPeriods
void test_multiple_quantile_priors_different_return_periods() {
    QuantilePrior prior_10yr(0.1, std::make_unique<Normal>(30000.0, 5000.0));
    QuantilePrior prior_100yr(0.01, std::make_unique<Normal>(75000.0, 15000.0));
    QuantilePrior prior_500yr(0.002, std::make_unique<Normal>(120000.0, 30000.0));

    CHECK_TRUE(prior_10yr.validate().is_valid);
    CHECK_TRUE(prior_100yr.validate().is_valid);
    CHECK_TRUE(prior_500yr.validate().is_valid);

    // Mean values should increase with decreasing AEP
    CHECK_TRUE(prior_10yr.mean_value() < prior_100yr.mean_value());
    CHECK_TRUE(prior_100yr.mean_value() < prior_500yr.mean_value());
}

// ---------------------------------------------------------------------------
// ISimulatable<TData>: contract via a minimal stub
// ---------------------------------------------------------------------------

// Deterministic stub: element i is seed + i when seed > 0, else i (no RNG needed to
// exercise the contract).
class StubSimulatable final : public ISimulatable<std::vector<double>> {
   public:
    std::vector<double> generate_random_values(int sample_size, int seed = -1) const override {
        if (sample_size < 1)
            throw std::invalid_argument("sampleSize must be at least 1.");
        std::vector<double> out(static_cast<std::size_t>(sample_size));
        for (int i = 0; i < sample_size; ++i)
            out[static_cast<std::size_t>(i)] = (seed > 0 ? seed : 0) + i;
        return out;
    }
};

void test_isimulatable_stub_contract() {
    StubSimulatable stub;

    // Polymorphic dispatch through the interface reference, including the default seed.
    const ISimulatable<std::vector<double>>& sim = stub;
    std::vector<double> defaulted = sim.generate_random_values(4);
    CHECK_EQ(defaulted.size(), std::size_t{4});
    CHECK_NEAR(defaulted[0], 0.0, 0.0);
    CHECK_NEAR(defaulted[3], 3.0, 0.0);

    std::vector<double> seeded = sim.generate_random_values(3, 7);
    CHECK_EQ(seeded.size(), std::size_t{3});
    CHECK_NEAR(seeded[0], 7.0, 0.0);
    CHECK_NEAR(seeded[2], 9.0, 0.0);

    // Same inputs -> same outputs (the contract's reproducibility promise for seed > 0).
    std::vector<double> seeded_again = sim.generate_random_values(3, 7);
    CHECK_TRUE(seeded == seeded_again);

    CHECK_THROWS(sim.generate_random_values(0));
}

// The Numerics distribution base already exposes the same generate_random_values shape;
// a seeded draw from it is deterministic (sanity that the contract matches that precedent).
void test_isimulatable_matches_distribution_precedent() {
    Normal normal(0.0, 1.0);
    std::vector<double> a = normal.generate_random_values(5, 12345);
    std::vector<double> b = normal.generate_random_values(5, 12345);
    CHECK_EQ(a.size(), std::size_t{5});
    CHECK_TRUE(a == b);
}

// ---------------------------------------------------------------------------
// IUnivariateModel: contract via a minimal stub
// ---------------------------------------------------------------------------

class StubUnivariateModel final : public IUnivariateModel {
   public:
    // DataFrame does not exist until M4; the accessors are declared against the forward
    // declaration, so a stub that never returns is the only implementation possible here.
    corehydro::models::DataFrame& data_frame() override {
        throw std::logic_error("DataFrame arrives in M4");
    }
    const corehydro::models::DataFrame& data_frame() const override {
        throw std::logic_error("DataFrame arrives in M4");
    }
    const UnivariateDistributionBase* distribution() const override { return dist_.get(); }
    bool is_nonstationary() const override { return false; }
    ValidationResult validate() const override {
        if (dist_) return ValidationResult{};
        return ValidationResult{false, {"Error: The model has not been estimated."}};
    }

    void set_fitted(std::unique_ptr<UnivariateDistributionBase> dist) { dist_ = std::move(dist); }

   private:
    std::unique_ptr<UnivariateDistributionBase> dist_;
};

void test_iunivariate_model_stub_contract() {
    StubUnivariateModel stub;
    IUnivariateModel& model = stub;

    // Not yet estimated: Distribution is null (the C# `UnivariateDistributionBase?`).
    CHECK_TRUE(model.distribution() == nullptr);
    CHECK_TRUE(!model.is_nonstationary());

    ValidationResult before = model.validate();
    CHECK_TRUE(!before.is_valid);
    CHECK_EQ(before.validation_messages.size(), std::size_t{1});

    // DataFrame accessor is callable through the interface (M4 delivers the real type).
    CHECK_THROWS(model.data_frame());

    // "Estimated": Distribution becomes available through the interface.
    stub.set_fitted(std::make_unique<Normal>(10.0, 2.0));
    CHECK_TRUE(model.distribution() != nullptr);
    CHECK_TRUE(model.distribution()->type() == UnivariateDistributionType::Normal);
    CHECK_NEAR(model.distribution()->mean(), 10.0, 1e-12);

    ValidationResult after = model.validate();
    CHECK_TRUE(after.is_valid);
    CHECK_EQ(after.validation_messages.size(), std::size_t{0});
}

// ---------------------------------------------------------------------------
// IQuantilePriors: contract via a minimal stub
// ---------------------------------------------------------------------------

class StubQuantilePriors final : public IQuantilePriors {
   public:
    std::vector<QuantilePrior>& quantile_priors() override { return priors_; }
    const std::vector<QuantilePrior>& quantile_priors() const override { return priors_; }
    void set_quantile_priors(std::vector<QuantilePrior> priors) override {
        priors_ = std::move(priors);
    }

    bool enable_quantile_priors() const override { return enable_quantile_priors_; }
    void set_enable_quantile_priors(bool enable) override { enable_quantile_priors_ = enable; }

    bool use_single_quantile() const override { return use_single_quantile_; }
    void set_use_single_quantile(bool use_single) override { use_single_quantile_ = use_single; }

    void process_quantile_priors() override { ++process_calls_; }

    void set_default_quantile_priors() override {
        priors_.clear();
        priors_.emplace_back(0.01, std::make_unique<Normal>(75000.0, 15000.0));
    }

    int process_calls() const { return process_calls_; }

   private:
    std::vector<QuantilePrior> priors_;
    bool enable_quantile_priors_ = false;
    bool use_single_quantile_ = false;
    int process_calls_ = 0;
};

void test_iquantile_priors_stub_contract() {
    StubQuantilePriors stub;
    IQuantilePriors& qp = stub;

    // Flags round-trip through the interface.
    CHECK_TRUE(!qp.enable_quantile_priors());
    qp.set_enable_quantile_priors(true);
    CHECK_TRUE(qp.enable_quantile_priors());

    CHECK_TRUE(!qp.use_single_quantile());
    qp.set_use_single_quantile(true);
    CHECK_TRUE(qp.use_single_quantile());

    // The QuantilePriors list holds QuantilePrior by value (deep-copying type).
    CHECK_EQ(qp.quantile_priors().size(), std::size_t{0});
    qp.quantile_priors().emplace_back(0.1, std::make_unique<Normal>(30000.0, 5000.0));
    CHECK_EQ(qp.quantile_priors().size(), std::size_t{1});
    CHECK_NEAR(qp.quantile_priors()[0].alpha(), 0.1, 0.0);

    // Setter replaces the whole list (the C# `{ get; set; }` property).
    std::vector<QuantilePrior> replacement;
    replacement.emplace_back(0.002, std::make_unique<Normal>(120000.0, 30000.0));
    replacement.emplace_back(0.5, std::make_unique<Uniform>(0.0, 1000.0));
    qp.set_quantile_priors(std::move(replacement));
    CHECK_EQ(qp.quantile_priors().size(), std::size_t{2});
    CHECK_NEAR(qp.quantile_priors()[0].alpha(), 0.002, 0.0);

    // SetDefaultQuantilePriors + ProcessQuantilePriors dispatch through the interface.
    qp.set_default_quantile_priors();
    CHECK_EQ(qp.quantile_priors().size(), std::size_t{1});
    CHECK_NEAR(qp.quantile_priors()[0].alpha(), 0.01, 0.0);

    qp.process_quantile_priors();
    qp.process_quantile_priors();
    CHECK_EQ(stub.process_calls(), 2);
}

}  // namespace

int main() {
    // QuantilePrior (transcribed from QuantilePriorTests.cs)
    test_constructor_empty_creates_default_instance();
    test_constructor_with_parameters_sets_values();
    test_alpha_set_and_get();
    test_distribution_set_and_get();
    test_upper_value_returns_95th_percentile();
    test_lower_value_returns_5th_percentile();
    test_mean_value_returns_mean();
    test_validate_valid_prior_returns_true();
    test_validate_alpha_zero_returns_false();
    test_validate_alpha_one_returns_false();
    test_validate_alpha_negative_returns_false();
    test_validate_alpha_greater_than_one_returns_false();
    test_validate_alpha_nan_returns_false();
    test_validate_alpha_infinity_returns_false();
    test_validate_invalid_distribution_returns_false();
    test_clone_creates_independent_copy();
    test_clone_preserves_distribution_type();
    test_clone_preserves_distribution_parameters();
    test_clone_distribution_is_deep_copy();
    test_quantile_prior_100_year_flood();
    test_quantile_prior_pmf();
    test_quantile_prior_historical_flood();
    test_quantile_prior_regional_information();
    test_quantile_prior_very_small_alpha();
    test_quantile_prior_alpha_near_one();
    test_quantile_prior_wide_prior_distribution();
    test_quantile_prior_narrow_prior_distribution();
    test_multiple_quantile_priors_different_return_periods();

    // Interface contracts (stub implementations)
    test_isimulatable_stub_contract();
    test_isimulatable_matches_distribution_precedent();
    test_iunivariate_model_stub_contract();
    test_iquantile_priors_stub_contract();

    return chtest::summary("quantile_prior");
}
