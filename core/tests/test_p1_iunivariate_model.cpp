// P1 support ctest (C++-only): the IUnivariateModel marginal-interface unblock and the
// numerics::data::maximum helper. Both are internal support, so hardcoded oracles are correct
// here (public-API oracle values stay in fixtures/ only).
//
// Oracles:
//   - IUnivariateModel routing: transcribed from
//     upstream/RMC-BestFit/src/RMC.BestFit.Tests/Bivariate/
//     BivariateDistributionMarginalInterfaceTests.cs @ fc28c0c -- the
//     TargetModelClasses_ImplementIUnivariateModel guard and the BuildNormalMarginal
//     accessor-routing checks that exercise UnivariateDistribution AS an IUnivariateModel.
//     Anything asserting on BivariateDistribution construction or copula LL (the
//     Construct_WithMixedMarginalTypes / SwapMarginal methods) is DEFERRED to B1.
//   - maximum: transcribed from
//     upstream/Numerics/Test_Numerics/Data/Statistics/Test_Statistics.cs @ a2c4dbf
//     (Test_Maximum: max(_sample1) == 337.0 vs R base::max), plus hand-authored empty-input
//     and NaN-propagation cases pinning the C# guard semantics
//     (Numerics/Data/Statistics/Statistics.cs:90-105).
#include <cmath>
#include <limits>
#include <vector>

#include "bestfit/models/data_frame/data_frame.hpp"
#include "bestfit/models/support/i_univariate_model.hpp"
#include "bestfit/models/support/validation_result.hpp"
#include "bestfit/models/univariate_distribution/univariate_distribution_model.hpp"
#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "check.hpp"

using bestfit::models::DataFrame;
using bestfit::models::ExactSeries;
using bestfit::models::IUnivariateModel;
using bestfit::models::UnivariateDistributionModel;
using bestfit::models::ValidationResult;
using bestfit::numerics::distributions::UnivariateDistributionType;

namespace {

// C# SampleX (BivariateDistributionMarginalInterfaceTests.cs).
const std::vector<double> kSampleX = {98.1,  102.7, 115.3, 88.4,  104.9, 92.0,  110.5,
                                      99.2,  107.6, 101.3, 95.8,  108.1, 103.4, 97.6,
                                      112.9, 89.5,  106.2, 100.0, 93.7,  109.4};

// C# BuildNormalMarginal(mu, sigma, data): a pre-fit Normal UnivariateDistribution whose
// parameters are seeded so Validate() passes without an MLE call.
UnivariateDistributionModel build_normal_marginal(double mu, double sigma,
                                                  const std::vector<double>& data) {
    DataFrame df;
    df.set_exact_series(ExactSeries(data));
    UnivariateDistributionModel model(std::move(df), UnivariateDistributionType::Normal);
    model.set_parameter_values({mu, sigma});
    return model;
}

// C# TargetModelClasses_ImplementIUnivariateModel (the UnivariateDistribution row) +
// the accessor-routing checks. Binds a UnivariateDistributionModel through IUnivariateModel&
// and verifies every interface member routes to the concrete model.
void test_univariate_distribution_implements_iunivariate_model() {
    UnivariateDistributionModel model = build_normal_marginal(101.0, 7.5, kSampleX);
    IUnivariateModel& iface = model;

    // Distribution: non-null after construction, and the interface pointer is the SAME
    // underlying object the concrete reference accessor exposes (pointer identity), with
    // matching seeded parameters.
    CHECK_TRUE(iface.distribution() != nullptr);
    CHECK_TRUE(iface.distribution() == &model.distribution());
    std::vector<double> params = iface.distribution()->get_parameters();
    CHECK_EQ(params.size(), static_cast<std::size_t>(2));
    CHECK_NEAR(params[0], 101.0, 1e-9);
    CHECK_NEAR(params[1], 7.5, 1e-9);

    // IsNonstationary: false for the stationary construction, routed through the interface.
    CHECK_TRUE(iface.is_nonstationary() == false);
    CHECK_TRUE(iface.is_nonstationary() == model.is_nonstationary());

    // Validate: the interface result is consistent with the concrete call and valid for the
    // seeded Normal marginal.
    ValidationResult iface_valid = iface.validate();
    ValidationResult concrete_valid = model.validate();
    CHECK_TRUE(iface_valid.is_valid == concrete_valid.is_valid);
    CHECK_TRUE(iface_valid.is_valid);

    // DataFrame: the interface accessor reaches the model's frame.
    CHECK_EQ(iface.data_frame().exact_series().count(), kSampleX.size());
    CHECK_TRUE(&iface.data_frame() == &model.data_frame());
}

// A const IUnivariateModel& still reaches the nullable distribution pointer and the const
// frame accessor.
void test_iunivariate_model_const_routing() {
    UnivariateDistributionModel model = build_normal_marginal(101.0, 7.5, kSampleX);
    const IUnivariateModel& iface = model;

    CHECK_TRUE(iface.distribution() != nullptr);
    CHECK_EQ(iface.data_frame().exact_series().count(), kSampleX.size());
    CHECK_TRUE(iface.is_nonstationary() == false);
}

// C# Test_Maximum: max(_sample1) == 337.0 (R base::max oracle).
void test_maximum_sample1() {
    const std::vector<double> sample1 = {
        122,   244, 214, 173, 229, 156, 212, 263, 146, 183, 161, 205, 135, 331,
        225,   174, 98.8, 149, 238, 262, 132, 235, 216, 240, 230, 192, 195, 172,
        173,   172, 153, 142, 317, 161, 201, 204, 194, 164, 183, 161, 167, 179,
        185,   117, 192, 337, 125, 166, 99.1, 202, 230, 158, 262, 154, 164, 182,
        164,   183, 171, 250, 184, 205, 237, 177, 239, 187, 180, 173, 174};
    CHECK_NEAR(bestfit::numerics::data::maximum(sample1), 337.0, 1e-10);
}

// Unsorted small vector -> its max.
void test_maximum_unsorted() {
    CHECK_NEAR(bestfit::numerics::data::maximum({3.0, -7.5, 12.25, 0.0, 12.24, -100.0}), 12.25,
               1e-9);
    CHECK_NEAR(bestfit::numerics::data::maximum({-5.0, -1.0, -9.0}), -1.0, 1e-9);
}

// Empty input -> NaN (C# data.Count == 0 guard).
void test_maximum_empty_is_nan() {
    CHECK_TRUE(std::isnan(bestfit::numerics::data::maximum({})));
}

// Any NaN element short-circuits -> NaN (C# double.IsNaN(data[i]) guard).
void test_maximum_nan_propagates() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    CHECK_TRUE(std::isnan(bestfit::numerics::data::maximum({1.0, 2.0, nan, 4.0})));
    CHECK_TRUE(std::isnan(bestfit::numerics::data::maximum({nan})));
}

}  // namespace

int main() {
    test_univariate_distribution_implements_iunivariate_model();
    test_iunivariate_model_const_routing();
    test_maximum_sample1();
    test_maximum_unsorted();
    test_maximum_empty_is_nan();
    test_maximum_nan_propagates();
    return bftest::summary("p1_iunivariate_model");
}
