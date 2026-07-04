// Standalone tests for models/data_frame/threshold_diagnostics.hpp (M5): POT
// threshold-selection diagnostics (Mean Residual Life + GPD parameter stability).
//
// Oracle is the upstream C# test class @ fc28c0c:
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/DataFrame/ThresholdDiagnosticsTests.cs
// transcribed method-for-method below (same section order), values unaltered.
//
// The C# tests generate input data with `new Random(seed)` (System.Random). To keep the
// data -- and therefore every statistical assertion -- bit-identical to the C#, the
// DotNetRandom helper below is a faithful port of the .NET seeded-Random compat PRNG
// (Knuth subtractive; dotnet/runtime CompatPrng), verified against dotnet 10 output.
//
// NOT transcribed (2 methods): MRL_NullData_ThrowsArgumentNullException and
// ParameterStability_NullData_ThrowsArgumentNullException -- the C++ port takes the data
// by const std::vector<double>&, so a null IList<double> has no counterpart.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "bestfit/models/data_frame/threshold_diagnostics.hpp"
#include "bestfit/numerics/distributions/exponential.hpp"
#include "bestfit/numerics/distributions/generalized_pareto.hpp"
#include "check.hpp"

using bestfit::models::MeanResidualLifeResult;
using bestfit::models::ParameterStabilityResult;
using bestfit::models::ThresholdDiagnostics;

namespace {

// Test-only port of .NET's seeded System.Random (dotnet/runtime CompatPrng: the Knuth
// subtractive generator that `new Random(seed)` uses on every modern .NET). Verified to
// reproduce dotnet 10's NextDouble() stream exactly for the seeds used here.
class DotNetRandom {
   public:
    explicit DotNetRandom(int seed) {
        const int kMBig = 2147483647;   // int.MaxValue
        const int kMSeed = 161803398;
        int subtraction = (seed == (-2147483647 - 1)) ? 2147483647 : (seed < 0 ? -seed : seed);
        int mj = kMSeed - subtraction;
        seed_array_[55] = mj;
        int mk = 1;
        int ii = 0;
        for (int i = 1; i < 55; i++) {
            if ((ii += 21) >= 55) ii -= 55;
            seed_array_[ii] = mk;
            mk = mj - mk;
            if (mk < 0) mk += kMBig;
            mj = seed_array_[ii];
        }
        for (int k = 1; k < 5; k++) {
            for (int i = 1; i < 56; i++) {
                int n = i + 30;
                if (n >= 55) n -= 55;
                seed_array_[i] -= seed_array_[1 + n];
                if (seed_array_[i] < 0) seed_array_[i] += kMBig;
            }
        }
        inext_ = 0;
        inextp_ = 21;
    }

    // C# Random.NextDouble()
    double next_double() { return internal_sample() * (1.0 / 2147483647.0); }

   private:
    int internal_sample() {
        int loc_inext = inext_;
        if (++loc_inext >= 56) loc_inext = 1;
        int loc_inextp = inextp_;
        if (++loc_inextp >= 56) loc_inextp = 1;
        int ret_val = seed_array_[loc_inext] - seed_array_[loc_inextp];
        if (ret_val == 2147483647) ret_val--;
        if (ret_val < 0) ret_val += 2147483647;
        seed_array_[loc_inext] = ret_val;
        inext_ = loc_inext;
        inextp_ = loc_inextp;
        return ret_val;
    }

    int seed_array_[56] = {0};
    int inext_ = 0;
    int inextp_ = 0;
};

// Exponential(lambda=1) sample via inverse CDF, mirroring the C# arrange blocks.
// (C# `new Exponential(1.0)` is the scale-only ctor: location 0, scale 1.)
std::vector<double> exponential_sample(int n, int seed) {
    DotNetRandom rng(seed);
    bestfit::numerics::distributions::Exponential exp_dist(0.0, 1.0);
    std::vector<double> data(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++)
        data[static_cast<std::size_t>(i)] = exp_dist.inverse_cdf(rng.next_double());
    return data;
}

// GPD(xi, alpha, kappa) sample via inverse CDF, mirroring the C# arrange blocks.
std::vector<double> gpd_sample(int n, int seed, double xi, double alpha, double kappa) {
    DotNetRandom rng(seed);
    bestfit::numerics::distributions::GeneralizedPareto gpd(xi, alpha, kappa);
    std::vector<double> data(static_cast<std::size_t>(n));
    for (int i = 0; i < n; i++)
        data[static_cast<std::size_t>(i)] = gpd.inverse_cdf(rng.next_double());
    return data;
}

}  // namespace

// --------------------------------------------------------------------------
// MRL Tests
// --------------------------------------------------------------------------

// C# MRL_Exponential_MeanExcessIsConstant: for Exponential(1) data the mean excess is
// ~1.0 at every threshold (memoryless property).
static void test_mrl_exponential_mean_excess_is_constant() {
    // Arrange -- generate Exponential(lambda=1) data using inverse CDF
    std::vector<double> data = exponential_sample(10000, 12345);
    double u_min = 0.5;
    double u_max = 3.0;

    // Act
    MeanResidualLifeResult result =
        ThresholdDiagnostics::compute_mean_residual_life(data, u_min, u_max, 50);

    // Assert -- all mean excess values should be approximately 1.0
    CHECK_TRUE(result.points.size() > 0);
    for (const auto& point : result.points) CHECK_NEAR(1.0, point.mean_excess, 0.15);
}

// C# MRL_Exponential_CIContainsTrueValue: the 95% CI contains the true mean excess
// (1.0) for at least 90% of thresholds.
static void test_mrl_exponential_ci_contains_true_value() {
    // Arrange
    std::vector<double> data = exponential_sample(10000, 54321);

    // Act
    MeanResidualLifeResult result =
        ThresholdDiagnostics::compute_mean_residual_life(data, 0.5, 2.5, 40);

    // Assert -- 95% CI should contain 1.0 for most thresholds
    int contains_true = 0;
    for (const auto& point : result.points)
        if (point.lower_ci <= 1.0 && point.upper_ci >= 1.0) contains_true++;

    double coverage = static_cast<double>(contains_true) /
                      static_cast<double>(result.points.size());
    CHECK_TRUE(coverage >= 0.90);
}

// C# MRL_CIWidthNarrowsWithSampleSize.
static void test_mrl_ci_width_narrows_with_sample_size() {
    // Arrange -- small and large samples
    std::vector<double> data_small = exponential_sample(200, 11111);
    std::vector<double> data_large = exponential_sample(10000, 22222);

    // Act
    MeanResidualLifeResult result_small =
        ThresholdDiagnostics::compute_mean_residual_life(data_small, 0.5, 1.5, 10);
    MeanResidualLifeResult result_large =
        ThresholdDiagnostics::compute_mean_residual_life(data_large, 0.5, 1.5, 10);

    // Assert -- average CI width should be smaller for large sample
    double sum_small = 0.0;
    for (const auto& p : result_small.points) sum_small += p.upper_ci - p.lower_ci;
    double avg_width_small = sum_small / static_cast<double>(result_small.points.size());
    double sum_large = 0.0;
    for (const auto& p : result_large.points) sum_large += p.upper_ci - p.lower_ci;
    double avg_width_large = sum_large / static_cast<double>(result_large.points.size());

    CHECK_TRUE(avg_width_large < avg_width_small);
}

// C# MRL_SkipsThresholdsWithFewExceedances: thresholds with fewer than 5 exceedances
// are excluded.
static void test_mrl_skips_thresholds_with_few_exceedances() {
    // Arrange -- small dataset where high thresholds will have < 5 exceedances
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    // Act -- use a range that includes thresholds near the max (few exceedances)
    MeanResidualLifeResult result =
        ThresholdDiagnostics::compute_mean_residual_life(data, 1.0, 9.5, 20);

    // Assert -- all returned points should have at least 5 exceedances
    for (const auto& point : result.points) CHECK_TRUE(point.exceedance_count >= 5);
}

// --------------------------------------------------------------------------
// Parameter Stability Tests
// --------------------------------------------------------------------------

// C# ParameterStability_GPDData_ParametersAreStable: shape ~0.1 and approximately
// constant across thresholds for GPD(0, 10, 0.1) data.
static void test_parameter_stability_gpd_data_parameters_are_stable() {
    // Arrange -- generate GPD(xi=0, alpha=10, kappa=0.1) data
    int n = 5000;
    std::vector<double> data = gpd_sample(n, 99999, 0.0, 10.0, 0.1);

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double u_min = sorted[static_cast<std::size_t>(n * 0.1)];
    double u_max = sorted[static_cast<std::size_t>(n * 0.7)];

    // Act
    ParameterStabilityResult result =
        ThresholdDiagnostics::compute_parameter_stability(data, u_min, u_max, 30);

    // Assert -- shape should be approximately constant (~0.1)
    CHECK_TRUE(result.points.size() > 5);

    // Check that shape values don't vary too wildly
    double sum_shape = 0.0;
    for (const auto& p : result.points) sum_shape += p.shape;
    double mean_shape = sum_shape / static_cast<double>(result.points.size());
    double max_deviation = 0.0;
    for (const auto& p : result.points)
        max_deviation = std::max(max_deviation, std::fabs(p.shape - mean_shape));

    CHECK_NEAR(0.1, mean_shape, 0.1);
    CHECK_TRUE(max_deviation < 0.3);
}

// C# ParameterStability_SkipsThresholdsWithFewExceedances: thresholds with fewer than
// 10 exceedances are excluded.
static void test_parameter_stability_skips_thresholds_with_few_exceedances() {
    // Arrange -- small dataset
    std::vector<double> data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

    // Act -- use a range where high thresholds will have < 10 exceedances
    ParameterStabilityResult result =
        ThresholdDiagnostics::compute_parameter_stability(data, 1.0, 12.0, 20);

    // Assert -- all returned points should have at least 10 exceedances
    for (const auto& point : result.points) CHECK_TRUE(point.exceedance_count >= 10);
}

// C# ParameterStability_ShapeCIContainsTrueValue: the 95% CI contains the true shape
// (0.1) for at least 80% of thresholds.
static void test_parameter_stability_shape_ci_contains_true_value() {
    // Arrange -- generate GPD(xi=0, alpha=10, kappa=0.1) data
    int n = 5000;
    std::vector<double> data = gpd_sample(n, 77777, 0.0, 10.0, 0.1);

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double u_min = sorted[static_cast<std::size_t>(n * 0.1)];
    double u_max = sorted[static_cast<std::size_t>(n * 0.5)];

    // Act
    ParameterStabilityResult result =
        ThresholdDiagnostics::compute_parameter_stability(data, u_min, u_max, 20);

    // Assert -- 95% CI should contain true shape (0.1) for most thresholds
    int contains_true = 0;
    for (const auto& point : result.points)
        if (point.shape_lower_ci <= 0.1 && point.shape_upper_ci >= 0.1) contains_true++;

    double coverage = static_cast<double>(contains_true) /
                      static_cast<double>(result.points.size());
    CHECK_TRUE(coverage >= 0.80);
}

// --------------------------------------------------------------------------
// Input Validation Tests
// --------------------------------------------------------------------------

// C# MRL_EmptyData_ThrowsArgumentException.
static void test_mrl_empty_data_throws() {
    CHECK_THROWS(ThresholdDiagnostics::compute_mean_residual_life({}, 0, 1));
}

// C# MRL_UMaxLessThanUMin_ThrowsArgumentException.
static void test_mrl_umax_less_than_umin_throws() {
    CHECK_THROWS(ThresholdDiagnostics::compute_mean_residual_life({1, 2, 3}, 5.0, 3.0));
}

// C# MRL_NThresholdsLessThanTwo_ThrowsArgumentException.
static void test_mrl_n_thresholds_less_than_two_throws() {
    CHECK_THROWS(ThresholdDiagnostics::compute_mean_residual_life({1, 2, 3}, 0.5, 2.5, 1));
}

// C# MRL_InvalidConfidenceLevel_ThrowsArgumentException.
static void test_mrl_invalid_confidence_level_throws() {
    CHECK_THROWS(
        ThresholdDiagnostics::compute_mean_residual_life({1, 2, 3}, 0.5, 2.5, 100, 1.5));
}

// C# ParameterStability_EmptyData_ThrowsArgumentException.
static void test_parameter_stability_empty_data_throws() {
    CHECK_THROWS(ThresholdDiagnostics::compute_parameter_stability({}, 0, 1));
}

// --------------------------------------------------------------------------
// Result Structure Tests
// --------------------------------------------------------------------------

// C# MRL_PointsAreOrderedByThreshold.
static void test_mrl_points_are_ordered_by_threshold() {
    // Arrange
    std::vector<double> data = exponential_sample(1000, 33333);

    // Act
    MeanResidualLifeResult result =
        ThresholdDiagnostics::compute_mean_residual_life(data, 0.5, 3.0, 30);

    // Assert
    for (std::size_t i = 1; i < result.points.size(); i++)
        CHECK_TRUE(result.points[i].threshold > result.points[i - 1].threshold);
}

// C# MRL_LowerCILessThanUpperCI.
static void test_mrl_lower_ci_less_than_upper_ci() {
    // Arrange
    std::vector<double> data = exponential_sample(2000, 44444);

    // Act
    MeanResidualLifeResult result =
        ThresholdDiagnostics::compute_mean_residual_life(data, 0.5, 3.0, 30);

    // Assert
    for (const auto& point : result.points) {
        CHECK_TRUE(point.lower_ci < point.upper_ci);
        CHECK_TRUE(point.lower_ci <= point.mean_excess &&
                   point.mean_excess <= point.upper_ci);
    }
}

// C# ParameterStability_ResultsAreFinite.
static void test_parameter_stability_results_are_finite() {
    // Arrange
    int n = 2000;
    std::vector<double> data = gpd_sample(n, 55555, 0.0, 10.0, 0.05);

    std::vector<double> sorted = data;
    std::sort(sorted.begin(), sorted.end());
    double u_min = sorted[static_cast<std::size_t>(n * 0.1)];
    double u_max = sorted[static_cast<std::size_t>(n * 0.5)];

    // Act
    ParameterStabilityResult result =
        ThresholdDiagnostics::compute_parameter_stability(data, u_min, u_max, 20);

    // Assert
    for (const auto& point : result.points) {
        CHECK_TRUE(!std::isnan(point.modified_scale));
        CHECK_TRUE(!std::isinf(point.modified_scale));
        CHECK_TRUE(!std::isnan(point.shape));
        CHECK_TRUE(!std::isinf(point.shape));
        CHECK_TRUE(point.modified_scale_lower_ci < point.modified_scale_upper_ci);
        CHECK_TRUE(point.shape_lower_ci < point.shape_upper_ci);
    }
}

int main() {
    // MRL tests
    test_mrl_exponential_mean_excess_is_constant();
    test_mrl_exponential_ci_contains_true_value();
    test_mrl_ci_width_narrows_with_sample_size();
    test_mrl_skips_thresholds_with_few_exceedances();

    // Parameter stability tests
    test_parameter_stability_gpd_data_parameters_are_stable();
    test_parameter_stability_skips_thresholds_with_few_exceedances();
    test_parameter_stability_shape_ci_contains_true_value();

    // Input validation tests
    test_mrl_empty_data_throws();
    test_mrl_umax_less_than_umin_throws();
    test_mrl_n_thresholds_less_than_two_throws();
    test_mrl_invalid_confidence_level_throws();
    test_parameter_stability_empty_data_throws();

    // Result structure tests
    test_mrl_points_are_ordered_by_threshold();
    test_mrl_lower_ci_less_than_upper_ci();
    test_parameter_stability_results_are_finite();

    return bftest::summary("threshold_diagnostics");
}
