// P2 support ctest (C++-only): the BoxCox power transform. BoxCox is internal support
// (not a public-API distribution), so hardcoded oracles transcribed from the upstream C#
// test are correct here (public-API oracle values stay in fixtures/ only).
//
// Oracles transcribed VALUES-UNALTERED from
//   upstream/Numerics/Test_Numerics/Data/Statistics/Test_BoxCox.cs @ a2c4dbf
//   (Test_Fit, Test_Fit_R, Test_Transform, Test_Transform_R). The 1.670035 / 0.1314482 /
//   trueVals literals are R-rounded (EnvStats::boxcox / sae::bxcx), so their C# tolerances
//   (1E-4 / 1E-6 / 0.001) are kept; the exact analytic leaf checks (Transform(5,1)==4,
//   InverseTransform(4,1)==5) are tightened to 1e-12 per the tolerance policy.
//
// Deferred to P5 (documented, no regression): the P4 brief's section-1 "public-path
// corroboration" (dump BoxCox.Transform values through the REAL C# via the oracle emitter to
// back these transcribed leaf oracles) is NOT wired. It is redundant defense-in-depth -- the
// oracles above are transcribed VALUES-UNALTERED from the upstream Test_BoxCox.cs literals and
// recomputed inline from the identical closed-form expression, so they already ARE the C#
// public-path values. Routing them through the emitter/verify_oracles gate (which reproduces
// FIXTURES) would require either a fixture -- violating the binding "internal support gets
// C++-only ctests, public-API oracles live ONLY in fixtures/" constraint -- or new four-way
// harness wiring for a non-distribution support class. Tracked as a P5 follow-up.
//
// v2.1.4 (upstream-sync Task 2, 2a0357a) added test_fit_invalid_samples_returns_nan and
// test_log_likelihood_invalid_samples_returns_neg_infinity below, transcribed the same way
// from Test_BoxCox.cs's new Test_Fit_InvalidSamples_ReturnsNaN and
// Test_LogLikelihood_InvalidSamples_ReturnsNegativeInfinity (minus the null-sample case --
// a std::vector reference can't be null). The public-facing BoxCoxLambda FitLambda oracle
// path for these same NaN semantics is ALSO fixture-covered (fixtures/data/
// statistics_utilities.json), since BoxCoxLambda (unlike LogLikelihood) is bound in R/Python.
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/data/box_cox.hpp"
#include "check.hpp"

using corehydro::numerics::data::BoxCox;

namespace {

// C# Test_Fit / Test_Transform sample (199 values).
const std::vector<double> kSample199 = {
    142.25, 141.23, 141.33, 140.82, 141.31, 140.58, 141.58, 142.15, 143.07, 142.85, 143.17,
    142.54, 143.07, 142.26, 142.97, 143.86, 142.57, 142.19, 142.35, 142.63, 144.15, 144.73,
    144.7,  144.97, 145.12, 144.78, 145.06, 143.94, 143.77, 144.8,  145.67, 145.44, 145.56,
    145.61, 146.05, 145.74, 145.83, 143.88, 140.39, 139.34, 140.05, 137.93, 138.78, 139.59,
    140.54, 141.31, 140.42, 140.18, 138.43, 138.97, 139.31, 139.26, 140.08, 141.1,  143.48,
    143.28, 143.5,  143.12, 142.14, 142.54, 142.24, 142.16, 142.97, 143.69, 143.67, 144.65,
    144.33, 144.82, 143.74, 144.9,  145.83, 146.97, 146.6,  146.55, 148.22, 148.37, 148.23,
    148.73, 149.49, 149.09, 149.64, 148.42, 148.9,  149.97, 150.75, 150.88, 150.58, 150.64,
    150.73, 149.75, 150.86, 150.7,  150.8,  151.38, 152.01, 152.58, 152.7,  152.95, 152.53,
    151.5,  151.94, 151.46, 153.67, 153.88, 153.54, 153.74, 152.86, 151.56, 149.58, 150.93,
    150.67, 150.5,  152.06, 153.14, 153.38, 152.55, 153.58, 151.08, 151.52, 150.24, 150.21,
    148.13, 150.38, 150.9,  150.87, 152.18, 152.4,  152.38, 153.16, 152.29, 150.75, 152.37,
    154.57, 154.99, 154.93, 154.23, 155.2,  154.89, 154.18, 153.12, 152.02, 150.19, 148.21,
    145.93, 148.33, 145.18, 146.76, 147.28, 144.21, 145.94, 148.41, 147.43, 144.39, 146.5,
    145.7,  142.72, 139.79, 145.5,  145.17, 144.6,  146.01, 147.34, 146.48, 147.85, 146.16,
    144.37, 145.45, 147.65, 147.45, 148.2,  147.95, 146.48, 146.52, 146.24, 147.29, 148.55,
    147.96, 148.31, 148.83, 153.41, 153.34, 152.71, 152.42, 150.81, 152.25, 152.91, 152.85,
    152.6,  154.61, 153.81, 154.11, 155.03, 155.39, 155.6,  156.04, 156.93, 155.46, 156.27,
    154.41, 154.98};

// C# Test_Transform trueVals (BoxCox.Transform(sample, 1.670035)).
const std::vector<double> kTrue199 = {
    2359.592, 2331.397, 2334.155, 2320.102, 2333.603, 2313.5,   2341.056, 2356.821, 2382.357,
    2376.241, 2385.139, 2367.633, 2382.357, 2359.869, 2379.576, 2404.372, 2368.466, 2357.93,
    2362.364, 2370.131, 2412.474, 2428.711, 2427.87,  2435.442, 2439.653, 2430.112, 2437.968,
    2406.606, 2401.86,  2430.673, 2455.118, 2448.646, 2452.022, 2453.429, 2465.826, 2457.089,
    2459.624, 2404.931, 2308.279, 2279.513, 2298.949, 2241.111, 2264.23,  2286.349, 2312.4,
    2333.603, 2309.103, 2302.514, 2254.699, 2269.41,  2278.693, 2277.327, 2299.771, 2327.813,
    2393.772, 2388.201, 2394.33,  2383.748, 2356.545, 2367.633, 2359.315, 2357.099, 2379.576,
    2399.628, 2399.07,  2426.468, 2417.508, 2431.234, 2401.023, 2433.478, 2459.624, 2491.827,
    2481.357, 2479.943, 2527.33,  2531.603, 2527.614, 2541.873, 2563.607, 2552.158, 2567.905,
    2533.029, 2546.728, 2577.372, 2599.803, 2603.549, 2594.907, 2596.635, 2599.226, 2571.059,
    2602.972, 2598.362, 2601.243, 2617.977, 2636.202, 2652.735, 2656.221, 2663.489, 2651.283,
    2621.444, 2634.174, 2620.289, 2684.466, 2690.597, 2680.673, 2686.509, 2660.871, 2623.179,
    2566.185, 2604.99,  2597.498, 2592.605, 2637.65,  2669.018, 2676.009, 2651.864, 2681.84,
    2609.316, 2622.023, 2585.127, 2584.265, 2524.767, 2589.153, 2604.125, 2603.26,  2641.128,
    2647.509, 2646.929, 2669.6,   2644.318, 2599.803, 2646.639, 2710.78,  2723.095, 2721.334,
    2700.827, 2729.26,  2720.16,  2699.364, 2668.436, 2636.492, 2583.69,  2527.045, 2462.442,
    2530.464, 2441.338, 2485.882, 2500.613, 2414.152, 2462.724, 2532.744, 2504.868, 2419.187,
    2478.53,  2455.962, 2372.63,  2291.823, 2450.334, 2441.057, 2425.068, 2464.697, 2502.315,
    2477.965, 2516.8,   2468.929, 2418.628, 2448.927, 2511.115, 2505.436, 2526.76,  2519.644,
    2477.965, 2479.095, 2471.187, 2500.896, 2536.736, 2519.929, 2529.894, 2544.728, 2676.883,
    2674.843, 2656.512, 2648.09,  2601.531, 2643.158, 2662.326, 2660.581, 2653.316, 2711.952,
    2688.552, 2697.318, 2724.269, 2734.844, 2741.021, 2753.98,  2780.268, 2736.903, 2760.764,
    2706.094, 2722.801};

// C# Test_Fit_R / Test_Transform_R sample (23 values).
const std::vector<double> kSample23 = {1, 3, 59, 1, 6,  4,  9, 13, 5, 84, 35, 8,
                                       31, 34, 9, 1, 35, 66, 1, 65, 4, 68, 46};

// C# Test_Transform_R trueVals (BoxCox.Transform(sample, 0.1314482)).
const std::vector<double> kTrue23 = {
    0.000000, 1.181898, 5.394755, 0.000000, 2.020349, 1.520639, 2.547415, 3.050330,
    1.792351, 6.012795, 4.532207, 2.391402, 4.340082, 4.486038, 2.547415, 0.000000,
    4.532207, 5.587797, 0.000000, 5.561342, 1.520639, 5.639679, 4.976243};

// C# Test_Fit.
void test_fit() {
    CHECK_NEAR(BoxCox::fit_lambda(kSample199), 1.670035, 1E-4);
}

// C# Test_Fit_R.
void test_fit_r() {
    CHECK_NEAR(BoxCox::fit_lambda(kSample23), 0.1314482, 1E-6);
}

// C# Test_Transform: forward transform matches trueVals, inverse round-trips to sample.
void test_transform() {
    std::vector<double> vals = BoxCox::transform(kSample199, 1.670035);
    CHECK_EQ(vals.size(), kTrue199.size());
    for (std::size_t i = 0; i < vals.size(); ++i) CHECK_NEAR(vals[i], kTrue199[i], 0.001);

    std::vector<double> rev = BoxCox::inverse_transform(vals, 1.670035);
    for (std::size_t i = 0; i < rev.size(); ++i) CHECK_NEAR(rev[i], kSample199[i], 0.001);
}

// C# Test_Transform_R: exact leaf checks (tightened to 1e-12) + the 23-value R-parity check.
void test_transform_r() {
    double x = BoxCox::transform(5.0, 1.0);
    double y = BoxCox::inverse_transform(x, 1.0);
    CHECK_NEAR(x, 4.0, 1e-12);
    CHECK_NEAR(y, 5.0, 1e-12);

    std::vector<double> vals = BoxCox::transform(kSample23, 0.1314482);
    CHECK_EQ(vals.size(), kTrue23.size());
    for (std::size_t i = 0; i < vals.size(); ++i) CHECK_NEAR(vals[i], kTrue23[i], 1E-6);

    std::vector<double> rev = BoxCox::inverse_transform(vals, 0.1314482);
    for (std::size_t i = 0; i < rev.size(); ++i) CHECK_NEAR(rev[i], kSample23[i], 0.001);
}

// Scalar guard behaviour (Numerics/Data/Statistics/BoxCox.cs:104-135): non-positive value
// and |lambda|>5 -> NaN; |lambda|<1e-8 -> log/exp branch.
void test_guards() {
    CHECK_TRUE(std::isnan(BoxCox::transform(0.0, 1.0)));
    CHECK_TRUE(std::isnan(BoxCox::transform(-2.0, 1.0)));
    CHECK_TRUE(std::isnan(BoxCox::transform(5.0, 6.0)));
    CHECK_TRUE(std::isnan(BoxCox::inverse_transform(1.0, 6.0)));
    CHECK_NEAR(BoxCox::transform(2.0, 0.0), std::log(2.0), 1e-12);
    CHECK_NEAR(BoxCox::inverse_transform(0.5, 0.0), std::exp(0.5), 1e-12);
}

// LogJacobian: sum (lambda-1)*log(x_i); -inf on any x_i <= 0 (BoxCox.cs:80-97).
void test_log_jacobian() {
    std::vector<double> x = {2.0, 4.0, 8.0};
    double expected = (1.5 - 1.0) * (std::log(2.0) + std::log(4.0) + std::log(8.0));
    CHECK_NEAR(BoxCox::log_jacobian(x, 1.5), expected, 1e-12);
    CHECK_TRUE(std::isinf(BoxCox::log_jacobian({2.0, -1.0, 8.0}, 1.5)));
}

// v2.1.4: FitLambda reports invalid/degenerate samples with NaN instead of computing
// through BrentSearch (Test_BoxCox.cs Test_Fit_InvalidSamples_ReturnsNaN, minus the null
// case -- a std::vector reference can't be null).
void test_fit_invalid_samples_returns_nan() {
    CHECK_TRUE(std::isnan(BoxCox::fit_lambda({1.0})));
    CHECK_TRUE(std::isnan(BoxCox::fit_lambda({1.0, 1.0, 1.0})));
    CHECK_TRUE(std::isnan(BoxCox::fit_lambda({0.0, 10.0, 11.0})));
    CHECK_TRUE(std::isnan(BoxCox::fit_lambda({-1.0, 10.0, 11.0})));
    CHECK_TRUE(std::isnan(BoxCox::fit_lambda({1.0, std::numeric_limits<double>::quiet_NaN(), 2.0})));
    CHECK_TRUE(
        std::isnan(BoxCox::fit_lambda({1.0, std::numeric_limits<double>::infinity(), 2.0})));
}

// v2.1.4: LogLikelihood returns -Infinity for samples outside the Box-Cox domain
// (Test_BoxCox.cs Test_LogLikelihood_InvalidSamples_ReturnsNegativeInfinity).
void test_log_likelihood_invalid_samples_returns_neg_infinity() {
    CHECK_TRUE(std::isinf(BoxCox::log_likelihood({0.0, 1.0}, 1.0)));
    CHECK_TRUE(BoxCox::log_likelihood({0.0, 1.0}, 1.0) < 0.0);
    CHECK_TRUE(std::isinf(BoxCox::log_likelihood({1.0, 1.0}, 1.0)));
    CHECK_TRUE(std::isinf(
        BoxCox::log_likelihood({1.0, std::numeric_limits<double>::quiet_NaN()}, 1.0)));
    CHECK_TRUE(std::isinf(
        BoxCox::log_likelihood({1.0, 2.0}, std::numeric_limits<double>::quiet_NaN())));
}

}  // namespace

int main() {
    test_fit();
    test_fit_r();
    test_transform();
    test_transform_r();
    test_guards();
    test_log_jacobian();
    test_fit_invalid_samples_returns_nan();
    test_log_likelihood_invalid_samples_returns_neg_infinity();
    return chtest::summary("box_cox");
}
