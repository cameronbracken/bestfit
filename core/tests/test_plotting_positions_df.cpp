// Standalone tests for DataFrame::calculate_plotting_positions() (M5): the
// Hirsch-Stedinger censored plotting positions (Bulletin 17C Appendix 5).
//
// Oracle is the upstream C# test class @ c2e6192 (BestFit v2.0.0):
//   upstream/RMC-BestFit/src/RMC.BestFit.Tests/DataFrame/PlottingPositionTests.cs
// transcribed method-for-method below (same order), values unaltered. The two B17C
// examples validate the full censored machinery (exact + interval + perception
// thresholds); the six named-formula tests validate the uncensored path against the
// ported Numerics PlottingPositions formulas using the bit-exact seeded Mersenne
// Twister random stream (`GenerateRandomValues(30, 12345)`). Notably, none of these
// eight cases needed re-pinning for the v2.0.0 ARRANGE2/PPLOT2/PLPOS rewrite (T12) --
// they reproduce byte-identical to their pre-rewrite values, confirming the rewrite
// preserves the formula shape for single-threshold/no-threshold, non-degenerate frames
// (see data_frame_plotting.hpp's file header).
//
// One extra test (not in the C# class) covers ApplyLangbeinConversion, which lives in
// the same ported region but has no upstream unit test; it checks the direct formula
// PP' = 1 - exp(-lambda * PP) on every series.
//
// BestFit v2.0.0 additions (T12, still transcribed from PlottingPositionTests.cs): the
// bootstrap-edge/tie-separation regressions (Test_PlottingPositions_
// Example5BootstrapEdge_RemainsStrictWithoutChangingSample /
// ...Example5BootstrapSample_TiesAreSeparated -- these two exercise
// EnsureDistinctPlottingPositions, the load-bearing new duplicate-position-spreading
// logic), the three small ARRANGE2 classification cases already covered cross-language
// via fixtures/estimation/plotting_position.json (Test_PlottingPositions_
// ValueBelowOwnThreshold_UsesCensoredBranch / ...AggregateThresholdCounts_AffectRanks /
// ...OutsideThresholdWindow_IsDetected -- re-verified here too, C++-only, as a fast
// sanity check independent of the fixture harness), Test_PlottingPositions_
// InvalidInputs_Throw (the DataFrame-level half; the PlottingParameter setter's own
// eager validation is covered by test_data_frame_set_plotting_parameter_rejects_out_of_
// range in test_data_frame.cpp), and a plotting_position_version() bump check (no
// upstream unit test -- PlottingPositionVersion has no dedicated C# test either).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/numerics/data/plotting_positions.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/utilities/extension_methods.hpp"
#include "check.hpp"

using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::IntervalData;
using corehydro::models::ThresholdData;
namespace pp = corehydro::numerics::data::plotting_positions;

// C# Test_PlottingPositions_B17C_Ex4: Bulletin 17C Example 4 -- 81 years of
// systematic record, 4 historical flood intervals, 4 perception thresholds,
// Weibull plotting parameter (default 0).
static void test_plotting_positions_b17c_ex4() {
    const std::vector<int> sys_years = {
        1895, 1896, 1897, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1906, 1907,
        1908, 1909, 1910, 1911, 1912, 1913, 1914, 1915, 1916, 1917, 1918, 1919, 1920,
        1922, 1923, 1924, 1925, 1926, 1927, 1928, 1929, 1930, 1931, 1932, 1933, 1934,
        1935, 1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947,
        1948, 1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960,
        1961, 1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973,
        1974, 1975, 1976};
    const std::vector<double> sys_values = {
        6100,  16500, 4300,  7500,  8800, 7600,  11100, 30000, 10500, 8500,  8000,
        11000, 6600,  7600,  5800,  8400, 3700,  10500, 7800,  7500,  17000, 8900,
        6800,  9600,  6300,  8500,  8850, 25600, 6510,  4930,  4520,  12400, 7800,
        10500, 6050,  3560,  4380,  8630, 2580,  9880,  11200, 9300,  11200, 2910,
        3860,  7560,  10300, 3320,  5980, 9290,  7050,  7280,  10900, 12800, 8700,
        9300,  4740,  6770,  10200, 11100, 8010, 9070,  4540,  2820,  5260,  5760,
        3540,  8360,  2840,  23500, 10600, 5870, 5190,  6620,  6300,  3360,  3360,
        6760,  5440,  10200, 12800};
    const std::vector<double> sys_pp = {
        0.690016355873821,  0.0819715154724692, 0.856930625787917,  0.558869429512745,
        0.380032711747642,  0.511179638108717,  0.165428650429517,  0.0285079600148093,
        0.213118441833545,  0.427722503151669,  0.475412294555697,  0.177351098280524,
        0.642326564469793,  0.523102085959724,  0.737706147277848,  0.439644951002676,
        0.880775521489931,  0.236963337535559,  0.49925719025771,   0.546946981661738,
        0.0700490676214623, 0.356187816045628,  0.594636773065766,  0.296575576790593,
        0.666171460171807,  0.415800055300662,  0.368110263896635,  0.0359126249537208,
        0.6542490123208,    0.797318386532883,  0.833085730085903,  0.11773885902549,
        0.487334742406703,  0.225040889684552,  0.701938803724828,  0.892697969340938,
        0.84500817793691,   0.403877607449655,  0.988077552148993,  0.284653128939586,
        0.141583754727504,  0.320420472492607,  0.129661306876497,  0.952310208595972,
        0.868853073638924,  0.535024533810731,  0.248885785386566,  0.940387760744965,
        0.713861251575834,  0.332342920343614,  0.582714325214759,  0.570791877363752,
        0.189273546131531,  0.105816411174483,  0.391955159598648,  0.3084980246416,
        0.80924083438389,   0.606559220916773,  0.272730681088579,  0.153506202578511,
        0.46348984670469,   0.344265368194621,  0.821163282234897,  0.976155104297986,
        0.773473490830869,  0.749628595128855,  0.904620417191945,  0.451567398853683,
        0.964232656446979,  0.0433172898926324, 0.201195993982538,  0.725783699426841,
        0.785395938681876,  0.630404116618786,  0.678093908022814,  0.928465312893959,
        0.916542865042952,  0.618481668767779,  0.761551042979862,  0.260808233237573,
        0.0938939633234761};
    const std::vector<double> int_pp = {0.0091324200913242, 0.0507219548315439,
                                        0.0211032950758978, 0.0045662100456621};

    // Create data frame
    DataFrame df;

    // Add exact data
    for (std::size_t i = 0; i < sys_values.size(); i++)
        df.exact_series().add(ExactData(sys_years[i], sys_values[i]));

    // Add interval data
    df.interval_series().add(IntervalData(1864, 41000, 49598.38707, 60000));
    df.interval_series().add(IntervalData(1893, 20000, 22360.67977, 25000));
    df.interval_series().add(IntervalData(1894, 35000, 37416.57387, 40000));
    df.interval_series().add(IntervalData(1921, 80000, 90774.44574, 103000));

    // Add thresholds
    df.threshold_series().add(ThresholdData(1165, 1858, 150000));
    df.threshold_series().add(ThresholdData(1859, 1892, 40000));
    df.threshold_series().add(ThresholdData(1893, 1894, 19900));
    df.threshold_series().add(ThresholdData(1977, 2004, 20000));

    // Process thresholds
    df.process_threshold_series();

    // Create plotting positions
    df.calculate_plotting_positions();

    // Test exact data plotting positions
    for (std::size_t i = 0; i < sys_pp.size(); i++)
        CHECK_NEAR(df.exact_series()[i].plotting_position(), sys_pp[i], 1E-12);

    // Test interval data plotting positions
    for (std::size_t i = 0; i < int_pp.size(); i++)
        CHECK_NEAR(df.interval_series()[i].plotting_position(), int_pp[i], 1E-12);
}

// C# Test_PlottingPositions_B17C_Ex7: Bulletin 17C Example 7 -- systematic record
// 1905-1997 plus 5 paleoflood intervals back to 605 AD, Cunnane parameter (0.4),
// multiple discontinuous perception-threshold periods.
static void test_plotting_positions_b17c_ex7() {
    const std::vector<int> sys_years = {
        1905, 1906, 1907, 1908, 1909, 1911, 1914, 1915, 1916, 1917, 1919, 1920, 1921,
        1922, 1923, 1924, 1925, 1926, 1927, 1928, 1930, 1931, 1932, 1933, 1934, 1935,
        1936, 1937, 1938, 1939, 1940, 1941, 1942, 1943, 1944, 1945, 1946, 1947, 1948,
        1949, 1950, 1951, 1952, 1953, 1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961,
        1962, 1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
        1975, 1976, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1997};
    const std::vector<double> sys_values = {
        24200,  59700,  156000, 10300,  119000, 81300,  74100,  47900,  40700,  42300,
        67500,  20100,  39200,  31600,  39000,  14000,  99500,  27400,  67700,  163000,
        24400,  9900,   21100,  16500,  22600,  60900,  58300,  33000,  114000, 10900,
        89200,  38800,  83200,  152000, 20100,  94400,  42200,  27900,  21000,  37500,
        34400,  180000, 37200,  49700,  42600,  10800,  219000, 42000,  54000,  20000,
        75000,  8000,   40000,  240000, 24000,  260000, 6500,   46000,  30000,  120000,
        122000, 48000,  12000,  69000,  55000,  46000,  15000,  40000,  33000,  175000,
        20000,  152000, 93000,  88000,  17000,  259000, 298000};
    const std::vector<double> sys_pp = {
        0.739808094808377,   0.354830275902403,  0.0948343583050208, 0.952209650066845,
        0.15570381784759,    0.261904595476824,  0.288454789884132,  0.434480859124329,
        0.527406539549909,   0.487581247938946,  0.328280081495095,  0.806183580826648,
        0.567231831160871,   0.673432608790105,  0.580506928364526,  0.899109261252228,
        0.182254012254898,   0.713257900401068,  0.315004984291441,  0.0846981249153794,
        0.726532997604722,   0.965484747270499,  0.779633386419339,  0.872559066844919,
        0.766358289215685,   0.341555178698749,  0.368105373106058,  0.646882414382797,
        0.168978915051244,   0.925659455659536,  0.222079303865861,  0.59378202556818,
        0.248629498273169,   0.104970591694662,  0.819458678030302,  0.195529109458552,
        0.5008563451426,     0.699982803197414,  0.792908483622994,  0.607057122771834,
        0.633607317179143,   0.0644256581360965, 0.620332219975488,  0.40793066471702,
        0.474306150735292,   0.93893455286319,   0.0542894247464551, 0.514131442346254,
        0.394655567513366,   0.832733775233956,  0.275179692680478,  0.978759844474153,
        0.540681636753563,   0.0441531913568137, 0.753083192012031,  0.0238807245775308,
        0.992034941677807,   0.461031053531637,  0.68670770599376,   0.142428720643935,
        0.129153623440281,   0.421205761920675,  0.912384358455882,  0.301729887087786,
        0.381380470309712,   0.447755956327983,  0.885834164048573,  0.553956733957217,
        0.660157511586451,   0.0745618915257379, 0.846008872437611,  0.115106825084304,
        0.208804206662207,   0.235354401069515,  0.859283969641265,  0.0340169579671723,
        0.00833768638161468};
    const std::vector<double> int_pp = {0.00025, 0.0013043186695279, 0.00264484978540773,
                                        0.00398538090128755, 0.0142509977329467};

    // Create data frame
    DataFrame df;

    // Add exact data
    for (std::size_t i = 0; i < sys_values.size(); i++)
        df.exact_series().add(ExactData(sys_years[i], sys_values[i]));

    // Add interval data
    df.interval_series().add(IntervalData(605, 600000, 714142.8429, 850000));
    df.interval_series().add(IntervalData(1437, 400000, 469041.576, 550000));
    df.interval_series().add(IntervalData(1574, 400000, 469041.576, 550000));
    df.interval_series().add(IntervalData(1711, 400000, 469041.576, 550000));
    df.interval_series().add(IntervalData(1862, 262000, 280356.9154, 300000));

    // Add thresholds
    df.threshold_series().add(ThresholdData(1, 1301, 599000));
    df.threshold_series().add(ThresholdData(1302, 1847, 399000));
    df.threshold_series().add(ThresholdData(1848, 1904, 261000));
    df.threshold_series().add(ThresholdData(1910, 1910, 150000));
    df.threshold_series().add(ThresholdData(1912, 1913, 150000));
    df.threshold_series().add(ThresholdData(1918, 1918, 150000));
    df.threshold_series().add(ThresholdData(1929, 1929, 150000));
    df.threshold_series().add(ThresholdData(1977, 1977, 150000));
    df.threshold_series().add(ThresholdData(1987, 1996, 150000));
    df.threshold_series().add(ThresholdData(1998, 2000, 150000));

    // Process thresholds
    df.process_threshold_series();

    // Create plotting positions
    df.set_plotting_parameter(0.4);
    df.calculate_plotting_positions();

    // Test exact data plotting positions
    for (std::size_t i = 0; i < sys_pp.size(); i++)
        CHECK_NEAR(df.exact_series()[i].plotting_position(), sys_pp[i], 1E-12);

    // Test interval data plotting positions
    for (std::size_t i = 0; i < int_pp.size(); i++)
        CHECK_NEAR(df.interval_series()[i].plotting_position(), int_pp[i], 1E-12);
}

// Shared body of the six named-formula tests (C# Test_Blom / Test_Cunnane /
// Test_Gringorten / Test_Hazen / Test_Median / Test_Weibull differ only in alpha and
// the reference formula): 30 seeded Normal(100, 15) variates sorted descending, then
// CalculatePlottingPositions with the formula's alpha must reproduce the ported
// Numerics PlottingPositions reference values.
static void check_named_formula(double alpha, const std::vector<double>& reference_pp) {
    // Create random (bit-exact with the C# MersenneTwister stream)
    const int n = 30;
    corehydro::numerics::distributions::Normal norm(100, 15);
    std::vector<double> data = norm.generate_random_values(n, 12345);
    std::sort(data.begin(), data.end());
    std::reverse(data.begin(), data.end());

    // Create data frame and add data
    DataFrame df;
    for (std::size_t i = 0; i < data.size(); i++)
        df.exact_series().add(ExactData(static_cast<int>(i), data[i]));

    // Create plotting positions
    df.set_plotting_parameter(alpha);
    df.calculate_plotting_positions();

    // Test exact data plotting positions
    for (std::size_t i = 0; i < data.size(); i++)
        CHECK_NEAR(reference_pp[i], df.exact_series()[i].plotting_position(), 1E-6);
}

static void test_blom() { check_named_formula(0.375, pp::blom(30)); }
static void test_cunnane() { check_named_formula(0.4, pp::cunnane(30)); }
static void test_gringorten() { check_named_formula(0.44, pp::gringorten(30)); }
static void test_hazen() { check_named_formula(0.50, pp::hazen(30)); }
static void test_median() { check_named_formula(0.3175, pp::median(30)); }
static void test_weibull() { check_named_formula(0.0, pp::weibull(30)); }

// Extra (no upstream unit test): ApplyLangbeinConversion transforms every exact,
// uncertain, and interval plotting position via PP' = 1 - exp(-lambda * PP).
static void test_apply_langbein_conversion() {
    const int n = 30;
    corehydro::numerics::distributions::Normal norm(100, 15);
    std::vector<double> data = norm.generate_random_values(n, 12345);
    std::sort(data.begin(), data.end());
    std::reverse(data.begin(), data.end());

    DataFrame df;
    for (std::size_t i = 0; i < data.size(); i++)
        df.exact_series().add(ExactData(static_cast<int>(i), data[i]));
    df.calculate_plotting_positions();

    const double lambda = 1.7;
    std::vector<double> before = df.exact_series().plotting_positions_to_list();
    df.apply_langbein_conversion(lambda);
    for (std::size_t i = 0; i < data.size(); i++) {
        CHECK_NEAR(df.exact_series()[i].plotting_position(),
                   1.0 - std::exp(-lambda * before[i]), 1E-15);
    }
}

// ===========================================================================================
// BestFit v2.0.0: ARRANGE2/PPLOT2/PLPOS rewrite regressions (T12)
// ===========================================================================================

// C# Test_PlottingPositions_ValueBelowOwnThreshold_UsesCensoredBranch: one detection and
// one censored observation at a common threshold give Weibull-style 0.25/0.75 exceedance
// (also cross-language-verified via fixtures/estimation/plotting_position.json).
static void test_plotting_positions_value_below_own_threshold_uses_censored_branch() {
    DataFrame df;
    df.exact_series().add(ExactData(0, 50.0));
    df.exact_series().add(ExactData(1, 200.0));
    df.threshold_series().add(ThresholdData(0, 1, 100.0));

    df.calculate_plotting_positions();

    CHECK_NEAR(df.exact_series()[0].plotting_position(), 0.75, 1E-12);
    CHECK_NEAR(df.exact_series()[1].plotting_position(), 0.25, 1E-12);
    CHECK_NEAR(df.exact_series()[0].value(), 50.0, 0.0);
    CHECK_NEAR(df.exact_series()[1].value(), 200.0, 0.0);
}

// C# Test_PlottingPositions_AggregateThresholdCounts_AffectRanks: three left-censored
// placeholders, one finite detection, one right-censored placeholder give a detection
// probability of 2/(2+3); the finite detection ranks before the right-censored placeholder.
static void test_plotting_positions_aggregate_threshold_counts_affect_ranks() {
    DataFrame df;
    df.exact_series().add(ExactData(2, 150.0));
    ThresholdData threshold(0, 4, 100.0);
    threshold.set_number_above(1);
    df.threshold_series().add(threshold);

    df.calculate_plotting_positions();

    CHECK_NEAR(df.exact_series()[0].plotting_position(), 4.0 / 15.0, 1E-12);
    CHECK_EQ(df.threshold_series()[0].number_below(), 3);
    CHECK_EQ(df.threshold_series()[0].number_above(), 1);
}

// C# Test_PlottingPositions_OutsideThresholdWindow_IsDetected: an observation outside
// every perception window is detected regardless of magnitude vs. an unrelated finite
// threshold (the synthetic negative-infinity threshold ARRANGE2 uses).
static void test_plotting_positions_outside_threshold_window_is_detected() {
    DataFrame df;
    df.exact_series().add(ExactData(10, 50.0));
    df.threshold_series().add(ThresholdData(0, 1, 100.0));

    df.calculate_plotting_positions();

    CHECK_NEAR(df.exact_series()[0].plotting_position(), 0.5, 1E-12);
}

// C# Test_PlottingPositions_InvalidInputs_Throw (the DataFrame-level half; the
// PlottingParameter setter's own eager validation is
// test_data_frame_set_plotting_parameter_rejects_out_of_range in test_data_frame.cpp).
static void test_plotting_positions_invalid_inputs_throw() {
    // Processed counts that cannot fit the threshold's own duration.
    DataFrame infeasible;
    ThresholdData infeasible_threshold(0, 0, 100.0);  // Duration 1
    infeasible_threshold.set_number_above(2);         // 2 > Duration: infeasible once processed
    infeasible.threshold_series().add(infeasible_threshold);
    CHECK_THROWS(infeasible.calculate_plotting_positions());

    // Overlapping threshold windows ([0, 2] and [2, 4] share index 2).
    DataFrame overlapping;
    overlapping.threshold_series().add(ThresholdData(0, 2, 100.0));
    overlapping.threshold_series().add(ThresholdData(2, 4, 200.0));
    CHECK_THROWS(overlapping.calculate_plotting_positions());
}

// C# Test_PlottingPositions_Example5BootstrapEdge_RemainsStrictWithoutChangingSample:
// verifies a bootstrap-derived arranged-count edge case (the former K=43/K=6 condition
// that made the prior recurrence calculate Q=1 at the 743-cfs level) keeps every
// resampled value unchanged and assigns only strict open-interval plotting positions.
// Two observations fall below their own perception thresholds and are classified as
// censored only for plotting; they remain exact observations with their original values.
static void test_plotting_positions_example5_bootstrap_edge_remains_strict() {
    std::vector<double> values(50);
    for (int i = 0; i < 8; i++)
        values[static_cast<std::size_t>(i)] = (i == 7) ? 1000.0 : 2000.0 + i;
    for (int i = 8; i < 44; i++) values[static_cast<std::size_t>(i)] = 3000.0 + i;
    for (int i = 44; i < 49; i++)
        values[static_cast<std::size_t>(i)] = 800.0 + (10.0 * (i - 44));
    values[49] = 500.0;

    DataFrame df;
    for (std::size_t i = 0; i < values.size(); i++)
        df.exact_series().add(ExactData(1965 + static_cast<int>(i), values[i]));

    df.threshold_series().add(ThresholdData(1965, 1972, 1180.0));
    df.threshold_series().add(ThresholdData(1973, 1991, 705.0));
    df.threshold_series().add(ThresholdData(1992, 2001, 714.0));
    df.threshold_series().add(ThresholdData(2002, 2002, 743.0));
    df.threshold_series().add(ThresholdData(2003, 2003, 560.0));
    df.threshold_series().add(ThresholdData(2004, 2005, 700.0));
    df.threshold_series().add(ThresholdData(2006, 2009, 710.0));
    df.threshold_series().add(ThresholdData(2010, 2012, 661.0));
    df.threshold_series().add(ThresholdData(2013, 2014, 700.0));

    std::vector<double> original_values(df.exact_series().count());
    for (std::size_t i = 0; i < df.exact_series().count(); i++)
        original_values[i] = df.exact_series()[i].value();

    df.set_plotting_parameter(0.4);
    df.calculate_plotting_positions();

    for (std::size_t i = 0; i < df.exact_series().count(); i++)
        CHECK_NEAR(df.exact_series()[i].value(), original_values[i], 0.0);
    CHECK_EQ(df.exact_series().count(), std::size_t{50});

    bool any_above_0_98 = false;
    for (std::size_t i = 0; i < df.exact_series().count(); i++) {
        double position = df.exact_series()[i].plotting_position();
        CHECK_TRUE(std::isfinite(position) && position > 0.0 && position < 1.0);
        if (position > 0.98) any_above_0_98 = true;
    }
    CHECK_TRUE(any_above_0_98);
}

// C# Test_PlottingPositions_Example5BootstrapSample_TiesAreSeparated: a frozen bootstrap
// sample that previously assigned the EXACT same probability to the events at indexes
// 1975 and 1998 -- the EnsureDistinctPlottingPositions regression this rewrite adds.
static void test_plotting_positions_example5_bootstrap_sample_ties_are_separated() {
    const std::vector<int> years = {
        1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974,
        1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984,
        1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994,
        1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
        2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014};
    const std::vector<double> values = {
        1398.5976334510967, 966.59552180918161,  1974.3223495966256, 2891.8338540508757,
        2505.9334402560103, 1885.0116761582647,  1680.6543191030453, 3847.0027056032231,
        930.55140459080849, 2460.2531376046609,  564.28401727153459, 1396.8182016011826,
        1591.5650583425017, 4578.067551996458,   2582.6599531978427, 2497.8881953825239,
        1491.8593917890207, 2380.1796316909777,  2654.2379192321528, 2646.1646459149301,
        3905.9886563585524, 899.41749596844386,  1138.9218977761075, 1753.1788985139399,
        2995.6252828849888, 1166.0571810678148,  2978.705067413357,  1557.4501735296865,
        3102.3195776956154, 3424.0748502645079,  2213.4043771087458, 2695.8435706029095,
        2637.9755363911408, 429.59797890129613,  4654.4548499288721, 3030.215825029497,
        3272.014817282608,  763.26237263475105,  2099.7317325275308, 2128.9063276074667,
        1465.0224395732935, 4738.3332968838067,  2611.3950878331016, 1765.2295530597592,
        1889.0450114112809, 2561.4319968329551,  2567.6539372706479, 1989.7098547510266,
        1387.028035261912,  1679.0903337032385};

    DataFrame source;
    source.set_low_outlier_threshold(1200.0);
    source.set_plotting_parameter(0.4);
    for (std::size_t i = 0; i < years.size(); i++)
        source.exact_series().add(ExactData(years[i], values[i]));

    source.threshold_series().add(ThresholdData(1965, 1972, 1180.0));
    source.threshold_series().add(ThresholdData(1973, 1991, 705.0));
    source.threshold_series().add(ThresholdData(1992, 2001, 714.0));
    source.threshold_series().add(ThresholdData(2002, 2002, 743.0));
    source.threshold_series().add(ThresholdData(2003, 2003, 560.0));
    source.threshold_series().add(ThresholdData(2004, 2005, 700.0));
    source.threshold_series().add(ThresholdData(2006, 2009, 710.0));
    source.threshold_series().add(ThresholdData(2010, 2012, 661.0));
    source.threshold_series().add(ThresholdData(2013, 2014, 700.0));

    source.calculate_plotting_positions();

    std::vector<double> positions;
    positions.reserve(source.exact_series().count());
    for (std::size_t i = 0; i < source.exact_series().count(); i++)
        positions.push_back(source.exact_series()[i].plotting_position());
    std::sort(positions.begin(), positions.end());

    const int higher_value_index = 1975;
    const int lower_value_index = 1998;
    const double expected_center = 0.97714285714285709;

    const ExactData* higher_value_event = nullptr;
    const ExactData* lower_value_event = nullptr;
    for (std::size_t i = 0; i < source.exact_series().count(); i++) {
        if (source.exact_series()[i].index() == higher_value_index)
            higher_value_event = &source.exact_series()[i];
        if (source.exact_series()[i].index() == lower_value_index)
            lower_value_event = &source.exact_series()[i];
    }
    CHECK_TRUE(higher_value_event != nullptr);
    CHECK_TRUE(lower_value_event != nullptr);

    CHECK_TRUE(higher_value_event->value() > lower_value_event->value());
    CHECK_TRUE(higher_value_event->plotting_position() < lower_value_event->plotting_position());
    CHECK_NEAR((higher_value_event->plotting_position() + lower_value_event->plotting_position()) / 2.0,
              expected_center, 1E-15);

    for (std::size_t i = 1; i < positions.size(); i++) {
        CHECK_TRUE(!corehydro::numerics::utilities::almost_equals(positions[i - 1], positions[i]));
    }

    // Positions must remain distinct even at the app's six-decimal display precision.
    std::vector<double> rounded;
    rounded.reserve(positions.size());
    for (double position : positions) rounded.push_back(std::round(position * 1e6) / 1e6);
    std::sort(rounded.begin(), rounded.end());
    std::size_t distinct_count =
        static_cast<std::size_t>(std::unique(rounded.begin(), rounded.end()) - rounded.begin());
    CHECK_EQ(distinct_count, positions.size());
}

// No dedicated C# unit test (PlottingPositionVersion has none either); confirms the
// monotonically increasing version counter Task 15's BivariateDistribution cache
// consumes bumps on every calculate_plotting_positions() call.
static void test_plotting_position_version_increments_on_calculate() {
    DataFrame df;
    df.exact_series().add(ExactData(0, 10.0));
    df.exact_series().add(ExactData(1, 20.0));

    std::int64_t before = df.plotting_position_version();
    df.calculate_plotting_positions();
    std::int64_t after_first = df.plotting_position_version();
    CHECK_TRUE(after_first > before);

    df.calculate_plotting_positions();
    CHECK_TRUE(df.plotting_position_version() > after_first);
}

int main() {
    test_plotting_positions_b17c_ex4();
    test_plotting_positions_b17c_ex7();
    test_blom();
    test_cunnane();
    test_gringorten();
    test_hazen();
    test_median();
    test_weibull();
    test_apply_langbein_conversion();
    test_plotting_positions_value_below_own_threshold_uses_censored_branch();
    test_plotting_positions_aggregate_threshold_counts_affect_ranks();
    test_plotting_positions_outside_threshold_window_is_detected();
    test_plotting_positions_invalid_inputs_throw();
    test_plotting_positions_example5_bootstrap_edge_remains_strict();
    test_plotting_positions_example5_bootstrap_sample_ties_are_separated();
    test_plotting_position_version_increments_on_calculate();
    return chtest::summary("plotting_positions_df");
}
