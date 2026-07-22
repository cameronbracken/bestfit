// Transcribed from: upstream/RMC-BestFit/src/RMC.BestFit.Tests/Univariate/
// Bulletin17CDistributionTests.cs (@ c2e6192, v2.0.0's new "#region CloneWithDataFrame")
// for the upstream-sync Task 18 port of `Bulletin17CDistribution.CloneWithDataFrame`
// (0dc8594 "Preserve B17C model state on bootstrap clones") and the companion
// `SetDefaultParameters` ROS-trigger narrowing (c420d48 "Improving default parameter
// settings for the B17 distribution"). Split into its own file rather than appended to the
// already-973-line test_bulletin17c.cpp (the repo's file-size cap), mirroring the split
// already applied to the production header/companion pair.
//
// This is a C++-only ctest, not a JSON oracle fixture, following the T9
// (test_multivariate_normal_api.cpp) / T13 (test_gmm.cpp PART 3) precedent for stateful
// multi-object sequences that don't fit the model_estimation fixture shape (build ONE
// model, dispatch scalar assertions against it): these tests build a PARENT model, mutate
// its penalty state, clone it against a SECOND independent frame, and compare parent vs.
// clone state -- there is no single cached (model, estimator) pair for the JSON dispatcher
// to hang assertions off. The values under test are also not upstream-C#-computed
// quantities (no optimizer, no numerical integration): CloneWithDataFrame is a pure
// state-preservation contract, so a hand-verified structural transcription of the C# test
// literals is the correct oracle here, exactly like the precedent cases.
//
// Transcribed structural methods (values/tolerances unaltered):
//   CloneWithDataFrame_BindsSuppliedFrame_AndPreservesParameters (adapted -- see below),
//   CloneWithDataFrame_PreservesPenaltyConfiguration,
//   CloneWithDataFrame_ThenSetRandomPenaltyFunction_InstallsPenalty,
//   CloneWithDataFrame_ProcessesSuppliedFrameThresholds,
//   SetDefaultParameters_DuplicateLowOutlierSample_ComputesFiniteInitials (adapted).
//
// SKIPPED upstream methods (each with reason):
//   - CloneWithDataFrame_NullFrame_Throws: the C++ signature takes `const DataFrame&` (a
//     move-only VALUE type, the project-wide M4 DataFrame-port decision), so a null frame
//     is structurally unrepresentable -- same precedent as the ctors' skipped
//     Constructor_NullDataFrame_Throws (see test_bulletin17c.cpp's header).
//   - XElementConstructor_WithUnprocessedThresholdFrame_PreservesSerializedState: XML
//     (de)serialization is a project-wide deliberate skip; clone_with_data_frame's C++
//     equivalent is exercised directly by the tests above instead of via a round trip.
//
// Adaptations:
//   - Assert.AreSame(bootFrame, clone.DataFrame) / Assert.AreSame(parentFrame,
//     parent.DataFrame): reference identity has no analog for the value-typed DataFrame
//     (the clone deep-copies the supplied frame, per the header's documented deviation);
//     the tests instead assert the clone's observable frame state (record count / threshold
//     processing) and the parent's frame is untouched by checking its record count is
//     unaffected by the clone call.
//
// SUPPLEMENT (no upstream analog; discriminates the v2.0.0 ROS-trigger narrowing directly --
// a full GMM fit is not guaranteed to diverge from a different starting point, so the fixture
// surface cannot reliably prove the narrowing landed; see the header note above
// compute_default_initials() in bulletin17c_distribution.hpp):
//   - test_set_default_parameters_uncertain_only_skips_ros_override.
#include <cmath>
#include <memory>
#include <optional>
#include <vector>

#include "corehydro/models/data_frame/data_frame.hpp"
#include "corehydro/models/data_frame/data_types/threshold_data.hpp"
#include "corehydro/models/data_frame/data_types/uncertain_data.hpp"
#include "corehydro/models/univariate_distribution/bulletin17c_distribution.hpp"
#include "corehydro/numerics/distributions/log_normal.hpp"
#include "corehydro/numerics/distributions/normal.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"
#include "check.hpp"

using corehydro::models::Bulletin17CDistribution;
using corehydro::models::DataFrame;
using corehydro::models::ExactData;
using corehydro::models::ThresholdData;
using corehydro::models::UncertainData;
using corehydro::numerics::distributions::Normal;
using corehydro::numerics::distributions::UnivariateDistributionType;
using corehydro::numerics::sampling::MersenneTwister;

namespace {

constexpr int kFixtureSize = 50;

// C# CreateFloodDataFrame: a deterministic LP3-like flood sample (fixed RNG seed so the
// file does not depend on external TestData), indexed as simulated water years 1970-2019.
std::vector<double> inline_flood_data() {
    return corehydro::numerics::distributions::LogNormal(8.0, 0.4)
        .generate_random_values(kFixtureSize, 12345);
}

DataFrame create_flood_data_frame() {
    DataFrame df;
    std::vector<double> data = inline_flood_data();
    for (std::size_t i = 0; i < data.size(); ++i)
        df.exact_series().add(ExactData(1970 + static_cast<int>(i), data[i]));
    return df;
}

// C# CreateParentWithSkewPenalty: a fitted-like LP3 parent with an enabled regional-skew
// parameter penalty (index 2 = kappa/skew), mimicking the state before a bootstrap run.
// Returns by unique_ptr: Bulletin17CDistribution's move ctor/assignment are deleted (the
// penalty-function closures capture `this`; see the class header), so it cannot be
// returned by value.
std::unique_ptr<Bulletin17CDistribution> create_parent_with_skew_penalty(DataFrame df) {
    auto model = std::make_unique<Bulletin17CDistribution>(
        std::move(df), UnivariateDistributionType::LogPearsonTypeIII);
    model->set_parameter_values({3.3, 0.14, 0.4});
    model->parameter_penalties()[2].set_enabled(true);
    model->parameter_penalties()[2].set_mean(0.421);
    model->parameter_penalties()[2].set_mse(0.302);
    return model;
}

// C# CreateBootDataFrameWithThreshold: a second, independent frame standing in for a
// bootstrap resample, including an unprocessed historical threshold (years 1930-1940,
// non-overlapping with the flood frame's 1970-2019 exact range).
DataFrame create_boot_data_frame_with_threshold() {
    DataFrame df = create_flood_data_frame();
    ThresholdData threshold(1930, 1940, 25000.0);
    threshold.set_number_above(2);
    df.threshold_series().add(std::move(threshold));
    return df;
}

// ---------------------------------------------------------------------------------------
// CloneWithDataFrame
// ---------------------------------------------------------------------------------------

// C# CloneWithDataFrame_BindsSuppliedFrame_AndPreservesParameters (adapted: reference
// identity -> observable frame state; see the file header).
void test_clone_with_data_frame_binds_supplied_frame_and_preserves_parameters() {
    DataFrame parent_frame_copy_for_check = create_flood_data_frame();  // same content, own object
    std::unique_ptr<Bulletin17CDistribution> parent =
        create_parent_with_skew_penalty(create_flood_data_frame());
    int parent_record_length_before = parent->data_frame().total_record_length();
    DataFrame boot_frame = create_boot_data_frame_with_threshold();
    // total_record_length() needs process_threshold_series() first (see its own doc
    // comment); the clone gets this via clone_with_data_frame()'s internal reprocessing, so
    // reproduce it here too for an apples-to-apples comparison.
    boot_frame.process_threshold_series();
    int boot_frame_record_length = boot_frame.total_record_length();

    std::unique_ptr<Bulletin17CDistribution> clone = parent->clone_with_data_frame(boot_frame);

    // The clone is bound to (a deep copy of) the supplied frame -- its threshold series
    // carries over, the parent's frame never had one.
    CHECK_EQ(static_cast<int>(clone->data_frame().threshold_series().count()), 1);
    CHECK_EQ(static_cast<int>(parent->data_frame().threshold_series().count()), 0);
    CHECK_EQ(clone->data_frame().total_record_length(), boot_frame_record_length);
    // The parent must keep its original frame, untouched by the clone call.
    CHECK_EQ(parent->data_frame().total_record_length(), parent_record_length_before);
    CHECK_EQ(parent->data_frame().total_record_length(),
             parent_frame_copy_for_check.total_record_length());

    CHECK_EQ(clone->number_of_parameters(), parent->number_of_parameters());
    for (int i = 0; i < parent->number_of_parameters(); ++i) {
        std::size_t si = static_cast<std::size_t>(i);
        // Parameter i must survive the clone (no SetDefaultParameters rebuild).
        CHECK_NEAR(clone->parameters()[si].value(), parent->parameters()[si].value(), 1e-12);
        CHECK_NEAR(clone->parameters()[si].lower_bound(), parent->parameters()[si].lower_bound(),
                   1e-12);
        CHECK_NEAR(clone->parameters()[si].upper_bound(), parent->parameters()[si].upper_bound(),
                   1e-12);
    }
}

// C# CloneWithDataFrame_PreservesPenaltyConfiguration: the regression guarded here is the
// bootstrap penalty wipe, where assigning the boot frame through the public DataFrame
// setter rebuilt every parameter penalty with Enabled = false.
void test_clone_with_data_frame_preserves_penalty_configuration() {
    std::unique_ptr<Bulletin17CDistribution> parent =
        create_parent_with_skew_penalty(create_flood_data_frame());

    std::unique_ptr<Bulletin17CDistribution> clone =
        parent->clone_with_data_frame(create_boot_data_frame_with_threshold());

    CHECK_EQ(static_cast<int>(clone->parameter_penalties().size()),
             static_cast<int>(parent->parameter_penalties().size()));
    CHECK_TRUE(clone->parameter_penalties()[2].enabled());  // the enabled skew penalty survives
    CHECK_NEAR(clone->parameter_penalties()[2].mean(), 0.421, 1e-12);
    CHECK_NEAR(clone->parameter_penalties()[2].mse(), 0.302, 1e-12);
    CHECK_EQ(static_cast<int>(clone->quantile_penalties().size()),
             static_cast<int>(parent->quantile_penalties().size()));
}

// C# CloneWithDataFrame_ThenSetRandomPenaltyFunction_InstallsPenalty: after
// clone_with_data_frame, set_random_penalty_function sees the preserved enabled penalty and
// installs a non-null randomized penalty function, so bootstrap replicates propagate
// regional-skew prior uncertainty as designed.
void test_clone_with_data_frame_then_set_random_penalty_function_installs_penalty() {
    std::unique_ptr<Bulletin17CDistribution> parent =
        create_parent_with_skew_penalty(create_flood_data_frame());
    std::vector<double> theta_hat;
    for (const auto& p : parent->parameters()) theta_hat.push_back(p.value());

    std::unique_ptr<Bulletin17CDistribution> clone =
        parent->clone_with_data_frame(create_boot_data_frame_with_threshold());
    clone->set_parameter_values(theta_hat);
    MersenneTwister prng(1);
    clone->set_random_penalty_function(theta_hat, &prng);

    // The bootstrap penalty must be installed from the preserved penalty configuration.
    CHECK_TRUE(static_cast<bool>(clone->penalty_function()));
}

// C# CloneWithDataFrame_ProcessesSuppliedFrameThresholds: clone_with_data_frame processes
// the supplied frame's threshold series during the DataFrame assignment, so the clone's
// moment conditions see effective counts.
void test_clone_with_data_frame_processes_supplied_frame_thresholds() {
    std::unique_ptr<Bulletin17CDistribution> parent =
        create_parent_with_skew_penalty(create_flood_data_frame());
    DataFrame boot_frame = create_boot_data_frame_with_threshold();

    std::unique_ptr<Bulletin17CDistribution> clone = parent->clone_with_data_frame(boot_frame);

    CHECK_EQ(static_cast<int>(clone->data_frame().threshold_series().count()), 1);
    const ThresholdData& threshold = clone->data_frame().threshold_series()[0];
    CHECK_EQ(threshold.number_above(), 2);  // the user-supplied exceedance count is retained
    // ProcessThresholdSeries must derive NumberBelow = Duration (11) - NumberAbove (2) for a
    // non-overlapping threshold (1930-1940 vs. the flood frame's 1970-2019 exact range).
    CHECK_EQ(threshold.number_below(), 9);
}

// ---------------------------------------------------------------------------------------
// SUPPLEMENT: SetDefaultParameters_DuplicateLowOutlierSample_ComputesFiniteInitials
// ---------------------------------------------------------------------------------------

// Duplicate bootstrap-like values must not poison the ROS moment path used by B17C default
// parameter setup when low outliers are present (still true post-narrowing: low-outlier
// data keeps the ROS override).
void test_set_default_parameters_duplicate_low_outlier_sample_computes_finite_initials() {
    DataFrame df;
    df.set_exact_series(corehydro::models::ExactSeries(
        std::vector<double>{10.0, 100.0, 100.0, 125.0, 125.0, 150.0, 150.0, 200.0, 200.0, 250.0}));
    df.set_low_outlier_threshold(50.0);
    df.set_low_outliers_from_threshold();
    df.calculate_plotting_positions();

    std::optional<std::vector<double>> moments = df.get_nonparametric_moments_ros(true);

    CHECK_EQ(df.number_of_low_outliers(), 1);
    CHECK_TRUE(moments.has_value());
    for (double m : *moments) CHECK_TRUE(std::isfinite(m));

    Bulletin17CDistribution model(std::move(df), UnivariateDistributionType::LogPearsonTypeIII);
    model.set_default_parameters();

    CHECK_EQ(model.number_of_parameters(), 3);
    for (const auto& p : model.parameters()) CHECK_TRUE(std::isfinite(p.value()));
}

// ---------------------------------------------------------------------------------------
// SUPPLEMENT: v2.0.0 ROS-trigger narrowing discrimination (uncertain-only data)
// ---------------------------------------------------------------------------------------

// Pre-v2.0.0, ANY of {low outliers, uncertain data, interval data, threshold series} fired
// the nonparametric ROS override. v2.0.0 narrows this to {low outliers, threshold series}
// only, so a frame with ONLY uncertain data (no low outliers, no threshold) must now seed
// from the plain IMaximumLikelihoodEstimation constraint initials -- NOT the ROS moments.
// Independently recomputes both CANDIDATE initials and confirms (a) they genuinely differ
// (so the test discriminates old vs. new behavior) and (b) SetDefaultParameters picked the
// post-narrowing candidate.
void test_set_default_parameters_uncertain_only_skips_ros_override() {
    DataFrame df;
    std::vector<double> exact_values = {88.0, 95.0, 101.0, 104.0, 110.0, 118.0, 121.0, 130.0};
    for (std::size_t i = 0; i < exact_values.size(); ++i)
        df.exact_series().add(ExactData(static_cast<int>(i), exact_values[i]));
    df.uncertain_series().add(
        UncertainData(static_cast<int>(exact_values.size()), std::make_unique<Normal>(112.0, 6.0)));
    df.uncertain_series().add(UncertainData(static_cast<int>(exact_values.size()) + 1,
                                            std::make_unique<Normal>(97.0, 5.0)));

    CHECK_EQ(df.number_of_low_outliers(), 0);
    CHECK_EQ(static_cast<int>(df.threshold_series().count()), 0);
    CHECK_TRUE(df.uncertain_series().count() > 0);  // would have fired the pre-v2.0.0 trigger

    // get_nonparametric_moments_ros() (Candidate B below) reads each item's CACHED plotting
    // position; the model construction path below does NOT need this call for this frame
    // shape post-narrowing (see the file header), but the independent Candidate B
    // computation does -- same explicit-invalidation contract test_bulletin17c.cpp's header
    // documents.
    df.calculate_plotting_positions();

    // Candidate A: the plain MLE constraint initials (exact + uncertain point values).
    std::vector<double> data = exact_values;
    data.push_back(112.0);
    data.push_back(97.0);
    Normal probe;
    std::vector<double> mle_initials, mle_lowers, mle_uppers;
    probe.get_parameter_constraints(data, mle_initials, mle_lowers, mle_uppers);

    // Candidate B: the ROS nonparametric-moment override (the pre-v2.0.0 behavior).
    std::optional<std::vector<double>> np_moments = df.get_nonparametric_moments_ros(false);
    CHECK_TRUE(np_moments.has_value());
    std::vector<double> ros_initials = probe.parameters_from_moments(*np_moments);

    // Sanity: the two candidates actually differ, so this test discriminates.
    CHECK_TRUE(std::abs(mle_initials[0] - ros_initials[0]) > 1e-6 ||
               std::abs(mle_initials[1] - ros_initials[1]) > 1e-6);

    Bulletin17CDistribution model(std::move(df), UnivariateDistributionType::Normal);
    CHECK_EQ(model.number_of_parameters(), 2);
    // v2.0.0: uncertain-only data no longer overrides -- the plain MLE constraint initials
    // win.
    CHECK_NEAR(model.parameters()[0].value(), mle_initials[0], 1e-9);
    CHECK_NEAR(model.parameters()[1].value(), mle_initials[1], 1e-9);
}

}  // namespace

int main() {
    test_clone_with_data_frame_binds_supplied_frame_and_preserves_parameters();
    test_clone_with_data_frame_preserves_penalty_configuration();
    test_clone_with_data_frame_then_set_random_penalty_function_installs_penalty();
    test_clone_with_data_frame_processes_supplied_frame_thresholds();

    test_set_default_parameters_duplicate_low_outlier_sample_computes_finite_initials();
    test_set_default_parameters_uncertain_only_skips_ros_override();

    return chtest::summary("bulletin17c_clone_with_data_frame");
}
