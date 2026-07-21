// P0/v2.1.4 support ctest (C++-only): BrentSearch's Bracket() bracketing helper. Like
// Powell/BFGS (test_optimizers_local.cpp), BrentSearch is internal support (not a public-
// API distribution/model), so hardcoded oracles transcribed from the upstream C# test are
// correct here -- public-API oracle values stay in fixtures/ only. No caller currently
// reachable from fixtures/ hits any of the failure paths exercised below (see
// brent_search.hpp's file header), so there is no fixtures/ entry for this file, matching
// the box_cox.hpp/test_optimizers_local.cpp precedent for internal-support ports.
//
// Oracles transcribed VALUES-UNALTERED from
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_BrentSearch.cs
//   @ 2a0357a (upstream-sync Task 11, 651035e "Harden MVNDST status and Brent bracketing"):
//   Test_Bracket_GeometricExpansionRight, Test_Bracket_GeometricExpansionLeft,
//   Test_Bracket_UsesCustomExpansionFactor, Test_Bracket_DistantMinimumReducesEvaluations,
//   Test_Bracket_PlateauTerminates, Test_Bracket_MonotoneObjectiveReachesIterationLimit,
//   Test_Bracket_RejectsInvalidInputs, Test_Bracket_RejectsNonFiniteSearchState.
//
// Status-surface adaptation: this port's BrentSearch has NO OptimizationStatus/
// ReportFailure surface (see brent_search.hpp's header) -- every guard throws
// UNCONDITIONALLY, matching the C# DEFAULT (`ReportFailure = true`) observable behavior.
// The C# tests that exercise `ReportFailure = false` (non-throwing, `Status` queried
// instead) are adapted below to their throwing counterparts: each still asserts the core
// "leaves bounds unchanged on failure" contract by reading lower_bound()/upper_bound()
// after catching the exception. Test_Minimize/Test_Maximize/Test_DeJong (pre-existing,
// not part of this diff) are out of scope for this file.
#include <cmath>
#include <limits>
#include <vector>

#include "corehydro/numerics/math/optimization/brent_search.hpp"
#include "check.hpp"

using corehydro::numerics::math::optimization::BrentSearch;

namespace {

// Test_Bracket_GeometricExpansionRight: exact trial sequence when the minimum is uphill
// to the right of the starting interval.
void test_bracket_geometric_expansion_right() {
    std::vector<double> evaluations;
    BrentSearch solver(
        [&](double x) {
            evaluations.push_back(x);
            return (x - 10.0) * (x - 10.0);
        },
        0.0, 1.0);

    solver.bracket(1.0, 2.0);

    std::vector<double> expected = {0.0, 1.0, 2.0, 4.0, 8.0, 16.0};
    CHECK_EQ(evaluations.size(), expected.size());
    for (std::size_t i = 0; i < expected.size() && i < evaluations.size(); ++i)
        CHECK_NEAR(evaluations[i], expected[i], 0.0);
    CHECK_NEAR(solver.lower_bound(), 4.0, 0.0);
    CHECK_NEAR(solver.upper_bound(), 16.0, 0.0);
}

// Test_Bracket_GeometricExpansionLeft: a positive initial step reverses direction and
// expands geometrically when the minimum is to the left.
void test_bracket_geometric_expansion_left() {
    std::vector<double> evaluations;
    BrentSearch solver(
        [&](double x) {
            evaluations.push_back(x);
            return (x + 10.0) * (x + 10.0);
        },
        0.0, 1.0);

    solver.bracket(1.0, 2.0);

    std::vector<double> expected = {0.0, 1.0, -1.0, -3.0, -7.0, -15.0};
    CHECK_EQ(evaluations.size(), expected.size());
    for (std::size_t i = 0; i < expected.size() && i < evaluations.size(); ++i)
        CHECK_NEAR(evaluations[i], expected[i], 0.0);
    CHECK_NEAR(solver.lower_bound(), -15.0, 0.0);
    CHECK_NEAR(solver.upper_bound(), -3.0, 0.0);
}

// Test_Bracket_UsesCustomExpansionFactor: a caller-provided k controls the geometric
// trial sequence.
void test_bracket_uses_custom_expansion_factor() {
    std::vector<double> evaluations;
    BrentSearch solver(
        [&](double x) {
            evaluations.push_back(x);
            return (x - 10.0) * (x - 10.0);
        },
        0.0, 1.0);

    solver.bracket(1.0, 3.0);

    std::vector<double> expected = {0.0, 1.0, 2.0, 5.0, 14.0, 41.0};
    CHECK_EQ(evaluations.size(), expected.size());
    for (std::size_t i = 0; i < expected.size() && i < evaluations.size(); ++i)
        CHECK_NEAR(evaluations[i], expected[i], 0.0);
    CHECK_NEAR(solver.lower_bound(), 5.0, 0.0);
    CHECK_NEAR(solver.upper_bound(), 41.0, 0.0);
}

// Test_Bracket_DistantMinimumReducesEvaluations: geometric bracketing finds a distant
// minimum in logarithmic (not linear) objective work -- the upstream-measured regression
// this task ports is ~16 evaluations instead of 10,002 for a minimum this far away.
// Counts are a structural assertion (not an oracle value), per the task brief.
void test_bracket_distant_minimum_reduces_evaluations() {
    int evaluations = 0;
    BrentSearch solver(
        [&](double x) {
            ++evaluations;
            return (x - 1000.0) * (x - 1000.0);
        },
        0.0, 1.0);

    solver.bracket(0.1, 2.0);
    int bracket_evaluations = evaluations;

    CHECK_TRUE(bracket_evaluations <= 20);
    CHECK_TRUE(solver.lower_bound() <= 1000.0 && solver.upper_bound() >= 1000.0);

    solver.minimize();
    CHECK_NEAR(solver.best_parameter(), 1000.0, 1E-4);
    CHECK_NEAR(solver.best_fitness(), 0.0, 1E-8);
}

// Test_Bracket_PlateauTerminates: a non-strict bracket (fc >= fb) terminates immediately
// for a flat objective.
void test_bracket_plateau_terminates() {
    int evaluations = 0;
    BrentSearch solver(
        [&](double) {
            ++evaluations;
            return 1.0;
        },
        0.0, 1.0);

    solver.bracket(1.0, 2.0);

    CHECK_EQ(evaluations, 3);
    CHECK_NEAR(solver.lower_bound(), 0.0, 0.0);
    CHECK_NEAR(solver.upper_bound(), 2.0, 0.0);
}

// Test_Bracket_MonotoneObjectiveReachesIterationLimit (adapted, see file header): a
// monotone objective never turns back up, so the search is bounded by max_iterations
// and fails -- transactionally preserving the original bounds. The pre-Task-11 C++
// bracket() used an unconditional `while (true)` here (a live hang risk this task fixes),
// so this case is exactly the upstream-measured regression fix.
void test_bracket_monotone_objective_reaches_iteration_limit() {
    BrentSearch solver([](double x) { return -x; }, 0.0, 1.0);
    solver.max_iterations = 10;

    CHECK_THROWS(solver.bracket(1.0, 2.0));
    CHECK_NEAR(solver.lower_bound(), 0.0, 0.0);
    CHECK_NEAR(solver.upper_bound(), 1.0, 0.0);
}

// Test_Bracket_RejectsInvalidInputs: validates the public step/expansion-factor contract.
void test_bracket_rejects_invalid_inputs() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double inf = std::numeric_limits<double>::infinity();

    for (double invalid_step : {0.0, nan, -inf, inf}) {
        BrentSearch solver([](double x) { return x * x; }, 0.0, 1.0);
        CHECK_THROWS(solver.bracket(invalid_step, 2.0));
    }

    for (double invalid_expansion : {-1.0, 0.0, 1.0, nan, -inf, inf}) {
        BrentSearch solver([](double x) { return x * x; }, 0.0, 1.0);
        CHECK_THROWS(solver.bracket(1.0, invalid_expansion));
    }
}

// Test_Bracket_RejectsNonFiniteSearchState (adapted, see file header): deterministic
// failure for NaN objectives and coordinate overflow, without changing the original
// bounds.
void test_bracket_rejects_non_finite_search_state() {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const double max = std::numeric_limits<double>::max();

    BrentSearch nan_solver([nan](double) { return nan; }, 0.0, 1.0);
    CHECK_THROWS(nan_solver.bracket());
    CHECK_NEAR(nan_solver.lower_bound(), 0.0, 0.0);
    CHECK_NEAR(nan_solver.upper_bound(), 1.0, 0.0);

    BrentSearch overflow_solver([](double x) { return -x; }, max, max);
    CHECK_THROWS(overflow_solver.bracket(max, 2.0));
    CHECK_NEAR(overflow_solver.lower_bound(), max, 0.0);
    CHECK_NEAR(overflow_solver.upper_bound(), max, 0.0);
}

}  // namespace

int main() {
    test_bracket_geometric_expansion_right();
    test_bracket_geometric_expansion_left();
    test_bracket_uses_custom_expansion_factor();
    test_bracket_distant_minimum_reduces_evaluations();
    test_bracket_plateau_terminates();
    test_bracket_monotone_objective_reaches_iteration_limit();
    test_bracket_rejects_invalid_inputs();
    test_bracket_rejects_non_finite_search_state();
    return chtest::summary("brent_search");
}
