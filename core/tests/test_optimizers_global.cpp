// Transcribed C# oracle tests for the MLSL global optimizer (Task B6):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Global/Test_MLSL.cs @ a2c4dbf
//
// All 14 upstream [TestMethod]s are transcribed with their exact fitness + coordinate
// oracles and tolerances, unaltered -- including the loosened hard-multimodal ones
// (Eggholder uses SampleSize = 200 with fitness tolerance 0.5 and coordinate tolerance
// 1E-1; TP2 asserts an OR-match against either symmetric solution). MLSL is seeded
// (PRNGSeed = 12345 default), so every run is deterministic and the upstream oracles
// reproduce exactly. These are internal-support ports validated against the C# test
// oracles themselves, so there is no fixtures/ entry for this file (fixtures/ is the
// public estimation API surface only). Skipped upstream methods: none (Test_MLSL.cs has
// no XML or INotifyPropertyChanged tests).
#include <cmath>
#include <vector>

#include "bestfit/numerics/math/optimization/mlsl.hpp"
#include "check.hpp"
#include "optimization_test_functions.hpp"

using bestfit::numerics::math::optimization::MLSL;

namespace {

// ============================== MLSL (Test_MLSL.cs) ==============================

// Test the MLSL algorithm with a simple 3-dimensional test function.
void mlsl_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = MLSL(test_functions::fxyz, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double z = solution[2];
    double validX = 0.125;
    double validY = 0.2;
    double validZ = 0.35;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
    CHECK_NEAR(z, validZ, 1E-4);
}

// Test the MLSL algorithm with the De Jong Function in 5-D.
void mlsl_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = MLSL(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the MLSL algorithm with the Sum of Power functions in 3-D.
void mlsl_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5, 0.5};
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = MLSL(test_functions::sum_of_power_functions, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the MLSL algorithm with the Rosenbrock Function (5-D).
void mlsl_rosenbrock() {
    std::vector<double> initial = {0, 0, 0, 0, 0};
    std::vector<double> lower = {-2.048, -2.048, -2.048, -2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048, 2.048, 2.048, 2.048};
    auto solver = MLSL(test_functions::rosenbrock, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0, 1.0, 1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the MLSL algorithm with the Booth Function.
void mlsl_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = MLSL(test_functions::booth, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 1.0;
    double validY = 3.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the MLSL algorithm with the Matyas Function.
void mlsl_matyas() {
    std::vector<double> initial = {2.0, -2.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = MLSL(test_functions::matyas, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = 0.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the MLSL algorithm with the McCormick Function.
void mlsl_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = MLSL(test_functions::mccormick, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    CHECK_NEAR(x, validX, 1E-3);
    CHECK_NEAR(y, validY, 1E-3);
}

// Test the MLSL algorithm with the Beale Function.
void mlsl_beale() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = MLSL(test_functions::beale, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 3.0;
    double validY = 0.5;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the MLSL algorithm with the Goldstein-Price Function.
void mlsl_goldstein_price() {
    std::vector<double> initial = {-1.0, 1.0};
    std::vector<double> lower = {-2.0, -2.0};
    std::vector<double> upper = {2.0, 2.0};
    auto solver = MLSL(test_functions::goldstein_price, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 3.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = -1.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the MLSL algorithm with the Rastrigin Function.
void mlsl_rastrigin() {
    std::vector<double> initial = {1, 1, 1, 1, 1};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    // Need to run a lot of starts
    auto solver = MLSL(test_functions::rastrigin, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the MLSL algorithm with the Ackley Function.
void mlsl_ackley() {
    std::vector<double> initial = {1.0, 1.0};
    std::vector<double> lower = {-5.0, -5.0};
    std::vector<double> upper = {5.0, 5.0};
    auto solver = MLSL(test_functions::ackley, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = 0.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the MLSL algorithm with the three hump camel Function.
void mlsl_three_hump_camel() {
    std::vector<double> initial = {2.0, -2.0};
    std::vector<double> lower = {-5.0, -5.0};
    std::vector<double> upper = {5.0, 5.0};
    auto solver = MLSL(test_functions::three_hump_camel, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 0.0;
    double validY = 0.0;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the MLSL algorithm with the Eggholder Function.
void mlsl_eggholder() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-512.0, -512.0};
    std::vector<double> upper = {512.0, 512.0};
    // Need to run a lot of starts
    auto solver = MLSL(test_functions::eggholder, 2, initial, lower, upper);
    solver.sample_size = 200;
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -959.6407;
    CHECK_NEAR(F, trueF, 0.5);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 512.0;
    double validY = 404.2319;
    CHECK_NEAR(x, validX, 1E-1);
    CHECK_NEAR(y, validY, 1E-1);
}

// Test the MLSL algorithm with the tp2 Function.
void mlsl_tp2() {
    std::vector<double> initial = {2.0, 2.0};
    std::vector<double> lower = {0.0, 0.0};
    std::vector<double> upper = {2.0, 2.0};
    auto solver = MLSL(test_functions::tp2, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = 1.0;
    double validY = 0.666667;

    // The upstream test accepts either of the two symmetric solutions (OR-match).
    bool match1 = std::fabs(x - validX) < 1E-4 && std::fabs(y - validY) < 1E-4;
    bool match2 = std::fabs(x - validY) < 1E-4 && std::fabs(y - validX) < 1E-4;
    CHECK_TRUE(match1 || match2);
}

}  // namespace

int main() {
    // Test_MLSL.cs
    mlsl_fxyz();
    mlsl_de_jong();
    mlsl_sum_of_power_functions();
    mlsl_rosenbrock();
    mlsl_booth();
    mlsl_matyas();
    mlsl_mccormick();
    mlsl_beale();
    mlsl_goldstein_price();
    mlsl_rastrigin();
    mlsl_ackley();
    mlsl_three_hump_camel();
    mlsl_eggholder();
    mlsl_tp2();
    return bftest::summary("test_optimizers_global");
}
