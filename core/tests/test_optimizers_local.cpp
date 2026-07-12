// Transcribed C# oracle tests for the local optimizers (Task B5 added BFGS; Task B6 added
// the Powell section; MLSL lives in test_optimizers_global.cpp):
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_BFGS.cs @ a2c4dbf
//   upstream/Numerics/Test_Numerics/Mathematics/Optimization/Local/Test_Powell.cs @ a2c4dbf
//
// All 8 upstream [TestMethod]s of each class are transcribed with their exact fitness +
// coordinate oracles and tolerances (1E-4 throughout, except the BFGS SumOfPowerFunctions
// coordinates at 1E-3), unaltered. These are internal-support ports validated against the
// C# test oracles themselves, so there is no fixtures/ entry for this file (fixtures/ is
// the public estimation API surface only). Skipped upstream methods: none (neither
// Test_BFGS.cs nor Test_Powell.cs has XML or INotifyPropertyChanged tests).
//
// SUPPLEMENT (clearly marked, not from Test_BFGS.cs): direct unit checks for the two
// Tools.cs functions this task ports -- sum_product (Tools.SumProduct, used by the BFGS
// strong-Wolfe line search) and normalized_distance (Tools.NormalizedDistance, whose only
// caller, MLSL, arrives in B6) -- against hand-computed values, since no upstream C# test
// exercises them directly at this layer.
#include <cmath>
#include <vector>

#include "corehydro/numerics/math/optimization/bfgs.hpp"
#include "corehydro/numerics/math/optimization/powell.hpp"
#include "corehydro/numerics/tools.hpp"
#include "check.hpp"
#include "optimization_test_functions.hpp"

using corehydro::numerics::math::optimization::BFGS;
using corehydro::numerics::math::optimization::Powell;

namespace {

// ============================== BFGS (Test_BFGS.cs) ==============================

// Test the BFGS algorithm with a simple 3-dimensional test function.
void bfgs_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = BFGS(test_functions::fxyz, 3, initial, lower, upper);
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

// Test the BFGS algorithm with the De Jong Function in 5-D.
void bfgs_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = BFGS(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the BFGS algorithm with the Sum of Power functions in 3-D.
void bfgs_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5, 0.5};
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = BFGS(test_functions::sum_of_power_functions, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-3);
}

// Test the BFGS algorithm with the Rosenbrock Function in 2-D.
void bfgs_rosenbrock() {
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048};
    auto solver = BFGS(test_functions::rosenbrock, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the BFGS algorithm with the Booth Function.
void bfgs_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = BFGS(test_functions::booth, 2, initial, lower, upper);
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

// Test the BFGS algorithm with the Matyas Function.
void bfgs_matyas() {
    std::vector<double> initial = {1.0, -1.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = BFGS(test_functions::matyas, 2, initial, lower, upper);
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

// Test the BFGS algorithm with the McCormick Function.
void bfgs_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = BFGS(test_functions::mccormick, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the BFGS algorithm with the Beale Function.
void bfgs_beale() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = BFGS(test_functions::beale, 2, initial, lower, upper);
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

// ============================== Powell (Test_Powell.cs) ==============================

// Test the Powell algorithm with a simple 3-dimensional test function.
void powell_fxyz() {
    std::vector<double> initial = {0.2, 0.5, 0.5};
    std::vector<double> lower = {0.0, 0.0, 0.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = Powell(test_functions::fxyz, 3, initial, lower, upper);
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

// Test the Powell algorithm with the De Jong Function in 5-D.
void powell_de_jong() {
    std::vector<double> initial = {1.0, -1.0, 2.0, -2.0, 1.0};
    std::vector<double> lower = {-5.12, -5.12, -5.12, -5.12, -5.12};
    std::vector<double> upper = {5.12, 5.12, 5.12, 5.12, 5.12};
    auto solver = Powell(test_functions::de_jong, 5, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Powell algorithm with the Sum of Power functions in 3-D.
void powell_sum_of_power_functions() {
    std::vector<double> initial = {0.5, -0.5, 0.5};
    std::vector<double> lower = {-1.0, -1.0, -1.0};
    std::vector<double> upper = {1.0, 1.0, 1.0};
    auto solver = Powell(test_functions::sum_of_power_functions, 3, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {0.0, 0.0, 0.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Powell algorithm with the Rosenbrock Function in 2-D.
void powell_rosenbrock() {
    std::vector<double> initial = {0, 0};
    std::vector<double> lower = {-2.048, -2.048};
    std::vector<double> upper = {2.048, 2.048};
    auto solver = Powell(test_functions::rosenbrock, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = 0.0;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    std::vector<double> valid = {1.0, 1.0};
    for (std::size_t i = 0; i < valid.size(); i++) CHECK_NEAR(solution[i], valid[i], 1E-4);
}

// Test the Powell algorithm with the Booth Function.
void powell_booth() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = Powell(test_functions::booth, 2, initial, lower, upper);
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

// Test the Powell algorithm with the Matyas Function.
void powell_matyas() {
    std::vector<double> initial = {1.0, -1.0};
    std::vector<double> lower = {-10.0, -10.0};
    std::vector<double> upper = {10.0, 10.0};
    auto solver = Powell(test_functions::matyas, 2, initial, lower, upper);
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

// Test the Powell algorithm with the McCormick Function.
void powell_mccormick() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-1.5, -3.0};
    std::vector<double> upper = {4.0, 4.0};
    auto solver = Powell(test_functions::mccormick, 2, initial, lower, upper);
    solver.minimize();
    double F = solver.best_parameter_set().fitness;
    double trueF = -1.9133;
    CHECK_NEAR(F, trueF, 1E-4);
    auto solution = solver.best_parameter_set().values;
    double x = solution[0];
    double y = solution[1];
    double validX = -0.54719;
    double validY = -1.54719;
    CHECK_NEAR(x, validX, 1E-4);
    CHECK_NEAR(y, validY, 1E-4);
}

// Test the Powell algorithm with the Beale Function.
void powell_beale() {
    std::vector<double> initial = {0.0, 0.0};
    std::vector<double> lower = {-4.5, -4.5};
    std::vector<double> upper = {4.5, 4.5};
    auto solver = Powell(test_functions::beale, 2, initial, lower, upper);
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

// ==================== SUPPLEMENT: Tools additions (B5, hand-computed) ====================
// Not from Test_BFGS.cs -- direct unit checks for the two Tools.cs functions B5 ports.

// Tools.SumProduct: dot product of two equal-length lists.
void tools_sum_product() {
    // (1*4) + (2*5) + (3*6) = 32
    CHECK_NEAR(corehydro::numerics::sum_product({1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}), 32.0, 1e-12);
    // Single element: (-2.5)*4 = -10
    CHECK_NEAR(corehydro::numerics::sum_product({-2.5}, {4.0}), -10.0, 1e-12);
    // Empty first list -> NaN (C# returns double.NaN).
    CHECK_TRUE(std::isnan(corehydro::numerics::sum_product({}, {})));
    // Mismatched lengths -> NaN (C# returns double.NaN).
    CHECK_TRUE(std::isnan(corehydro::numerics::sum_product({1.0, 2.0}, {1.0})));
}

// Tools.NormalizedDistance: Euclidean distance after min-max normalizing each dimension.
void tools_normalized_distance() {
    // Dimension 1: range 10, normalized delta (3-1)/10 = 0.2.
    // Dimension 2: range 4,  normalized delta (2-0)/4  = 0.5.
    // Distance = sqrt(0.2^2 + 0.5^2) = sqrt(0.29).
    CHECK_NEAR(corehydro::numerics::normalized_distance({1.0, 0.0}, {3.0, 2.0}, {0.0, -2.0},
                                                      {10.0, 2.0}),
               std::sqrt(0.29), 1e-12);
    // Identical points -> 0.
    CHECK_NEAR(corehydro::numerics::normalized_distance({0.5, 0.5}, {0.5, 0.5}, {0.0, 0.0},
                                                      {1.0, 1.0}),
               0.0, 1e-15);
    // Degenerate dimension (range <= 0) contributes nothing: only dim 2 counts,
    // delta (4-1)/10 = 0.3.
    CHECK_NEAR(corehydro::numerics::normalized_distance({1.0, 1.0}, {2.0, 4.0}, {5.0, 0.0},
                                                      {5.0, 10.0}),
               0.3, 1e-12);
    // NaN range likewise contributes nothing -> all dims degenerate -> 0.
    CHECK_NEAR(corehydro::numerics::normalized_distance({1.0}, {2.0}, {0.0}, {std::nan("")}),
               0.0, 1e-15);
}

}  // namespace

int main() {
    // Test_BFGS.cs
    bfgs_fxyz();
    bfgs_de_jong();
    bfgs_sum_of_power_functions();
    bfgs_rosenbrock();
    bfgs_booth();
    bfgs_matyas();
    bfgs_mccormick();
    bfgs_beale();
    // Test_Powell.cs
    powell_fxyz();
    powell_de_jong();
    powell_sum_of_power_functions();
    powell_rosenbrock();
    powell_booth();
    powell_matyas();
    powell_mccormick();
    powell_beale();
    // Supplement: Tools additions (B5)
    tools_sum_product();
    tools_normalized_distance();
    return chtest::summary("test_optimizers_local");
}
