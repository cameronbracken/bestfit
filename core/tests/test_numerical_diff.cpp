// Standalone tests for corehydro::estimation::NumericalDiff (Phase 4, Task T3; T13
// edge-hardening supplement).
//
// PART 1 (T3): no C# oracle literal exists for NumericalDiff at that point (upstream it
// was exercised only via the MLE/MAP estimator tests, which weren't ported yet), so this
// suite checks self-consistency against ANALYTIC derivatives of simple polynomials:
//   - ComputeHessian of quadratics with known constant Hessians.
//   - ComputePointwiseGradients of a linear-in-parameters vector function, where each
//     row's gradient is known exactly and the summed gradient must equal the gradient
//     of the summed function.
//   - A flat-spot case (a parameter the function does not depend on at all) verifying
//     the adaptive step-growth logic terminates with a finite (not NaN/Inf) result
//     rather than dividing by a zero effective step.
//
// PART 2 (T13, BestFit v2.0.0) transcribes upstream's brand-new
// `RMC.BestFit.Tests/ModelEstimation/NumericalDiffTests.cs` (introduced 0d6821d..7efa9d0,
// the same range that hardened the two adaptive step searches) verbatim -- values,
// tolerances, and evaluation-count caps unaltered. These exercise the bounded-attempt
// search, the last-finite-step fallback, the no-finite-central-stencil throw, and the
// Jacobian one-sided fallback that PART 1 predates.
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/estimation/numerical_diff.hpp"
#include "corehydro/estimation/optimization_method.hpp"
#include "check.hpp"

using corehydro::estimation::NumericalDiff;
using corehydro::estimation::OptimizationMethod;

namespace {

// --- OptimizationMethod: 6 values, C# order preserved -----------------------

void test_optimization_method_enum() {
    CHECK_EQ(static_cast<int>(OptimizationMethod::Brent), 0);
    CHECK_EQ(static_cast<int>(OptimizationMethod::BFGS), 1);
    CHECK_EQ(static_cast<int>(OptimizationMethod::NelderMead), 2);
    CHECK_EQ(static_cast<int>(OptimizationMethod::Powell), 3);
    CHECK_EQ(static_cast<int>(OptimizationMethod::DifferentialEvolution), 4);
    CHECK_EQ(static_cast<int>(OptimizationMethod::MultilevelSingleLinkage), 5);
}

// --- ComputeHessian: quadratics with known constant Hessians ----------------

// f(x0,x1) = x0^2 + 3*x1^2 + x0*x1  =>  Hessian = [[2,1],[1,6]] everywhere.
void test_hessian_coupled_quadratic() {
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) {
        return x[0] * x[0] + 3.0 * x[1] * x[1] + x[0] * x[1];
    };

    for (const auto& point : std::vector<std::vector<double>>{{0.0, 0.0}, {1.5, -2.3}, {10.0, 5.0}}) {
        auto hessian = NumericalDiff::compute_hessian(f, point, 2);
        CHECK_NEAR(hessian(0, 0), 2.0, 1e-4);
        CHECK_NEAR(hessian(0, 1), 1.0, 1e-4);
        CHECK_NEAR(hessian(1, 0), 1.0, 1e-4);
        CHECK_NEAR(hessian(1, 1), 6.0, 1e-4);
    }
}

// f(x0,x1) = 2*x0^2 + 5*x1^2  =>  Hessian = [[4,0],[0,10]] everywhere.
void test_hessian_diagonal_quadratic() {
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) {
        return 2.0 * x[0] * x[0] + 5.0 * x[1] * x[1];
    };

    for (const auto& point : std::vector<std::vector<double>>{{0.0, 0.0}, {3.0, -1.0}, {-4.5, 7.2}}) {
        auto hessian = NumericalDiff::compute_hessian(f, point, 2);
        CHECK_NEAR(hessian(0, 0), 4.0, 1e-4);
        CHECK_NEAR(hessian(0, 1), 0.0, 1e-4);
        CHECK_NEAR(hessian(1, 0), 0.0, 1e-4);
        CHECK_NEAR(hessian(1, 1), 10.0, 1e-4);
    }
}

// f(x0,x1) = x0^2  (constant in x1) -- the x1 diagonal entry is a genuine flat spot:
// the adaptive step must grow all the way to MaxStep and still return a finite value
// (0), never NaN/Inf from a degenerate 0/0.
void test_hessian_flat_spot_is_finite() {
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& x) {
        return x[0] * x[0];
    };
    std::vector<double> point{2.0, 0.5};
    auto hessian = NumericalDiff::compute_hessian(f, point, 2);

    CHECK_TRUE(std::isfinite(hessian(0, 0)));
    CHECK_TRUE(std::isfinite(hessian(0, 1)));
    CHECK_TRUE(std::isfinite(hessian(1, 0)));
    CHECK_TRUE(std::isfinite(hessian(1, 1)));
    CHECK_NEAR(hessian(0, 0), 2.0, 1e-4);
    CHECK_NEAR(hessian(1, 1), 0.0, 1e-6);
}

// --- ComputePointwiseGradients: linear-in-parameters vector function --------

// g_i(x0,x1) = a_i*x0 + b_i*x1  =>  row gradient [a_i, b_i] exactly, for every i.
void test_pointwise_gradients_linear_rows() {
    const std::vector<double> a{1.0, -2.0, 0.5};
    const std::vector<double> b{4.0, 3.0, -1.5};
    const int n = 3;
    const int p = 2;

    std::function<std::vector<double>(const std::vector<double>&)> g =
        [&](const std::vector<double>& x) {
            std::vector<double> out(static_cast<std::size_t>(n));
            for (int i = 0; i < n; ++i)
                out[static_cast<std::size_t>(i)] = a[static_cast<std::size_t>(i)] * x[0] +
                                                    b[static_cast<std::size_t>(i)] * x[1];
            return out;
        };

    std::vector<double> point{1.2, -0.7};
    auto gradients = NumericalDiff::compute_pointwise_gradients(g, point, n, p);

    CHECK_EQ(static_cast<int>(gradients.size()), n);
    double sum_a = 0.0, sum_b = 0.0;
    for (int i = 0; i < n; ++i) {
        CHECK_NEAR(gradients[static_cast<std::size_t>(i)][0], a[static_cast<std::size_t>(i)], 1e-6);
        CHECK_NEAR(gradients[static_cast<std::size_t>(i)][1], b[static_cast<std::size_t>(i)], 1e-6);
        sum_a += a[static_cast<std::size_t>(i)];
        sum_b += b[static_cast<std::size_t>(i)];
    }

    // Summed gradient must equal the gradient of the summed (total) function.
    double total_grad_0 = 0.0, total_grad_1 = 0.0;
    for (int i = 0; i < n; ++i) {
        total_grad_0 += gradients[static_cast<std::size_t>(i)][0];
        total_grad_1 += gradients[static_cast<std::size_t>(i)][1];
    }
    CHECK_NEAR(total_grad_0, sum_a, 1e-6);
    CHECK_NEAR(total_grad_1, sum_b, 1e-6);
}

// One row is constant in x1 (a genuine flat spot for that column/row combination):
// the result must still be finite, never NaN/Inf.
void test_pointwise_gradients_flat_spot_is_finite() {
    const int n = 2;
    const int p = 2;
    std::function<std::vector<double>(const std::vector<double>&)> g =
        [](const std::vector<double>& x) {
            return std::vector<double>{x[0], x[0] + 2.0 * x[1]};
        };
    std::vector<double> point{1.0, -3.0};
    auto gradients = NumericalDiff::compute_pointwise_gradients(g, point, n, p);

    for (int i = 0; i < n; ++i)
        for (int j = 0; j < p; ++j)
            CHECK_TRUE(std::isfinite(gradients[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]));

    // row 0: g_0 = x0            -> gradient [1, 0]
    // row 1: g_1 = x0 + 2*x1     -> gradient [1, 2]
    CHECK_NEAR(gradients[0][0], 1.0, 1e-6);
    CHECK_NEAR(gradients[0][1], 0.0, 1e-6);
    CHECK_NEAR(gradients[1][0], 1.0, 1e-6);
    CHECK_NEAR(gradients[1][1], 2.0, 1e-6);
}

// ======================================================================================
// PART 2 -- upstream NumericalDiffTests.cs transcription (T13, BestFit v2.0.0)
// ======================================================================================

// upstream ComputeHessian_SmoothQuadratic_ReturnsExpectedCurvature: interior smooth
// values are IDENTICAL to the pre-T13 search (see numerical_diff.hpp's header), so this
// is the same style of check as test_hessian_coupled_quadratic above, transcribed at the
// upstream test's own point/tolerance.
void test_upstream_hessian_smooth_quadratic() {
    std::vector<double> parameters{1.2, -0.7};
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& theta) {
        return 3.0 * theta[0] * theta[0] + 2.0 * theta[1] * theta[1] + 5.0 * theta[0] * theta[1];
    };

    auto hessian = NumericalDiff::compute_hessian(f, parameters, 2);

    CHECK_NEAR(hessian(0, 0), 6.0, 1e-6);
    CHECK_NEAR(hessian(1, 1), 4.0, 1e-6);
    CHECK_NEAR(hessian(0, 1), 5.0, 1e-6);
    CHECK_NEAR(hessian(1, 0), 5.0, 1e-6);
}

// upstream ComputeHessian_FlatFiniteStepThenNonFiniteLargerStep_DoesNotCycle: a flat but
// FINITE value inside |theta0| <= 2.5e-4, NaN outside -- the initial step (1e-4) is
// finite-but-flat, so the search grows (x4 each time: 1e-4 -> 4e-4 -> 1.6e-3, which
// exceeds 2.5e-4 and returns non-finite), and must fall back to the LAST finite stencil
// (step 1e-4, value 7.0 on both sides -> second derivative exactly 0) rather than cycling
// forever between the finite flat step and the non-finite larger one.
void test_upstream_hessian_flat_then_nonfinite_does_not_cycle() {
    int evaluations = 0;
    std::function<double(const std::vector<double>&)> f = [&](const std::vector<double>& theta) {
        ++evaluations;
        return std::fabs(theta[0]) <= 2.5e-4 ? 7.0 : std::numeric_limits<double>::quiet_NaN();
    };

    auto hessian = NumericalDiff::compute_hessian(f, {0.0}, 1);

    CHECK_NEAR(hessian(0, 0), 0.0, 1e-10);
    CHECK_TRUE(evaluations <= 8);
}

// upstream ComputeHessian_InitialStepNonFinite_ShrinksToFiniteStep: NaN outside
// |theta0| <= 7.5e-5 forces the initial 1e-4 step to shrink (x0.5 retries) until it lands
// inside the finite window, where f = theta0^2 has an exact Hessian of 2.
void test_upstream_hessian_initial_step_nonfinite_shrinks_to_finite() {
    std::function<double(const std::vector<double>&)> f = [](const std::vector<double>& theta) {
        return std::fabs(theta[0]) <= 7.5e-5 ? theta[0] * theta[0]
                                             : std::numeric_limits<double>::quiet_NaN();
    };

    auto hessian = NumericalDiff::compute_hessian(f, {0.0}, 1);

    CHECK_NEAR(hessian(0, 0), 2.0, 1e-8);
}

// upstream ComputeHessian_NoFiniteCentralStencil_ThrowsInvalidOperationException: f is
// non-finite everywhere except the exact origin, so NO step (however small) ever produces
// a finite forward/backward pair -- the bounded search must throw quickly (T13's
// std::runtime_error, C# InvalidOperationException) rather than retry forever.
void test_upstream_hessian_no_finite_central_stencil_throws() {
    int evaluations = 0;
    std::function<double(const std::vector<double>&)> f = [&](const std::vector<double>& theta) {
        ++evaluations;
        return theta[0] == 0.0 ? 0.0 : std::numeric_limits<double>::quiet_NaN();
    };

    bool threw = false;
    std::string message;
    try {
        NumericalDiff::compute_hessian(f, {0.0}, 1);
    } catch (const std::runtime_error& ex) {
        threw = true;
        message = ex.what();
    }
    CHECK_TRUE(threw);
    CHECK_TRUE(message.find("no finite central stencil") != std::string::npos);
    CHECK_TRUE(evaluations < 100);
}

// upstream ComputeJacobian_InitialStepExceedsMaxStep_UsesCappedStep: at theta =
// (1000, -500), the raw relative initial step (1e-4 * (|theta|+1)) is far above MaxStep
// (1e-2), so T13's `std::min(normalize_step(initial_step(...)), kMaxStep)` caps it before
// the first evaluation; the linear g still differentiates exactly at the capped step.
void test_upstream_jacobian_initial_step_exceeds_max_step_uses_capped_step() {
    std::function<std::vector<double>(const std::vector<double>&)> g =
        [](const std::vector<double>& theta) {
            return std::vector<double>{2.0 * theta[0] - 3.0 * theta[1], theta[0] + 4.0 * theta[1]};
        };

    auto jacobian = NumericalDiff::compute_jacobian(g, {1000.0, -500.0}, 2);

    CHECK_NEAR(jacobian[0][0], 2.0, 1e-10);
    CHECK_NEAR(jacobian[0][1], -3.0, 1e-10);
    CHECK_NEAR(jacobian[1][0], 1.0, 1e-10);
    CHECK_NEAR(jacobian[1][1], 4.0, 1e-10);
}

// upstream ComputeJacobian_ParameterAtLowerBound_UsesOneSidedStep: parameter pinned
// exactly at its lower bound leaves zero room on the left, forcing the (pre-existing)
// "room_right only" one-sided forward difference; expected value derived the same way
// the upstream test derives it, off the public initial_step().
void test_upstream_jacobian_parameter_at_lower_bound_uses_one_sided_step() {
    double parameter = 2.0;
    std::function<std::vector<double>(const std::vector<double>&)> g =
        [](const std::vector<double>& theta) { return std::vector<double>{theta[0] * theta[0]}; };

    auto jacobian =
        NumericalDiff::compute_jacobian(g, {parameter}, 1, {parameter}, {10.0});

    double expected_forward_difference = 2.0 * parameter + NumericalDiff::initial_step(parameter);
    CHECK_NEAR(jacobian[0][0], expected_forward_difference, 1e-10);
}

// upstream ComputeJacobian_OneCentralSideNonFinite_UsesFiniteOneSidedStep: exercises the
// NEW T13 one-sided fallback (distinct from the bound-driven one-sided branch above) --
// both sides have room, but g is only defined for theta0 >= 0, so the backward evaluation
// at theta0 - h is non-finite while the base value and the forward evaluation are both
// finite; the column must fall back to the one-sided FORWARD difference rather than
// reporting failure.
void test_upstream_jacobian_one_central_side_nonfinite_uses_finite_one_sided_step() {
    std::function<std::vector<double>(const std::vector<double>&)> g =
        [](const std::vector<double>& theta) {
            return theta[0] < 0.0 ? std::vector<double>{std::numeric_limits<double>::quiet_NaN()}
                                  : std::vector<double>{theta[0] * theta[0]};
        };

    auto jacobian = NumericalDiff::compute_jacobian(g, {0.0}, 1);

    CHECK_NEAR(jacobian[0][0], NumericalDiff::initial_step(0.0), 1e-12);
}

}  // namespace

int main() {
    test_optimization_method_enum();

    test_hessian_coupled_quadratic();
    test_hessian_diagonal_quadratic();
    test_hessian_flat_spot_is_finite();

    test_pointwise_gradients_linear_rows();
    test_pointwise_gradients_flat_spot_is_finite();

    test_upstream_hessian_smooth_quadratic();
    test_upstream_hessian_flat_then_nonfinite_does_not_cycle();
    test_upstream_hessian_initial_step_nonfinite_shrinks_to_finite();
    test_upstream_hessian_no_finite_central_stencil_throws();
    test_upstream_jacobian_initial_step_exceeds_max_step_uses_capped_step();
    test_upstream_jacobian_parameter_at_lower_bound_uses_one_sided_step();
    test_upstream_jacobian_one_central_side_nonfinite_uses_finite_one_sided_step();

    return chtest::summary("numerical_diff");
}
