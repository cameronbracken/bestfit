// Standalone tests for corehydro::estimation::NumericalDiff (Phase 4, Task T3).
//
// No C# oracle literal exists for NumericalDiff (upstream it's exercised only via the
// MLE/MAP estimator tests, which aren't ported yet). Instead this suite checks
// self-consistency against ANALYTIC derivatives of simple polynomials:
//   - ComputeHessian of quadratics with known constant Hessians.
//   - ComputePointwiseGradients of a linear-in-parameters vector function, where each
//     row's gradient is known exactly and the summed gradient must equal the gradient
//     of the summed function.
//   - A flat-spot case (a parameter the function does not depend on at all) verifying
//     the adaptive step-growth logic terminates with a finite (not NaN/Inf) result
//     rather than dividing by a zero effective step.
#include <cmath>
#include <functional>
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

}  // namespace

int main() {
    test_optimization_method_enum();

    test_hessian_coupled_quadratic();
    test_hessian_diagonal_quadratic();
    test_hessian_flat_spot_is_finite();

    test_pointwise_gradients_linear_rows();
    test_pointwise_gradients_flat_spot_is_finite();

    return chtest::summary("numerical_diff");
}
