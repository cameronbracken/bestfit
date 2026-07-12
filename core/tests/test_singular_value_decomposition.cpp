// Transcribed from: upstream/Numerics/Test_Numerics/Mathematics/Linear Algebra/
// Test_SingularValueDecomp.cs (@ a2c4dbf) for the Phase 6 Task B9 SVD port (a discovered
// dependency of LinearRegression.FitSVD; see the B9 report). All 9 upstream methods are
// transcribed, values and tolerances unaltered: Test_SVDWithLU, Test_Rank, Test_Nullity,
// Test_Range, Test_Nullspace, Test_SolveVector, Test_SolveMatrix, Test_Decompose,
// Test_LogDeterminant, Test_LogPseudoDeterminant.
#include <cmath>
#include <vector>

#include "corehydro/numerics/math/linalg/lu_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/math/linalg/singular_value_decomposition.hpp"
#include "corehydro/numerics/math/linalg/vector.hpp"
#include "check.hpp"

using corehydro::numerics::math::linalg::LUDecomposition;
using corehydro::numerics::math::linalg::Matrix;
using corehydro::numerics::math::linalg::SingularValueDecomposition;
using corehydro::numerics::math::linalg::Vector;

namespace {

// C# Test_SVDWithLU.
void test_svd_with_lu() {
    Matrix a(3);
    a(0, 0) = 25.0; a(0, 1) = 15.0; a(0, 2) = -5.0;
    a(1, 0) = 15.0; a(1, 1) = 18.0; a(1, 2) = 0.0;
    a(2, 0) = -5.0; a(2, 1) = 0.0;  a(2, 2) = 11.0;
    LUDecomposition lu(a);
    SingularValueDecomposition svd(a);

    Vector b(std::vector<double>{6.0, -4.0, 27.0});
    Vector true_x = lu.solve(b);
    Vector x = svd.solve(b);
    for (int i = 0; i < x.length(); ++i) CHECK_NEAR(x[i], true_x[i], 0.0001);
}

// C# Test_Rank.
void test_rank() {
    Matrix a(2);
    a(0, 0) = 3; a(0, 1) = 0;
    a(1, 0) = 4; a(1, 1) = 5;
    SingularValueDecomposition svd(a);
    CHECK_EQ(svd.rank(), 2);
}

// C# Test_Nullity.
void test_nullity() {
    Matrix a(3);
    a(0, 0) = 10; a(0, 1) = 20; a(0, 2) = 10;
    a(1, 0) = 20; a(1, 1) = 40; a(1, 2) = 20;
    a(2, 0) = 30; a(2, 1) = 50; a(2, 2) = 0;
    SingularValueDecomposition svd(a);
    CHECK_EQ(svd.nullity(), 1);
}

// C# Test_Range. (The upstream method builds an unused 4x3 `ranB` scratch matrix; only the
// expected `true_range` values are asserted, so the scratch matrix is dropped here.)
void test_range() {
    Matrix a(2);
    a(0, 0) = 3; a(0, 1) = 0;
    a(1, 0) = 4; a(1, 1) = 5;
    SingularValueDecomposition svd(a);

    Matrix true_range(2, 2);
    true_range(0, 0) = 1.0 / std::sqrt(10.0);
    true_range(0, 1) = -3.0 / std::sqrt(10.0);
    true_range(1, 0) = 3.0 / std::sqrt(10.0);
    true_range(1, 1) = 1.0 / std::sqrt(10.0);

    Matrix range = svd.range();
    for (int i = 0; i < range.number_of_rows(); ++i)
        for (int j = 0; j < range.number_of_columns(); ++j)
            CHECK_NEAR(range(i, j), true_range(i, j), 0.0001);
}

// C# Test_Nullspace.
void test_nullspace() {
    Matrix b(2);
    b(0, 0) = 2; b(0, 1) = 0;
    b(1, 0) = 2; b(1, 1) = 0;
    SingularValueDecomposition svd(b);
    Matrix nullspace = svd.nullspace();

    const double b_null[2] = {0.0, 1.0};
    for (int i = 0; i < nullspace.number_of_rows(); ++i)
        CHECK_NEAR(nullspace(i, 0), b_null[i], 0.0001);
}

// C# Test_SolveVector.
void test_solve_vector() {
    Matrix a(3);
    a(0, 0) = 2; a(0, 1) = -3; a(0, 2) = 1;
    a(1, 0) = 1; a(1, 1) = -1; a(1, 2) = -2;
    a(2, 0) = 3; a(2, 1) = 1;  a(2, 2) = -1;
    SingularValueDecomposition svd(a);

    Vector b(std::vector<double>{7.0, -2.0, 0.0});
    const double true_x[3] = {1.0, -1.0, 2.0};
    Vector x = svd.solve(b);
    for (int i = 0; i < x.length(); ++i) CHECK_NEAR(x[i], true_x[i], 0.0001);
}

// C# Test_SolveMatrix.
void test_solve_matrix() {
    Matrix a(3);
    a(0, 0) = 3;  a(0, 1) = -2; a(0, 2) = 1;
    a(1, 0) = -2; a(1, 1) = 3;  a(1, 2) = 2;
    a(2, 0) = 1;  a(2, 1) = 2;  a(2, 2) = 2;
    SingularValueDecomposition svd(a);

    Matrix mat_b(3, 1);
    mat_b(0, 0) = 3; mat_b(1, 0) = -3; mat_b(2, 0) = 2;

    const double true_mat_x[3] = {2.0, 1.0, -1.0};
    Matrix mat_x = svd.solve(mat_b);
    for (int i = 0; i < mat_x.number_of_rows(); ++i)
        CHECK_NEAR(mat_x(i, 0), true_mat_x[i], 0.0001);
}

// C# Test_Decompose.
void test_decompose() {
    Matrix a(2);
    a(0, 0) = 3; a(0, 1) = 0;
    a(1, 0) = 4; a(1, 1) = 5;
    SingularValueDecomposition svd(a);

    const double s10 = 1.0 / std::sqrt(10.0);
    const double s2 = 1.0 / std::sqrt(2.0);
    const double true_u[2][2] = {{1 * s10, -3 * s10}, {3 * s10, 1 * s10}};
    const double true_w[2] = {std::sqrt(45.0), std::sqrt(5.0)};
    const double true_v[2][2] = {{1 * s2, -1 * s2}, {1 * s2, 1 * s2}};

    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(true_u[i][j], svd.u()(i, j), 0.0001);
    for (int i = 0; i < 2; ++i) CHECK_NEAR(true_w[i], svd.w()[i], 0.0001);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) CHECK_NEAR(true_v[i][j], svd.v()(i, j), 0.0001);
}

// C# Test_LogDeterminant.
void test_log_determinant() {
    Matrix a(3);
    a(0, 0) = 16; a(0, 1) = 4;  a(0, 2) = 8;
    a(1, 0) = 4;  a(1, 1) = 5;  a(1, 2) = -4;
    a(2, 0) = 8;  a(2, 1) = -4; a(2, 2) = 22;
    SingularValueDecomposition svd(a);
    CHECK_NEAR(svd.log_determinant(), 6.356108, 0.0001);  // Math.Log(576)
}

// C# Test_LogPseudoDeterminant.
void test_log_pseudo_determinant() {
    Matrix a(3);
    a(0, 0) = 16; a(0, 1) = 4;  a(0, 2) = 8;
    a(1, 0) = 4;  a(1, 1) = 5;  a(1, 2) = -4;
    a(2, 0) = 8;  a(2, 1) = -4; a(2, 2) = 22;
    SingularValueDecomposition svd(a);
    CHECK_NEAR(svd.log_pseudo_determinant(), 6.356108, 0.0001);  // Math.Log(576)
}

}  // namespace

int main() {
    test_svd_with_lu();
    test_rank();
    test_nullity();
    test_range();
    test_nullspace();
    test_solve_vector();
    test_solve_matrix();
    test_decompose();
    test_log_determinant();
    test_log_pseudo_determinant();
    return chtest::summary("singular_value_decomposition");
}
