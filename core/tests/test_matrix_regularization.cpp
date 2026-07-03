// Standalone (non-fixture) tests for
// bestfit::numerics::math::linalg::MatrixRegularization::make_symmetric_positive_definite.
//
// There is no curated C# oracle literal for this method (see the task brief and the
// provenance header in matrix_regularization.hpp), so these tests assert the
// deterministic INVARIANTS the C# algorithm guarantees rather than a numeric literal:
// symmetry of the output, Cholesky-factorizability (the actual purpose of the method),
// and near-identity to an already-SPD input given the tiny trace-scaled ridge.
#include <cmath>

#include "bestfit/numerics/math/linalg/cholesky_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/matrix_regularization.hpp"
#include "check.hpp"

using bestfit::numerics::math::linalg::CholeskyDecomposition;
using bestfit::numerics::math::linalg::Matrix;
using bestfit::numerics::math::linalg::MatrixRegularization;

namespace {

bool is_cholesky_factorizable(const Matrix& m) {
    try {
        CholeskyDecomposition chol(m);
        return chol.is_positive_definite();
    } catch (const std::exception&) {
        return false;
    }
}

// Case 1: an already-symmetric-positive-definite input should come back essentially
// unchanged (only the tiny trace-scaled ridge added to the diagonal), symmetric, and
// Cholesky-factorizable.
void test_already_spd_input_returns_near_identical_result() {
    Matrix m(2, 2, {2.0, 0.3, 0.3, 1.0});
    Matrix out = MatrixRegularization::make_symmetric_positive_definite(m);

    CHECK_TRUE(is_cholesky_factorizable(out));
    CHECK_EQ(out(0, 1), out(1, 0));

    // baseRidge = 1e-10 * trace / p = 1e-10 * 3.0 / 2 = 1.5e-10; allow generous slack.
    CHECK_NEAR(out(0, 0), 2.0, 1e-6);
    CHECK_NEAR(out(1, 1), 1.0, 1e-6);
    CHECK_NEAR(out(0, 1), 0.3, 1e-9);
}

// Case 2a: a symmetric, near-singular/mildly-indefinite input (eigenvalues ~2.0005 and
// ~-0.0005 -- the kind of tiny negative eigenvalue finite-difference Hessian noise
// produces, which is what this method exists to repair) must come back
// Cholesky-factorizable (the core guarantee of the method) and symmetric.
//
// NOTE: the brief's suggested example [[1,2],[2,1]] (eigenvalues 3 and -1) was tried
// first and does NOT actually get repaired by the ported (== C#) algorithm: its 8-try
// ridge ladder tops out at baseRidge*1e7 == 1e-3*trace/p, and the last-resort ridge is
// only 1e-4*trace/p, both far too small to overcome an eigenvalue gap of 1 when trace/p
// is only 1. This is a faithful limitation of the C# source, not a porting bug (confirmed
// by hand-tracing MakeSymmetricPositiveDefinite's loop against this input) -- the method
// is designed to nudge near-PD covariance estimates over the line, not rescue matrices
// that are strongly indefinite relative to their trace. The milder near-singular input
// below is a case the algorithm actually guarantees to fix, matching the "near-singular"
// alternative the brief itself offered.
void test_near_singular_input_becomes_cholesky_factorizable() {
    Matrix m(2, 2, {1.0, 1.0005, 1.0005, 1.0});
    Matrix out = MatrixRegularization::make_symmetric_positive_definite(m);

    CHECK_TRUE(is_cholesky_factorizable(out));
    CHECK_EQ(out(0, 1), out(1, 0));
}

// Case 2b: a zero matrix (singular, trace == 0) must also come back Cholesky-factorizable.
void test_zero_matrix_becomes_cholesky_factorizable() {
    Matrix m(3, 3);
    Matrix out = MatrixRegularization::make_symmetric_positive_definite(m);

    CHECK_TRUE(is_cholesky_factorizable(out));
    for (int i = 0; i < out.number_of_rows(); ++i)
        for (int j = 0; j < out.number_of_columns(); ++j) CHECK_EQ(out(i, j), out(j, i));
}

// Case 3: a slightly non-symmetric input must return an EXACTLY symmetric result
// (S = 0.5*(M + Mtranspose) is exact regardless of the ridge search outcome).
void test_non_symmetric_input_returns_exactly_symmetric_result() {
    Matrix m(2, 2, {2.0, 0.31, 0.29, 1.0});
    Matrix out = MatrixRegularization::make_symmetric_positive_definite(m);

    CHECK_EQ(out(0, 1), out(1, 0));
    CHECK_TRUE(is_cholesky_factorizable(out));
}

}  // namespace

int main() {
    test_already_spd_input_returns_near_identical_result();
    test_near_singular_input_becomes_cholesky_factorizable();
    test_zero_matrix_becomes_cholesky_factorizable();
    test_non_symmetric_input_returns_exactly_symmetric_result();
    return bftest::summary("test_matrix_regularization");
}
