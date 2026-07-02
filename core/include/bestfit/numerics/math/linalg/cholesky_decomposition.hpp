// ported from: Numerics/Mathematics/Linear Algebra/CholeskyDecomposition.cs @ a2c4dbf
//
// Cholesky decomposition A = L*L^T of a symmetric positive-definite matrix, ported
// verbatim (loop order and epsilon checks unchanged). Member order mirrors the C# source
// exactly: ctor, l(), a(), is_positive_definite(), solve(), backward(), forward(),
// inverse_a(), determinant(), log_determinant() -- note the C# source declares
// `Backward` before `Forward`, which differs from the task brief's prose order.
#pragma once
#include <cmath>
#include <stdexcept>

#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"

namespace bestfit::numerics::math::linalg {

class CholeskyDecomposition {
   public:
    // Constructs new Cholesky Decomposition. Throws std::invalid_argument if A is not
    // square, and std::runtime_error if A is not positive-definite.
    explicit CholeskyDecomposition(const Matrix& A)
        : n_(A.number_of_rows()), a_(A.to_array()), l_(A.to_array()) {
        if (A.number_of_columns() != A.number_of_rows())
            throw std::invalid_argument("The matrix A must be square.");

        // Decomposing a matrix into lower triangular form.
        for (int i = 0; i < n_; ++i) {
            for (int j = i; j < n_; ++j) {
                double sum = l_(i, j);
                for (int k = i - 1; k >= 0; --k) sum -= l_(i, k) * l_(j, k);  // Cholesky formula
                if (i == j) {
                    if (std::isnan(sum) || sum <= 0.0)
                        throw std::runtime_error(
                            "Cholesky Decomposition failed. The input matrix is not "
                            "positive-definite.");
                    l_(i, i) = std::sqrt(sum);
                } else {
                    l_(j, i) = sum / l_(i, i);  // Upper triangular entry (transposed into L)
                }
            }
        }

        // Making sure 0 entries for the upper triangular part.
        for (int i = 0; i < n_; ++i)
            for (int j = 0; j < i; ++j) l_(j, i) = 0.0;

        // Failure of the decomposition indicates that A is not positive-definite; reaching
        // here means it is.
        is_positive_definite_ = true;
    }

    // Stores the decomposition (lower triangular L, with A = L*L^T).
    const Matrix& l() const { return l_; }

    // Stores the input matrix A that was decomposed.
    const Matrix& a() const { return a_; }

    // Determines whether the input matrix A is positive definite.
    bool is_positive_definite() const { return is_positive_definite_; }

    // Solves the set of n linear equations A*x=b using the stored decomposition A=L*L^T.
    Vector solve(const Vector& b) const {
        if (b.length() != n_)
            throw std::invalid_argument(
                "The vector b must have the same number of rows as the matrix A.");
        Vector x(n_);
        for (int i = 0; i < n_; ++i) {
            double sum = b[i];
            for (int k = i - 1; k >= 0; --k) sum -= l_(i, k) * x[k];
            x[i] = sum / l_(i, i);
        }
        for (int i = n_ - 1; i >= 0; --i) {
            double sum = x[i];
            for (int k = i + 1; k < n_; ++k) sum -= l_(k, i) * x[k];
            x[i] = sum / l_(i, i);
        }
        return x;
    }

    // Solves the L^T * x = y equation with backward substitution.
    Vector backward(const Vector& y) const {
        if (y.length() != n_)
            throw std::invalid_argument(
                "The vector y must have the same number of rows as the matrix A.");
        Vector x(n_);
        for (int i = n_ - 1; i >= 0; --i) {
            double sum = y[i];
            for (int j = n_ - 1; j > i; --j) sum -= x[j] * l_(j, i);
            x[i] = sum / l_(i, i);
        }
        return x;
    }

    // Solves the L * y = b equation using forward substitution.
    Vector forward(const Vector& b) const {
        if (b.length() != n_)
            throw std::invalid_argument(
                "The vector b must have the same number of rows as the matrix A.");
        Vector y(n_);
        for (int i = 0; i < n_; ++i) {
            double sum = b[i];
            for (int j = 0; j < i; ++j) sum -= l_(i, j) * y[j];
            y[i] = sum / l_(i, i);
        }
        return y;
    }

    // Matrix inverse A^-1 using the stored Cholesky decomposition.
    Matrix inverse_a() const {
        Matrix Ainv(n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j <= i; ++j) {
                double sum = i == j ? 1.0 : 0.0;
                for (int k = i - 1; k >= j; --k) sum -= l_(i, k) * Ainv(j, k);
                Ainv(j, i) = sum / l_(i, i);
            }
        }
        for (int i = n_ - 1; i >= 0; --i) {
            for (int j = 0; j <= i; ++j) {
                double sum = i < j ? 0.0 : Ainv(j, i);
                for (int k = i + 1; k < n_; ++k) sum -= l_(k, i) * Ainv(j, k);
                Ainv(j, i) = sum / l_(i, i);
                Ainv(i, j) = Ainv(j, i);
            }
        }
        return Ainv;
    }

    // Using the stored Cholesky decomposition, returns the determinant of the matrix A.
    double determinant() const {
        double d = 1.0;
        for (int i = 0; i < n_; ++i) d *= l_(i, i);
        return std::pow(d, 2.0);
    }

    // Using the stored Cholesky decomposition, returns the logarithm of the determinant.
    double log_determinant() const {
        double sum = 0.0;
        for (int i = 0; i < n_; ++i) sum += std::log(l_(i, i));
        return 2.0 * sum;
    }

   private:
    int n_;  // Number of rows in A
    Matrix a_;
    Matrix l_;
    bool is_positive_definite_ = false;
};

}  // namespace bestfit::numerics::math::linalg
