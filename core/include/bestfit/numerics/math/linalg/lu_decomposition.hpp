// ported from: Numerics/Mathematics/Linear Algebra/LUDecomposition.cs @ a2c4dbf
//
// LU decomposition (Crout's algorithm with implicit partial pivoting -- Numerical
// Recipes-style outer-product Gaussian elimination) of a square matrix A into a
// row-permuted L*U, ported verbatim: the same k-i-j loop order, the same `vv` implicit
// scaling-factor precompute, the same TINY (1e-40) singular-pivot substitution, and the
// same permutation bookkeeping (`index_`, `d_` for the pivot-swap parity). Member order
// mirrors the C# source: ctor, `lu()`/`a()` accessors, `solve(Vector)`, `solve(Matrix)`,
// `inverse_a()`, `determinant()`.
//
// This header and matrix.hpp are mutually dependent: LUDecomposition's constructor takes
// a `Matrix`, and `Matrix::determinant()`/`Matrix::inverse()` (C# `Matrix.Determinant()`/
// `Matrix.Inverse()` parity) are implemented in terms of LUDecomposition. matrix.hpp
// declares those two methods but does not define them; this header supplies the
// out-of-line definitions at the bottom, once both classes are complete. Any caller of
// `Matrix::determinant()`/`Matrix::inverse()` must include this header (transitively or
// directly) for those definitions to be visible at link time.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/math/linalg/matrix.hpp"
#include "bestfit/numerics/math/linalg/vector.hpp"

namespace bestfit::numerics::math::linalg {

class LUDecomposition {
   public:
    // Constructs new LU Decomposition. The input matrix is not altered; a copy is made,
    // on which outer-product Gaussian elimination is then done in-place (mirrors the C#
    // ctor's remarks). Throws std::invalid_argument if a row is entirely zero (C#
    // `ArgumentException("Singular matrix in LU decomposition.")`).
    explicit LUDecomposition(const Matrix& A)
        : n_(A.number_of_rows()), a_(A.to_array()), lu_(A.to_array()), index_(static_cast<std::size_t>(n_)) {
        std::vector<double> vv(static_cast<std::size_t>(n_));
        constexpr double kTiny = 1.0e-40;
        int imax, i, j, k;
        double big, temp;
        d_ = 1.0;

        // Loop over rows to get the implicit scaling information.
        for (i = 0; i < n_; ++i) {
            big = 0.0;
            for (j = 0; j < n_; ++j) {
                temp = std::fabs(lu_(i, j));
                if (temp > big) big = temp;
            }
            if (big == 0.0) throw std::invalid_argument("Singular matrix in LU decomposition.");
            vv[static_cast<std::size_t>(i)] = 1.0 / big;
        }

        // This is the outermost k-i-j loop.
        for (k = 0; k < n_; ++k) {
            big = 0.0;
            imax = k;
            for (i = k; i < n_; ++i) {
                temp = vv[static_cast<std::size_t>(i)] * std::fabs(lu_(i, k));
                // Is the figure of merit for the pivot better than the best so far?
                if (temp > big) {
                    big = temp;
                    imax = i;
                }
            }

            if (k != imax) {
                // Do we need to interchange rows? Yes, do so...
                for (j = 0; j < n_; ++j) {
                    temp = lu_(imax, j);
                    lu_(imax, j) = lu_(k, j);
                    lu_(k, j) = temp;
                }
                d_ = -d_;
                vv[static_cast<std::size_t>(imax)] = vv[static_cast<std::size_t>(k)];
            }

            index_[static_cast<std::size_t>(k)] = imax;

            // If the pivot element is zero, the matrix is singular (at least to the
            // precision of the algorithm). For some applications on singular matrices, it
            // is desirable to substitute TINY for zero.
            if (lu_(k, k) == 0.0) lu_(k, k) = kTiny;

            // Perform Gaussian elimination step
            for (i = k + 1; i < n_; ++i) {
                lu_(i, k) /= lu_(k, k);  // True Lower triangular matrix
                for (j = k + 1; j < n_; ++j) lu_(i, j) -= lu_(i, k) * lu_(k, j);  // True Upper triangular matrix
            }
        }
    }

    // Stores the decomposition.
    const Matrix& lu() const { return lu_; }

    // Stores the input matrix A that was LU decomposed.
    const Matrix& a() const { return a_; }

    // Solves the set of n linear equations A*x=b using the stored LU decomposition of A.
    Vector solve(const Vector& b) const {
        if (b.length() != n_)
            throw std::invalid_argument(
                "The vector b must have the same number of rows as the matrix A.");
        int i, ii = 0, ip, j;
        double sum;
        Vector x(n_);
        for (i = 0; i < n_; ++i) x[i] = b[i];
        for (i = 0; i < n_; ++i) {
            ip = index_[static_cast<std::size_t>(i)];
            sum = x[ip];
            x[ip] = x[i];
            if (ii != 0)
                for (j = ii - 1; j < i; ++j) sum -= lu_(i, j) * x[j];
            else if (sum != 0.0)
                ii = i + 1;
            x[i] = sum;
        }
        for (i = n_ - 1; i >= 0; --i) {
            sum = x[i];
            for (j = i + 1; j < n_; ++j) sum -= lu_(i, j) * x[j];
            x[i] = sum / lu_(i, i);
        }
        return x;
    }

    // Solves m sets of n linear equations A*X=B using the stored LU decomposition of A.
    Matrix solve(const Matrix& B) const {
        if (B.number_of_rows() != n_)
            throw std::invalid_argument(
                "The vector b must have the same number of rows as the matrix A.");
        int m = B.number_of_columns();
        Matrix x(n_);
        for (int j = 0; j < m; ++j) {
            Vector xx(n_);
            for (int i = 0; i < n_; ++i) xx[i] = B(i, j);
            xx = solve(xx);
            for (int i = 0; i < n_; ++i) x(i, j) = xx[i];
        }
        return x;
    }

    // Returns the matrix inverse A^-1 using the stored LU decomposition.
    Matrix inverse_a() const {
        Matrix Ainv(n_);
        for (int i = 0; i < n_; ++i) {
            for (int j = 0; j < n_; ++j) Ainv(i, j) = 0.0;
            Ainv(i, i) = 1.0;
        }
        return solve(Ainv);
    }

    // Using the stored LU decomposition, returns the determinant of the matrix A.
    double determinant() const {
        double dd = d_;
        for (int i = 0; i < n_; ++i) dd *= lu_(i, i);
        return dd;
    }

   private:
    int n_;               // Number of rows in A
    Matrix a_;             // Input matrix A
    Matrix lu_;            // The decomposition
    double d_;              // used by determinant routine (+-1, pivot-swap parity)
    std::vector<int> index_;  // stores the permutation
};

// Out-of-line definitions of Matrix::determinant()/Matrix::inverse() (declared in
// matrix.hpp), now that LUDecomposition is complete -- see this file's header comment
// and matrix.hpp's own comment on the two headers' mutual dependency. Mirrors C#
// `Matrix.Determinant()`/`Matrix.Inverse()` exactly (construct an LUDecomposition of
// `this`, delegate).
inline double Matrix::determinant() const {
    if (!is_square()) throw std::invalid_argument("The matrix must be square.");
    return LUDecomposition(*this).determinant();
}

inline Matrix Matrix::inverse() const {
    if (!is_square()) throw std::invalid_argument("The matrix must be square.");
    return LUDecomposition(*this).inverse_a();
}

}  // namespace bestfit::numerics::math::linalg
