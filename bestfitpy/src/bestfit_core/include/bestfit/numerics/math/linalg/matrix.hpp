// ported from: Numerics/Mathematics/Linear Algebra/Support/Matrix.cs @ a2c4dbf
//
// Phase 0 shipped a minimal dense square-matrix inverse (Gauss-Jordan with partial
// pivoting) as free-standing `Matrix2D`/`inverse()`; GEV still depends on those exactly
// as they are, so they are kept untouched below. Phase 2 adds the real `Matrix` class
// alongside it for CholeskyDecomposition and the multivariate distributions.
//
// `Matrix` ports the members Phase 2 actually uses: the `(rows, cols)` / `(n)` / 2D-data
// constructors, `identity`, element access, `number_of_rows`/`number_of_columns`,
// `is_square`/`is_symmetric`, instance and static `diagonal`, elementwise `sqrt`,
// `multiply` (+ `operator*` for Matrix*Matrix and Matrix*Vector), `transpose`, and
// `clone`. A `(rows, cols, flat)` constructor is also added -- not present in the C#
// source -- to match this port's fixture convention of passing matrices as flattened
// row-major `args` (see `fixtures/special_functions/cholesky.json`).
//
// Omitted (UI/serialization or not yet needed by Phase 2; add if a later target needs
// them): `Header`, `Array` (raw-array reference property; `to_array()`/`ToArray()`, a
// copy, is provided instead), `ToXElement`/XElement ctor, the single-column-array/
// `List<double[]>` ctors, `Row`/`Column`, `UpperTriangle`/`LowerTriangle`/`Trace`,
// `ColumnMeans`, `Apply`/`Sqr`/`Log`/`Exp`, `Sum`, `Outer`, `Add`/`Subtract` (+ their
// operators), and the scalar `Multiply`/`Divide` (+ their operators). Also omitted,
// because they depend on the not-yet-ported `LUDecomposition`: `Matrix.Determinant()`/
// `Matrix.Inverse()` and `operator!` -- CholeskyDecomposition computes its own
// determinant/inverse directly from `L` without needing them. `operator~` (the C#
// transpose alias) is omitted too, but for an unrelated reason: it has no LU dependency
// -- it is just a spelling for the already-ported `transpose()` below -- and is left out
// only because no caller needs the operator form yet.
#pragma once
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/math/linalg/vector.hpp"

namespace bestfit::numerics::math::linalg {

using Matrix2D = std::vector<std::vector<double>>;

inline Matrix2D inverse(const Matrix2D& a) {
    const int n = static_cast<int>(a.size());
    // Augment [a | I]
    std::vector<std::vector<double>> m(n, std::vector<double>(2 * n, 0.0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) m[i][j] = a[i][j];
        m[i][n + i] = 1.0;
    }
    for (int col = 0; col < n; ++col) {
        // Partial pivot
        int pivot = col;
        double best = std::fabs(m[col][col]);
        for (int r = col + 1; r < n; ++r) {
            if (std::fabs(m[r][col]) > best) {
                best = std::fabs(m[r][col]);
                pivot = r;
            }
        }
        if (best == 0.0) throw std::runtime_error("matrix is singular; cannot invert");
        if (pivot != col) std::swap(m[pivot], m[col]);

        double diag = m[col][col];
        for (int j = 0; j < 2 * n; ++j) m[col][j] /= diag;
        for (int r = 0; r < n; ++r) {
            if (r == col) continue;
            double factor = m[r][col];
            if (factor == 0.0) continue;
            for (int j = 0; j < 2 * n; ++j) m[r][j] -= factor * m[col][j];
        }
    }
    Matrix2D inv(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) inv[i][j] = m[i][n + j];
    return inv;
}

class Matrix {
   public:
    // Construct a new matrix with specified number of rows and columns, zero-initialized.
    Matrix(int number_of_rows, int number_of_columns)
        : data_(static_cast<std::size_t>(number_of_rows),
                std::vector<double>(static_cast<std::size_t>(number_of_columns), 0.0)) {}

    // Constructs a new square matrix.
    explicit Matrix(int size) : Matrix(size, size) {}

    // Construct a new matrix based on an initial 2D array.
    explicit Matrix(Matrix2D initial_array) : data_(std::move(initial_array)) {}

    // Construct a new matrix from flattened row-major data (see file header comment --
    // this ctor is not in the C# source; it exists for this port's fixture convention).
    Matrix(int number_of_rows, int number_of_columns, const std::vector<double>& flat)
        : data_(static_cast<std::size_t>(number_of_rows),
                std::vector<double>(static_cast<std::size_t>(number_of_columns), 0.0)) {
        if (static_cast<int>(flat.size()) != number_of_rows * number_of_columns)
            throw std::invalid_argument("flattened data length must equal rows*columns");
        for (int i = 0; i < number_of_rows; ++i)
            for (int j = 0; j < number_of_columns; ++j)
                data_[i][j] = flat[static_cast<std::size_t>(i * number_of_columns + j)];
    }

    // Gets the number of rows.
    int number_of_rows() const { return static_cast<int>(data_.size()); }

    // Gets the number of columns.
    int number_of_columns() const { return data_.empty() ? 0 : static_cast<int>(data_[0].size()); }

    // Get/set the element at the specific row and column index.
    double operator()(int row_index, int column_index) const { return data_[row_index][column_index]; }
    double& operator()(int row_index, int column_index) { return data_[row_index][column_index]; }

    // Evaluates whether this matrix is symmetric.
    bool is_symmetric() const {
        if (!is_square()) return false;
        const double epsilon = 1e-12;
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = i + 1; j < number_of_columns(); ++j)
                if (std::fabs(data_[i][j] - data_[j][i]) > epsilon) return false;
        return true;
    }

    // Determines whether this matrix is square.
    bool is_square() const { return number_of_rows() == number_of_columns(); }

    // Returns a copy of this matrix.
    Matrix clone() const { return Matrix(data_); }

    // Returns the matrix as a 2D array (copy).
    Matrix2D to_array() const { return data_; }

    // Returns the elements of the diagonal. For non-square matrices, returns
    // min(rows, columns) elements where i == j.
    std::vector<double> diagonal() const {
        int n = std::min(number_of_rows(), number_of_columns());
        std::vector<double> result(static_cast<std::size_t>(n));
        for (int i = 0; i < n; ++i) result[i] = data_[i][i];
        return result;
    }

    // Returns the transpose of the matrix.
    Matrix transpose() const {
        Matrix t(number_of_columns(), number_of_rows());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) t(j, i) = data_[i][j];
        return t;
    }

    // Returns the diagonal matrix from a square matrix A (off-diagonal entries zeroed).
    static Matrix diagonal(const Matrix& a) {
        if (!a.is_square()) throw std::invalid_argument("The matrix must be square.");
        Matrix d(a.number_of_columns());
        for (int i = 0; i < a.number_of_rows(); ++i)
            for (int j = 0; j < a.number_of_columns(); ++j) d(i, j) = i == j ? a(i, j) : 0.0;
        return d;
    }

    // Returns the diagonal matrix built from a vector.
    static Matrix diagonal(const Vector& a) {
        Matrix d(a.length());
        for (int i = 0; i < d.number_of_rows(); ++i)
            for (int j = 0; j < d.number_of_columns(); ++j) d(i, j) = i == j ? a[i] : 0.0;
        return d;
    }

    // Returns the identity matrix of size n.
    static Matrix identity(int size) {
        Matrix I(size);
        for (int j = 0; j < size; ++j) I(j, j) = 1.0;
        return I;
    }

    // Computes the square root of the matrix elementwise, in place (mirrors the C#
    // `void Sqrt()`, which mutates via `Apply(Math.Sqrt)`).
    void sqrt() {
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < number_of_columns(); ++j) data_[i][j] = std::sqrt(data_[i][j]);
    }

    // Multiply by another matrix.
    Matrix multiply(const Matrix& other) const {
        if (number_of_columns() != other.number_of_rows())
            throw std::invalid_argument(
                "The number of rows in the right-hand matrix must be equal to the number of "
                "columns in this matrix.");
        Matrix result(number_of_rows(), other.number_of_columns());
        for (int i = 0; i < number_of_rows(); ++i)
            for (int j = 0; j < other.number_of_columns(); ++j) {
                double sum = 0.0;
                for (int k = 0; k < number_of_columns(); ++k) sum += data_[i][k] * other(k, j);
                result(i, j) = sum;
            }
        return result;
    }

    // Multiply by a vector.
    Vector multiply(const Vector& vector) const {
        if (number_of_columns() != vector.length())
            throw std::invalid_argument(
                "The number of rows in vector must be equal to the number of columns in the "
                "matrix.");
        std::vector<double> result(static_cast<std::size_t>(number_of_rows()));
        for (int i = 0; i < number_of_rows(); ++i) {
            double sum = 0.0;
            for (int j = 0; j < number_of_columns(); ++j) sum += data_[i][j] * vector[j];
            result[i] = sum;
        }
        return Vector(std::move(result));
    }

   private:
    Matrix2D data_;
};

// Multiplies matrix A and B and returns the result as a matrix.
inline Matrix operator*(const Matrix& a, const Matrix& b) { return a.multiply(b); }

// Multiplies matrix A with vector b and returns the result as a vector.
inline Vector operator*(const Matrix& a, const Vector& b) { return a.multiply(b); }

}  // namespace bestfit::numerics::math::linalg
