// ported from: Numerics/Mathematics/Linear Algebra/Support/Matrix.cs @ <pending-sha>
//
// Minimal dense square-matrix inverse (Gauss-Jordan with partial pivoting). Sufficient
// for the small parameter-covariance matrices in Phase 0; the full Matrix type and its
// decompositions land in a later phase.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

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

}  // namespace bestfit::numerics::math::linalg
