// ported from: Numerics/Mathematics/Linear Algebra/MatrixRegularization.cs @ a2c4dbf
//
// Ports ONLY `MakeSymmetricPositiveDefinite` -- the sole member the in-scope BestFit
// Estimation layer (MaximumLikelihood covariance/sandwich estimator, MaximumAPosteriori
// covariance) calls. Its C# body: symmetrize S = 0.5*(M + Mtranspose); compute a
// trace-scaled base ridge (1e-10 * trace(S)/p, or 1e-10 if trace <= 0); try up to 8
// increasing ridges (baseRidge * 10^k for k = 0..7) added to the diagonal of a clone of
// S, returning the first that Cholesky-factorizes; if all 8 fail, add a last-resort
// ridge (1e-4 * trace(S)/p, or 1e-4 if trace <= 0) and return unconditionally. The
// private `Symmetrize` helper is inlined here via Matrix's own `operator+`/`transpose`/
// scalar `operator*`, matching the C# source (which itself inlines `0.5 * (M +
// M.Transpose())` rather than calling its own `Symmetrize` private method for this
// particular member). C#'s bare `catch { }` around `new CholeskyDecomposition(T)` is
// mirrored as `catch (const std::exception&)`, the faithful analogue per the task brief
// (CholeskyDecomposition's ctor only ever throws std::invalid_argument or
// std::runtime_error, both std::exception, so this is not narrower than the C#).
//
// DEFERRED (severable omission, tracked with the pivotal-bootstrap/GMM follow-up): both
// `Regularize` overloads (Matrix and double[,]) and their private `MedianFromVector`
// helper. They depend on `EigenValueDecomposition` (symmetric Jacobi EVD), which is not
// ported in this repo -- it was severed along with the pivotal bootstrap and Generalized
// Method of Moments (GMM) work at the end of Phase 3. No in-scope estimator calls
// `Regularize`; only the severed GMM and pivotal-bootstrap paths do. Port `Regularize` +
// `MedianFromVector` + `EigenValueDecomposition` together when that follow-up lands.
#pragma once
#include <cmath>

#include "bestfit/numerics/math/linalg/cholesky_decomposition.hpp"
#include "bestfit/numerics/math/linalg/matrix.hpp"

namespace bestfit::numerics::math::linalg {

class MatrixRegularization {
   public:
    // Makes the matrix symmetric and positive definite (C# `MakeSymmetricPositiveDefinite`).
    static Matrix make_symmetric_positive_definite(const Matrix& m) {
        // Symmetrize.
        Matrix s = (m + m.transpose()) * 0.5;

        // Tiny trace-scaled ridge.
        double tr = 0.0;
        for (int i = 0; i < s.number_of_rows(); ++i) tr += s(i, i);
        double base_ridge = (tr > 0 ? 1e-10 * tr / s.number_of_rows() : 1e-10);

        // Try increasing ridge until Cholesky succeeds.
        for (int k = 0; k < 8; ++k) {
            Matrix t = s.clone();
            double ridge = base_ridge * std::pow(10.0, static_cast<double>(k));
            for (int i = 0; i < t.number_of_rows(); ++i) t(i, i) += ridge;
            try {
                CholeskyDecomposition chol(t);
                (void)chol;
                return t;
            } catch (const std::exception&) {
                // retry bigger ridge
            }
        }

        // Last resort: add a biggish ridge.
        Matrix u = s.clone();
        double big = (tr > 0 ? 1e-4 * tr / u.number_of_rows() : 1e-4);
        for (int i = 0; i < u.number_of_rows(); ++i) u(i, i) += big;
        return u;
    }
};

}  // namespace bestfit::numerics::math::linalg
