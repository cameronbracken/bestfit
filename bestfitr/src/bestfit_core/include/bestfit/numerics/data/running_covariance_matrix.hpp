// ported from: Numerics/Data/Statistics/RunningCovarianceMatrix.cs @ a2c4dbf
//
// Welford's online algorithm generalized to a running mean vector and covariance matrix,
// ported verbatim -- push() updates `mean_` using the OLD mean (captured before the
// update) in the covariance term, exactly matching the C# `oldMean`/`Mean` two-step
// order. ARWMH adapts its proposal covariance through this class's push().
//
// `x = new Matrix(values.ToArray())` (C#'s single-column-array ctor, not ported -- see
// matrix.hpp's omission note) is replaced here by the already-ported `(rows, cols, flat)`
// ctor called as `Matrix(size, 1, values)`: passing a length-N vector as a flattened Nx1
// matrix is bit-identical to the single-column ctor's construction, so this is a like-for-
// like substitution, not a behavioral change.
//
// `sample_correlation()`/`population_correlation()` call `sample_covariance()`/
// `population_covariance()` only ONCE each (into a local `sc`/`pc`), where the C# getters
// call the equivalent `SampleCovariance`/`PopulationCovariance` PROPERTY twice (once for
// `Matrix.Diagonal(...)`, once for the final product) -- both calls recompute the exact
// same value with no side effects, so this is a value-preserving redundant-computation
// elimination, not a fidelity deviation.
//
// Member order mirrors the C# source: ctor, `n()`, `mean()`, `covariance()`,
// `sample_covariance()`, `population_covariance()`, `sample_correlation()`,
// `population_correlation()`, `push()`.
#pragma once
#include <vector>

#include "bestfit/numerics/math/linalg/lu_decomposition.hpp"  // Matrix::inverse() definition (C# `!D`)
#include "bestfit/numerics/math/linalg/matrix.hpp"

namespace bestfit::numerics::data {

class RunningCovarianceMatrix {
   public:
    // Construct a covariance matrix. The mean vector starts zero-initialized (the
    // `(rows, cols)` Matrix ctor); the covariance starts as the identity matrix.
    explicit RunningCovarianceMatrix(int size)
        : mean_(size, 1), covariance_(math::linalg::Matrix::identity(size)) {}

    // The sample size N.
    int n() const { return n_; }

    // The mean vector.
    const math::linalg::Matrix& mean() const { return mean_; }

    // The covariance matrix. This is unadjusted by the sample size.
    const math::linalg::Matrix& covariance() const { return covariance_; }

    // The sample covariance matrix corrected by the sample size with the degrees of
    // freedom adjustment.
    math::linalg::Matrix sample_covariance() const {
        return covariance_ * (1.0 / static_cast<double>(n_ - 1));
    }

    // The population covariance matrix corrected by the total sample size.
    math::linalg::Matrix population_covariance() const {
        return covariance_ * (1.0 / static_cast<double>(n_));
    }

    // The sample correlation matrix.
    math::linalg::Matrix sample_correlation() const {
        math::linalg::Matrix sc = sample_covariance();
        math::linalg::Matrix d = math::linalg::Matrix::diagonal(sc);
        d.sqrt();
        math::linalg::Matrix inv_sqrt_d = d.inverse();
        return (inv_sqrt_d * sc) * inv_sqrt_d;
    }

    // The population correlation matrix.
    math::linalg::Matrix population_correlation() const {
        math::linalg::Matrix pc = population_covariance();
        math::linalg::Matrix d = math::linalg::Matrix::diagonal(pc);
        d.sqrt();
        math::linalg::Matrix inv_sqrt_d = d.inverse();
        return (inv_sqrt_d * pc) * inv_sqrt_d;
    }

    // Add a new vector to the running statistics. The length of `values` must be the
    // same as the number of rows (enforced implicitly by the Matrix subtraction below,
    // mirroring the C# source, which has no explicit length check either).
    void push(const std::vector<double>& values) {
        n_ += 1;
        math::linalg::Matrix x(static_cast<int>(values.size()), 1, values);
        // Update mean
        math::linalg::Matrix old_mean = mean_;
        mean_ = n_ == 1 ? x : mean_ + (x - mean_) * (1.0 / static_cast<double>(n_));
        // Update covariance
        covariance_ = covariance_ + (x - old_mean) * (x - mean_).transpose();
    }

   private:
    int n_ = 0;
    math::linalg::Matrix mean_;
    math::linalg::Matrix covariance_;
};

}  // namespace bestfit::numerics::data
