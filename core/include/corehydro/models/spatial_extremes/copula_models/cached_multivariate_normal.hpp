// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/CopulaModels/CachedMultivariateNormal.cs @ c2e6192
//
// Cached multivariate normal likelihood engine for the SpatialExtremes copula slice. Caches
// the Cholesky factor L and log-determinant of the covariance matrix so repeated likelihood
// evaluations (MCMC) avoid recomputation; the cache is invalidated when the covariance changes.
//
// This class uses its OWN Cholesky path -- it does NOT delegate to the ported
// MultivariateNormal distribution. It builds the ported Matrix + CholeskyDecomposition and rolls
// its own forward-substitution solve (SolveCholesky), mirroring the C# bodies term-for-term.
//
// Numerical degradation (mirrors the C#): a non-positive-definite or degenerate covariance
// yields LogPDF = -infinity, PDF = 0, GetLogDeterminant = -infinity. The ported
// CholeskyDecomposition THROWS std::runtime_error on a non-PD matrix (rather than clearing an
// is_positive_definite flag as the C# does), so the C# try/catch-around-UpdateCache -- which
// logs via Debug.WriteLine and returns false on any exception -- is ported as a silent
// no-throw guard (catch(...) { cache invalid; return false; }). The explicit
// !is_positive_definite() branch is kept for the structural mirror even though the ported
// decomposition reaches it only via the throw.
//
// Deliberately NOT ported (project-wide convention): XML / INPC / Debug.WriteLine plumbing.
#pragma once
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

#include "corehydro/numerics/math/linalg/cholesky_decomposition.hpp"
#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/tools.hpp"

namespace corehydro::models::spatial_extremes {

class CachedMultivariateNormal {
   public:
    // Constructs a new cached multivariate normal (C# ctor): zero mean, zero covariance,
    // cache invalid.
    explicit CachedMultivariateNormal(int dimension)
        : dimension_(dimension),
          mean_(static_cast<std::size_t>(dimension), 0.0),
          covariance_(static_cast<std::size_t>(dimension),
                      std::vector<double>(static_cast<std::size_t>(dimension), 0.0)),
          is_cache_valid_(false) {}

    // Constructs a new cached multivariate normal (C# ctor): the C# null-guards are vacuous in
    // C++ (vector arguments cannot be null); the dimension guard (mean length == covariance
    // rows == covariance cols) is meaningful and ported. Both arrays are value-copied
    // (mirrors the C# `.Clone()`).
    CachedMultivariateNormal(const std::vector<double>& mean,
                             const std::vector<std::vector<double>>& covariance)
        : dimension_(static_cast<int>(mean.size())),
          mean_(mean),
          covariance_(covariance),
          is_cache_valid_(false) {
        std::size_t rows = covariance.size();
        std::size_t cols = rows == 0 ? 0 : covariance[0].size();
        if (mean.size() != rows || rows != cols) {
            throw std::invalid_argument("Dimension mismatch between mean and covariance.");
        }
    }

    // Gets the dimension of the distribution (C# Dimension).
    int dimension() const { return dimension_; }

    // Gets whether the cache is currently valid (C# IsCacheValid).
    bool is_cache_valid() const { return is_cache_valid_; }

    // Sets the mean vector (C# SetMean): dimension-checked, value-copy.
    void set_mean(const std::vector<double>& mean) {
        if (static_cast<int>(mean.size()) != dimension_) {
            throw std::invalid_argument("Mean vector dimension mismatch.");
        }
        mean_ = mean;
    }

    // Sets the covariance matrix and invalidates the cache (C# SetCovariance):
    // dimension-checked, value-copy.
    void set_covariance(const std::vector<std::vector<double>>& covariance) {
        std::size_t rows = covariance.size();
        std::size_t cols = rows == 0 ? 0 : covariance[0].size();
        if (static_cast<int>(rows) != dimension_ || static_cast<int>(cols) != dimension_) {
            throw std::invalid_argument("Covariance matrix dimension mismatch.");
        }
        covariance_ = covariance;
        is_cache_valid_ = false;
    }

    // Computes the log probability density function (C# LogPDF).
    double log_pdf(const std::vector<double>& x) {
        if (static_cast<int>(x.size()) != dimension_) {
            throw std::invalid_argument("Input vector dimension mismatch.");
        }

        // Update cache if needed.
        if (!is_cache_valid_) {
            if (!update_cache()) return neg_infinity();
        }

        // Compute (x - mu).
        std::vector<double> centered(static_cast<std::size_t>(dimension_));
        for (int i = 0; i < dimension_; ++i) centered[i] = x[i] - mean_[i];

        // Solve L * y = (x - mu) for y.
        std::optional<std::vector<double>> y = solve_cholesky(centered);
        if (!y.has_value()) return neg_infinity();

        // Quadratic form: (x-mu)' Sigma^-1 (x-mu) = y' y.
        double quad_form = 0.0;
        for (int i = 0; i < dimension_; ++i) quad_form += (*y)[i] * (*y)[i];

        // Log-PDF: -0.5 * [n*log(2*pi) + log|Sigma| + quad].
        return -0.5 * (dimension_ * std::log(2.0 * numerics::kPi) + log_determinant_ + quad_form);
    }

    // Computes the probability density function (C# PDF).
    double pdf(const std::vector<double>& x) {
        double lp = log_pdf(x);
        if (lp == neg_infinity()) return 0.0;
        return std::exp(lp);
    }

    // Gets the log-determinant of the covariance matrix (C# GetLogDeterminant), or -infinity
    // if the cache cannot be refreshed.
    double get_log_determinant() {
        if (!is_cache_valid_) {
            if (!update_cache()) return neg_infinity();
        }
        return log_determinant_;
    }

    // Invalidates the cache -- forces recomputation on next use (C# InvalidateCache).
    void invalidate_cache() { is_cache_valid_ = false; }

   private:
    static double neg_infinity() { return -std::numeric_limits<double>::infinity(); }

    // Updates the Cholesky decomposition and log-determinant cache (C# UpdateCache):
    // returns true if the decomposition succeeded, false otherwise.
    bool update_cache() {
        try {
            numerics::math::linalg::Matrix cov_matrix(covariance_);
            numerics::math::linalg::CholeskyDecomposition chol(cov_matrix);

            // The ported CholeskyDecomposition reaches this branch only via the throw handled
            // below; kept for the structural mirror of the C# `if (!chol.IsPositiveDefinite)`.
            if (!chol.is_positive_definite()) {
                is_cache_valid_ = false;
                return false;
            }

            cholesky_l_ = chol.l();

            // log|Sigma| = 2 * sum(log(L_ii)).
            log_determinant_ = 0.0;
            for (int i = 0; i < dimension_; ++i) {
                double lii = cholesky_l_(i, i);
                if (lii <= 0) {
                    is_cache_valid_ = false;
                    return false;
                }
                log_determinant_ += std::log(lii);
            }
            log_determinant_ *= 2.0;

            is_cache_valid_ = true;
            return true;
        } catch (...) {
            // C# logs via Debug.WriteLine and returns false on any exception (e.g. the ported
            // CholeskyDecomposition throwing for a non-positive-definite / ill-conditioned
            // matrix). Ported as a silent no-throw guard -- failures degrade to false, not
            // propagate.
            is_cache_valid_ = false;
            return false;
        }
    }

    // Solves L * y = b via forward substitution using the cached L directly (C#
    // SolveCholesky). Returns the "no solution" sentinel (nullopt; C# null) when a diagonal
    // entry is effectively zero; LogPDF maps that to -infinity.
    std::optional<std::vector<double>> solve_cholesky(const std::vector<double>& b) const {
        std::vector<double> y(static_cast<std::size_t>(dimension_));
        for (int i = 0; i < dimension_; ++i) {
            double sum = b[i];
            for (int j = 0; j < i; ++j) sum -= cholesky_l_(i, j) * y[j];

            double lii = cholesky_l_(i, i);
            if (std::fabs(lii) < 1e-15) return std::nullopt;

            y[i] = sum / lii;
        }
        return y;
    }

    int dimension_;
    std::vector<double> mean_;
    std::vector<std::vector<double>> covariance_;
    numerics::math::linalg::Matrix cholesky_l_{0};  // C# `Matrix _choleskyL = null!`
    double log_determinant_ = 0.0;
    bool is_cache_valid_ = false;
};

}  // namespace corehydro::models::spatial_extremes
