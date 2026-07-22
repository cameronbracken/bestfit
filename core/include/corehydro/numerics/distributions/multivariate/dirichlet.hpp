// ported from: Numerics/Distributions/Multivariate/Dirichlet.cs @ 2a0357a
//
// The Dirichlet distribution, a multivariate generalization of the Beta distribution,
// defined on the (K-1)-simplex. Conjugate prior for mixture-model weights / categorical
// and multinomial likelihoods.
//
// Simplification vs. the C# source: Mean/Variance/Mode/CovarianceMatrix are computed
// directly on each call rather than lazily cached in mutable backing fields -- the C#
// caching is a micro-optimization orthogonal to the math and not worth the extra
// `mutable` state here; results are identical. Clone() likewise reconstructs via the
// public alpha-vector constructor instead of replicating the C# private-parameterless-
// ctor-plus-field-copy pattern.
#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/gamma_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"
#include "corehydro/numerics/math/special/gamma.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::distributions {

namespace sf_dir = corehydro::numerics::math::special;

class Dirichlet : public MultivariateDistribution {
   public:
    // Symmetric Dirichlet: all K concentration parameters equal to `alpha`.
    Dirichlet(int dimension, double alpha) {
        if (dimension < 2) throw std::out_of_range("The dimension must be at least 2.");
        if (std::isnan(alpha) || std::isinf(alpha) || alpha <= 0.0)
            throw std::out_of_range("The concentration parameter must be positive.");
        alpha_.assign(static_cast<std::size_t>(dimension), alpha);
    }

    // Dirichlet with the given concentration parameter vector alpha_1..alpha_K.
    explicit Dirichlet(std::vector<double> alpha) {
        if (alpha.size() < 2)
            throw std::out_of_range(
                "The concentration parameter vector must have at least 2 elements.");
        for (double a : alpha) {
            if (std::isnan(a) || std::isinf(a) || a <= 0.0)
                throw std::out_of_range("Concentration parameter alpha[i] must be positive.");
        }
        alpha_ = std::move(alpha);
    }

    const std::vector<double>& alpha() const { return alpha_; }

    double alpha_sum() const {
        double s = 0.0;
        for (double a : alpha_) s += a;
        return s;
    }

    // --- Identity / parameters ---
    int dimension() const override { return static_cast<int>(alpha_.size()); }
    MultivariateDistributionType type() const override {
        return MultivariateDistributionType::Dirichlet;
    }
    std::string display_name() const override { return "Dirichlet"; }
    std::string short_display_name() const override { return "Dir"; }

    bool parameters_valid() const override {
        if (alpha_.size() < 2) return false;
        for (double a : alpha_) {
            if (std::isnan(a) || std::isinf(a) || a <= 0.0) return false;
        }
        return true;
    }

    // --- Moments ---

    // Mean[i] = alpha[i] / S, where S = sum(alpha).
    std::vector<double> mean() const {
        double s = alpha_sum();
        std::vector<double> m(alpha_.size());
        for (std::size_t i = 0; i < alpha_.size(); ++i) m[i] = alpha_[i] / s;
        return m;
    }

    // Var[i] = alpha[i] * (S - alpha[i]) / (S^2 * (S + 1)).
    std::vector<double> variance() const {
        double s = alpha_sum();
        double denom = s * s * (s + 1.0);
        std::vector<double> v(alpha_.size());
        for (std::size_t i = 0; i < alpha_.size(); ++i) v[i] = alpha_[i] * (s - alpha_[i]) / denom;
        return v;
    }

    // Mode[i] = (alpha[i] - 1) / (S - K), defined only when every alpha[i] > 1.
    std::vector<double> mode() const {
        for (double a : alpha_) {
            if (a <= 1.0)
                throw std::domain_error(
                    "The mode is only defined in the interior of the simplex when all "
                    "alpha[i] > 1.");
        }
        double denom = alpha_sum() - static_cast<double>(alpha_.size());
        std::vector<double> mo(alpha_.size());
        for (std::size_t i = 0; i < alpha_.size(); ++i) mo[i] = (alpha_[i] - 1.0) / denom;
        return mo;
    }

    // Cov(Xi, Xj) = -alpha_i * alpha_j / (S^2 * (S + 1)); Cov(Xi, Xi) = Var(Xi).
    double covariance(int i, int j) const {
        std::size_t k = alpha_.size();
        if (i < 0 || static_cast<std::size_t>(i) >= k || j < 0 || static_cast<std::size_t>(j) >= k)
            throw std::out_of_range("Index out of range.");
        double s = alpha_sum();
        double denom = s * s * (s + 1.0);
        std::size_t ui = static_cast<std::size_t>(i), uj = static_cast<std::size_t>(j);
        if (i == j) return alpha_[ui] * (s - alpha_[ui]) / denom;
        return -alpha_[ui] * alpha_[uj] / denom;
    }

    std::vector<std::vector<double>> covariance_matrix() const {
        std::size_t k = alpha_.size();
        std::vector<std::vector<double>> cov(k, std::vector<double>(k));
        for (std::size_t i = 0; i < k; ++i)
            for (std::size_t j = 0; j < k; ++j)
                cov[i][j] = covariance(static_cast<int>(i), static_cast<int>(j));
        return cov;
    }

    // --- Distribution functions ---

    double pdf(const std::vector<double>& x) const override { return std::exp(log_pdf(x)); }

    // log f(x) = sum((alpha_i - 1) * log(x_i)) - log B(alpha); -inf off the simplex.
    double log_pdf(const std::vector<double>& x) const override {
        if (x.size() != alpha_.size()) return -kInf;

        double sum = 0.0;
        for (std::size_t i = 0; i < alpha_.size(); ++i) {
            if (x[i] <= 0.0 || x[i] > 1.0) return -kInf;
            sum += x[i];
        }
        if (std::fabs(sum - 1.0) > 1e-10) return -kInf;

        double result = -log_normalization();
        for (std::size_t i = 0; i < alpha_.size(); ++i) result += (alpha_[i] - 1.0) * std::log(x[i]);
        return result;
    }

    // The Dirichlet CDF has no closed-form expression (mirrors the C# NotImplementedException).
    double cdf(const std::vector<double>& /*x*/) const override {
        throw std::logic_error(
            "The CDF of the Dirichlet distribution does not have a closed-form expression.");
    }

    std::unique_ptr<MultivariateDistribution> clone() const override {
        return std::make_unique<Dirichlet>(alpha_);
    }

    // Draws sample_size simplex vectors via K independent Gamma(alpha_i, 1) variates,
    // normalized to sum to 1. seed<=0 seeds from a clock (no reproducible draw; mirrors
    // the LatinHypercube port's same unseeded-fallback rationale -- this C++ port's
    // MersenneTwister has no clock-seeded default constructor).
    std::vector<std::vector<double>> generate_random_values(int sample_size, int seed = -1) const {
        sampling::MersenneTwister rng = make_master(seed);
        std::size_t k = alpha_.size();

        // GammaDistribution uses scale/shape parameterization: GammaDistribution(theta,
        // kappa); theta=1 (scale=1) matches `new GammaDistribution(1.0, _alpha[i])` in C#.
        std::vector<GammaDistribution> gammas;
        gammas.reserve(k);
        for (double a : alpha_) gammas.emplace_back(1.0, a);

        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                 std::vector<double>(k));
        for (int s = 0; s < sample_size; ++s) {
            double sum = 0.0;
            std::vector<double> y(k);
            for (std::size_t i = 0; i < k; ++i) {
                y[i] = gammas[i].inverse_cdf(rng.next_double());
                if (y[i] < 0.0) y[i] = 0.0;  // Guard against numerical issues
                sum += y[i];
            }

            // Normalize to simplex
            std::size_t us = static_cast<std::size_t>(s);
            if (sum == 0.0) {
                for (std::size_t i = 0; i < k; ++i) sample[us][i] = 1.0 / static_cast<double>(k);
                continue;
            }
            for (std::size_t i = 0; i < k; ++i) sample[us][i] = y[i] / sum;
        }
        return sample;
    }

    // log B(alpha) = sum(logGamma(alpha_i)) - logGamma(sum(alpha_i)).
    static double log_multivariate_beta(const std::vector<double>& alpha) {
        double log_b = 0.0;
        double sum = 0.0;
        for (double a : alpha) {
            log_b += sf_dir::log_gamma(a);
            sum += a;
        }
        log_b -= sf_dir::log_gamma(sum);
        return log_b;
    }

   private:
    double log_normalization() const { return log_multivariate_beta(alpha_); }

    static sampling::MersenneTwister make_master(int seed) {
        if (seed > 0) return sampling::MersenneTwister(static_cast<std::uint32_t>(seed));
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return sampling::MersenneTwister(static_cast<std::uint32_t>(ticks));
    }

    std::vector<double> alpha_;
};

}  // namespace corehydro::numerics::distributions
