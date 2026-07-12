// ported from: Numerics/Distributions/Multivariate/Multinomial.cs @ a2c4dbf
//
// The Multinomial distribution: outcome counts across K categories over N trials with
// fixed per-category probabilities p_1..p_K (sum p_i = 1).
//
// Simplification vs. the C# source: Mean/Variance are computed directly on each call
// rather than cached (see the Dirichlet port header for the same rationale). Sample()
// and generate_random_values() take corehydro::numerics::sampling::MersenneTwister
// directly rather than the C# System.Random base type -- this port has no generic RNG
// interface hierarchy (every other ported sampling routine also takes MersenneTwister
// concretely).
#pragma once
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/binomial.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution.hpp"
#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"
#include "corehydro/numerics/math/special/factorial.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::distributions {

namespace sf_mult = corehydro::numerics::math::special;

class Multinomial : public MultivariateDistribution {
   public:
    Multinomial(int number_of_trials, std::vector<double> probabilities) {
        if (number_of_trials < 1)
            throw std::out_of_range("The number of trials must be positive.");
        if (probabilities.size() < 2)
            throw std::out_of_range("The probability vector must have at least 2 elements.");
        double sum = 0.0;
        for (double p : probabilities) {
            if (std::isnan(p) || p < 0.0 || p > 1.0)
                throw std::out_of_range("Probability p[i] must be in [0, 1].");
            sum += p;
        }
        if (std::fabs(sum - 1.0) > 1e-10)
            throw std::out_of_range("Probabilities must sum to 1.");

        n_ = number_of_trials;
        p_ = std::move(probabilities);
    }

    int number_of_trials() const { return n_; }
    const std::vector<double>& probabilities() const { return p_; }

    // --- Identity / parameters ---
    int dimension() const override { return static_cast<int>(p_.size()); }
    MultivariateDistributionType type() const override {
        return MultivariateDistributionType::Multinomial;
    }
    std::string display_name() const override { return "Multinomial"; }
    std::string short_display_name() const override { return "Mult"; }

    bool parameters_valid() const override {
        if (p_.size() < 2 || n_ < 1) return false;
        double sum = 0.0;
        for (double p : p_) {
            if (std::isnan(p) || p < 0.0 || p > 1.0) return false;
            sum += p;
        }
        return std::fabs(sum - 1.0) <= 1e-10;
    }

    // --- Moments ---

    // Mean[i] = N * p[i].
    std::vector<double> mean() const {
        std::vector<double> m(p_.size());
        for (std::size_t i = 0; i < p_.size(); ++i) m[i] = static_cast<double>(n_) * p_[i];
        return m;
    }

    // Var[i] = N * p[i] * (1 - p[i]).
    std::vector<double> variance() const {
        std::vector<double> v(p_.size());
        for (std::size_t i = 0; i < p_.size(); ++i)
            v[i] = static_cast<double>(n_) * p_[i] * (1.0 - p_[i]);
        return v;
    }

    // Cov(Xi, Xj) = -N * pi * pj; Cov(Xi, Xi) = Var(Xi).
    double covariance(int i, int j) const {
        std::size_t k = p_.size();
        if (i < 0 || static_cast<std::size_t>(i) >= k || j < 0 || static_cast<std::size_t>(j) >= k)
            throw std::out_of_range("Index out of range.");
        std::size_t ui = static_cast<std::size_t>(i), uj = static_cast<std::size_t>(j);
        if (i == j) return static_cast<double>(n_) * p_[ui] * (1.0 - p_[ui]);
        return -static_cast<double>(n_) * p_[ui] * p_[uj];
    }

    // --- Distribution functions ---

    // PMF: P(X = x) = N! / (x1! ... xK!) * p1^x1 * ... * pK^xK, computed in log-space
    // (the C# names this "PDF" per IMultivariateDistribution; LogPMF is a distinct
    // public method that LogPDF forwards to).
    double pdf(const std::vector<double>& x) const override { return std::exp(log_pmf(x)); }

    double log_pmf(const std::vector<double>& x) const {
        if (x.size() != p_.size()) return -kInf;

        // Validate: all non-negative integers that sum to N
        int sum = 0;
        std::vector<int> xi(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            int v = static_cast<int>(std::lround(x[i]));
            if (v < 0 || std::fabs(x[i] - static_cast<double>(v)) > 1e-10) return -kInf;
            xi[i] = v;
            sum += v;
        }
        if (sum != n_) return -kInf;

        // Compute in log-space: log(N!) - sum(log(xi!)) + sum(xi * log(pi))
        double result = sf_mult::factorial::log_factorial(n_);
        for (std::size_t i = 0; i < x.size(); ++i) {
            result -= sf_mult::factorial::log_factorial(xi[i]);
            if (xi[i] > 0) {
                if (p_[i] <= 0.0) return -kInf;
                result += static_cast<double>(xi[i]) * std::log(p_[i]);
            }
        }
        return result;
    }

    double log_pdf(const std::vector<double>& x) const override { return log_pmf(x); }

    // The Multinomial CDF has no closed-form expression (mirrors the C# NotImplementedException).
    double cdf(const std::vector<double>& /*x*/) const override {
        throw std::logic_error(
            "The CDF of the multinomial distribution does not have a closed-form expression.");
    }

    std::unique_ptr<MultivariateDistribution> clone() const override {
        return std::make_unique<Multinomial>(n_, p_);
    }

    // Sequential binomial sampling: category i ~ Binomial(n_remaining, p_i / p_remaining);
    // the last category gets the remainder. Exact; matches the C# draw order.
    std::vector<std::vector<double>> generate_random_values(int sample_size, int seed = -1) const {
        sampling::MersenneTwister rng = make_master(seed);
        std::size_t k = p_.size();
        std::vector<std::vector<double>> sample(static_cast<std::size_t>(sample_size),
                                                 std::vector<double>(k));

        for (int s = 0; s < sample_size; ++s) {
            int n_remaining = n_;
            double p_remaining = 1.0;
            std::size_t us = static_cast<std::size_t>(s);

            for (std::size_t i = 0; i + 1 < k; ++i) {
                if (n_remaining == 0 || p_remaining <= 0.0) {
                    sample[us][i] = 0.0;
                    continue;
                }

                double conditional_p = p_[i] / p_remaining;
                if (conditional_p >= 1.0) {
                    sample[us][i] = n_remaining;
                    n_remaining = 0;
                } else {
                    int count = binomial_sample(n_remaining, conditional_p, rng);
                    sample[us][i] = count;
                    n_remaining -= count;
                }
                p_remaining -= p_[i];
            }

            // Last category gets the remainder
            sample[us][k - 1] = static_cast<double>(n_remaining);
        }
        return sample;
    }

    // Weighted categorical draw: index i is chosen with probability weight[i]/sum(weights).
    // Used by the NUTS algorithm to select a trajectory state weighted by exp(H).
    static int sample(const std::vector<double>& weights, sampling::MersenneTwister& rng) {
        if (weights.empty()) throw std::invalid_argument("Weights array must be non-empty.");

        double total_weight = 0.0;
        for (double w : weights) {
            if (w < 0.0) throw std::invalid_argument("Weights must be non-negative.");
            total_weight += w;
        }
        if (total_weight <= 0.0)
            throw std::invalid_argument("At least one weight must be positive.");

        double u = rng.next_double() * total_weight;
        double cumulative = 0.0;
        for (std::size_t i = 0; i < weights.size(); ++i) {
            cumulative += weights[i];
            if (u <= cumulative) return static_cast<int>(i);
        }
        // Should not reach here, but return last index as a safeguard
        return static_cast<int>(weights.size()) - 1;
    }

   private:
    static sampling::MersenneTwister make_master(int seed) {
        if (seed > 0) return sampling::MersenneTwister(static_cast<std::uint32_t>(seed));
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return sampling::MersenneTwister(static_cast<std::uint32_t>(ticks));
    }

    // Binomial(n, p) draw: direct simulation for small n, inverse-CDF via Binomial for
    // large n (n >= 25).
    static int binomial_sample(int n, double p, sampling::MersenneTwister& rng) {
        if (n <= 0 || p <= 0.0) return 0;
        if (p >= 1.0) return n;

        if (n < 25) {
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (rng.next_double() < p) ++count;
            }
            return count;
        }

        Binomial binom(p, n);
        return static_cast<int>(std::lround(binom.inverse_cdf(rng.next_double())));
    }

    int n_ = 0;
    std::vector<double> p_;
};

}  // namespace corehydro::numerics::distributions
