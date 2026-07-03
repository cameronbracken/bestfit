// ported from: Numerics/Sampling/MCMC/Support/MCMCDiagnostics.cs @ a2c4dbf
//
// Convergence diagnostics: effective sample size (both C# overloads), the Gelman-Rubin
// R-hat, and the Raftery-Lewis minimum-sample-size heuristic.
//
// The multi-chain EffectiveSampleSize overload's C# signature is `out double[][,]
// averageACF` (an out-parameter). C++ has no equally ergonomic out-parameter idiom, so
// this port returns a small struct (EffectiveSampleSizeResult) bundling both outputs
// (ess + average_acf) instead -- call sites destructure the two fields where C# would
// destructure the return value + out-var.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/math/fourier/fourier.hpp"
#include "bestfit/numerics/math/optimization/support/parameter_set.hpp"

namespace bestfit::numerics::sampling::mcmc {

namespace fourier = bestfit::numerics::math::fourier;
using bestfit::numerics::math::optimization::ParameterSet;

// Computes the effective sample size of a single posterior series.
// https://www.rdocumentation.org/packages/LaplacesDemon/versions/16.1.4/topics/ESS
inline double effective_sample_size(const std::vector<double>& series) {
    int N = static_cast<int>(series.size());
    auto acf = fourier::autocorrelation(series, static_cast<int>(std::ceil(static_cast<double>(N) / 2.0)));
    if (!acf.has_value()) return static_cast<double>(N);
    double rho = 0.0;
    for (std::size_t i = 1; i < acf->size(); ++i) {
        if ((*acf)[i][1] < 0.0) break;
        rho += (*acf)[i][1];
    }
    return std::min(static_cast<double>(N) / (1.0 + 2.0 * rho), static_cast<double>(N));
}

// Bundles the multi-chain EffectiveSampleSize overload's two outputs -- see file header.
struct EffectiveSampleSizeResult {
    std::vector<double> ess;
    // One entry per parameter; each entry has 51 {lag, value} rows (mirrors the C#
    // `new double[51, 2]` fixed allocation). Column 0 (lag) is never populated by the C#
    // source (only `[j, 1]` -- the value -- is ever assigned), so it is transcribed
    // verbatim as always-zero here too.
    std::vector<std::vector<std::array<double, 2>>> average_acf;
};

// Computes the effective sample size for each model parameter across multiple Markov
// chains. `markov_chains` must be non-empty and every chain at least 2 iterations long
// (chains may differ in length; the shortest determines N, mirroring C#'s `chain =>
// chain.Count` Min).
inline EffectiveSampleSizeResult effective_sample_size(
    const std::vector<std::vector<ParameterSet>>& markov_chains) {
    int M = static_cast<int>(markov_chains.size());
    if (M == 0) throw std::invalid_argument("No chains provided.");
    int P = static_cast<int>(markov_chains[0][0].values.size());
    int N = static_cast<int>(markov_chains[0].size());
    for (const auto& chain : markov_chains) N = std::min(N, static_cast<int>(chain.size()));

    if (N < 2) throw std::out_of_range("There must be at least two iterations to evaluate.");
    if (P < 1) throw std::out_of_range("There must be at least one parameter to evaluate.");

    EffectiveSampleSizeResult result;
    result.ess.assign(static_cast<std::size_t>(P), 0.0);
    result.average_acf.assign(static_cast<std::size_t>(P),
                               std::vector<std::array<double, 2>>(51, std::array<double, 2>{0.0, 0.0}));

    for (int p = 0; p < P; ++p) {
        double mean_rho = 0.0;
        for (int i = 0; i < M; ++i) {
            const auto& chain = markov_chains[static_cast<std::size_t>(i)];
            std::vector<double> values(chain.size());
            for (std::size_t k = 0; k < chain.size(); ++k)
                values[k] = chain[k].values[static_cast<std::size_t>(p)];

            auto acf = fourier::autocorrelation(values, static_cast<int>(std::ceil(static_cast<double>(N) / 2.0)));
            if (!acf.has_value()) continue;

            for (std::size_t j = 0; j < acf->size(); ++j) {
                if (j > 50) break;
                result.average_acf[static_cast<std::size_t>(p)][j][1] += (*acf)[j][1] / M;
            }
            double rho = 0.0;
            for (std::size_t j = 1; j < acf->size(); ++j) {
                if ((*acf)[j][1] < 0.0) break;
                rho += (*acf)[j][1];
            }
            mean_rho += rho / M;
        }
        result.ess[static_cast<std::size_t>(p)] =
            std::min(static_cast<double>(N) * M / (1.0 + 2.0 * mean_rho), static_cast<double>(N) * M);
    }

    return result;
}

// The Gelman-Rubin diagnostic: tests for lack of convergence by comparing the between-chain
// variance to the within-chain variance. `warmup_iterations`: the number of leading
// iterations discarded from each chain before computing the statistic.
inline std::vector<double> gelman_rubin(const std::vector<std::vector<ParameterSet>>& markov_chains,
                                          int warmup_iterations = 0) {
    int M = static_cast<int>(markov_chains.size());
    if (M == 0) throw std::invalid_argument("No chains provided.");
    int P = static_cast<int>(markov_chains[0][0].values.size());
    int N = static_cast<int>(markov_chains[0].size());
    for (const auto& chain : markov_chains) {
        if (static_cast<int>(chain.size()) != N)
            throw std::invalid_argument("All chains must have the same length.");
    }

    std::vector<double> rhat(static_cast<std::size_t>(P), std::numeric_limits<double>::quiet_NaN());

    // Not enough chains to compute between-chain variance: return all-NaN (matches C#,
    // which returns here BEFORE the N/P/warmup_iterations validation below -- a single
    // chain, or zero, is a valid (if uninformative) call, not an error).
    if (M < 2) return rhat;
    if (N < 2) throw std::out_of_range("There must be at least two iterations to evaluate.");
    if (P < 1) throw std::out_of_range("There must be at least one parameter to evaluate.");
    if (warmup_iterations < 0)
        throw std::out_of_range("The warm up iterations must be non-negative.");
    int start_index = std::max(0, warmup_iterations);

    for (int p = 0; p < P; ++p) {
        // Step 1. Compute between- and within-chain mean.
        std::vector<double> chain_means(static_cast<std::size_t>(M), 0.0);
        double overall_mean = 0.0;
        for (int i = 0; i < M; ++i) {
            for (int j = start_index; j < N; ++j)
                chain_means[static_cast<std::size_t>(i)] +=
                    markov_chains[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                        .values[static_cast<std::size_t>(p)];
            chain_means[static_cast<std::size_t>(i)] /= static_cast<double>(N - start_index);
            overall_mean += chain_means[static_cast<std::size_t>(i)];
        }
        overall_mean /= M;

        // Step 2. Compute between- and within-chain variance.
        int n = N - start_index;
        double B = 0.0, W = 0.0;
        for (int i = 0; i < M; ++i) {
            double sum = 0.0;
            for (int j = start_index; j < N; ++j) {
                double d = markov_chains[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                               .values[static_cast<std::size_t>(p)] -
                           chain_means[static_cast<std::size_t>(i)];
                sum += d * d;
            }
            // Within-chain variance.
            W += sum / (n - 1);
            // Between-chain variance.
            double bd = chain_means[static_cast<std::size_t>(i)] - overall_mean;
            B += bd * bd;
        }
        W /= M;
        B *= n / static_cast<double>(M - 1);

        // Step 3. Compute the pooled variance.
        double V = ((n - 1.0) / n) * W + (1.0 / n) * B;

        // Step 4. Compute R-hat.
        rhat[static_cast<std::size_t>(p)] = std::sqrt(V / W);
    }

    return rhat;
}

// Computes the minimum sample size, rounded to the nearest 100, via the Raftery-Lewis
// method. `quantile`: the posterior quantile of interest (e.g. 0.975). `tolerance`: the
// acceptable tolerance for that quantile (e.g. +-0.005). `probability`: probability of
// being within the tolerance range (e.g. 0.95).
inline int minimum_sample_size(double quantile, double tolerance, double probability) {
    double q = quantile;
    double r = tolerance;
    double s = probability;
    double N =
        (q * (1.0 - q) * std::pow(distributions::Normal::standard_z(0.5 * (s + 1.0)), 2.0)) / std::pow(r, 2.0);
    int n_min = static_cast<int>(std::round(N / 100.0)) * 100;
    return n_min;
}

}  // namespace bestfit::numerics::sampling::mcmc
