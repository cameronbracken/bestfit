// ported from: Numerics/Data/Statistics/Correlation.cs @ a2c4dbf
//
// Pearson, Spearman, and Kendall's Tau correlation coefficients for two equal-length
// samples. Feeds copula parameter constraint bounds and dependence estimation.
//
// The matrix overloads (Pearson(double[,]) / Spearman(double[,]), column-pairwise
// correlation matrices over an [n, p] observation table) are omitted: no caller ported
// so far needs them, and they are not trivial one-liners over the pairwise versions
// (they fold in a mean/cross-product pass over all p columns at once). Add them if a
// later task needs a correlation matrix.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"

namespace bestfit::numerics::data {

// Computes the Pearson correlation coefficient.
inline double pearson(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample2.size() != sample1.size())
        throw std::invalid_argument("The sample arrays must be the same length.");

    int n = static_cast<int>(sample1.size());
    // Find means.
    double ax = 0.0, ay = 0.0;
    for (int i = 0; i < n; ++i) {
        ax += sample1[static_cast<std::size_t>(i)];
        ay += sample2[static_cast<std::size_t>(i)];
    }
    ax /= n;
    ay /= n;
    // Compute the correlation coefficient.
    double sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < n; ++i) {
        double xt = sample1[static_cast<std::size_t>(i)] - ax;
        double yt = sample2[static_cast<std::size_t>(i)] - ay;
        sxx += xt * xt;
        syy += yt * yt;
        sxy += xt * yt;
    }
    return sxy / std::sqrt(sxx * syy);
}

// Computes the Spearman ranked correlation coefficient.
inline double spearman(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample2.size() != sample1.size())
        throw std::invalid_argument("The sample arrays must be the same length.");

    std::vector<double> rank1 = ranks_in_place(sample1);
    std::vector<double> rank2 = ranks_in_place(sample2);
    return pearson(rank1, rank2);
}

// Computes Kendall's Tau ranked correlation coefficient (the O(n^2) direct pair count).
inline double kendalls_tau(const std::vector<double>& sample1, const std::vector<double>& sample2) {
    if (sample2.size() != sample1.size())
        throw std::invalid_argument("The sample arrays must be the same length.");

    int n = static_cast<int>(sample1.size());
    int i = 0, n1 = 0, n2 = 0;
    // Loop over first member of pair and second member.
    for (int j = 0; j <= n - 2; ++j) {
        for (int k = j + 1; k <= n - 1; ++k) {
            double a1 = sample1[static_cast<std::size_t>(j)] - sample1[static_cast<std::size_t>(k)];
            double a2 = sample2[static_cast<std::size_t>(j)] - sample2[static_cast<std::size_t>(k)];
            double aa = a1 * a2;
            if (aa != 0.0) {
                // Neither has a tie.
                n1 += 1;
                n2 += 1;
                i = aa > 0.0 ? i + 1 : i - 1;
            } else {
                // One or both arrays have ties.
                if (a1 != 0.0) n1 += 1;
                if (a2 != 0.0) n2 += 1;
            }
        }
    }
    return static_cast<double>(i) / (std::sqrt(static_cast<double>(n1)) * std::sqrt(static_cast<double>(n2)));
}

}  // namespace bestfit::numerics::data
