// ported from: Numerics/Data/Statistics/Statistics.cs @ a2c4dbf
//
// Sample statistics needed by distribution estimation: product moments
// (mean, stdev, bias-corrected skew & excess kurtosis), linear (L-)moments, and
// rank statistics (ranks_in_place, which Correlation::spearman consumes).
//
// ranks_in_place ports only the single-return-value RanksInPlace(double[]) overload
// (exact-equality tie runs); the RanksInPlace(double[], out ties) overload (tolerance-
// based ties via AlmostEquals, returning a tie-count array) is omitted -- no caller
// ported so far needs the tie-count output.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace bestfit::numerics::data {

// Returns {mean, stdev (sample), bias-corrected skewness, bias-corrected excess kurtosis}.
inline std::vector<double> product_moments(const std::vector<double>& data) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    double N = static_cast<double>(data.size());
    if (N < 4) return {kNaN, kNaN, kNaN, kNaN};

    double X1 = 0, X2 = 0, X3 = 0, X4 = 0;
    for (double x : data) {
        double x2 = x * x;
        X1 += x;
        X2 += x2;
        X3 += x2 * x;
        X4 += x2 * x2;
    }
    double U1 = X1 / N, U2 = X2 / N, U3 = X3 / N, U4 = X4 / N;
    double m2 = (U2 - U1 * U1) * (N / (N - 1));  // sample variance
    double S = std::sqrt(m2);
    double U1_2 = U1 * U1, U1_3 = U1_2 * U1, U1_4 = U1_3 * U1;
    double S3 = S * S * S, S4 = S3 * S;
    double c3 = U3 - 3 * U1 * U2 + 2 * U1_3;
    double c4 = U4 - 4 * U1 * U3 + 6 * U2 * U1_2 - 3 * U1_4;
    double G = (N * N) / ((N - 1) * (N - 2)) * (c3 / S3);
    double K = ((N * N) * (N + 1)) / ((N - 1) * (N - 2) * (N - 3)) * (c4 / S4) -
               3.0 * (N - 1) * (N - 1) / ((N - 2) * (N - 3));
    return {U1, S, G, K};
}

// Returns {L1 (L-mean), L2 (L-scale), T3 (L-skewness), T4 (L-kurtosis)}.
inline std::vector<double> linear_moments(const std::vector<double>& data) {
    constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
    double N = static_cast<double>(data.size());
    if (N < 4) return {kNaN, kNaN, kNaN, kNaN};

    std::vector<double> sorted(data);
    std::sort(sorted.begin(), sorted.end());

    double B0 = 0, B1 = 0, B2 = 0, B3 = 0;
    for (int i = 1; i <= static_cast<int>(N); ++i) {
        B0 += sorted[i - 1];
        if (i > 1) B1 += (i - 1) / (N - 1) * sorted[i - 1];
        if (i > 2) B2 += (i - 2) * (i - 1) / ((N - 2) * (N - 1)) * sorted[i - 1];
        if (i > 3)
            B3 += (i - 3) * (i - 2) * (i - 1) / ((N - 3) * (N - 2) * (N - 1)) * sorted[i - 1];
    }
    B0 /= N;
    B1 /= N;
    B2 /= N;
    B3 /= N;
    double L1 = B0;
    double L2 = 2 * B1 - B0;
    double T3 = 2 * (3 * B2 - B0) / (2 * B1 - B0) - 3;
    double T4 = 5 * (2 * (2 * B3 - 3 * B2) + B0) / (2 * B1 - B0) + 6;
    return {L1, L2, T3, T4};
}

// Returns the rank of each entry of the unsorted data array. Tied values (exact
// equality) receive the average rank of their run (mirrors C# Statistics.RanksInPlace).
inline std::vector<double> ranks_in_place(const std::vector<double>& data) {
    const int n = static_cast<int>(data.size());
    std::vector<double> ranks(static_cast<std::size_t>(n), 0.0);

    // Co-sorted index array: index[k] is the original position of the k-th smallest
    // value (mirrors Array.Sort(work, index) sorting a cloned `work` and permuting a
    // parallel `index` array to match).
    std::vector<int> index(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) index[static_cast<std::size_t>(i)] = i;
    std::sort(index.begin(), index.end(),
              [&data](int a, int b) { return data[static_cast<std::size_t>(a)] < data[static_cast<std::size_t>(b)]; });

    std::vector<double> work(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i)
        work[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(index[static_cast<std::size_t>(i)])];

    // Assign the average rank (b+a-1)/2 + 1 to the tie run [a, b) (mirrors C# RanksTies).
    auto ranks_ties = [&ranks, &index](int a, int b) {
        double rank = (b + a - 1) / 2.0 + 1;
        for (int k = a; k < b; ++k) ranks[static_cast<std::size_t>(index[static_cast<std::size_t>(k)])] = rank;
    };

    int previous_index = 0;
    for (int i = 1; i < n; ++i) {
        if (std::fabs(work[static_cast<std::size_t>(i)] - work[static_cast<std::size_t>(previous_index)]) <= 0)
            continue;

        if (i == previous_index + 1) {
            ranks[static_cast<std::size_t>(index[static_cast<std::size_t>(previous_index)])] =
                static_cast<double>(i);
        } else {
            ranks_ties(previous_index, i);
        }
        previous_index = i;
    }
    ranks_ties(previous_index, n);

    return ranks;
}

}  // namespace bestfit::numerics::data
