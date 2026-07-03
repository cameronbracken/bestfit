// ported from: Numerics/Mathematics/Fourier Methods/Fourier.cs @ a2c4dbf
//
// FFT-based autocorrelation, ported for the closure Autocorrelation actually needs: FFT
// (in-place complex FFT, Numerical Recipes style, "data" packed as [re0, im0, re1, im1,
// ...]), RealFFT (real-input FFT built on FFT), Correlation (cross-correlation via
// RealFFT), and Autocorrelation (the normalized autocorrelation function feeding MCMC's
// effective-sample-size diagnostics). All four members of Fourier.cs are ported -- the
// class has no other public members to omit.
//
// FFT/RealFFT mutate `data` in place, mirroring the C# `double[]` in/out parameter
// (Numerical Recipes' classic style); this port keeps that shape (mutable `vector<double>&`)
// rather than returning a new vector, since Correlation calls RealFFT twice in place on its
// own scratch copies and a copying API would just add an extra allocation per call with no
// behavioral difference.
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/statistics.hpp"
#include "bestfit/numerics/tools.hpp"

namespace bestfit::numerics::math::fourier {

// Performs the fast Fourier transform (FFT) on a complex data array in place. `data` is a
// complex array of length n stored as a real array of length 2*n ([re, im, re, im, ...]); n
// must be a power of two. If `inverse` is true, replaces `data` by n times its inverse DFT
// (matching the C# doc comment -- the caller must divide by n to get the true inverse).
inline void fft(std::vector<double>& data, bool inverse = false) {
    int n = static_cast<int>(data.size() / 2);
    if (n < 2 || !bestfit::numerics::is_power_of_two(n))
        throw std::out_of_range("The data array length must be a power of 2.");

    int nn = n << 1;
    int isign = inverse ? -1 : 1;
    int j = 1;
    // Bit-reversal section.
    for (int i = 1; i < nn; i += 2) {
        if (j > i) {
            double tempr = data[static_cast<std::size_t>(j - 1)];
            double tempi = data[static_cast<std::size_t>(j)];
            data[static_cast<std::size_t>(j - 1)] = data[static_cast<std::size_t>(i - 1)];
            data[static_cast<std::size_t>(j)] = data[static_cast<std::size_t>(i)];
            data[static_cast<std::size_t>(i - 1)] = tempr;
            data[static_cast<std::size_t>(i)] = tempi;
        }
        int m = n;
        while (m >= 2 && j > m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
    // Danielson-Lanczos section.
    int mmax = 2;
    while (nn > mmax) {
        int istep = mmax << 1;
        double theta = isign * (2.0 * bestfit::numerics::kPi / mmax);
        double wpr = -2.0 * std::pow(std::sin(0.5 * theta), 2.0);
        double wpi = std::sin(theta);
        double wr = 1.0, wi = 0.0;
        for (int m = 1; m < mmax; m += 2) {
            for (int i = m; istep >= 0 ? i <= nn : i >= nn; i += istep) {
                int j2 = i + mmax;
                double tempr = wr * data[static_cast<std::size_t>(j2 - 1)] - wi * data[static_cast<std::size_t>(j2)];
                double tempi = wr * data[static_cast<std::size_t>(j2)] + wi * data[static_cast<std::size_t>(j2 - 1)];
                data[static_cast<std::size_t>(j2 - 1)] = data[static_cast<std::size_t>(i - 1)] - tempr;
                data[static_cast<std::size_t>(j2)] = data[static_cast<std::size_t>(i)] - tempi;
                data[static_cast<std::size_t>(i - 1)] += tempr;
                data[static_cast<std::size_t>(i)] += tempi;
            }
            double wtemp = wr;
            wr = wr * wpr - wi * wpi + wr;
            wi = wi * wpr + wtemp * wpi + wi;
        }
        mmax = istep;
    }
}

// Calculates the Fourier transform of n real-valued data points in place (n a power of
// two). Replaces `data` by the positive-frequency half of its complex transform: the
// real-valued first and last components are packed into data[0]/data[1]. If `inverse` is
// true, computes the inverse transform of a complex array that is the transform of real
// data (the result must be multiplied by 2/n by the caller for the true inverse -- mirrors
// the C# doc comment; Correlation below relies on this exact scaling convention).
inline void real_fft(std::vector<double>& data, bool inverse = false) {
    int n = static_cast<int>(data.size());
    int isign = inverse ? -1 : 1;
    double theta = bestfit::numerics::kPi / (n >> 1);
    double c1 = 0.5, c2;
    if (isign == 1) {
        c2 = -0.5;
        fft(data);
    } else {
        c2 = 0.5;
        theta = -theta;
    }
    double wpr = -2.0 * std::pow(std::sin(0.5 * theta), 2.0);
    double wpi = std::sin(theta);
    double wr = 1.0 + wpr;
    double wi = wpi;
    for (int i = 1; i < (n >> 2); ++i) {
        int i1 = i + i;
        int i2 = 1 + i1;
        int i3 = n - i1;
        int i4 = 1 + i3;
        double h1r = c1 * (data[static_cast<std::size_t>(i1)] + data[static_cast<std::size_t>(i3)]);
        double h1i = c1 * (data[static_cast<std::size_t>(i2)] - data[static_cast<std::size_t>(i4)]);
        double h2r = -c2 * (data[static_cast<std::size_t>(i2)] + data[static_cast<std::size_t>(i4)]);
        double h2i = c2 * (data[static_cast<std::size_t>(i1)] - data[static_cast<std::size_t>(i3)]);
        data[static_cast<std::size_t>(i1)] = h1r + wr * h2r - wi * h2i;
        data[static_cast<std::size_t>(i2)] = h1i + wr * h2i + wi * h2r;
        data[static_cast<std::size_t>(i3)] = h1r - wr * h2r + wi * h2i;
        data[static_cast<std::size_t>(i4)] = -h1i + wr * h2i + wi * h2r;
        double wtemp = wr;
        wr = wr * wpr - wi * wpi + wr;
        wi = wi * wpr + wtemp * wpi + wi;
    }

    if (isign == 1) {
        double h1r = data[0];
        data[0] = h1r + data[1];
        data[1] = h1r - data[1];
    } else {
        double h1r = data[0];
        data[0] = c1 * (h1r + data[1]);
        data[1] = c1 * (h1r - data[1]);
        fft(data, true);
    }
}

// Computes the correlation of two equal-length real datasets (length a power of two). NOT
// a normalized correlation coefficient. The answer is returned in wraparound order:
// correlations at increasing negative lags are in [n-1] down to [n/2], while correlations
// at increasingly positive lags are in [0] up to [n/2-1].
inline std::vector<double> correlation(const std::vector<double>& data1, const std::vector<double>& data2) {
    int n = static_cast<int>(data1.size());
    std::vector<double> ans(data1.begin(), data1.begin() + n);
    std::vector<double> temp(data2.begin(), data2.begin() + n);
    real_fft(ans);
    real_fft(temp);
    int no2 = n >> 1;
    for (int i = 2; i < n; i += 2) {
        double tmp = ans[static_cast<std::size_t>(i)];
        ans[static_cast<std::size_t>(i)] =
            (ans[static_cast<std::size_t>(i)] * temp[static_cast<std::size_t>(i)] +
             ans[static_cast<std::size_t>(i + 1)] * temp[static_cast<std::size_t>(i + 1)]) /
            no2;
        ans[static_cast<std::size_t>(i + 1)] =
            (ans[static_cast<std::size_t>(i + 1)] * temp[static_cast<std::size_t>(i)] -
             tmp * temp[static_cast<std::size_t>(i + 1)]) /
            no2;
    }
    ans[0] = ans[0] * temp[0] / no2;
    ans[1] = ans[1] * temp[1] / no2;
    real_fft(ans, true);
    return ans;
}

// Computes the autocorrelation function (ACF) of `series`. Returns std::nullopt where the
// C# returns `null` (lagMax < 1, or n < 2 after defaulting). `lag_max < 0` (the default)
// auto-selects `floor(min(10*log10(n), n-1))`, the first lag begins at zero. The returned
// vector has one {lag, acf} pair per row, lag == row index (kept as a pair, mirroring the
// C# 2-column `double[,]`, rather than collapsing to a plain vector<double>, purely for
// closer structural fidelity to the source).
inline std::optional<std::vector<std::array<double, 2>>> autocorrelation(const std::vector<double>& series,
                                                                          int lag_max = -1) {
    int n = static_cast<int>(series.size());
    if (lag_max < 0) {
        // n == 0 would make std::log10(0) == -inf; casting that to int is undefined
        // behavior in C++ (unlike C#, where the analogous out-of-range cast is merely
        // unspecified). The n < 2 check right below discards this value regardless of
        // what it is when n < 1, so route n < 1 to a safe sentinel instead of computing
        // the log10 expression; observable behavior is unchanged.
        lag_max = n >= 1 ? static_cast<int>(std::floor(
                               std::min(10.0 * std::log10(static_cast<double>(n)), static_cast<double>(n - 1))))
                          : -1;
    }
    if (lag_max < 1 || n < 2) return std::nullopt;

    // Pad the length to a power of 2 to facilitate FFT speed.
    int new_length = static_cast<int>(
        std::lround(std::pow(2.0, std::ceil(std::log2(static_cast<double>(n))))));
    std::vector<double> normalized_series(static_cast<std::size_t>(new_length), 0.0);
    double m = bestfit::numerics::data::mean(series);
    for (int i = 0; i < new_length; ++i) {
        if (i < n) normalized_series[static_cast<std::size_t>(i)] = series[static_cast<std::size_t>(i)] - m;
    }
    std::vector<double> corr = correlation(normalized_series, normalized_series);
    double max_value = *std::max_element(corr.begin(), corr.end());
    std::vector<std::array<double, 2>> acf(static_cast<std::size_t>(lag_max + 1));
    for (int i = 0; i <= lag_max; ++i) {
        acf[static_cast<std::size_t>(i)] = {static_cast<double>(i),
                                             corr[static_cast<std::size_t>(i)] / max_value};
    }
    return acf;
}

}  // namespace bestfit::numerics::math::fourier
