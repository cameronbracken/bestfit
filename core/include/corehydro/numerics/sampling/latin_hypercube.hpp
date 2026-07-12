// ported from: Numerics/Sampling/LatinHypercube.cs @ a2c4dbf
//
// Latin hypercube sampling (LHS): [0, 1) is stratified into sample_size equal-
// probability bins per dimension, then bin order is shuffled with an in-place
// Fisher-Yates draw from a column-specific MersenneTwister seeded by the master
// generator's next(). RNG draw order (master.next() once per column, then per-bin
// next_double()/shuffle draws within that column) mirrors the C# source exactly, so a
// seeded call reproduces the same stream in R/Python as it does in the real C# library
// (a later task pins this with seeded sampling oracles).
//
// Divergence note: when seed <= 0, the C# source seeds a clock-based MersenneTwister
// (`new MersenneTwister()`, which uses `DateTime.UtcNow.Ticks`). The C++
// MersenneTwister port has no default/clock constructor -- Task 3 extends it with
// next()/next(int) only, no ABI/layout change -- so the unseeded path here instead
// seeds from std::chrono::steady_clock. This only affects non-reproducible, unseeded
// calls; seeded calls (the case the oracle gate and later sampling fixtures care
// about) are unaffected.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <vector>

#include "corehydro/numerics/math/linalg/matrix.hpp"
#include "corehydro/numerics/sampling/mersenne_twister.hpp"

namespace corehydro::numerics::sampling {

class LatinHypercube {
   public:
    // Generate Latin Hypercube samples with uniform stratified sampling.
    // Returns an [sample_size x dimension] matrix (rows = samples, cols = dimensions).
    static math::linalg::Matrix2D random(int sample_size, int dimension = 1, int seed = -1) {
        math::linalg::Matrix2D lhs(static_cast<std::size_t>(sample_size),
                                    std::vector<double>(static_cast<std::size_t>(dimension)));
        MersenneTwister rnd_master = make_master(seed);

        for (int col = 0; col < dimension; ++col) {
            // Create bins.
            std::vector<double> bins(static_cast<std::size_t>(sample_size));
            MersenneTwister rnd(static_cast<std::uint32_t>(rnd_master.next()));

            for (int i = 0; i < sample_size; ++i) {
                double delta = rnd.next_double();
                bins[static_cast<std::size_t>(i)] = (i + delta) / sample_size;
            }

            // Shuffle bins.
            shuffle(bins, rnd);

            // Assign to LHS matrix.
            for (int row = 0; row < sample_size; ++row)
                lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                    bins[static_cast<std::size_t>(row)];
        }

        return lhs;
    }

    // Generate Latin Hypercube samples using median bin locations.
    static math::linalg::Matrix2D median(int sample_size, int dimension = 1, int seed = -1) {
        math::linalg::Matrix2D lhs(static_cast<std::size_t>(sample_size),
                                    std::vector<double>(static_cast<std::size_t>(dimension)));
        MersenneTwister rnd_master = make_master(seed);

        for (int col = 0; col < dimension; ++col) {
            // Create median-centered bins.
            std::vector<double> bins(static_cast<std::size_t>(sample_size));
            for (int i = 0; i < sample_size; ++i)
                bins[static_cast<std::size_t>(i)] = (i + 0.5) / sample_size;

            // Shuffle bins.
            MersenneTwister rnd(static_cast<std::uint32_t>(rnd_master.next()));
            shuffle(bins, rnd);

            // Assign to LHS matrix.
            for (int row = 0; row < sample_size; ++row)
                lhs[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] =
                    bins[static_cast<std::size_t>(row)];
        }

        return lhs;
    }

   private:
    static MersenneTwister make_master(int seed) {
        if (seed > 0) return MersenneTwister(static_cast<std::uint32_t>(seed));
        auto ticks = std::chrono::steady_clock::now().time_since_epoch().count();
        return MersenneTwister(static_cast<std::uint32_t>(ticks));
    }

    // In-place Fisher-Yates shuffle (mirrors the C# private static Shuffle).
    static void shuffle(std::vector<double>& array, MersenneTwister& rnd) {
        for (int i = static_cast<int>(array.size()) - 1; i > 0; --i) {
            int j = rnd.next(i + 1);
            std::swap(array[static_cast<std::size_t>(i)], array[static_cast<std::size_t>(j)]);
        }
    }
};

}  // namespace corehydro::numerics::sampling
