// ported from: Numerics/Utilities/ExtensionMethods.cs @ a2c4dbf
//
// C# extends `System.Random`/arrays/`Vector`/`Matrix` with instance-style extension
// methods; C++ has no such mechanism, so each ported member becomes a free function
// taking the receiver as its first parameter. This header ports only the three random-
// sampling helpers the MCMC/Bootstrap prerequisites need: `next_doubles(rng, n)`,
// `next_doubles(rng, n, dim)`, and `next_integers(rng, n)` (the `Random`-region
// overloads; `GetRow` is not ported -- neither of the two `NextDoubles` overloads calls
// it, unlike the brief's initial guess).
//
// `next_doubles(rng, n, dim)` transcribes a specific quirk exactly (source ~lines
// 130-157): rather than drawing n*dim values off ONE stream, it constructs `dim`
// independent sub-`MersenneTwister`s -- each seeded by a single `rng.next()` draw off the
// PARENT generator, taken in dimension order -- and fills column `i` by advancing that
// sub-generator's own stream `n` times. This sub-stream-per-column pattern must stay
// bit-exact: a later task (SNIS) depends on reproducing it against the real C# output.
//
// Omitted (no ported caller needs them; the C# source has many more extension-method
// regions -- Enum, Double, 1-D/2-D Array, Vector, Matrix -- covering `Apply`/`Map`/
// `Subset`/`RandomSubset`/`Fill`/`GetColumn`/`SetRow`/`SetColumn`/etc.): every
// `ExtensionMethods.cs` member outside the three ported below.
#pragma once
#include <cstdint>
#include <vector>

#include "bestfit/numerics/sampling/mersenne_twister.hpp"

namespace bestfit::numerics::utilities {

// Returns an array of `length` random doubles (matches C# `NextDoubles(this Random,
// int length)`).
inline std::vector<double> next_doubles(sampling::MersenneTwister& random, int length) {
    std::vector<double> values(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) values[static_cast<std::size_t>(i)] = random.next_double();
    return values;
}

// Returns a 2-D array of random doubles shaped [length][dimension] (row = sample index,
// column = dimension index -- matches C#'s `values[j, i]` indexing, where `i` is the
// dimension/outer loop and `j` the sample/inner loop). See the file header comment for
// the sub-MersenneTwister-per-column quirk this transcribes exactly (matches C#
// `NextDoubles(this Random, int length, int dimension)`).
inline std::vector<std::vector<double>> next_doubles(sampling::MersenneTwister& random, int length,
                                                       int dimension) {
    std::vector<std::vector<double>> values(
        static_cast<std::size_t>(length), std::vector<double>(static_cast<std::size_t>(dimension)));
    for (int i = 0; i < dimension; ++i) {
        sampling::MersenneTwister sub(static_cast<std::uint32_t>(random.next()));
        for (int j = 0; j < length; ++j)
            values[static_cast<std::size_t>(j)][static_cast<std::size_t>(i)] = sub.next_double();
    }
    return values;
}

// Returns an array of `length` random integers (matches C# `NextIntegers(this Random,
// int length)`, which draws each element via the parameterless `Next()`).
inline std::vector<int> next_integers(sampling::MersenneTwister& random, int length) {
    std::vector<int> values(static_cast<std::size_t>(length));
    for (int i = 0; i < length; ++i) values[static_cast<std::size_t>(i)] = random.next();
    return values;
}

}  // namespace bestfit::numerics::utilities
