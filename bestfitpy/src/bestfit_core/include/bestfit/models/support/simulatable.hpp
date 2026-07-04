// ported from: RMC-BestFit/src/RMC.BestFit/Models/Support/ISimulatable.cs @ fc28c0c
//
// Capability interface for models that can generate simulated data. Kept separate from
// ModelBase because not every model that defines a likelihood can simulate, and the output
// type varies by model (double[] for univariate models, double[,] for bivariate, ...).
//
// The C# generic interface `ISimulatable<TData>` ports as a TEMPLATE mixin (one abstract
// class template, instantiated per TData) rather than a plain abstract class per data type:
// that is the direct C++ analogue of a C# generic interface, and it matches how this
// codebase already expresses C# capability interfaces as abstract mixins (see
// numerics/distributions/base/i_estimation.hpp). Later model classes inherit e.g.
// `ISimulatable<std::vector<double>>` (C# `ISimulatable<double[]>`, the instantiation
// UnivariateDistribution uses); a future bivariate model would inherit an
// `ISimulatable<Matrix>`-style instantiation.
//
// `generate_random_values` is const, matching the precedent set by
// UnivariateDistributionBase::generate_random_values (same C# signature on the Numerics
// base): generating samples reads the model's current parameter values, it does not
// change them.
#pragma once

namespace bestfit::models {

template <typename TData>
class ISimulatable {
   public:
    virtual ~ISimulatable() = default;

    // Generates random samples from the model using its current parameter values.
    // `seed` <= 0 means clock-seeded (non-reproducible); `seed` > 0 seeds the PRNG
    // deterministically (C# `GenerateRandomValues(int sampleSize, int seed = -1)`).
    // Implementations throw std::out_of_range / std::runtime_error where the C#
    // documents ArgumentOutOfRangeException (sampleSize < 1) / InvalidOperationException
    // (model not in a valid state to simulate), per the model-layer exception mapping.
    virtual TData generate_random_values(int sample_size, int seed = -1) const = 0;
};

}  // namespace bestfit::models
