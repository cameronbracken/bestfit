// ported from: Numerics/Mathematics/Optimization/Support/ParameterSet.cs @ 2a0357a
//
// A trial optimization parameter set: values + fitness (+ optional weight). C#'s struct
// is a value type whose `Values` array is nullable by convention -- `null` means "not yet
// computed" and is checked at several call sites before any real ParameterSet exists
// (Optimizer::evaluate()'s best-tracking guard; the MCMC layer's initial MAP placeholder,
// a later task). This port represents "unset" as an EMPTY `values` vector rather than a
// pointer/optional: every C# call site that constructs a not-yet-computed placeholder
// passes an explicitly empty array too (`new ParameterSet()` or `new ParameterSet([],
// ...)`), and every real trial parameter set has `values.size() == NumberOfParameters >=
// 1` (Optimizer's ctor rejects `numberOfParameters < 1`), so an empty vector is a direct,
// unambiguous stand-in with no null-vs-empty ambiguity to resolve.
//
// The XElement (de)serialization ctor and ToXElement() are NOT ported -- GUI/XML
// persistence, out of scope for this numerical port (see the task brief).
//
// clone(bool deep): C# arrays are reference types, so Clone(false) shares the SAME
// underlying `Values` array with the original -- a deliberate perf optimization at call
// sites (in the MCMC layer, a later task) that never mutate a ParameterSet's `Values`
// array in place after cloning. `std::vector<double>` has value semantics, so every C++
// copy is unconditionally independent of its source; there is no aliasing mode to mirror.
// `deep` is kept as a parameter (default true, matching the C# default) purely for
// call-site API parity with later ports -- it does not change behavior here.
#pragma once
#include <utility>
#include <vector>

namespace corehydro::numerics::math::optimization {

struct ParameterSet {
    // The trial parameter set values. Empty means "unset" (see file header).
    std::vector<double> values;

    // The objective function result (or fitness) given the trial parameter set.
    double fitness = 0.0;

    // An optional weight given to the parameter set values.
    double weight = 0.0;

    // Constructs an empty parameter set.
    ParameterSet() = default;

    // Constructs a parameter set.
    ParameterSet(std::vector<double> values, double fitness)
        : values(std::move(values)), fitness(fitness) {}

    // Constructs a parameter set.
    ParameterSet(std::vector<double> values, double fitness, double weight)
        : values(std::move(values)), fitness(fitness), weight(weight) {}

    // Returns a clone of the point. `deep` is retained for C# API parity; std::vector's
    // value semantics make the deep/shallow distinction moot in C++ (see file header).
    ParameterSet clone(bool deep = true) const {
        (void)deep;
        return ParameterSet(values, fitness, weight);
    }
};

}  // namespace corehydro::numerics::math::optimization
