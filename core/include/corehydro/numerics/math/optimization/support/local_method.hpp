// ported from: Numerics/Mathematics/Optimization/Support/LocalMethod.cs @ 2a0357a
//
// The enumeration of local optimization methods for use in global optimizers. All five
// C# members are ported in the C# declaration order; MLSL's GetLocalOptimizer only
// constructs the BFGS / NelderMead / Powell branches (ADAM and GradientDescent throw
// "Unsupported local method" there, exactly as upstream).
#pragma once

namespace corehydro::numerics::math::optimization {

enum class LocalMethod {
    // The Adaptive Movement (Adam) optimization algorithm.
    ADAM,
    // The Broyden-Fletcher-Goldfarb-Shanno (BFGS) algorithm.
    BFGS,
    // The Gradient Descent algorithm.
    GradientDescent,
    // The Nelder-Mead downhill simplex algorithm.
    NelderMead,
    // The Powell optimization algorithm.
    Powell
};

}  // namespace corehydro::numerics::math::optimization
