// ported from: RMC-BestFit/src/RMC.BestFit/Estimation/OptimizationMethod.cs @ fc28c0c
//
// Enumeration of optimization methods available to the BestFit MLE/MAP estimators.
// Verbatim port; member order mirrors the C# source. This header only defines the
// enum -- gating which methods are actually supported by a given estimator happens
// later, in the estimator itself (Phase 4 Task T7), not here.
#pragma once

namespace bestfit::estimation {

enum class OptimizationMethod {
    // The Brent method for single parameter models.
    Brent,

    // The Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimization method.
    BFGS,

    // The Nelder-Mead downhill simplex method.
    NelderMead,

    // The Powell optimization method.
    Powell,

    // The Differential Evolution (DE) global optimization method.
    DifferentialEvolution,

    // The multi-level single linkage (MLSL) global optimization method.
    MultilevelSingleLinkage,
};

}  // namespace bestfit::estimation
