// ported from: Numerics/Mathematics/Optimization/Support/OptimizationStatus.cs @ 2a0357a
//
// Enumeration of optimization statuses. Verbatim port; member order mirrors the C# source.
#pragma once

namespace corehydro::numerics::math::optimization {

enum class OptimizationStatus {
    // Optimization has not been performed yet.
    None,

    // The optimization method ended successfully.
    Success,

    // The optimization method was stopped because the maximum number of iterations was
    // reached.
    MaximumIterationsReached,

    // The optimization method was stopped because the maximum number of objective
    // function evaluations was reached.
    MaximumFunctionEvaluationsReached,

    // The optimization method was stopped due to internal failure.
    Failure,
};

}  // namespace corehydro::numerics::math::optimization
