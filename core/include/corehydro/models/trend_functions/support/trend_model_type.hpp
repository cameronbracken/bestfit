// ported from: RMC-BestFit/src/RMC.BestFit/Models/TrendFunctions/Support/TrendModelType.cs @ c2e6192
//
// Enumeration of supported trend model types used to represent how distribution
// parameters vary with time. Member names and order mirror the C# enum exactly.
#pragma once

namespace corehydro::models::trend_functions {

enum class TrendModelType {
    // Constant in time.
    Constant,
    // Cubic polynomial in time.
    Cubic,
    // Exponential function in time.
    Exponential,
    // Linear function in time.
    Linear,
    // Logistic (sigmoid) function in time.
    Logistic,
    // Power-law function in time.
    Power,
    // Quadratic polynomial in time.
    Quadratic,
    // Reciprocal function in time.
    Reciprocal,
    // Sinusoidal function in time.
    Sinusoidal,
    // Step function with a single change point.
    StepFunction,
    // General linear function with arbitrary covariates.
    // Used for spatial regression surfaces and covariate modeling.
    GeneralLinear
};

}  // namespace corehydro::models::trend_functions
