// ported from: RMC-BestFit/src/RMC.BestFit/Models/SpatialExtremes/SpatialCorrelation/CorrelationFunctionType.cs @ fc28c0c
//
// Enumeration of spatial correlation function types. C# member order preserved, so the
// default underlying values match (Exponential=0, PoweredExponential=1, Spherical=2).
#pragma once

namespace corehydro::models::spatial_extremes {

enum class CorrelationFunctionType { Exponential, PoweredExponential, Spherical };

}  // namespace corehydro::models::spatial_extremes
