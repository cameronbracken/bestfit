// ported from: Numerics/Distributions/Univariate/Base/UnivariateDistributionType.cs @ a2c4dbf
//
// Enumeration of every univariate distribution. The full set is mirrored verbatim
// (even those not yet ported) so the factory switch and any persisted type names line
// up one-for-one with the C# source; unported entries simply have no factory case yet.
#pragma once

namespace bestfit::numerics::distributions {

enum class UnivariateDistributionType {
    ChiSquared,
    Bernoulli,
    Beta,
    Binomial,
    Cauchy,
    CompetingRisks,
    Deterministic,
    Empirical,
    Exponential,
    GammaDistribution,
    GeneralizedBeta,
    GeneralizedExtremeValue,
    GeneralizedLogistic,
    GeneralizedNormal,
    GeneralizedPareto,
    Geometric,
    Gumbel,
    InverseChiSquared,
    InverseGamma,
    KappaFour,
    KernelDensity,
    LnNormal,
    Logistic,
    LogNormal,
    LogPearsonTypeIII,
    Mixture,
    NoncentralT,
    Normal,
    Pareto,
    PearsonTypeIII,
    Pert,
    PertPercentile,
    PertPercentileZ,
    Poisson,
    Rayleigh,
    StudentT,
    Triangular,
    TruncatedNormal,
    Uniform,
    UniformDiscrete,
    UserDefined,
    VonMises,
    Weibull
};

}  // namespace bestfit::numerics::distributions
