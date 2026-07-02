// ported from: Numerics/Distributions/Multivariate/Base/MultivariateDistributionType.cs @ a2c4dbf
//
// Multivariate distribution type tags (mirrors the C# enum members verbatim).
#pragma once

namespace bestfit::numerics::distributions {

enum class MultivariateDistributionType {
    BivariateEmpiricalDistribution,
    MultivariateNormal,
    Dirichlet,
    Multinomial,
    MultivariateStudentT
};

}  // namespace bestfit::numerics::distributions
