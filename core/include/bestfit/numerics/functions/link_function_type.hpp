// ported from: Numerics/Functions/Link Functions/LinkFunctionType.cs @ a2c4dbf
//
// Enumeration of standard link function types for generalized linear models. Each value
// corresponds to a canonical link for a specific GLM family; use
// LinkFunctionFactory::create to obtain an ILinkFunction instance from an enum value.
// Value order mirrors the C# declaration (YeoJohnson before FisherZ).
#pragma once

namespace bestfit::numerics::functions {

enum class LinkFunctionType {
    // Identity link: eta = x. Canonical link for the Normal (Gaussian) family.
    Identity,

    // Log link: eta = log(x). Canonical link for the Poisson and Exponential families.
    Log,

    // Logit link: eta = log(x / (1 - x)). Canonical link for the Binomial family.
    Logit,

    // Probit link: eta = Phi^-1(x). Alternative link for the Binomial family using the
    // standard normal quantile function.
    Probit,

    // Complementary log-log link: eta = log(-log(1 - x)). Used for asymmetric binary
    // response models.
    ComplementaryLogLog,

    // Yeo-Johnson power-transformation link function for real-valued skew or shape
    // parameters.
    YeoJohnson,

    // Fisher z link: eta = atanh(x), commonly used for correlation parameters.
    FisherZ
};

}  // namespace bestfit::numerics::functions
