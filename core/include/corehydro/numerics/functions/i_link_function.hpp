// ported from: Numerics/Functions/Link Functions/ILinkFunction.cs @ a2c4dbf
//
// Interface for link functions that transform parameters between real-space and
// link-space. A link function h provides a bijective, differentiable mapping from the
// parameter's natural domain to an unconstrained link-space (GLMs, Bayesian MCMC
// estimation, parametric bootstrap). Implementations must satisfy the round-trip
// identity InverseLink(Link(x)) = x on the valid domain and the derivative consistency
// DLink(x) = dLink(x)/dx. ToXElement is dropped (serialization is a desktop concern).
#pragma once

namespace corehydro::numerics::functions {

class ILinkFunction {
   public:
    virtual ~ILinkFunction() = default;

    // Evaluates the link function mapping real-space to link-space: eta = h(x).
    virtual double link(double x) const = 0;

    // Evaluates the inverse link function mapping link-space back to real-space:
    // x = h^-1(eta).
    virtual double inverse_link(double eta) const = 0;

    // Evaluates the derivative of the link function with respect to x: h'(x) = deta/dx.
    virtual double d_link(double x) const = 0;
};

}  // namespace corehydro::numerics::functions
