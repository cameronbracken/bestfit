// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/YeoJohnsonLink.cs @ fc28c0c
//
// A Yeo-Johnson power-transformation link function for bootstrap pivot methods. The
// power parameter lambda can be estimated from bootstrap parameter samples by maximum
// likelihood, making the transformed values approximately normal.
// Reference: Yeo, I.-K. and Johnson, R.A. (2000). A new family of power transformations
// to improve normality or symmetry. Biometrika, 87(4), 954-959.
//
// COLLAPSE NOTE: upstream RMC.BestFit historically carried its own near-duplicate copy
// of the Numerics YeoJohnson transform (a CS0104 name-collision workaround); at fc28c0c
// the class calls Numerics.Data.Statistics.YeoJohnson directly, and this port likewise
// collapses onto the B1 machinery `bestfit::numerics::data::YeoJohnson`
// (numerics/data/yeo_johnson.hpp) for Transform / InverseTransform / FitLambda. DLink
// is the C# inline two-branch power formula, kept inline per the C# file structure (it
// coincides with YeoJohnson::derivative except for that helper's |lambda| > 5 NaN
// guard, which the C# DLink does not have).
//
// NAME COLLISION: this class deliberately coexists with the Numerics-layer
// `bestfit::numerics::functions::YeoJohnsonLink` (numerics/functions/
// yeo_johnson_link.hpp). They differ per their C# sources: this models-layer variant
// keeps a mutable, UNVALIDATED Lambda (default 1.0 = identity) and a values constructor
// that does not require finite inputs, while the Numerics variant validates lambda into
// [-5, 5] and is immutable. Keep includes/usings unambiguous when both are visible.
//
// The XElement constructor / ToXElement are dropped (serialization is a desktop
// concern). The C# ArgumentNullException on a null values array has no analog (a C++
// std::vector cannot be null); ArgumentException -> std::invalid_argument.
#pragma once
#include <cmath>
#include <stdexcept>
#include <vector>

#include "bestfit/numerics/data/yeo_johnson.hpp"
#include "bestfit/numerics/functions/i_link_function.hpp"

namespace bestfit::models::link_functions {

class YeoJohnsonLink : public numerics::functions::ILinkFunction {
   public:
    // Constructs a new YeoJohnsonLink with a default identity transform (lambda = 1).
    YeoJohnsonLink() = default;

    // Constructs a new YeoJohnsonLink with a specified power parameter.
    explicit YeoJohnsonLink(double lambda) : lambda_(lambda) {}

    // Constructs a new YeoJohnsonLink by fitting the power parameter from a sample of
    // values (typically the bootstrap parameter estimates for a single parameter across
    // all B replicates) via maximum profile log-likelihood.
    explicit YeoJohnsonLink(const std::vector<double>& values) {
        if (values.size() < 2)
            throw std::invalid_argument("At least 2 values are required to fit lambda.");

        lambda_ = numerics::data::YeoJohnson::fit_lambda(values);
    }

    // The fitted Yeo-Johnson power parameter.
    double lambda() const { return lambda_; }
    void set_lambda(double value) { lambda_ = value; }

    // Applies the link function h(x) = YeoJohnson.Transform(x, lambda), mapping
    // real-space to link-space.
    double link(double x) const override {
        return numerics::data::YeoJohnson::transform(x, lambda_);
    }

    // Applies the inverse link h^-1(eta), mapping link-space back to real-space.
    double inverse_link(double eta) const override {
        return numerics::data::YeoJohnson::inverse_transform(eta, lambda_);
    }

    // Computes the derivative of the link function deta/dx at point x.
    // For x >= 0: deta/dx = (x + 1)^(lambda - 1).
    // For x < 0:  deta/dx = (-x + 1)^(1 - lambda).
    // Both cases reduce to 1 when lambda = 1 (identity transform).
    double d_link(double x) const override {
        return x >= 0 ? std::pow(x + 1.0, lambda_ - 1.0) : std::pow(-x + 1.0, 1.0 - lambda_);
    }

   private:
    double lambda_ = 1.0;
};

}  // namespace bestfit::models::link_functions
