// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/CenteredLink.cs @ c2e6192
//
// Affine wrapper that centers and scales any ILinkFunction:
//   z = (x - mu0)/s, then eta = Inner.Link(z).
//   Inverse: x = mu0 + s * Inner.InverseLink(eta).
//   Derivative: deta/dx = Inner.DLink(z) * (1/s).
// Allows recentering link computations around a focal point (e.g., parent estimate)
// without recomputing the inner link's tuning parameters.
//
// Ownership: the C# `Inner` is a get-private-set reference; the C++ port takes SOLE
// OWNERSHIP via unique_ptr moved in at construction (the ModelParameter precedent),
// exposed as a raw observer pointer. A null inner throws (C# ArgumentNullException ->
// std::invalid_argument). The C# XElement constructor's null-means-IdentityLink
// fallback survives only through BestFitLinkFunctionFactory::create(Centered), which
// builds a CenteredLink around a default IdentityLink; the XElement constructor and
// ToXElement themselves are dropped (serialization is a desktop concern). NOTE: only
// the constructor clamps scale to >= 1e-12; the set_scale setter is unvalidated,
// exactly like the C# auto-property.
#pragma once
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <utility>

#include "corehydro/numerics/functions/i_link_function.hpp"

namespace corehydro::models::link_functions {

class CenteredLink final : public numerics::functions::ILinkFunction {
   public:
    // Constructs a centered link wrapper around an inner link function (must not be
    // null; throws std::invalid_argument, the C# ArgumentNullException).
    CenteredLink(std::unique_ptr<numerics::functions::ILinkFunction> inner, double mu0,
                 double scale = 1.0)
        : inner_(std::move(inner)), mu0_(mu0), scale_(std::max(1e-12, scale)) {
        if (!inner_) throw std::invalid_argument("inner must not be null.");
    }

    // The inner link being centered/scaled (e.g., SESLink). Owned; C# get-private-set.
    const numerics::functions::ILinkFunction* inner() const { return inner_.get(); }

    // Center mu0 in x-units (e.g., parent estimate).
    double mu0() const { return mu0_; }
    void set_mu0(double value) { mu0_ = value; }

    // Scale s > 0 in x-units (sets how quickly the link departs from linearity).
    double scale() const { return scale_; }
    void set_scale(double value) { scale_ = value; }

    double link(double x) const override {
        double z = (x - mu0_) / scale_;
        return inner_->link(z);
    }

    double inverse_link(double eta) const override {
        double z = inner_->inverse_link(eta);
        return mu0_ + scale_ * z;
    }

    double d_link(double x) const override {
        // deta/dx = (deta/dz) * (dz/dx) = Inner.DLink(z) * (1/Scale)
        double z = (x - mu0_) / scale_;
        return inner_->d_link(z) / scale_;
    }

   private:
    std::unique_ptr<numerics::functions::ILinkFunction> inner_;
    double mu0_ = 0.0;
    double scale_ = 1.0;
};

}  // namespace corehydro::models::link_functions
