// ported from: RMC-BestFit/src/RMC.BestFit/Models/LinkFunctions/BestFitLinkFunctionFactory.cs
//              @ fc28c0c
//
// Factory for creating ILinkFunction instances, supporting both standard Numerics link
// types and the BestFit-specific types (ASinHLink, SESLink, LogSESLink, LogASinHLink,
// CenteredLink, YeoJohnsonLink). The upstream factory dispatches on serialized XElement
// names and falls through to the Numerics LinkFunctionFactory for standard types; per
// the B2 brief only the type dispatch is ported (XML routing skipped), so the C# string
// switch becomes a BestFitLinkFunctionType enum and the fall-through becomes an
// overload taking the Numerics LinkFunctionType. Each creation yields the type's
// default construction (the C# `new XElement("<Name>")` no-attribute case); Centered
// wraps a default IdentityLink inner (the C# no-child-element fallback). Unknown enum
// values throw std::out_of_range (the B1 LinkFunctionFactory convention; C# raises
// NotSupportedException on unknown names).
#pragma once
#include <memory>
#include <stdexcept>

#include "corehydro/models/link_functions/asinh_link.hpp"
#include "corehydro/models/link_functions/centered_link.hpp"
#include "corehydro/models/link_functions/log_asinh_link.hpp"
#include "corehydro/models/link_functions/log_ses_link.hpp"
#include "corehydro/models/link_functions/ses_link.hpp"
#include "corehydro/models/link_functions/yeo_johnson_link.hpp"
#include "corehydro/numerics/functions/i_link_function.hpp"
#include "corehydro/numerics/functions/identity_link.hpp"
#include "corehydro/numerics/functions/link_function_factory.hpp"
#include "corehydro/numerics/functions/link_function_type.hpp"

namespace corehydro::models::link_functions {

// The BestFit-specific link function types (the C# switch cases, in declaration order).
enum class BestFitLinkFunctionType {
    // Centered sinh-arcsinh link for unbounded parameters (ASinHLink).
    ASinH,

    // Skew-exponential-sinh link with tunable asymmetry and fat tails (SESLink).
    SES,

    // Log-space SES link for positive scale parameters (LogSESLink).
    LogSES,

    // Positive-support sinh-arcsinh link on log-relative scale (LogASinHLink).
    LogASinH,

    // Affine centering/scaling wrapper around an inner link (CenteredLink).
    Centered,

    // Yeo-Johnson power-transformation link for bootstrap pivots (YeoJohnsonLink).
    YeoJohnson
};

class BestFitLinkFunctionFactory {
   public:
    // Creates an ILinkFunction instance for a BestFit-specific link function type.
    static std::unique_ptr<numerics::functions::ILinkFunction> create(
        BestFitLinkFunctionType type) {
        switch (type) {
            case BestFitLinkFunctionType::ASinH:
                return std::make_unique<ASinHLink>();
            case BestFitLinkFunctionType::SES:
                return std::make_unique<SESLink>();
            case BestFitLinkFunctionType::LogSES:
                return std::make_unique<LogSESLink>();
            case BestFitLinkFunctionType::LogASinH:
                return std::make_unique<LogASinHLink>();
            case BestFitLinkFunctionType::Centered:
                // The C# XElement path defaults to an IdentityLink inner when no child
                // element is present (null-means-identity), Mu0 = 0, Scale = 1.
                return std::make_unique<CenteredLink>(
                    std::make_unique<numerics::functions::IdentityLink>(), 0.0, 1.0);
            case BestFitLinkFunctionType::YeoJohnson:
                return std::make_unique<YeoJohnsonLink>();
            default:
                throw std::out_of_range("Unknown BestFit link function type.");
        }
    }

    // Creates an ILinkFunction instance for a standard Numerics link function type (the
    // C# default-case fall-through to the Numerics LinkFunctionFactory).
    static std::unique_ptr<numerics::functions::ILinkFunction> create(
        numerics::functions::LinkFunctionType type) {
        return numerics::functions::LinkFunctionFactory::create(type);
    }
};

}  // namespace corehydro::models::link_functions
