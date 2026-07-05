// ported from: Numerics/Functions/Link Functions/LinkFunctionFactory.cs @ a2c4dbf
//
// Factory for creating ILinkFunction instances from LinkFunctionType enum values. The
// XElement overload (CreateFromXElement) is dropped (serialization is a desktop
// concern). Returns an owned pointer, matching the distribution/copula factory pattern.
// C# ArgumentOutOfRangeException -> std::out_of_range.
#pragma once
#include <memory>
#include <stdexcept>

#include "bestfit/numerics/functions/complementary_log_log_link.hpp"
#include "bestfit/numerics/functions/fisher_z_link.hpp"
#include "bestfit/numerics/functions/i_link_function.hpp"
#include "bestfit/numerics/functions/identity_link.hpp"
#include "bestfit/numerics/functions/link_function_type.hpp"
#include "bestfit/numerics/functions/log_link.hpp"
#include "bestfit/numerics/functions/logit_link.hpp"
#include "bestfit/numerics/functions/probit_link.hpp"
#include "bestfit/numerics/functions/yeo_johnson_link.hpp"

namespace bestfit::numerics::functions {

class LinkFunctionFactory {
   public:
    // Creates an ILinkFunction instance corresponding to the specified link function type.
    static std::unique_ptr<ILinkFunction> create(LinkFunctionType type) {
        switch (type) {
            case LinkFunctionType::Identity:
                return std::make_unique<IdentityLink>();
            case LinkFunctionType::Log:
                return std::make_unique<LogLink>();
            case LinkFunctionType::Logit:
                return std::make_unique<LogitLink>();
            case LinkFunctionType::Probit:
                return std::make_unique<ProbitLink>();
            case LinkFunctionType::ComplementaryLogLog:
                return std::make_unique<ComplementaryLogLogLink>();
            case LinkFunctionType::YeoJohnson:
                return std::make_unique<YeoJohnsonLink>();
            case LinkFunctionType::FisherZ:
                return std::make_unique<FisherZLink>();
            default:
                throw std::out_of_range("Unknown link function type.");
        }
    }
};

}  // namespace bestfit::numerics::functions
