// ported from: Numerics/Distributions/Univariate/Base/UnivariateDistributionFactory.cs @ <pending-sha>
//
// Constructs a default-parameterized distribution from its type. The C# if/else chain
// becomes a switch; the XElement overload is dropped (serialization is a desktop concern).
// Only ported distributions have a case -- requesting an unported type throws rather than
// silently returning a Deterministic placeholder, so gaps surface immediately.
#pragma once
#include <memory>
#include <stdexcept>
#include <string>

#include "bestfit/numerics/distributions/base/univariate_distribution_base.hpp"
#include "bestfit/numerics/distributions/base/univariate_distribution_type.hpp"
#include "bestfit/numerics/distributions/cauchy.hpp"
#include "bestfit/numerics/distributions/exponential.hpp"
#include "bestfit/numerics/distributions/triangular.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/gumbel.hpp"
#include "bestfit/numerics/distributions/logistic.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/pareto.hpp"
#include "bestfit/numerics/distributions/rayleigh.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"

namespace bestfit::numerics::distributions {

inline std::unique_ptr<UnivariateDistributionBase> create_distribution(
    UnivariateDistributionType type) {
    switch (type) {
        case UnivariateDistributionType::Cauchy:
            return std::make_unique<Cauchy>();
        case UnivariateDistributionType::Exponential:
            return std::make_unique<Exponential>();
        case UnivariateDistributionType::GeneralizedExtremeValue:
            return std::make_unique<GeneralizedExtremeValue>();
        case UnivariateDistributionType::Gumbel:
            return std::make_unique<Gumbel>();
        case UnivariateDistributionType::Logistic:
            return std::make_unique<Logistic>();
        case UnivariateDistributionType::Normal:
            return std::make_unique<Normal>();
        case UnivariateDistributionType::Pareto:
            return std::make_unique<Pareto>();
        case UnivariateDistributionType::Rayleigh:
            return std::make_unique<Rayleigh>();
        case UnivariateDistributionType::Triangular:
            return std::make_unique<Triangular>();
        case UnivariateDistributionType::Uniform:
            return std::make_unique<Uniform>();
        default:
            throw std::invalid_argument("univariate distribution type not yet ported");
    }
}

// Construct from the C# type name (the value stored in fixtures' "target" field).
inline std::unique_ptr<UnivariateDistributionBase> create_distribution(const std::string& name) {
    if (name == "Cauchy") return create_distribution(UnivariateDistributionType::Cauchy);
    if (name == "Exponential") return create_distribution(UnivariateDistributionType::Exponential);
    if (name == "GeneralizedExtremeValue")
        return create_distribution(UnivariateDistributionType::GeneralizedExtremeValue);
    if (name == "Gumbel") return create_distribution(UnivariateDistributionType::Gumbel);
    if (name == "Logistic") return create_distribution(UnivariateDistributionType::Logistic);
    if (name == "Normal") return create_distribution(UnivariateDistributionType::Normal);
    if (name == "Pareto") return create_distribution(UnivariateDistributionType::Pareto);
    if (name == "Rayleigh") return create_distribution(UnivariateDistributionType::Rayleigh);
    if (name == "Triangular") return create_distribution(UnivariateDistributionType::Triangular);
    if (name == "Uniform") return create_distribution(UnivariateDistributionType::Uniform);
    throw std::invalid_argument("unknown distribution name: " + name);
}

}  // namespace bestfit::numerics::distributions
