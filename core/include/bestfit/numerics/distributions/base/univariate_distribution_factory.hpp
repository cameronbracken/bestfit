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
#include "bestfit/numerics/distributions/bernoulli.hpp"
#include "bestfit/numerics/distributions/cauchy.hpp"
#include "bestfit/numerics/distributions/deterministic.hpp"
#include "bestfit/numerics/distributions/exponential.hpp"
#include "bestfit/numerics/distributions/triangular.hpp"
#include "bestfit/numerics/distributions/generalized_extreme_value.hpp"
#include "bestfit/numerics/distributions/generalized_logistic.hpp"
#include "bestfit/numerics/distributions/generalized_pareto.hpp"
#include "bestfit/numerics/distributions/geometric.hpp"
#include "bestfit/numerics/distributions/chi_squared.hpp"
#include "bestfit/numerics/distributions/gamma_distribution.hpp"
#include "bestfit/numerics/distributions/gumbel.hpp"
#include "bestfit/numerics/distributions/inverse_chi_squared.hpp"
#include "bestfit/numerics/distributions/inverse_gamma.hpp"
#include "bestfit/numerics/distributions/ln_normal.hpp"
#include "bestfit/numerics/distributions/log_normal.hpp"
#include "bestfit/numerics/distributions/logistic.hpp"
#include "bestfit/numerics/distributions/normal.hpp"
#include "bestfit/numerics/distributions/pareto.hpp"
#include "bestfit/numerics/distributions/log_pearson_type_iii.hpp"
#include "bestfit/numerics/distributions/pearson_type_iii.hpp"
#include "bestfit/numerics/distributions/poisson.hpp"
#include "bestfit/numerics/distributions/rayleigh.hpp"
#include "bestfit/numerics/distributions/truncated_normal.hpp"
#include "bestfit/numerics/distributions/uniform.hpp"
#include "bestfit/numerics/distributions/uniform_discrete.hpp"
#include "bestfit/numerics/distributions/weibull.hpp"

namespace bestfit::numerics::distributions {

inline std::unique_ptr<UnivariateDistributionBase> create_distribution(
    UnivariateDistributionType type) {
    switch (type) {
        case UnivariateDistributionType::Bernoulli:
            return std::make_unique<Bernoulli>();
        case UnivariateDistributionType::Cauchy:
            return std::make_unique<Cauchy>();
        case UnivariateDistributionType::Deterministic:
            return std::make_unique<Deterministic>();
        case UnivariateDistributionType::Exponential:
            return std::make_unique<Exponential>();
        case UnivariateDistributionType::GeneralizedExtremeValue:
            return std::make_unique<GeneralizedExtremeValue>();
        case UnivariateDistributionType::GeneralizedLogistic:
            return std::make_unique<GeneralizedLogistic>();
        case UnivariateDistributionType::GeneralizedPareto:
            return std::make_unique<GeneralizedPareto>();
        case UnivariateDistributionType::Geometric:
            return std::make_unique<Geometric>();
        case UnivariateDistributionType::ChiSquared:
            return std::make_unique<ChiSquared>();
        case UnivariateDistributionType::GammaDistribution:
            return std::make_unique<GammaDistribution>();
        case UnivariateDistributionType::Gumbel:
            return std::make_unique<Gumbel>();
        case UnivariateDistributionType::InverseChiSquared:
            return std::make_unique<InverseChiSquared>();
        case UnivariateDistributionType::InverseGamma:
            return std::make_unique<InverseGamma>();
        case UnivariateDistributionType::LogPearsonTypeIII:
            return std::make_unique<LogPearsonTypeIII>();
        case UnivariateDistributionType::LnNormal:
            return std::make_unique<LnNormal>();
        case UnivariateDistributionType::LogNormal:
            return std::make_unique<LogNormal>();
        case UnivariateDistributionType::Logistic:
            return std::make_unique<Logistic>();
        case UnivariateDistributionType::Normal:
            return std::make_unique<Normal>();
        case UnivariateDistributionType::Pareto:
            return std::make_unique<Pareto>();
        case UnivariateDistributionType::PearsonTypeIII:
            return std::make_unique<PearsonTypeIII>();
        case UnivariateDistributionType::Poisson:
            return std::make_unique<Poisson>();
        case UnivariateDistributionType::Rayleigh:
            return std::make_unique<Rayleigh>();
        case UnivariateDistributionType::Triangular:
            return std::make_unique<Triangular>();
        case UnivariateDistributionType::TruncatedNormal:
            return std::make_unique<TruncatedNormal>();
        case UnivariateDistributionType::Uniform:
            return std::make_unique<Uniform>();
        case UnivariateDistributionType::UniformDiscrete:
            return std::make_unique<UniformDiscrete>();
        case UnivariateDistributionType::Weibull:
            return std::make_unique<Weibull>();
        default:
            throw std::invalid_argument("univariate distribution type not yet ported");
    }
}

// Construct from the C# type name (the value stored in fixtures' "target" field).
inline std::unique_ptr<UnivariateDistributionBase> create_distribution(const std::string& name) {
    if (name == "Bernoulli") return create_distribution(UnivariateDistributionType::Bernoulli);
    if (name == "Cauchy") return create_distribution(UnivariateDistributionType::Cauchy);
    if (name == "Deterministic") return create_distribution(UnivariateDistributionType::Deterministic);
    if (name == "Exponential") return create_distribution(UnivariateDistributionType::Exponential);
    if (name == "GeneralizedExtremeValue")
        return create_distribution(UnivariateDistributionType::GeneralizedExtremeValue);
    if (name == "GeneralizedLogistic")
        return create_distribution(UnivariateDistributionType::GeneralizedLogistic);
    if (name == "GeneralizedPareto")
        return create_distribution(UnivariateDistributionType::GeneralizedPareto);
    if (name == "Geometric") return create_distribution(UnivariateDistributionType::Geometric);
    if (name == "ChiSquared") return create_distribution(UnivariateDistributionType::ChiSquared);
    if (name == "GammaDistribution") return create_distribution(UnivariateDistributionType::GammaDistribution);
    if (name == "Gumbel") return create_distribution(UnivariateDistributionType::Gumbel);
    if (name == "InverseChiSquared") return create_distribution(UnivariateDistributionType::InverseChiSquared);
    if (name == "InverseGamma") return create_distribution(UnivariateDistributionType::InverseGamma);
    if (name == "LogPearsonTypeIII")
        return create_distribution(UnivariateDistributionType::LogPearsonTypeIII);
    if (name == "LnNormal") return create_distribution(UnivariateDistributionType::LnNormal);
    if (name == "LogNormal") return create_distribution(UnivariateDistributionType::LogNormal);
    if (name == "Logistic") return create_distribution(UnivariateDistributionType::Logistic);
    if (name == "Normal") return create_distribution(UnivariateDistributionType::Normal);
    if (name == "Pareto") return create_distribution(UnivariateDistributionType::Pareto);
    if (name == "PearsonTypeIII") return create_distribution(UnivariateDistributionType::PearsonTypeIII);
    if (name == "Poisson") return create_distribution(UnivariateDistributionType::Poisson);
    if (name == "Rayleigh") return create_distribution(UnivariateDistributionType::Rayleigh);
    if (name == "Triangular") return create_distribution(UnivariateDistributionType::Triangular);
    if (name == "TruncatedNormal") return create_distribution(UnivariateDistributionType::TruncatedNormal);
    if (name == "Uniform") return create_distribution(UnivariateDistributionType::Uniform);
    if (name == "UniformDiscrete")
        return create_distribution(UnivariateDistributionType::UniformDiscrete);
    if (name == "Weibull") return create_distribution(UnivariateDistributionType::Weibull);
    throw std::invalid_argument("unknown distribution name: " + name);
}

}  // namespace bestfit::numerics::distributions
