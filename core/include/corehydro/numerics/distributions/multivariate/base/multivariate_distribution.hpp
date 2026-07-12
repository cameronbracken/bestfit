// ported from: Numerics/Distributions/Multivariate/Base/IMultivariateDistribution.cs @ a2c4dbf
//           +  Numerics/Distributions/Multivariate/Base/MultivariateDistribution.cs @ a2c4dbf
//
// Abstract base for every multivariate distribution. Folds IMultivariateDistribution's
// members directly into the base -- the Phase 1 pattern (interfaces fold into the
// abstract base rather than being ported as a separate file), applicable here because
// every C# multivariate distribution class derives from
// `MultivariateDistribution : IMultivariateDistribution` and nothing implements only the
// interface.
//
// Declares the distribution-core surface as pure virtuals (dimension, type, display
// names, parameter validity, PDF/CDF) and promotes the shared concrete helpers (LogPDF,
// LogCDF, CCDF, LogCCDF) with the exact C# NaN/Inf guard behavior. Unlike
// UnivariateDistributionBase, `parameters_valid()` here is a pure virtual (not a cached
// protected field) -- Dirichlet/Multinomial compute it live from their parameter arrays
// on every call, matching the C# `ParametersValid` getters.
#pragma once
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "corehydro/numerics/distributions/multivariate/base/multivariate_distribution_type.hpp"

namespace corehydro::numerics::distributions {

class MultivariateDistribution {
   public:
    virtual ~MultivariateDistribution() = default;

    // --- Identity / parameters ---
    virtual int dimension() const = 0;
    virtual MultivariateDistributionType type() const = 0;
    virtual std::string display_name() const = 0;
    virtual std::string short_display_name() const = 0;
    virtual bool parameters_valid() const = 0;

    // --- Distribution functions ---
    virtual double pdf(const std::vector<double>& x) const = 0;
    virtual double cdf(const std::vector<double>& x) const = 0;

    virtual double log_pdf(const std::vector<double>& x) const {
        double f = pdf(x);
        if (std::isnan(f) || std::isinf(f) || f <= 0.0) return -kInf;
        return std::log(f);
    }

    virtual double log_cdf(const std::vector<double>& x) const {
        double F = cdf(x);
        if (std::isnan(F) || std::isinf(F) || F <= 0.0) return -kInf;
        return std::log(F);
    }

    virtual double ccdf(const std::vector<double>& x) const { return 1.0 - cdf(x); }

    virtual double log_ccdf(const std::vector<double>& x) const {
        double cF = ccdf(x);
        if (std::isnan(cF) || std::isinf(cF) || cF <= 0.0) return -kInf;
        return std::log(cF);
    }

    virtual std::unique_ptr<MultivariateDistribution> clone() const = 0;

   protected:
    static constexpr double kInf = std::numeric_limits<double>::infinity();
};

}  // namespace corehydro::numerics::distributions
